"""Independent frozen LLM baseline. No research-index imports or SQLite access."""
import os
import re
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx
from .settings import load

app = FastAPI(title="JARVIS LLM Baseline", version="0.3.2")

class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    profile: str = Field(default="fast", pattern="^(fast|deep)$")


def public_content(raw: str) -> str:
    """Never expose explicitly delimited/unfinished model reasoning."""
    raw = raw or ""
    if "<think>" in raw and "</think>" not in raw:
        return ""
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()

@app.get("/health")
async def health():
    cfg = load()
    models = []
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            response = await client.get(os.getenv("OLLAMA_URL", "http://ollama:11434") + "/api/tags")
            response.raise_for_status()
            models = [x.get("name") for x in response.json().get("models", [])]
    except httpx.HTTPError:
        pass
    return {"status":"ok", "version":"0.3.2", "module":"llm", "profiles":cfg["profiles"],
            "models_downloaded":models, "index_access":False}

@app.post("/chat")
async def chat(req: ChatIn):
    cfg = load()
    profile = cfg["profiles"][req.profile]
    model = profile["model"]
    payload = {"model": model,
               "messages": [{"role":"system","content":cfg["system_prompt"]},
                            {"role":"user","content":req.text}],
               "stream": False,
               "think": bool(profile["think"]),
               "keep_alive": "30m",
               "options": {k:profile[k] for k in ("temperature","num_predict","num_ctx")}}
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("OLLAMA_TIMEOUT", "240"))) as client:
            response = await client.post(os.getenv("OLLAMA_URL","http://ollama:11434")+"/api/chat",json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(503, f"Model {model} is missing; pull it inside Ollama container")
        raise HTTPException(502, f"Ollama status {e.response.status_code}")
    except httpx.HTTPError as e:
        raise HTTPException(503, f"Ollama unavailable: {type(e).__name__}")
    data = response.json()
    answer = public_content(data.get("message", {}).get("content", ""))
    if not answer:
        answer = "Модель не сформировала окончательный ответ. Попробуй другой профиль."
    return {"answer":answer, "profile":req.profile,"model":model, "route":"llm_"+req.profile,
            "tokens":int(data.get("eval_count") or 0), "prompt_tokens":int(data.get("prompt_eval_count") or 0),
            "timings_ms":{
                "total_request":round((time.perf_counter()-start)*1000,2),
                "model_load":round((data.get("load_duration") or 0)/1e6,2),
                "prompt_eval":round((data.get("prompt_eval_duration") or 0)/1e6,2),
                "generation":round((data.get("eval_duration") or 0)/1e6,2)},
            "executed":False}
