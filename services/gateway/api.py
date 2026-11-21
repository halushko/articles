"""Experiment gateway: independent controls for LLM baseline and novel index."""
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import httpx

LLM_URL = os.getenv("LLM_URL", "http://llm:8000")
RESEARCH_URL = os.getenv("RESEARCH_URL", "http://research:8000")
TOKEN = os.getenv("JARVIS_TOKEN", "")
HOST_BIND = os.getenv("HOST_BIND", "127.0.0.1")

@asynccontextmanager
async def lifespan(app):
    if HOST_BIND not in {"127.0.0.1", "localhost", "::1"} and len(TOKEN) < 16:
        raise RuntimeError("Non-loopback access needs JARVIS_TOKEN length >= 16")
    yield

app = FastAPI(title="JARVIS Research Workbench", version="0.3.2", lifespan=lifespan, docs_url=None)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parents[2]/"web")), name="static")

class ChatIn(BaseModel):
    text: str = Field(min_length=1,max_length=3000)
    experiment: str = Field(default="hybrid", pattern="^(hybrid|research_only|llm_only)$")
    profile: str = Field(default="fast",pattern="^(fast|deep)$")
    ablation: str = Field(default="combined",pattern="^(exact|online|combined)$")

class LearnIn(BaseModel):
    phrase: str = Field(min_length=1,max_length=500)
    target_id: str = Field(min_length=1,max_length=100)
    negative_ids: list[str] = Field(default_factory=list,max_length=20)

class ConceptIn(BaseModel):
    name: str = Field(min_length=1,max_length=150)
    kind: str = Field(default="other",max_length=60)
    domain: str = Field(default="general",max_length=120)
    description: str = Field(default="",max_length=400)
    aliases: list[str] = Field(default_factory=list,max_length=20)

class FeedbackIn(BaseModel):
    decision_id: int
    rating: str = Field(pattern="^(good|bad)$")


def auth(x_jarvis_token: str | None = Header(None)):
    if TOKEN and not secrets.compare_digest(x_jarvis_token or "", TOKEN):
        raise HTTPException(401,"Missing or incorrect X-Jarvis-Token")

async def forward(url: str, method: str = "GET", payload: dict | None = None):
    try:
        async with httpx.AsyncClient(timeout=250) as client:
            response = await client.request(method,url,json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        # Propagate meaningful client mistakes without hiding a failed experiment.
        if e.response.status_code in {400,409,422}:
            raise HTTPException(e.response.status_code,e.response.text[:500])
        raise HTTPException(502,f"Upstream error {e.response.status_code}")
    except httpx.HTTPError as e:
        raise HTTPException(503,f"Upstream unavailable: {type(e).__name__}")


def is_memory_query(text: str) -> bool:
    normalized = re.sub(r"[^\w\s]", " ", text.casefold().replace("ё", "е"))
    memory_words = ("индекс", "объект", "обьект", "памят", "сохранил", "запомнил", "научился", "понят", "понятт", "зберег", "пам ят")
    query_words = ("какие", "что", "сколько", "покажи", "чему", "чем", "які", "що", "скільки", "покажи", "чого")
    return any(w in normalized for w in memory_words) and any(w in normalized for w in query_words)


def format_memory(mem: dict) -> str:
    counts = mem["counts"]
    lines = ["Состояние исследовательской памяти (фактические данные SQLite):",
             f"Подтверждений: {counts['corrections']}; актуальных соответствий: {counts['active_mappings']}; признаков весов: {counts['weight_features']}.",
             f"Понятий: {counts['concepts']}; синонимов: {counts['aliases']}; связей: {counts['relations']}; векторов: {counts['semantic_vectors']}."]
    if mem["recent_corrections"]:
        lines.append("Последние исправления:")
        lines += [f"• «{item['phrase']}» → {item['target_id']}" for item in mem["recent_corrections"][:5]]
    if mem["concepts"]:
        lines.append("Последние понятия: " + ", ".join(i["name"] for i in mem["concepts"][:7]))
    return "\n".join(lines)

@app.get("/")
async def root():
    return FileResponse(Path(__file__).resolve().parents[2]/"web"/"index.html")

@app.get("/api/health", dependencies=[Depends(auth)])
async def health():
    result = {"version":"0.3.2", "gateway":"ok", "research":None,"llm":None}
    async with httpx.AsyncClient(timeout=6) as client:
        for key,url in (("research",RESEARCH_URL),("llm",LLM_URL)):
            try:
                r = await client.get(url+"/health")
                r.raise_for_status()
                result[key] = r.json()
            except httpx.HTTPError as e:
                result[key] = {"status":"unavailable", "error":type(e).__name__}
    return result

@app.get("/api/memory", dependencies=[Depends(auth)])
async def memory():
    return await forward(RESEARCH_URL+"/memory")

@app.get("/api/targets", dependencies=[Depends(auth)])
async def targets():
    return await forward(RESEARCH_URL+"/targets")

@app.post("/api/learn", dependencies=[Depends(auth)])
async def learn(data: LearnIn):
    return await forward(RESEARCH_URL+"/learn","POST",data.model_dump())

@app.post("/api/concepts", dependencies=[Depends(auth)])
async def concepts(data: ConceptIn):
    return await forward(RESEARCH_URL+"/concepts","POST",data.model_dump())

@app.post("/api/feedback", dependencies=[Depends(auth)])
async def feedback(data: FeedbackIn):
    return await forward(RESEARCH_URL+"/feedback","POST",data.model_dump())

@app.post("/api/chat", dependencies=[Depends(auth)])
async def chat(data: ChatIn):
    start = time.perf_counter()
    text = data.text.strip()
    if data.experiment != "llm_only":
        if is_memory_query(text):
            mem = await forward(RESEARCH_URL+"/memory")
            result = {"answer":format_memory(mem),"route":"sqlite_inspection", "research":mem,
                      "executed":False, "tokens":0}
            result["total_ms"] = round((time.perf_counter()-start)*1000,2)
            return result
        result = await forward(RESEARCH_URL+"/resolve","POST",
                               {"text":text,"ablation":data.ablation,"record":True})
        if result["is_command"]:
            if result["status"] == "resolved":
                answer = (f"Распознал: {'включить' if result['action']=='on' else 'выключить'} "
                          f"«{result['target_name']}» ({result['target_id']}). "
                          "Это симуляция, устройство не переключалось.")
            elif result["status"] == "blocked":
                answer = "Команда запрещена или неоднозначна. Ничего не выполняю."
            else:
                answer = "Не уверен в устройстве. Выбери правильное из списка — сохраню исправление. Ничего не выполняю."
            result.update(answer=answer, tokens=0, total_ms=round((time.perf_counter()-start)*1000,2))
            return result
        if data.experiment == "research_only":
            return {"answer":"Исследовательский индекс обрабатывает изученные команды и факты; для свободного диалога нужен LLM. Я не буду угадывать ответ.",
                    "route":"research_abstain", "tokens":0,"executed":False,
                    "total_ms":round((time.perf_counter()-start)*1000,2)}
    answer = await forward(LLM_URL+"/chat","POST",{"text":text,"profile":data.profile})
    answer["total_ms"] = round((time.perf_counter()-start)*1000,2)
    answer["experiment"] = data.experiment
    return answer
