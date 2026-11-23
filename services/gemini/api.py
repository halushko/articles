"""Gemini synthesizes only public evidence; it never reads user memory."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.llm.gemini import configured, generate
from services.research.web_research import public_https

app = FastAPI(title="JARVIS Gemini Evidence Synthesizer", version="0.1", docs_url=None)


class SynthesizeIn(BaseModel):
    question: str = Field(min_length=2, max_length=1200)
    sources: list[dict] = Field(min_length=1, max_length=4)


@app.get("/health")
def health():
    return {"status": "ok", "module": "gemini", "configured": configured()}


@app.post("/v1/synthesize")
async def synthesize(req: SynthesizeIn):
    # This service intentionally has NO endpoint for general free-form
    # answers, Google Search, user-memory requests, or device actions.
    safe = []
    for item in req.sources:
        if (isinstance(item, dict)
                and item.get("status") == "read"
                and len(str(item.get("page_text") or "")) >= 100
                and public_https(item.get("url"))):
            safe.append({k: item.get(k) for k in
                         ("title", "url", "status", "page_text", "snippet")})
    if not safe:
        raise HTTPException(422, "Нет прочитанных публичных источников")
    result = await generate(req.question, search=False, sources=safe)
    # No new sources can be introduced by Gemini.
    result["sources"] = [{"title":s.get("title"),"url":s.get("url"),"status":"read"}
                         for s in safe]
    result["grounded_in"] = "supplied_public_evidence"
    return result
