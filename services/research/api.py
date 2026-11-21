"""Standalone research index HTTP API, no LLM chat model calls."""
import os
import sqlite3
import time
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .engine import Engine
from .settings import load
from .normalize import parse_command
from .embedding import embed

engine: Engine | None = None

@asynccontextmanager
async def lifespan(app):
    global engine
    engine = Engine(os.getenv("DATABASE_PATH", "/data/research.sqlite3"), load())
    yield

app = FastAPI(title="JARVIS Research Index", version="0.3.2", lifespan=lifespan)

class ResolveIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    ablation: str = Field(default="combined", pattern="^(exact|online|combined)$")
    record: bool = True

class LearnIn(BaseModel):
    phrase: str = Field(min_length=1, max_length=500)
    target_id: str = Field(min_length=1, max_length=100)
    negative_ids: list[str] = Field(default_factory=list, max_length=20)

class ConceptIn(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    kind: str = Field(default="other", max_length=60)
    domain: str = Field(default="general", max_length=120)
    description: str = Field(default="", max_length=400)
    aliases: list[str] = Field(default_factory=list, max_length=20)

class FeedbackIn(BaseModel):
    decision_id: int
    rating: str = Field(pattern="^(good|bad)$")

@app.get("/health")
def health():
    return {"status":"ok", "version":"0.3.2", "module":"research",
            "semantic_enabled":os.getenv("SEMANTIC_ENABLED","false").lower() in {"true","1","yes"},
            "settings": engine.settings}

@app.get("/targets")
def targets():
    return [{"id":id,"name":name,"demo":True} for id,name in engine.targets.items()]

@app.post("/resolve")
async def resolve(req: ResolveIn):
    start = time.perf_counter()
    result = engine.predict(req.text, req.ablation, record=False)
    if (req.ablation == "combined" and result["status"] == "needs_confirmation"
            and parse_command(req.text).get("safe") and os.getenv("SEMANTIC_ENABLED","false").lower() in {"true","1","yes"}):
        try:
            parsed = parse_command(req.text)
            vector = await embed(parsed["target_text"], engine.settings["semantic_model"])
            ranked = engine.semantic_rank(vector, engine.settings["semantic_model"]) if vector else []
            if ranked:
                top = ranked[0]
                margin = top["score"] - (ranked[1]["score"] if len(ranked)>1 else 0.0)
                if (top["score"] >= engine.settings["semantic_min_similarity"]
                        and margin >= engine.settings["semantic_min_margin"]
                        and top["distinct_examples"] >= engine.settings["semantic_min_distinct_examples"]):
                    result.update(status="resolved",route="frozen_semantic",target_id=top["target_id"],
                                  target_name=top["target_name"],candidates=ranked[:4])
                else:
                    result["semantic_candidates"] = ranked[:4]
        except (httpx.HTTPError, KeyError, ValueError):
            result["semantic_status"] = "unavailable; online/exact still work"
    result["latency_ms"] = round((time.perf_counter()-start)*1000,2)
    if req.record:
        result["decision_id"] = engine.log_decision(result,req.text,req.ablation)
    return result

@app.post("/learn")
async def learn(req: LearnIn):
    try:
        result = engine.learn(req.phrase, req.target_id, req.negative_ids)
    except ValueError as err:
        raise HTTPException(422, str(err))
    if os.getenv("SEMANTIC_ENABLED","false").lower() in {"true","1","yes"}:
        try:
            vector = await embed(result["phrase_key"], engine.settings["semantic_model"])
            if vector:
                engine.store_embedding(result["evidence_id"],engine.settings["semantic_model"],vector)
                result["vector_saved"] = True
        except (httpx.HTTPError, ValueError, KeyError):
            result["vector_saved"] = False
            result["embedding_warning"] = "Frozen embedder unavailable; correction remains saved"
    return result

@app.get("/memory")
def memory():
    return engine.memory_snapshot()

@app.post("/concepts")
def add_concept(req: ConceptIn):
    try:
        return engine.remember_concept(req.name,req.kind,req.domain,req.description,req.aliases)
    except ValueError as err:
        raise HTTPException(422, str(err))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Concept already exists")

@app.post("/feedback")
def feedback(req: FeedbackIn):
    try:
        return engine.record_feedback(req.decision_id, req.rating)
    except ValueError as err:
        raise HTTPException(422,str(err))
