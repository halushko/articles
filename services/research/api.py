"""Standalone research index HTTP API, no LLM chat model calls."""
import os
import sqlite3
import time
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .engine import Engine
from .apprentice import ApprenticeMemory
from .settings import load
from .normalize import parse_command
from .embedding import embed

engine: Engine | None = None
apprentice: ApprenticeMemory | None = None

@asynccontextmanager
async def lifespan(app):
    global engine, apprentice
    engine = Engine(os.getenv("DATABASE_PATH", "/data/research.sqlite3"), load())
    apprentice = ApprenticeMemory(engine)
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

# The apprenticeship index is domain-independent: it knows no Home Assistant
# entity IDs, network endpoints, shell commands, or executable code.
class ApprenticeProposalsIn(BaseModel):
    source_text: str = Field(min_length=1,max_length=3000)
    proposals: list[dict] = Field(default_factory=list,max_length=5)

class ApprenticeReviewIn(BaseModel):
    proposal_id: int = Field(gt=0)
    decision: str = Field(pattern="^(confirm|reject)$")
    name: str | None = Field(default=None,max_length=120)
    description: str | None = Field(default=None,max_length=1200)
    details: dict | None = None

class ApprenticeContextIn(BaseModel):
    query: str = Field(min_length=1,max_length=3000)

class ApprenticeGradeIn(BaseModel):
    question: str = Field(min_length=1,max_length=3000)
    answer: str = Field(min_length=1,max_length=3000)
    rating: str = Field(pattern="^(good|bad)$")
    teacher_note: str = Field(default="",max_length=1200)

@app.post("/apprentice/grade")
def apprentice_grade(req: ApprenticeGradeIn):
    try:
        return apprentice.grade(req.question,req.answer,req.rating,req.teacher_note)
    except ValueError as error:
        raise HTTPException(422,str(error))

class ApprenticeForgetIn(BaseModel):
    kind: str = Field(pattern="^(entity|action|fact|lesson|capability)$")
    item_id: int = Field(gt=0)

@app.post("/apprentice/forget")
def apprentice_forget(req: ApprenticeForgetIn):
    try:
        return apprentice.forget(req.kind,req.item_id)
    except ValueError as error:
        raise HTTPException(422,str(error))

@app.get("/apprentice/state")
def apprentice_state():
    return apprentice.snapshot()

@app.post("/apprentice/context")
def apprentice_context(req: ApprenticeContextIn):
    return apprentice.context(req.query)

@app.post("/apprentice/inspect")
def apprentice_inspect(req: ApprenticeContextIn):
    # Pure read: no lesson extraction, no model calls, no dialogue mutation.
    normalized = apprentice.dialogue.normalize_speech(req.query)
    graph = apprentice.graph.snapshot()
    roles = apprentice.dialogue.state()["roles"]
    result = apprentice.graph.inspect(normalized,roles,graph)
    return {"matched":result is not None,"inspection":result,
            "normalized_text":normalized,"executed":False}

@app.post("/apprentice/propose")
def apprentice_propose(req: ApprenticeProposalsIn):
    return {"proposals":apprentice.propose(req.source_text,req.proposals),"executed":False}

@app.post("/apprentice/review")
def apprentice_review(req: ApprenticeReviewIn):
    try:
        return apprentice.review(req.proposal_id,req.decision,req.name,req.description,req.details)
    except ValueError as error:
        raise HTTPException(422,str(error))


class SpeechCorrectionIn(BaseModel):
    heard: str = Field(min_length=2,max_length=80)
    correct: str = Field(min_length=2,max_length=100)

class SpeechCorrectionDeleteIn(BaseModel):
    item_id: int = Field(gt=0)

@app.post("/apprentice/speech-corrections")
def speech_correction_add(req: SpeechCorrectionIn):
    try:
        return apprentice.dialogue.add_correction(req.heard,req.correct)
    except ValueError as error:
        raise HTTPException(422,str(error))

@app.post("/apprentice/speech-corrections/delete")
def speech_correction_delete(req: SpeechCorrectionDeleteIn):
    try:
        return apprentice.dialogue.delete_correction(req.item_id)
    except ValueError as error:
        raise HTTPException(422,str(error))
