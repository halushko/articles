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
from .sessions import ConversationStore, markdown
from .findings import ResearchFindings
from .review_voice import VoiceReviews
from .approvals import OperationApprovals
from .dialogue import DialogueCoordinator
from .adaptive import BehaviorMemory, WEB_CONTEXT
from .actors import ActorRegistry
from .world import WorldMemory
from .utterance import classify
from .web_research import FreeWebResearch

engine: Engine | None = None
apprentice: ApprenticeMemory | None = None
sessions: ConversationStore | None = None
web_research: FreeWebResearch | None = None
voice_reviews: VoiceReviews | None = None
operation_approvals: OperationApprovals | None = None
actors: ActorRegistry | None = None
behavior: BehaviorMemory | None = None
dialogue: DialogueCoordinator | None = None
world: WorldMemory | None = None
findings: ResearchFindings | None = None

@asynccontextmanager
async def lifespan(app):
    global engine, apprentice, sessions, web_research, voice_reviews, operation_approvals, actors, behavior, dialogue, world, findings
    engine = Engine(os.getenv("DATABASE_PATH", "/data/research.sqlite3"), load())
    apprentice = ApprenticeMemory(engine)
    sessions = ConversationStore(engine)
    findings = ResearchFindings(engine)
    voice_reviews = VoiceReviews(engine,apprentice,sessions)
    behavior = BehaviorMemory(engine)
    operation_approvals = OperationApprovals(engine,sessions,behavior)
    dialogue = DialogueCoordinator(engine,sessions,operation_approvals,voice_reviews,behavior)
    actors = ActorRegistry(engine)
    world = WorldMemory(engine,apprentice,dialogue,behavior)
    web_research = FreeWebResearch(engine)
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
class IntentIn(BaseModel):
    text: str = Field(min_length=1,max_length=3000)

@app.post("/apprentice/intent")
def apprentice_intent(req: IntentIn):
    # Pure parser. No model latency, no teaching and no operation approval.
    return classify(req.text)

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
    session_id: str = Field(default="",max_length=36)
    actor_id: str = Field(default="",max_length=36)

class ApprenticeGradeIn(BaseModel):
    question: str = Field(min_length=1,max_length=3000)
    answer: str = Field(min_length=1,max_length=3000)
    rating: str = Field(pattern="^(good|bad)$")
    teacher_note: str = Field(default="",max_length=1200)
    actor_id: str = Field(default="",max_length=36)

@app.post("/apprentice/grade")
def apprentice_grade(req: ApprenticeGradeIn):
    try:
        if req.actor_id and not actors.find(req.actor_id):
            raise ValueError("Unknown speaker")
        return apprentice.grade(req.question,req.answer,req.rating,req.teacher_note,req.actor_id)
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

@app.post("/world/context")
@app.post("/apprentice/context")
def apprentice_context(req: ApprenticeContextIn):
    """Shared world read model for hybrid and apprentice; same source of truth."""
    speaker=actors.find(req.actor_id)
    if req.actor_id and not speaker:
        raise HTTPException(422,"Неизвестный профиль говорящего")
    # A shared device never silently inherits another person's identity.
    if speaker is None and len(actors.list()["people"])>1:
        speaker={"id":None,"display_name":None,"origin":"unselected"}
    result=world.context(req.query,speaker=speaker,
                         actor_id=req.actor_id,session_id=req.session_id)
    if req.session_id and sessions.exists(req.session_id):
        change=result.get("change_proposal")
        if change and change.get("id"):
            result["voice_review"]=voice_reviews.queue(req.session_id,changes=[change],actor_id=req.actor_id)
        else:
            voice_reviews.clear(req.session_id)
    return result


class GraphAliasIn(BaseModel):
    alias: str = Field(min_length=2,max_length=120)
    canonical: str = Field(min_length=2,max_length=120)

class GraphAliasDeleteIn(BaseModel):
    item_id: int = Field(gt=0)

@app.post("/apprentice/graph/aliases")
def apprentice_graph_alias(req: GraphAliasIn):
    try:
        return apprentice.graph.add_equivalence(req.alias,req.canonical)
    except ValueError as error:
        raise HTTPException(422,str(error))

@app.post("/apprentice/graph/aliases/delete")
def apprentice_graph_alias_delete(req: GraphAliasDeleteIn):
    try:
        return apprentice.graph.delete_equivalence(req.item_id)
    except ValueError as error:
        raise HTTPException(422,str(error))

@app.post("/apprentice/inspect")
def apprentice_inspect(req: ApprenticeContextIn):
    # Pure read: no lesson extraction, no model calls, no dialogue mutation.
    normalized = apprentice.dialogue.normalize_speech(req.query)
    graph = apprentice.graph.snapshot()
    roles = apprentice.dialogue.state()["roles"]
    result = apprentice.graph.inspect(normalized,roles,graph)
    return {"matched":result is not None,"inspection":result,
            "normalized_text":normalized,"executed":False}

class ModelChangesIn(BaseModel):
    source_text: str = Field(min_length=1,max_length=3000)
    changes: list[dict] = Field(default_factory=list,max_length=3)

class ChangeReviewIn(BaseModel):
    change_id: int = Field(gt=0)
    decision: str = Field(pattern="^(confirm|reject)$")
    new_name: str | None = Field(default=None,max_length=100)
    resolve_conflict: bool = False

@app.get("/apprentice/changes")
def apprentice_changes():
    return apprentice.changes.snapshot()

@app.post("/apprentice/changes/propose")
def apprentice_change_propose(req: ModelChangesIn):
    return {"changes":apprentice.changes.model_proposals(req.source_text,req.changes)}

@app.post("/apprentice/changes/review")
def apprentice_change_review(req: ChangeReviewIn):
    try:
        return apprentice.changes.review(req.change_id,req.decision,req.new_name,
                                         resolve_conflict=req.resolve_conflict)
    except ValueError as error:
        raise HTTPException(409,str(error))
    except sqlite3.IntegrityError as error:
        raise HTTPException(409,"Имя конфликтует с существующим объектом")

class VoiceReviewQueueIn(BaseModel):
    session_id: str = Field(min_length=36,max_length=36)
    actor_id: str = Field(default="",max_length=36)
    changes: list[dict] = Field(default_factory=list,max_length=5)
    proposals: list[dict] = Field(default_factory=list,max_length=5)

class VoiceReviewResolveIn(BaseModel):
    session_id: str = Field(default="",max_length=36)
    actor_id: str = Field(default="",max_length=36)
    text: str = Field(min_length=1,max_length=250)

@app.post("/apprentice/voice/queue")
def voice_review_queue(req: VoiceReviewQueueIn):
    try:
        return voice_reviews.queue(req.session_id,req.changes,req.proposals,actor_id=req.actor_id)
    except ValueError as error:
        raise HTTPException(422,str(error))

@app.post("/apprentice/voice/resolve")
def voice_review_resolve(req: VoiceReviewResolveIn):
    return voice_reviews.resolve(req.session_id,req.text,actor_id=req.actor_id)

@app.post("/apprentice/propose")
def apprentice_propose(req: ApprenticeProposalsIn):
    return {"proposals":apprentice.propose(req.source_text,req.proposals),"executed":False}

@app.post("/apprentice/review")
def apprentice_review(req: ApprenticeReviewIn):
    try:
        return apprentice.review(req.proposal_id,req.decision,req.name,req.description,req.details)
    except ValueError as error:
        raise HTTPException(422,str(error))


class BehaviorLearnIn(BaseModel):
    source_text: str = Field(min_length=1,max_length=3000)
    learning: list[dict] = Field(default_factory=list,max_length=3)
    actor_id: str = Field(default="",max_length=36)

class BehaviorForgetIn(BaseModel):
    rule_id: int = Field(gt=0)
    actor_id: str = Field(default="",max_length=36)

@app.post("/apprentice/behavior/learn")
def behavior_learn(req: BehaviorLearnIn):
    if req.actor_id and not actors.find(req.actor_id):
        raise HTTPException(422,"Unknown speaker")
    return {"rules": behavior.learn(req.source_text, req.learning, req.actor_id),
            "executed": False}

@app.get("/apprentice/behavior")
def behavior_state(actor_id: str = ""):
    if actor_id and not actors.find(actor_id):
        raise HTTPException(422,"Unknown speaker")
    return behavior.snapshot(actor_id)

@app.post("/apprentice/behavior/forget")
def behavior_forget(req: BehaviorForgetIn):
    try:
        return behavior.forget(req.rule_id, req.actor_id)
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


class DialogueResolveIn(BaseModel):
    session_id: str = Field(min_length=36,max_length=36)
    text: str = Field(min_length=1,max_length=250)
    actor_id: str = Field(default="",max_length=36)
    web_enabled: bool = True

@app.post("/research/dialogue/resolve")
def dialogue_resolve(req: DialogueResolveIn):
    if req.actor_id and not actors.find(req.actor_id):
        raise HTTPException(422,"Unknown speaker")
    return dialogue.resolve(req.session_id,req.text,req.actor_id,req.web_enabled)

@app.get("/research/dialogue/context/{sid}")
def dialogue_context(sid: str,actor_id: str = ""):
    if actor_id and not actors.find(actor_id):
        raise HTTPException(422,"Unknown speaker")
    return dialogue.context(sid,actor_id)


class SearchApprovalIn(BaseModel):
    session_id: str = Field(min_length=36,max_length=36)
    actor_id: str = Field(default="",max_length=36)
    query: str = Field(min_length=3,max_length=240)
    question: str = Field(default="",max_length=3000)

class SearchApprovalDecisionIn(BaseModel):
    session_id: str = Field(min_length=36,max_length=36)
    actor_id: str = Field(default="",max_length=36)
    phrase: str = Field(default="",max_length=250)
    approval_id: str = Field(default="",max_length=36)
    source: str = Field(default="voice",pattern="^(voice|button)$")

class SearchApprovalClearIn(BaseModel):
    session_id: str = Field(min_length=36,max_length=36)

@app.post("/research/operations/web/propose")
def web_operation_propose(req: SearchApprovalIn):
    try:
        return operation_approvals.propose_search(req.session_id,req.query,req.question,req.actor_id)
    except ValueError as error:
        raise HTTPException(422,str(error))

@app.post("/research/operations/web/resolve")
def web_operation_resolve(req: SearchApprovalDecisionIn):
    return operation_approvals.resolve(req.session_id,req.phrase,req.approval_id,req.source,req.actor_id)

class SearchExecutionIn(BaseModel):
    session_id: str = Field(min_length=36,max_length=36)
    actor_id: str = Field(default="",max_length=36)
    operation_id: str = Field(min_length=36,max_length=36)


class SearchExecutionFinishIn(SearchExecutionIn):
    status: str = Field(pattern="^(succeeded|failed)$")
    answer: str = Field(default="",max_length=5000)
    sources: list[dict] = Field(default_factory=list,max_length=8)
    error: str = Field(default="",max_length=400)


class SearchExecutionStatusIn(BaseModel):
    session_id: str = Field(min_length=36,max_length=36)
    actor_id: str = Field(default="",max_length=36)


@app.post("/research/operations/web/start")
def web_execution_start(req: SearchExecutionIn):
    return operation_approvals.execution_start(req.session_id,req.operation_id,req.actor_id)


@app.post("/research/operations/web/finish")
def web_execution_finish(req: SearchExecutionFinishIn):
    return operation_approvals.execution_finish(req.session_id,req.operation_id,
                req.actor_id,req.status,req.answer,req.sources,req.error)


@app.post("/research/operations/web/status")
def web_execution_status(req: SearchExecutionStatusIn):
    return operation_approvals.execution_status(req.session_id,req.actor_id)


@app.post("/research/operations/web/clear")
def web_operation_clear(req: SearchApprovalClearIn):
    operation_approvals.clear(req.session_id)
    return {"cleared":True}


class ActorCreateIn(BaseModel):
    display_name: str = Field(min_length=2,max_length=70)

class ActorDeviceIn(BaseModel):
    device_id: str = Field(min_length=36,max_length=36)
    actor_id: str = Field(min_length=36,max_length=36)
    label: str = Field(min_length=1,max_length=70)

@app.get("/research/actors")
def actor_list(device_id: str = ""):
    return actors.list(device_id)

@app.post("/research/actors")
def actor_create(req: ActorCreateIn):
    try:
        return actors.create(req.display_name)
    except ValueError as error:
        raise HTTPException(422,str(error))

@app.post("/research/actors/device")
def actor_device(req: ActorDeviceIn):
    try:
        return actors.add_device(req.device_id,req.actor_id,req.label)
    except ValueError as error:
        raise HTTPException(422,str(error))

# Research sessions and free public-web search are independent from the old
# online-weight index. Every answer's provenance is stored per-turn.
class SessionCreateIn(BaseModel):
    title: str = Field(default="",max_length=140)

class SessionTurnIn(BaseModel):
    question: str = Field(min_length=1,max_length=3000)
    actor_id: str = Field(default="",max_length=36)
    answer: str = Field(default="",max_length=6000)
    experiment: str = Field(max_length=40)
    profile: str = Field(max_length=40)
    ablation: str = Field(max_length=40)
    web_mode: str = Field(default="off",pattern="^(off|ask|auto)$")
    result: dict = Field(default_factory=dict)

class SessionEventIn(BaseModel):
    event_type: str = Field(pattern="^[a-z][a-z0-9_.]{0,50}$")
    data: dict = Field(default_factory=dict)
    turn_id: int | None = Field(default=None,gt=0)

class SessionGradeIn(BaseModel):
    turn_id: int = Field(gt=0)
    rating: str = Field(pattern="^(good|bad)$")
    teacher_note: str = Field(default="",max_length=1200)

class WebSearchIn(BaseModel):
    query: str = Field(min_length=3,max_length=240)
    session_id: str = Field(default="",max_length=60)

@app.post("/research/sessions")
def research_session_create(req: SessionCreateIn):
    return sessions.create(req.title)

@app.get("/research/sessions")
def research_session_list():
    return {"sessions":sessions.list()}

@app.get("/research/sessions/{sid}")
def research_session_read(sid: str):
    try:
        return sessions.read(sid)
    except ValueError as error:
        raise HTTPException(404,str(error))

@app.post("/research/sessions/{sid}/turns")
def research_session_turn(sid: str, req: SessionTurnIn):
    try:
        speaker=actors.find(req.actor_id)
        if req.actor_id and not speaker:
            raise ValueError("Неизвестный говорящий")
        result=sessions.save_turn(sid,req.question,req.answer,req.experiment,
                                  req.profile,req.ablation,req.web_mode,req.result,
                                  actor_id=req.actor_id)
        dialogue.record(sid,result["turn_id"],req.actor_id,req.question,
                        req.answer,req.result)
        # Successful public searches are reusable experience, not world facts.
        findings.remember(sid,req.actor_id,req.question,req.answer,req.result)
        if speaker:
            actors.record_turn_speaker(sid,req.actor_id)
        return result
    except ValueError as error:
        raise HTTPException(404,str(error))

class SessionClearIn(BaseModel):
    confirm: bool = Field(default=False)


@app.post("/research/sessions/{sid}/clear")
def research_session_clear(sid: str, req: SessionClearIn):
    if req.confirm is not True:
        raise HTTPException(422, "Для очистки нужен confirm=true")
    try:
        return sessions.clear(sid)
    except ValueError as error:
        raise HTTPException(404, str(error))


class FindingRecallIn(BaseModel):
    query: str = Field(min_length=2, max_length=1200)
    session_id: str = Field(default="", max_length=60)
    actor_id: str = Field(default="", max_length=36)


@app.post("/research/findings/recall")
def research_finding_recall(req: FindingRecallIn):
    # Unlike confirmed graph facts, these are explicitly unverified
    # public-web excerpts / answers, returned with original sources.
    if req.actor_id and not actors.find(req.actor_id):
        raise HTTPException(422, "Неизвестный профиль пользователя")
    return findings.recall(req.query, req.session_id, req.actor_id)


@app.post("/research/sessions/{sid}/events")
def research_session_event(sid: str,req: SessionEventIn):
    try:
        return sessions.event(sid,req.event_type,req.data,req.turn_id)
    except ValueError as error:
        raise HTTPException(422,str(error))

@app.post("/research/sessions/{sid}/grade")
def research_session_grade(sid: str,req: SessionGradeIn):
    try:
        return sessions.grade(sid,req.turn_id,req.rating,req.teacher_note)
    except ValueError as error:
        raise HTTPException(422,str(error))

@app.get("/research/sessions/{sid}/export")
def research_session_export(sid: str):
    try:
        data=sessions.read(sid)
    except ValueError as error:
        raise HTTPException(404,str(error))
    return {"markdown":markdown(data),"session":data}

@app.post("/research/web/search")
async def research_web_search(req: WebSearchIn):
    return await web_research.search(req.query,req.session_id)
