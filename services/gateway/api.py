"""Experiment gateway: independent controls for LLM baseline and novel index."""
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
import json
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

app = FastAPI(title="JARVIS Research Workbench", version="0.5.0", lifespan=lifespan, docs_url=None)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parents[2]/"web")), name="static")

class ChatIn(BaseModel):
    text: str = Field(min_length=1,max_length=3000)
    experiment: str = Field(default="hybrid", pattern="^(apprentice|hybrid|research_only|llm_only)$")
    profile: str = Field(default="fast",pattern="^(fast|deep)$")
    ablation: str = Field(default="combined",pattern="^(exact|online|combined)$")
    history: list[dict] = Field(default_factory=list,max_length=14)
    session_id: str | None = Field(default=None,max_length=60)
    web_mode: str = Field(default="off",pattern="^(off|ask|auto)$")
    allow_web_search: bool = False
    approved_web_query: str = Field(default="",max_length=240)
    approval_id: str = Field(default="",max_length=36)
    external_context: str = Field(default="",max_length=8000)
    request_id: str | None = Field(default=None,pattern=r"^[a-f0-9-]{36}$")

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

class SpeechCorrectionIn(BaseModel):
    heard: str = Field(min_length=2,max_length=80)
    correct: str = Field(min_length=2,max_length=100)

class SpeechCorrectionDeleteIn(BaseModel):
    item_id: int = Field(gt=0)

class ApprenticeGradeIn(BaseModel):
    question: str = Field(min_length=1,max_length=3000)
    answer: str = Field(min_length=1,max_length=3000)
    rating: str = Field(pattern="^(good|bad)$")
    teacher_note: str = Field(default="",max_length=1200)

class ApprenticeForgetIn(BaseModel):
    kind: str = Field(pattern="^(entity|action|fact|lesson|capability)$")
    item_id: int = Field(gt=0)

class ApprenticeReviewIn(BaseModel):
    proposal_id: int = Field(gt=0)
    decision: str = Field(pattern="^(confirm|reject)$")
    name: str | None = Field(default=None,max_length=120)
    description: str | None = Field(default=None,max_length=1200)
    details: dict | None = None

class GraphAliasIn(BaseModel):
    alias: str = Field(min_length=2,max_length=120)
    canonical: str = Field(min_length=2,max_length=120)

class GraphAliasDeleteIn(BaseModel):
    item_id: int = Field(gt=0)

class ChangeReviewIn(BaseModel):
    change_id: int = Field(gt=0)
    decision: str = Field(pattern="^(confirm|reject)$")
    new_name: str | None = Field(default=None,max_length=100)
    resolve_conflict: bool = False

class SessionCreateIn(BaseModel):
    title: str = Field(default="",max_length=140)

class SessionEventIn(BaseModel):
    event_type: str = Field(pattern="^[a-z][a-z0-9_.]{0,50}$")
    data: dict = Field(default_factory=dict)
    turn_id: int | None = Field(default=None,gt=0)

class SessionGradeIn(BaseModel):
    turn_id: int = Field(gt=0)
    rating: str = Field(pattern="^(good|bad)$")
    teacher_note: str = Field(default="",max_length=1200)

class MemoryInspectIn(BaseModel):
    query: str = Field(min_length=1,max_length=3000)

class FeedbackIn(BaseModel):
    decision_id: int
    rating: str = Field(pattern="^(good|bad)$")


def auth(x_jarvis_token: str | None = Header(None)):
    if TOKEN and not secrets.compare_digest(x_jarvis_token or "", TOKEN):
        raise HTTPException(401,"Missing or incorrect X-Jarvis-Token")

CHAT_PHASES = {}
PHASE_TTL = 900

def progress(request_id, phase, detail="", final=False):
    """Actual gateway pipeline stage; no invented completion percentages."""
    if not request_id:
        return
    now=time.monotonic()
    if len(CHAT_PHASES)>300:
        for key,item in list(CHAT_PHASES.items()):
            if now-item["updated"]>PHASE_TTL:
                CHAT_PHASES.pop(key,None)
        if len(CHAT_PHASES)>300:
            for key in list(CHAT_PHASES)[:100]:
                CHAT_PHASES.pop(key,None)
    previous=CHAT_PHASES.get(request_id)
    CHAT_PHASES[request_id]={
        "phase":phase,"detail":detail[:140],
        "started":previous["started"] if previous else now,
        "updated":now,"final":final
    }

@app.get("/api/chat/status/{request_id}", dependencies=[Depends(auth)])
async def chat_status(request_id: str):
    try:
        uuid.UUID(request_id)
    except ValueError:
        raise HTTPException(422,"Некорректный идентификатор запроса")
    state=CHAT_PHASES.get(request_id)
    if not state:
        return {"phase":"unknown","detail":"Запрос не найден в текущем процессе.",
                "running":False,"elapsed_seconds":0}
    now=time.monotonic()
    if now-state["updated"]>PHASE_TTL:
        CHAT_PHASES.pop(request_id,None)
        return {"phase":"expired","detail":"Статус запроса устарел.",
                "running":False,"elapsed_seconds":0}
    return {"phase":state["phase"],"detail":state["detail"],
            "running":not state["final"],
            "elapsed_seconds":round(now-state["started"],1),
            "phase_elapsed_seconds":round(now-state["updated"],1)}

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
    result = {"version":"0.5.0", "gateway":"ok", "research":None,"llm":None,
              "web_search_enabled":os.getenv("WEB_SEARCH_ENABLED","true").lower() in ("true","1","yes")}
    async with httpx.AsyncClient(timeout=6) as client:
        for key,url in (("research",RESEARCH_URL),("llm",LLM_URL)):
            try:
                r = await client.get(url+"/health")
                r.raise_for_status()
                result[key] = r.json()
            except httpx.HTTPError as e:
                result[key] = {"status":"unavailable", "error":type(e).__name__}
    return result

@app.post("/api/sessions", dependencies=[Depends(auth)])
async def create_session(req: SessionCreateIn):
    return await forward(RESEARCH_URL+"/research/sessions","POST",req.model_dump())

@app.get("/api/sessions", dependencies=[Depends(auth)])
async def list_sessions():
    return await forward(RESEARCH_URL+"/research/sessions")

@app.get("/api/sessions/{sid}", dependencies=[Depends(auth)])
async def read_session(sid: str):
    return await forward(RESEARCH_URL+"/research/sessions/"+sid)

@app.post("/api/sessions/{sid}/events", dependencies=[Depends(auth)])
async def add_session_event(sid: str,req: SessionEventIn):
    return await forward(RESEARCH_URL+"/research/sessions/"+sid+"/events",
                         "POST",req.model_dump())

@app.post("/api/sessions/{sid}/grade", dependencies=[Depends(auth)])
async def grade_session_turn(sid: str,req: SessionGradeIn):
    return await forward(RESEARCH_URL+"/research/sessions/"+sid+"/grade",
                         "POST",req.model_dump())

@app.get("/api/sessions/{sid}/export", dependencies=[Depends(auth)])
async def export_session(sid: str, format: str = "md"):
    if format not in ("md","json"):
        raise HTTPException(422,"Доступные форматы: md, json")
    data=await forward(RESEARCH_URL+"/research/sessions/"+sid+"/export")
    if format=="md":
        body=data["markdown"]
        kind="text/markdown; charset=utf-8"
    else:
        body=json.dumps(data["session"],ensure_ascii=False,indent=2)
        kind="application/json; charset=utf-8"
    return Response(content=body,media_type=kind,headers={
        "Content-Disposition":f'attachment; filename="jarvis-{sid}.{format}"',
        "Cache-Control":"no-store"
    })

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


@app.get("/api/apprentice/changes", dependencies=[Depends(auth)])
async def apprentice_changes():
    return await forward(RESEARCH_URL+"/apprentice/changes")

@app.post("/api/apprentice/changes/review", dependencies=[Depends(auth)])
async def apprentice_changes_review(req: ChangeReviewIn):
    return await forward(RESEARCH_URL+"/apprentice/changes/review","POST",
                         req.model_dump())

@app.get("/api/apprentice/state", dependencies=[Depends(auth)])
async def apprentice_state():
    return await forward(RESEARCH_URL+"/apprentice/state")

@app.post("/api/apprentice/graph/aliases", dependencies=[Depends(auth)])
async def apprentice_graph_alias(data: GraphAliasIn):
    return await forward(RESEARCH_URL+"/apprentice/graph/aliases","POST",data.model_dump())

@app.post("/api/apprentice/graph/aliases/delete", dependencies=[Depends(auth)])
async def apprentice_graph_alias_delete(data: GraphAliasDeleteIn):
    return await forward(RESEARCH_URL+"/apprentice/graph/aliases/delete","POST",data.model_dump())

@app.post("/api/apprentice/inspect", dependencies=[Depends(auth)])
async def apprentice_inspect(req: "MemoryInspectIn"):
    return await forward(RESEARCH_URL+"/apprentice/inspect","POST",req.model_dump())

@app.post("/api/apprentice/speech-corrections", dependencies=[Depends(auth)])
async def speech_correction_add(data: SpeechCorrectionIn):
    return await forward(RESEARCH_URL+"/apprentice/speech-corrections","POST",data.model_dump())

@app.post("/api/apprentice/speech-corrections/delete", dependencies=[Depends(auth)])
async def speech_correction_delete(data: SpeechCorrectionDeleteIn):
    return await forward(RESEARCH_URL+"/apprentice/speech-corrections/delete","POST",data.model_dump())

@app.post("/api/apprentice/grade", dependencies=[Depends(auth)])
async def apprentice_grade(data: ApprenticeGradeIn):
    return await forward(RESEARCH_URL+"/apprentice/grade","POST",data.model_dump())

@app.post("/api/apprentice/forget", dependencies=[Depends(auth)])
async def apprentice_forget(data: ApprenticeForgetIn):
    return await forward(RESEARCH_URL+"/apprentice/forget","POST",data.model_dump())

@app.post("/api/apprentice/review", dependencies=[Depends(auth)])
async def apprentice_review(data: ApprenticeReviewIn):
    return await forward(RESEARCH_URL+"/apprentice/review","POST",data.model_dump())

def unverified_external_activity(reply: str) -> bool:
    """Reject model claims of tool calls when no external execution tool exists."""
    past_action = re.search(
        r"(?iu)\bя\s+(?:(?:уже|успешно|только что)\s+)?"
        r"(?:запросил|подключился|подключила|подключил|проверил|"
        r"настроил|получил\s+доступ|отправил\s+запрос|выполнил\s+запрос)\b",
        reply,
    )
    external = re.search(
        r"(?iu)\b(?:api|url|http|сервер|сайт|сеть|доступ|устройств|"
        r"интерфейс|соединени|smart.?home|home.?assistant|подключени)\w*",
        reply,
    )
    unsafe_secret_request = re.search(
        r"(?iu)(?:предоставьте|предоставь|пришли|пришлите|отправь|"
        r"отправьте|вставьте|вставь)\s+(?:мне\s+)?(?:свой\s+)?"
        r"(?:токен|пароль|api.?ключ|секрет)\b",
        reply,
    )
    return bool((past_action and external) or unsafe_secret_request)


def requests_external_connection(text: str) -> bool:
    """Specific first-person tool request, not a lesson about network protocols."""
    wants_connection = re.search(
        r"(?iu)\b(?:подключ\w+|соединени[ея]|"
        r"предоставить\s+доступ|дать\s+доступ)\b", text)
    assistant_target = re.search(
        r"(?iu)\b(?:ты|тебе|тебя|твой|джарвис|jarvis|вам|вы)\b", text)
    return bool(wants_connection and assistant_target)


def proposed_web_query(text: str, parsed: dict) -> str:
    """Model proposes search terms. A small explicit-user-request fallback
    handles missed tool planning by weak Qwen profiles; neither is a result.
    """
    if not isinstance(parsed,dict):
        parsed={}
    query=str(parsed.get("web_query") or "").strip()
    if parsed.get("needs_web") is True:
        return (query or text)[:240]
    # Only explicit questions that request finding information on the web.
    if re.search(r"(?iu)\b(найди|найдите|поищи|поищите|загугли|"
                 r"проверь\s+в\s+интернете|search\s+the\s+web|"
                 r"ищи\s+в\s+сети)\b",text):
        return text[:240]
    return ""


async def answer_from_public_search(data: ChatIn, text: str,
                                    memory: dict, query: str, start: float):
    progress(data.request_id,"web_fetch","SearXNG: поиск и чтение публичных страниц")
    results=await forward(RESEARCH_URL+"/research/web/search","POST",{
        "query":query[:240],"session_id":data.session_id or ""})
    evidence=results.get("sources") or []
    if not results.get("available") or not evidence:
        return {"answer":"Не удалось выполнить публичный поиск: "+
                        (results.get("reason") or "источники недоступны")+
                        ". Не буду выдавать предположения за найденные сведения.",
                "route":"apprentice_web_unavailable","model":"SQLite",
                "search_query":query[:240],"search_status":results.get("status"),
                "search_error":results.get("reason",""),"sources":[],
                "tool_trace":{"tool":"public_web.search","status":results.get("status")},
                "proposals":[],"tokens":0,"executed":False,
                "total_ms":round((time.perf_counter()-start)*1000,2)}
    progress(data.request_id,"web_answer","Qwen: анализирует найденные источники")
    grounded=await forward(LLM_URL+"/web-answer","POST",{
        "question":text,"profile":data.profile,"sources":evidence,
        "memory":memory,"origin":"searxng"})
    grounded.update({"sources":[{"title":v.get("title"),"url":v.get("url"),
                                   "snippet":v.get("snippet"),"status":v.get("status"),
                                   "page_text":v.get("page_text","")[:4000]}
                                  for v in evidence],
                     "search_query":query[:240],"search_status":"ok",
                     "tool_trace":{"tool":"public_web.search",
                                   "status":"ok","pages_read":sum(
                                       x.get("status")=="read" for x in evidence)},
                     "web_requested":True,"proposals":[],
                     "total_ms":round((time.perf_counter()-start)*1000,2),
                     "executed":False})
    if unverified_external_activity(grounded.get("answer","")):
        grounded["answer"]=("Я изучил найденные публичные источники, но у меня "
            "нет подключения к устройствам или произвольному API. "
            "Ссылки и выдержки приведены ниже.")
        grounded["integrity_warning"]="Blocked unsupported external action claim"
    return grounded


async def apprentice_chat(data: ChatIn, start: float):
    # Tool permissions are checked before any model call or memory teaching.
    # A previous request is bound to this session + exact query, consumed once.
    # Bare "да" does NOT authorize a network request.
    is_web_reply = bool(re.match(
        r"(?iu)^\\s*(?:(?:команда|приказ)\\s*[:,-]?\\s*)?"
        r"(?:разрешаю|да\\s+разрешаю|подтверждаю\\s+поиск|"
        r"разрешить\\s+поиск|не\\s+разрешаю|"
        r"запрещаю\\s+поиск|отмена\\s+поиска|отмени\\s+поиск)\\b",
        data.text))
    if is_web_reply or data.approval_id or data.allow_web_search:
        if data.web_mode == "off":
            return {"answer":"Интернет отключён. Поиск не выполнялся.",
                    "route":"web_permission_denied","model":"SQLite",
                    "executed":False,"tokens":0}
        resolved=await forward(RESEARCH_URL+"/research/operations/web/resolve","POST",{
            "session_id":data.session_id or "",
            "phrase":data.text[:250],
            "approval_id":data.approval_id,
            "source":"button" if data.approval_id and data.allow_web_search else "voice"})
        if resolved.get("handled"):
            if resolved.get("status")=="approved":
                search=await answer_from_public_search(
                    data,resolved["question"],{},resolved["query"],start)
                search.update({"route":"apprentice_web_evidence"
                               if search.get("search_status")=="ok" else search["route"],
                               "approval_id":resolved["operation_id"],
                               "approval_source":"button" if data.approval_id else "voice",
                               "utterance_intent":{"intent":"command","reason":"approved_tool_operation",
                                                   "confidence":"high"},
                               "requested_action":"web.search"})
                return search
            return {"answer":resolved.get("answer","Поиск не разрешён."),
                    "route":"web_permission_"+resolved.get("status","denied"),
                    "utterance_intent":{"intent":"command","reason":"tool_permission"},
                    "model":"SQLite","tokens":0,"executed":False,
                    "approval_id":resolved.get("operation_id"),
                    "total_ms":round((time.perf_counter()-start)*1000,2)}
    # Only confirmed knowledge is given to the interpreter; guesses stay pending.
    progress(data.request_id,"memory","SQLite: проверяю знания и изменения")
    # Voice confirmation is scoped to the last review shown in this session,
    # and is never interpreted as a device command.
    voice_cue=bool(re.match(
        r"(?iu)^\s*(?:(?:команда|вопрос)\s*[:,-]?\s*)?"
        r"(?:да\b|нет\b|ага\b|угу\b|подтвер\w*|отклон\w*|"
        r"отмен\w*|применяй\b|согласен\b|согласна\b|сохрани\b)",data.text))
    if voice_cue:
        voice=await forward(RESEARCH_URL+"/apprentice/voice/resolve","POST",
                            {"text":data.text[:250],"session_id":data.session_id or ""})
        if voice.get("handled"):
            return {"answer":voice["answer"],"route":"apprentice_voice_review",
                    "intent":"review","utterance_intent":{"intent":"command",
                    "confidence":"high","reason":"session_scoped_voice_review"},
                    "voice_review_result":voice,"proposals":[],"tokens":0,
                    "model":"SQLite","executed":False,
                    "total_ms":round((time.perf_counter()-start)*1000,2)}
    memory = await forward(RESEARCH_URL+"/apprentice/context","POST",
                           {"query":data.text,"session_id":data.session_id or ""})
    text = memory.get("normalized_text", data.text)
    direct_reply = memory.get("direct_reply")
    # Questions about graph memory have authoritative SQLite responses.
    # The model must not hallucinate the contents of its own index.
    if direct_reply:
        inspection = memory.get("inspection")
        intent_info=memory.get("utterance_intent") or {}
        return {"answer":direct_reply,
                "route":"apprentice_graph_inspection" if inspection else "apprentice_confirmed_context",
                "intent":intent_info.get("intent") or "ask",
                "utterance_intent":intent_info,
                "voice_review":memory.get("voice_review"),
                "missing":"", "proposals":[], "model":"SQLite",
                "tokens":0, "executed":False,
                "retrieval_trace":inspection.get("trace") if inspection else None,
                "evidence_source":inspection.get("source") if inspection else "dialogue_roles",
                "normalized_input":text if text != data.text else None,
                "change_proposal":memory.get("change_proposal"),
                "total_ms":round((time.perf_counter()-start)*1000,2)}
    # Route explicit requests to a real, allowlisted tool before Qwen.
    # Never require Qwen to infer that "найди в интернете" needs a search.
    intent_info=memory.get("utterance_intent") or {}
    wanted_direct=(proposed_web_query(text,{})
                   if intent_info.get("intent")=="command" else "")
    if wanted_direct:
        if data.web_mode=="off":
            return {"answer":"Интернет отключён. Поиск не выполнялся.",
                    "route":"apprentice_web_disabled","model":"SQLite",
                    "utterance_intent":intent_info,"tokens":0,"executed":False}
        if data.web_mode=="auto":
            found=await answer_from_public_search(data,text,memory,wanted_direct,start)
            found["utterance_intent"]=intent_info
            return found
        if data.session_id:
            approval=await forward(RESEARCH_URL+"/research/operations/web/propose",
                 "POST",{"session_id":data.session_id,
                          "query":wanted_direct,"question":text})
            return {"answer":("Могу выполнить публичный поиск по запросу "
                      f"«{approval['query']}». Поиск ещё не запущен. "
                      "Скажи «Разрешаю поиск» или нажми «Разрешить этот поиск»."),
                    "route":"apprentice_web_permission","web_approval":approval,
                    "requested_action":"web.search",
                    "utterance_intent":intent_info,"model":"SQLite",
                    "tokens":0,"executed":False,
                    "total_ms":round((time.perf_counter()-start)*1000,2)}
        return {"answer":"Для безопасного согласия на поиск нужен сохранённый диалог.",
                "route":"apprentice_web_no_session","model":"SQLite",
                "utterance_intent":intent_info,"tokens":0,"executed":False}
    # An unrelated turn invalidates older, unanswered search requests.
    if data.session_id:
        await forward(RESEARCH_URL+"/research/operations/web/clear",
                      "POST",{"session_id":data.session_id})
    # User-pasted Google AI text is a manually supplied, unverified outside
    # observation. It is never saved as teacher-confirmed knowledge.
    if data.external_context:
        source={"title":"Google AI Mode · вставлено учителем",
                "url":"","status":"manually_pasted_unverified",
                "text":data.external_context[:8000]}
        progress(data.request_id,"web_answer","Qwen: анализирует вставленный ответ Google AI")
        grounded=await forward(LLM_URL+"/web-answer","POST",{
            "question":text,"profile":data.profile,"sources":[source],
            "memory":memory,"origin":"google_ai_manual"})
        grounded.update({"sources":[{"title":source["title"],"url":"",
                                      "status":source["status"],
                                      "snippet":source["text"][:4000]}],
                         "external_origin":"google_ai_manual",
                         "tool_trace":{"tool":"google.ai_mode",
                                       "status":"teacher_pasted_not_api"},
                         "executed":False,"proposals":[],
                         "total_ms":round((time.perf_counter()-start)*1000,2)})
        if unverified_external_activity(grounded.get("answer","")):
            grounded["answer"]=("Вставленный ответ Google AI — это материал "
                "для анализа, а не свидетельство моего доступа к внешним "
                "сервисам или устройствам. Он сохранён как непроверенный источник.")
            grounded["integrity_warning"]="Blocked unsupported external action claim"
        return grounded
    capabilities = {item.get("name"):item.get("status")
                    for item in memory.get("capabilities",[])}
    if (requests_external_connection(text)
            and capabilities.get("network.http") == "unavailable"
            and capabilities.get("external.action") == "unavailable"):
        return {"answer":(
                    "Понимаю: ты хочешь, чтобы я сам подключился к внешней системе. "
                    "Но у меня пока нет HTTP-инструмента и коннектора для действий "
                    "в других системах. Поэтому URL сам по себе не позволит мне "
                    "подключиться, а токен в чат отправлять нельзя. "
                    "Нужно добавить разрешённый инструмент подключения с отдельным "
                    "хранилищем доступа. До этого я могу только изучать описания."
                ), "route":"apprentice_capability_boundary",
                "intent":"request", "missing":"Отсутствует исполнитель network.http",
                "proposals":[], "model":"SQLite", "tokens":0,"executed":False,
                "normalized_input":text if text != data.text else None,
                "total_ms":round((time.perf_counter()-start)*1000,2)}
    progress(data.request_id,"planning","Qwen: обдумывает вопрос и формирует ответ")
    parsed = await forward(LLM_URL+"/analyze","POST",
                           {"text":text,"profile":data.profile,
                            "history":data.history,"memory":memory})
    # Tool calls cannot be represented as words. Block invented external work
    # before any model-generated "lessons" are proposed to the teacher.
    if unverified_external_activity(parsed.get("reply","")):
        parsed["reply"] = (
            "Вижу информацию, которую ты прислал, но не подключался и не "
            "запрашивал доступ. В этой версии у меня нет HTTP-инструмента "
            "или внешнего коннектора, поэтому я не могу проверить адрес "
            "и получить токен. Не отправляй секреты в чат: для реального "
            "подключения разработчик должен предоставить защищённый инструмент."
        )
        parsed["proposals"] = []
        parsed["missing"] = "Внешний инструмент подключения с отдельной настройкой доступа"
        parsed["integrity_warning"] = "Model claimed an external action without a tool call"
    # Web evidence is only looked up when policy allows it. In ask mode
    # JARVIS can propose a search but must wait for explicit user approval.
    wanted=proposed_web_query(text,parsed)
    if wanted and data.web_mode=="auto":
        result=await answer_from_public_search(data,text,memory,wanted,start)
        result["planner_model"]=parsed.get("model")
        result["planner_tokens"]=parsed.get("tokens",0)
        result["tokens"]=int(result.get("tokens") or 0)+int(parsed.get("tokens") or 0)
        return result
    if wanted and data.web_mode=="ask":
        if data.session_id:
            approval=await forward(RESEARCH_URL+"/research/operations/web/propose",
                 "POST",{"session_id":data.session_id,"query":wanted,"question":text})
            parsed["web_approval"]=approval
            parsed["reply"]=(
                "Для проверки нужен публичный поиск. "
                f"Предлагаю запрос: «{wanted}». "
                "Скажи «Разрешаю поиск» или нажми кнопку. "
                "До согласия запрос не отправляется.")
        else:
            parsed["reply"]="Для согласия на поиск нужен сохранённый диалог."
        parsed["web_requested"]=True
    if wanted and data.web_mode=="off":
        parsed["reply"]=(
            "Для актуального ответа мне нужен публичный поиск, но "
            "интернет отключён в этом эксперименте. "
            "Я не буду выдавать непроверенную информацию за найденную.")
    change_proposal=None
    if parsed.get("changes") and isinstance(parsed["changes"],list):
        saved_changes=await forward(RESEARCH_URL+"/apprentice/changes/propose","POST",
            {"source_text":text,"changes":parsed["changes"][:3]})
        if saved_changes.get("changes"):
            change_proposal=saved_changes["changes"][0]
            parsed["reply"]=(
                "Нужно обновить подтверждённое знание. "
                f"Предлагаю заменить «{change_proposal['expected_value']}» на "
                f"«{change_proposal['proposed_value']}». "
                "Пока изменение не применено: подтверди ниже.")
            parsed["proposals"]=[]
    if re.search(r"(?iu)\b(?:я\s+переименован|я\s+переименовал(?:ся)?|"
                 r"я\s+изменил\s+(?:сво[её]\s+)?имя)\b",
                 parsed.get("reply","")) and not change_proposal:
        parsed["reply"]=(
            "Я мог ошибочно заявить, что переименование выполнено. "
            "У меня нет подтверждённой записи об изменении в этом запросе. "
            "Попроси изменить имя существующего объекта — я предложу "
            "проверяемое изменение SQLite.")
        parsed["integrity_warning"]="Blocked unsupported self-mutation claim"
        parsed["proposals"]=[]
    # Speech transcription omits punctuation. The verified intent classifier
    # gates writes; a model saying "teach" is not a write permission.
    interpreted=memory.get("utterance_intent") or {}
    if interpreted and interpreted.get("intent")!="statement":
        parsed["proposals"]=[]
    # Do not turn information requests or model guesses into confirmed
    # teaching proposals: only explicit teaching can create candidates.
    if (parsed.get("intent") in ("ask","request")
            or "?" in text or re.match(
                r"(?iu)^(какие|какой|что|чем|сколько|как|почему|где|"
                r"когда|можешь ли|ты можешь|what|how)\b",text.strip())):
        parsed["proposals"]=[]
    proposals = []
    # The research service also has a transparent language-specific teaching
    # baseline. Call it for first-person declarations even when Qwen misses
    # a relationship. All extracted triples remain pending teacher review.
    declaration = (interpreted.get("intent")=="statement" and bool(
        re.search(r"(?iu)\bу\s+меня\b",text)))
    if parsed.get("proposals") or declaration:
        stored = await forward(RESEARCH_URL+"/apprentice/propose","POST",
                               {"source_text":text,"proposals":parsed.get("proposals") or []})
        proposals = stored.get("proposals",[])
    missing = parsed.get("missing","")
    if not isinstance(missing,str) or missing.casefold().startswith(
        ("конкретный недостающий ресурс", "конкретна недостаюча інформація")):
        missing = ""
    voice_review=None
    if data.session_id and (change_proposal or proposals):
        voice_review=await forward(RESEARCH_URL+"/apprentice/voice/queue","POST",
            {"session_id":data.session_id,"changes":[change_proposal] if change_proposal else [],
             "proposals":proposals})
    if voice_review and voice_review.get("ready"):
        parsed["reply"]=(str(parsed.get("reply") or "").rstrip()+" "+
                         voice_review.get("instruction","")).strip()
    return {"answer":parsed.get("reply",""),"route":"apprentice_tutor",
            "knowledge_evidence":memory.get("graph") or {},
            "normalized_input":text if text != data.text else None,
            "intent":parsed.get("intent","chat"),
            "missing":missing,
            "parse_warning":parsed.get("parse_warning"),
            "change_proposal":change_proposal,
            "utterance_intent":memory.get("utterance_intent"),
            "voice_review":voice_review,
            "integrity_warning":parsed.get("integrity_warning"),
            "web_approval":parsed.get("web_approval"),
            "web_requested":bool(wanted),
            "search_status":"approval_needed" if wanted and data.web_mode=="ask"
                             else "disabled" if wanted and data.web_mode=="off" else "not_needed",
            "proposals":proposals,"model":parsed.get("model"),
            "tokens":parsed.get("tokens",0),
            "prompt_tokens":parsed.get("prompt_tokens",0),
            "executed":False,
            "total_ms":round((time.perf_counter()-start)*1000,2)}

def probable_graph_inspection(text: str) -> bool:
    """Cheap gate; Research decides whether it can answer, not this heuristic."""
    q = text.casefold().replace("ё","е")
    # An explicitly named legacy index is a different experimental dataset.
    # Do not silently substitute the new apprentice graph for old index metrics.
    if "индекс" in q and "граф" not in q:
        return False
    has_verb = any(w in q for w in (
        "сколько","что","какие","перечис","покаж","назови","список",
        "помнишь","помниш","что у меня","які","перелік"))
    about_memory = any(w in q for w in (
        "сущност","сутност","entity","объект","обьект","связ",
        "отношен","факт","памят","помниш","помнишь","знан",
        "сохранил","дома","в доме","домашн","вдома"))
    return bool(has_verb and about_memory)


async def chat_core(data: ChatIn):
    start = time.perf_counter()
    progress(data.request_id,"routing","Выбираю маршрут ответа")
    text = data.text.strip()
    if data.experiment == "apprentice":
        return await apprentice_chat(data,start)
    if data.experiment != "llm_only":
        # Same authoritative graph can be inspected from all memory-aware modes.
        # llm_only remains an intentionally isolated frozen-model control.
        if probable_graph_inspection(text) and data.experiment in ("hybrid","research_only"):
            checked = await forward(RESEARCH_URL+"/apprentice/inspect",
                                    "POST",{"query":text})
            if checked.get("matched") and checked.get("inspection"):
                inspected = checked["inspection"]
                return {"answer":inspected["answer"],"route":"sqlite_graph_inspection",
                        "experiment":data.experiment,"retrieval_trace":inspected["trace"],
                        "evidence_source":inspected["source"],
                        "normalized_input":checked.get("normalized_text"),
                        "model":"SQLite","tokens":0,"executed":False,
                        "total_ms":round((time.perf_counter()-start)*1000,2)}
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
    progress(data.request_id,"planning","Qwen: выполняется контрольный диалог")
    answer = await forward(LLM_URL+"/chat","POST",{"text":text,"profile":data.profile})
    answer["total_ms"] = round((time.perf_counter()-start)*1000,2)
    answer["experiment"] = data.experiment
    return answer


@app.post("/api/chat", dependencies=[Depends(auth)])
async def chat(data: ChatIn):
    """Persist a complete model/retrieval turn without affecting control modes."""
    progress(data.request_id,"received","Запрос принят")
    try:
        result=await chat_core(data)
    except HTTPException as error:
        progress(data.request_id,"error","Ошибка обработки: "+str(error.detail)[:70],True)
        if data.session_id:
            try:
                await forward(
                    RESEARCH_URL+"/research/sessions/"+data.session_id+"/turns",
                    "POST",{
                        "question":data.text,
                        "answer":"[Не удалось получить ответ: "+str(error.detail)[:500]+"]",
                        "experiment":data.experiment,"profile":data.profile,
                        "ablation":data.ablation,"web_mode":data.web_mode,
                        "result":{"route":"chat_error","model":"none",
                                  "error_status":error.status_code,
                                  "search_status":"upstream_error"}})
            except HTTPException:
                pass
        raise
    if data.session_id:
        progress(data.request_id,"saving","SQLite: сохраняю ответ и источники")
        try:
            saved=await forward(
                RESEARCH_URL+"/research/sessions/"+data.session_id+"/turns",
                "POST",{
                    "question":data.text,"answer":str(result.get("answer",""))[:6000],
                    "experiment":data.experiment,"profile":data.profile,
                    "ablation":data.ablation,"web_mode":data.web_mode,
                    "result":result})
            result["turn_id"]=saved["turn_id"]
            result["session_id"]=data.session_id
        except HTTPException as error:
            # Never pretend unsaved turns were persisted. Return answer with
            # explicit warning so the UI can let the teacher export manually.
            result["persistence_warning"]=str(error.detail)[:300]
    progress(data.request_id,"done","Ответ готов",True)
    return result
