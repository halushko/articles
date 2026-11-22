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

app = FastAPI(title="JARVIS Research Workbench", version="0.4.0", lifespan=lifespan, docs_url=None)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parents[2]/"web")), name="static")

class ChatIn(BaseModel):
    text: str = Field(min_length=1,max_length=3000)
    experiment: str = Field(default="hybrid", pattern="^(apprentice|hybrid|research_only|llm_only)$")
    profile: str = Field(default="fast",pattern="^(fast|deep)$")
    ablation: str = Field(default="combined",pattern="^(exact|online|combined)$")
    history: list[dict] = Field(default_factory=list,max_length=14)

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
    result = {"version":"0.4.0", "gateway":"ok", "research":None,"llm":None}
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


@app.get("/api/apprentice/state", dependencies=[Depends(auth)])
async def apprentice_state():
    return await forward(RESEARCH_URL+"/apprentice/state")

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


async def apprentice_chat(data: ChatIn, start: float):
    # Only confirmed knowledge is given to the interpreter; guesses stay pending.
    memory = await forward(RESEARCH_URL+"/apprentice/context","POST",{"query":data.text})
    text = memory.get("normalized_text", data.text)
    direct_reply = memory.get("direct_reply")
    # Names of speakers and explicit pronunciation lessons are resolved in SQLite.
    # They must never require an LLM guess or create a spurious proposal.
    if direct_reply:
        return {"answer":direct_reply, "route":"apprentice_confirmed_context",
                "intent":"ask", "missing":"", "proposals":[], "model":"SQLite",
                "tokens":0, "executed":False,
                "normalized_input":text if text != data.text else None,
                "total_ms":round((time.perf_counter()-start)*1000,2)}
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
    proposals = []
    if parsed.get("proposals"):
        stored = await forward(RESEARCH_URL+"/apprentice/propose","POST",
                               {"source_text":text,"proposals":parsed["proposals"]})
        proposals = stored.get("proposals",[])
    missing = parsed.get("missing","")
    if not isinstance(missing,str) or missing.casefold().startswith(
        ("конкретный недостающий ресурс", "конкретна недостаюча інформація")):
        missing = ""
    return {"answer":parsed.get("reply",""),"route":"apprentice_tutor",
            "normalized_input":text if text != data.text else None,
            "intent":parsed.get("intent","chat"),
            "missing":missing,
            "parse_warning":parsed.get("parse_warning"),
            "integrity_warning":parsed.get("integrity_warning"),
            "proposals":proposals,"model":parsed.get("model"),
            "tokens":parsed.get("tokens",0),
            "prompt_tokens":parsed.get("prompt_tokens",0),
            "executed":False,
            "total_ms":round((time.perf_counter()-start)*1000,2)}

@app.post("/api/chat", dependencies=[Depends(auth)])
async def chat(data: ChatIn):
    start = time.perf_counter()
    text = data.text.strip()
    if data.experiment == "apprentice":
        return await apprentice_chat(data,start)
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
