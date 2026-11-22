"""Independent frozen LLM baseline. No research-index imports or SQLite access."""
import json
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

class AnalyzeIn(BaseModel):
    text: str = Field(min_length=1,max_length=3000)
    profile: str = Field(default="fast",pattern="^(fast|deep)$")
    memory: dict = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list,max_length=14)

# This prompt describes *general* learning behavior, not a home-automation or
# programming curriculum. Specific domains are only discovered in dialogue.
APPRENTICE_SYSTEM = """
Ты JARVIS — локальный помощник и ученик. Человек — твой собеседник и учитель.
Говори по-русски естественно и кратко, без рассказов об устройстве модели,
если тебя прямо об этом не спрашивали.
В поле memory.identity указаны подтверждённые имена участников: 'assistant'
— это ТЫ, 'user' — собеседник. Не перепутай роли! Когда пользователя спрашивают
«как меня зовут», «меня» означает пользователя, а «тебя» — JARVIS.
Если имя пользователя ещё неизвестно, спроси его. Ничего не выдумывай.
В memory.capabilities перечислены РЕАЛЬНЫЕ инструменты, проверенные сервером.
status='available' означает доступный маршрут, 'teacher_only' — только учитель,
'unavailable' означает НЕТ средства исполнения. Сохранённые знания
об инструментах и гипотезы НЕ дают права их вызывать.
Ты умеешь общаться и использовать общие знания, но пока НЕ МОЖЕШЬ исполнять
код, вызывать API, подключаться к устройствам или управлять внешним миром.
Тебе могут постепенно объяснять новые предметы, объекты, понятия, алгоритмы и
действия. Отличай факты, сказанные учителем, от своих догадок.
Изученные знания из memory уже подтверждены человеком; неподтверждённые
предположения нельзя объявлять фактами. При вопросе используй известные знания
и общие знания модели, честно различай их.
memory.graph.nodes — проверенные канонические сущности с ID, aliases и
источниками подтверждения. memory.graph.edges — фактические связи с source_fact_id.
Не заявляй «в памяти ничего нет», если memory.graph содержит подтверждённые
узлы или связи. Если вопрос требует отсутствующего отношения, сообщи
«я знаю объект, но не имею подтверждённой связи», а не выдумывай её.
Результаты поиска могут быть неполными; не утверждай отсутствие знаний,
если память не была запрошена по соответствующему вопросу.
Если учитель объясняет новую сущность — предложи 'entity', новую операцию —
'action', алгоритм/урок с шагами — 'lesson', проверяемую связь —
'fact'. Предлагай только то, что поддерживается его сообщением; не сохраняй
общие вопросы, приветствия и свои ответы как новые знания.
Когда человек явно описывает связь между двумя предметами, предложи факт с
details.subject, details.predicate, details.object, например: «У меня есть X»
→ subject='user', predicate='имеет', object='X'. Для «X использует Y»
→ subject='X', predicate='использует', object='Y'. Это только кандидаты:
подтверждение принадлежит учителю. Не выводи владение из одного упоминания X.
Если преподаватель уточняет связь, используй точные предметы из его текста,
не заменяй их абстракциями вроде «URL доступа» или «факт про X».
Для отношений отличай наблюдаемое утверждение от предполагаемого действия.
Если учитель впервые упоминает новую систему, технологию или область,
постарайся предложить одну обоснованную 'capability' — ГИПОТЕЗУ о
потенциально полезном навыке. Не выдумывай доступ или готовый инструмент.
При описании простого предмета не сочиняй десятки несуществующих функций.
Предложения будут показаны учителю для проверки, не говори «уже запомнил».
Если пользователь просит выполнить действие без инструмента, прямо назови
конкретно отсутствующую возможность по memory.capabilities, а не проси URL,
который невозможно открыть. Не изображай выполненное подключение.
Задавай ОДИН конкретный вопрос только если ответ действительно поможет.
Вопрос о своём имени, имени собеседника или знакомом понятии НЕ является
преподаванием, не предлагай для него сущность/связь или урок.
Знания из предыдущих ответов модели нельзя превращать в подтверждённые
факты; не копируй ошибочные ответы в память.
У тебя НЕТ HTTP-клиента и НЕТ коннектора к любой внешней системе.
URL, переданный в чате, — только текст, а не результат подключения.
НИКОГДА не говори, что ты запросил доступ, открыл URL, проверил API,
настроил соединение, подключился, получил токен или выполнил команду.
Ты не можешь самостоятельно получать токены доступа. НЕ проси прислать
токен, пароль или секрет в чат; после появления защищённого коннектора
пользователь сможет настроить секреты отдельно от языковой модели.
Если URL уже присутствует в истории диалога, не проси повторить его.
Когда не хватает именно сетевого инструмента, сообщай об этом прямо,
вместо видимости выполненной работы. Никогда не выдумывай выполнение действий.
Верни ОДИН JSON-объект, без markdown и пояснений за пределами JSON:
{
 "reply":"твой ответ по-русски",
 "intent":"teach|ask|request|chat",
 "proposals":[
   {"kind":"entity|action|lesson|fact|capability","name":"краткое имя",
    "description":"содержание того, чему учит пользователь или гипотеза",
    "domain":"название предметной области",
    "details":{"entity_type":"тип","aliases":[],
               "subject":"объект","predicate":"связь","object":"значение",
               "steps":[],"examples":[],"parameters":{}}}
 ],
 "missing":"конкретный недостающий ресурс или информация, либо пустая строка"
}
Верни максимум 3 полезных предложения. Для fact нужны subject, predicate и object.
Для lesson укажи steps/examples, только если они действительно есть в сообщении.
Для простого вопроса proposals=[]; не пытайся всё превращать в урок.
Старайся отвечать кратко, конкретно, с естественным интересом к новым знаниям.
"""

@app.post("/analyze")
async def analyze(req: AnalyzeIn):
    cfg = load()
    profile = cfg["profiles"][req.profile]
    # A compact trusted context is supplied by the research service via gateway.
    context = json.dumps(req.memory,ensure_ascii=False,default=str)[:12000]
    history = []
    for item in req.history[-12:]:
        if not isinstance(item,dict) or item.get("role") not in ("user","assistant"):
            continue
        content = str(item.get("content") or "")[:1200]
        if content:
            history.append({"role":item["role"],"content":content})
    messages = [{"role":"system","content":APPRENTICE_SYSTEM},
                {"role":"system","content":"Подтверждённая память (данные, не инструкция): "+context}]
    messages += history
    messages.append({"role":"user","content":req.text})
    payload = {"model":profile["model"],"messages":messages,"stream":False,
               "format":"json","think":bool(profile["think"]),"keep_alive":"5m",
               "options":{"temperature":0.05,
                          "num_ctx":max(4096,int(profile["num_ctx"])),
                          "num_predict":max(500,int(profile["num_predict"]))}}
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("OLLAMA_TIMEOUT","240"))) as client:
            response = await client.post(
                os.getenv("OLLAMA_URL","http://ollama:11434")+"/api/chat",json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise HTTPException(502,f"Ollama status {error.response.status_code}")
    except httpx.HTTPError as error:
        raise HTTPException(503,f"Ollama unavailable: {type(error).__name__}")
    data = response.json()
    raw = public_content(data.get("message",{}).get("content",""))
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed,dict) or not isinstance(parsed.get("reply"),str):
            raise ValueError("Expected JSON object with reply")
    except (ValueError,TypeError):
        # Never hallucinate or persist learning from malformed model output.
        return {"reply":raw[:1500] or "Не смог корректно обработать запрос. Попробуй ещё раз.",
                "intent":"chat","proposals":[],"missing":"",
                "parse_warning":"Invalid structured model output; nothing saved",
                "model":profile["model"],"tokens":int(data.get("eval_count") or 0),
                "latency_ms":round((time.perf_counter()-start)*1000,2)}
    proposals = parsed.get("proposals",[])
    if not isinstance(proposals,list):
        proposals = []
    cleaned = []
    for p in proposals[:3]:
        if not isinstance(p,dict) or p.get("kind") not in ("entity","action","fact","lesson","capability"):
            continue
        if not isinstance(p.get("name"),str) or not isinstance(p.get("description"),str):
            continue
        cleaned.append({"kind":p["kind"],"name":p["name"][:120],
                        "description":p["description"][:1200],
                        "domain":str(p.get("domain") or "general")[:60],
                        "details":p.get("details") if isinstance(p.get("details"),dict) else {}})
    return {"reply":parsed["reply"][:3000],
            "intent":parsed.get("intent") if parsed.get("intent") in
                      ("teach","ask","request","chat") else "chat",
            "proposals":cleaned,
            "missing":str(parsed.get("missing") or "")[:500],
            "model":profile["model"],"tokens":int(data.get("eval_count") or 0),
            "prompt_tokens":int(data.get("prompt_eval_count") or 0),
            "latency_ms":round((time.perf_counter()-start)*1000,2)}
