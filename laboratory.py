import json
import os
import random
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from common import post, serve

WORLD = os.getenv("WORLD_URL", "http://world:8000")
AGENT = os.getenv("AGENT_URL", "http://agent:8000")
MODEL = os.getenv("MODEL", "qwen3:4b")
DB = os.getenv("DB_PATH", "data/lab.sqlite3")
LOCK = threading.Lock()

def connect():
    return sqlite3.connect(DB, timeout=30)

def init_db():
    Path(DB).parent.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS knowledge (profile TEXT, world TEXT, model TEXT, body TEXT, PRIMARY KEY(profile,world,model))")
        for rid, body in db.execute("SELECT id,body FROM runs").fetchall():
            r = json.loads(body)
            if r["status"] in ("queued", "running"):
                r["status"] = "interrupted"
                db.execute("UPDATE runs SET body=? WHERE id=?", (json.dumps(r), rid))

def save(r):
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO runs VALUES (?,?)", (r["id"], json.dumps(r, ensure_ascii=False)))

def get_run(rid):
    with connect() as db:
        row = db.execute("SELECT body FROM runs WHERE id=?", (rid,)).fetchone()
    if row is None:
        raise ValueError("Unknown run")
    return json.loads(row[0])

def load_knowledge(profile, world):
    with connect() as db:
        row = db.execute("SELECT body FROM knowledge WHERE profile=? AND world=? AND model=?",
                         (profile, world, MODEL)).fetchone()
    return json.loads(row[0]) if row else {}

def save_knowledge(profile, world, knowledge):
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO knowledge VALUES (?,?,?,?)",
                   (profile, world, MODEL, json.dumps(knowledge, ensure_ascii=False)))

def world(path, world_id, **data):
    return post(WORLD, path, {"world": world_id, **data})

def ask(r, task, **data):
    prompt = json.dumps({"task": task, "interface": r["interface"], **data}, ensure_ascii=False)
    result = post(AGENT, "/ask", {"prompt": prompt, "seed": r["config"]["seed"]})
    r["llm_calls"].append({"task": task, "model": result.get("model"),
                           "tokens": result.get("tokens"), "prompt_tokens": result.get("prompt_tokens")})
    return result["answer"]

def prediction(value, allowed):
    if type(value) is not int or value not in allowed:
        raise ValueError("Invalid prediction: " + repr(value))
    return value

def knowledge_text(value):
    if not isinstance(value, str) or len(value) > 2400:
        raise ValueError("Knowledge must be a string of at most 2400 characters")
    return value

def test_model(r, cases, knowledge):
    # The agent never receives expected values or test feedback.
    rows = [{k: v for k, v in c.items() if k != "expected"} for c in cases]
    answer = ask(r, "Predict the output for every independent case. Return JSON with field predictions: an object mapping each case id (string) to its integer predicted output. If uncertain, make your best guess.",
                 knowledge=knowledge, cases=rows)
    p = answer["predictions"]
    if not isinstance(p, dict) or set(p) != {str(c["id"]) for c in cases}:
        raise ValueError("Predictions must contain exactly all case ids")
    results = []
    for c in cases:
        v = prediction(p[str(c["id"])], r["interface"]["values"])
        results.append({**c, "prediction": v, "correct": v == c["expected"]})
    return {"correct": sum(x["correct"] for x in results), "total": len(results), "cases": results}

def run_experiment(r):
    cfg = r["config"]
    wid = cfg["world"]
    rng = random.Random(cfg["seed"])
    try:
        r["status"] = "running"
        r["phase"] = "baseline"
        catalog = post(WORLD, "/catalog", {})
        r["interface"] = next(w for w in catalog["worlds"] if w["id"] == wid)
        r["world_audit"] = world("/audit", wid)
        cases = world("/cases", wid)["cases"]
        previous = load_knowledge(cfg["profile"], wid) if cfg["reuse"] else {}
        learned = previous.get("knowledge", "")
        r["initial_knowledge"] = learned
        r["knowledge"] = learned
        save(r)
        r["baseline"] = test_model(r, cases, learned)
        r["phase"] = "exploration"
        state = world("/start", wid)
        save(r)
        for i in range(cfg["steps"]):
            event = rng.choice(r["interface"]["events"]) if rng.random() < cfg["event_probability"] else None
            kind = "events" if event is not None else "actions"
            # Only public observations are passed; never state, expected or audit fields.
            recent = [{k: x[k] for k in ("before", "kind", "input", "after", "prediction")}
                      for x in r["steps"][-6:]]
            answer = ask(r, "Update concise general hypotheses using the recent evidence, choose an allowed input if it is your turn, and predict the next output before observing it. Return JSON fields knowledge (string, max 2400 characters), input (string), prediction (integer). If an external event is announced, input must equal that event. Aim to discover the laws, not maximize the output.",
                         knowledge=learned, recent=recent, y=state["y"], kind=kind, announced_event=event)
            learned = knowledge_text(answer["knowledge"])
            name = answer["input"]
            if not isinstance(name, str) or name not in r["interface"][kind] or (event is not None and name != event):
                raise ValueError("Invalid chosen input: " + repr(name))
            pred = prediction(answer["prediction"], r["interface"]["values"])
            new = world("/step", wid, state=state["state"], kind=kind, input=name)
            r["steps"].append({"step": i+1, "before": state["y"], "kind": kind,
                               "input": name, "prediction": pred, "after": new["y"],
                               "correct": pred == new["y"], "knowledge_before_result": learned,
                               "audit_state_before": state["state"], "audit_state_after": new["state"]})
            state = new
            r["knowledge"] = learned
            save(r)
        r["phase"] = "consolidation"
        save(r)
        recent = [{k: x[k] for k in ("before", "kind", "input", "after", "prediction")}
                  for x in r["steps"][-6:]]
        learned = knowledge_text(ask(r, "Consolidate your hypotheses from the latest observations and previous knowledge. Separate supported rules from uncertainty. Return JSON with knowledge (string, max 2400 characters).",
                                     knowledge=learned, recent=recent)["knowledge"])
        r["knowledge"] = learned
        r["phase"] = "evaluation"
        save(r)
        r["evaluation"] = test_model(r, cases, learned)
        save_knowledge(cfg["profile"], wid, {"knowledge": learned, "run_id": r["id"]})
        r["status"] = "done"
        r["phase"] = "done"
    except Exception as exc:
        r["status"] = "error"
        r["error"] = str(exc)
    finally:
        save(r)

def worker(r):
    try:
        run_experiment(r)
    finally:
        LOCK.release()

def start(data):
    wid = data["world"]
    profile = data.get("profile", "default")
    steps = data.get("steps", 12)
    seed = data.get("seed", 42)
    prob = data.get("event_probability", .25)
    reuse = data.get("reuse", False)
    if not isinstance(wid, str) or not isinstance(profile, str) or not 1 <= len(profile) <= 80:
        raise ValueError("Invalid world or profile")
    if type(steps) is not int or not 1 <= steps <= 100 or type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("Invalid steps or seed")
    if type(prob) not in (float, int) or not 0 <= prob <= 1 or type(reuse) is not bool:
        raise ValueError("Invalid probability or reuse flag")
    if not LOCK.acquire(blocking=False):
        raise ValueError("An experiment is already running")
    try:
        r = {"id": uuid.uuid4().hex[:12], "created": time.time(), "model": MODEL,
             "status": "queued", "phase": "queued",
             "config": {"world": wid, "profile": profile, "steps": steps, "seed": seed,
                        "event_probability": prob, "reuse": reuse},
             "steps": [], "llm_calls": [], "knowledge": ""}
        save(r)
        threading.Thread(target=worker, args=(r,), daemon=True).start()
        return {"id": r["id"]}
    except Exception:
        LOCK.release()
        raise

def dispatch(method, path, data):
    if method == "GET" and path == "/":
        return "text/html", PAGE
    if method == "GET" and path == "/api/worlds":
        return post(WORLD, "/catalog", {})
    if method == "POST" and path == "/api/runs":
        return start(data)
    if method == "GET" and path == "/api/runs":
        with connect() as db:
            runs = [json.loads(x[0]) for x in db.execute("SELECT body FROM runs")]
        return {"runs": [{"id": r["id"], "status": r["status"], "config": r["config"],
                         "created": r["created"]} for r in sorted(runs, key=lambda x: x["created"], reverse=True)]}
    if method == "GET" and path.startswith("/api/runs/"):
        return get_run(path.split("/")[-1])
    raise ValueError("Unknown endpoint")

PAGE = """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Process learning laboratory</title><style>
:root{color-scheme:dark}body{font:15px system-ui;max-width:1000px;margin:32px auto;padding:0 16px;background:#111820;color:#e8eef5}h1{font-size:28px}h2{font-size:18px}section{border:1px solid #344356;border-radius:12px;padding:18px;margin:16px 0}label{display:flex;flex-direction:column;gap:6px}input,select,button{font:inherit;padding:8px;border:1px solid #53657b;border-radius:6px;background:#1e2936;color:inherit}button{cursor:pointer;background:#245d88}button:disabled{opacity:.5}form,.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.metrics b{display:block;font-size:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1b2633;padding:12px;border-radius:6px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #344356}.ok{color:#74d5a0}.bad{color:#ff9090}small{color:#a8b6c7}a{color:#8cc7ff}#error{color:#ff9090}
</style><body><h1>Лаборатория обучения Qwen</h1><small>Неизвестный мир · наблюдение · гипотеза · проверка</small>
<section><h2>Новый эксперимент</h2><form id="form">
<label>Мир<select id="world" required></select></label>
<label>Профиль<input id="profile" value="default" required maxlength="80"></label>
<label>Шаги<input id="steps" type="number" value="12" min="1" max="100" required></label>
<label>Seed<input id="seed" type="number" value="42" min="0" max="2147483647" required></label>
<label>Вероятность события<input id="prob" type="number" value="0.25" min="0" max="1" step="0.01" required></label>
<label>Память<select id="reuse"><option value="false">Чистый старт</option><option value="true">Использовать знания</option></select></label>
<button id="start">Запустить</button></form><p id="error"></p></section>
<section><h2>История запусков</h2><select id="runs"></select> <a id="export" hidden>JSON отчёт</a></section>
<section><h2 id="status">Нет выбранного запуска</h2><div class="metrics"><div>До обучения<b id="baseline">—</b></div><div>После опыта<b id="evaluation">—</b></div><div>Прогнозы на шагах<b id="online">—</b></div></div>
<h2>Текущие знания модели</h2><pre id="knowledge">—</pre><h2>Наблюдения</h2><div style="overflow:auto"><table><thead><tr><th>Шаг</th><th>До</th><th>Вход</th><th>Прогноз</th><th>Факт</th></tr></thead><tbody id="rows"></tbody></table></div>
<details><summary>Полный отчёт (включая скрытые правила для исследователя)</summary><pre id="report"></pre></details></section>
<script>
const $=id=>document.getElementById(id);let selected='',polling=false;
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw Error(j.error||r.status);return j}
function opt(value,label){const o=document.createElement('option');o.value=value;o.textContent=label;return o}
function score(x){return x?x.correct+'/'+x.total:'—'}
async function refresh(){if(polling)return;polling=true;try{
const {runs}=await api('/api/runs');$('runs').replaceChildren(...runs.map(r=>opt(r.id,r.id+' · '+r.config.world+' · '+r.status)));
$('start').disabled=runs.some(r=>['queued','running'].includes(r.status));
if(!selected&&runs.length)selected=runs[0].id;if(!selected)return;$('runs').value=selected;
const r=await api('/api/runs/'+selected);$('status').textContent=r.status+' / '+r.phase+' · '+r.model+' · '+r.steps.length+'/'+r.config.steps+(r.error?' — '+r.error:'');
$('baseline').textContent=score(r.baseline);$('evaluation').textContent=score(r.evaluation);$('online').textContent=r.steps.filter(s=>s.correct).length+'/'+r.steps.length;
$('knowledge').textContent=r.knowledge||'—';$('rows').replaceChildren(...r.steps.map(s=>{const tr=document.createElement('tr');for(const v of [s.step,s.before,s.kind+': '+s.input,s.prediction,s.after]){const td=document.createElement('td');td.textContent=v;tr.append(td)}tr.className=s.correct?'ok':'bad';return tr}));
$('report').textContent=JSON.stringify(r,null,2);$('export').href='/api/runs/'+selected;$('export').download=selected+'.json';$('export').hidden=false;
}catch(e){$('error').textContent=e.message}finally{polling=false}}
$('runs').onchange=()=>{selected=$('runs').value;refresh()};
$('form').onsubmit=async e=>{e.preventDefault();$('error').textContent='';$('start').disabled=true;try{const r=await api('/api/runs',{world:$('world').value,profile:$('profile').value,steps:Number($('steps').value),seed:Number($('seed').value),event_probability:Number($('prob').value),reuse:$('reuse').value==='true'});selected=r.id;await refresh()}catch(e){$('error').textContent=e.message;$('start').disabled=false}};
async function boot(){try{const j=await api('/api/worlds');$('world').replaceChildren(...j.worlds.map(w=>opt(w.id,w.id+' | '+w.actions.join(', ')+' / '+w.events.join(', '))));await refresh()}catch(e){$('error').textContent=e.message;setTimeout(boot,3000)}}boot();setInterval(refresh,2000);
</script></body></html>"""

if __name__ == "__main__":
    init_db()
    serve(dispatch)
