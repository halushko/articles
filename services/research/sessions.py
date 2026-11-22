"""Persistent, auditable research conversations. Separate from confirmed knowledge.

Messages, model/tool routes, source provenance and teacher decisions are stored
in the existing research SQLite volume. No Qwen output or search snippet is
implicitly promoted to a confirmed teaching.
"""
import json
import re
import uuid


SCHEMA = """
CREATE TABLE IF NOT EXISTS research_sessions (
 id TEXT PRIMARY KEY,
 title TEXT NOT NULL DEFAULT 'Новый диалог',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS research_turns (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
 actor_id TEXT NOT NULL DEFAULT '',
 question TEXT NOT NULL,
 answer TEXT NOT NULL,
 experiment TEXT NOT NULL,
 profile TEXT NOT NULL,
 ablation TEXT NOT NULL,
 web_mode TEXT NOT NULL DEFAULT 'off',
 route TEXT NOT NULL DEFAULT '',
 model TEXT NOT NULL DEFAULT '',
 tokens INTEGER NOT NULL DEFAULT 0,
 prompt_tokens INTEGER NOT NULL DEFAULT 0,
 total_ms REAL NOT NULL DEFAULT 0,
 provenance TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 rating TEXT,
 teacher_note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_research_turns_session ON research_turns(session_id,id);
CREATE TABLE IF NOT EXISTS research_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
 turn_id INTEGER REFERENCES research_turns(id) ON DELETE SET NULL,
 event_type TEXT NOT NULL,
 data TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_research_events_session ON research_events(session_id,id);
"""

def _json(value):
    return json.dumps(value,ensure_ascii=False,default=str)

def _dict(value):
    try:
        x=json.loads(value)
        return x if isinstance(x,dict) else {}
    except (ValueError,TypeError):
        return {}


def markdown(session):
    """Escaped enough for prose export; raw input never executed."""
    lines=[
        "# JARVIS — журнал исследовательского диалога",
        "",
        f"Сессия: `{session['id']}`",
        f"Начало: {session['created_at']} UTC",
        f"Последнее сообщение: {session['updated_at']} UTC",
        "",
        "Данные из модели, поиска и браузера НЕ равны подтверждённым знаниям.",
        ""
    ]
    events=session["events"]
    indexed={}
    for ev in events:
        if ev["turn_id"] is not None:
            indexed.setdefault(ev["turn_id"],[]).append(ev)
    for turn in session["turns"]:
        lines+=["---","",f"## Учитель · {turn['created_at']} UTC",
                "",turn["question"],"",f"## JARVIS · {turn['created_at']} UTC",
                "",turn["answer"],"",
                f"- Режим: `{turn['experiment']}`; профиль: `{turn['profile']}`; ablation: `{turn['ablation']}`",
                f"- Маршрут: `{turn['route']}`; модель: `{turn['model']}`",
                f"- Интернет: `{turn['web_mode']}`; токены: {turn['tokens']} / prompt {turn['prompt_tokens']}; задержка: {turn['total_ms']} мс",
                f"- Оценка: {turn['rating'] or 'не поставлена'}"
                + (f"; пояснение: {turn['teacher_note']}" if turn["teacher_note"] else "")]
        evidence=turn["provenance"]
        if evidence.get("sources"):
            lines+=["", "Источники внешнего поиска (не подтверждены учителем):"]
            for i,src in enumerate(evidence["sources"],1):
                lines.append(f"{i}. {src.get('title','Источник')}: {src.get('url','')} "
                             f"— {src.get('status','search_result')}")
        if evidence.get("retrieval_trace"):
            lines.append("Retrieval trace: `"+_json(evidence["retrieval_trace"])[:1800]+"`")
        for ev in indexed.get(turn["id"],[]):
            lines.append(f"- Событие {ev['created_at']} UTC: {ev['event_type']}; {_json(ev['data'])[:2000]}")
        lines.append("")
    standalone=[e for e in events if e["turn_id"] is None]
    if standalone:
        lines+=["---","", "## События сессии", ""]
        for ev in standalone:
            lines.append(f"- {ev['created_at']} UTC · {ev['event_type']}: {_json(ev['data'])[:7000]}")
    return "\n".join(lines)+"\n"


class ConversationStore:
    def __init__(self, engine):
        self.engine=engine
        with self.engine.connect() as con:
            con.executescript(SCHEMA)
            # Non-destructive migration: old research journals keep all turns.
            names={row["name"] for row in con.execute("PRAGMA table_info(research_turns)")}
            if "actor_id" not in names:
                con.execute("ALTER TABLE research_turns ADD COLUMN actor_id TEXT NOT NULL DEFAULT ''")

    def create(self,title=""):
        sid=str(uuid.uuid4())
        title=(str(title or "").strip() or "Новый диалог")[:140]
        with self.engine.connect() as con:
            con.execute("INSERT INTO research_sessions(id,title) VALUES(?,?)",(sid,title))
        return {"id":sid,"title":title}

    def list(self,limit=50):
        with self.engine.connect() as con:
            return [dict(r) for r in con.execute("""
                SELECT s.id,s.title,s.created_at,s.updated_at,
                    COUNT(t.id) AS turns
                FROM research_sessions s LEFT JOIN research_turns t ON t.session_id=s.id
                GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ?
            """,(min(100,max(1,limit)),))]

    def exists(self,sid):
        if not isinstance(sid,str) or not re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",sid):
            return False
        with self.engine.connect() as con:
            return con.execute("SELECT 1 FROM research_sessions WHERE id=?",(sid,)).fetchone() is not None

    def read(self,sid):
        if not self.exists(sid):
            raise ValueError("Диалог не найден")
        with self.engine.connect() as con:
            session=dict(con.execute(
                "SELECT * FROM research_sessions WHERE id=?",(sid,)).fetchone())
            turns=[]
            for r in con.execute("SELECT * FROM research_turns WHERE session_id=? ORDER BY id",(sid,)):
                item=dict(r)
                item["provenance"]=_dict(item["provenance"])
                turns.append(item)
            events=[]
            for r in con.execute("SELECT * FROM research_events WHERE session_id=? ORDER BY id",(sid,)):
                item=dict(r)
                item["data"]=_dict(item["data"])
                events.append(item)
        return {**session,"turns":turns,"events":events}

    def save_turn(self,sid,question,answer,experiment,profile,ablation,web_mode,result,actor_id=""):
        if not self.exists(sid):
            raise ValueError("Диалог не найден")
        question=str(question)[:3000]
        if not question.strip():
            raise ValueError("Пустой вопрос")
        answer=str(answer)[:6000]
        result=result if isinstance(result,dict) else {}
        provenance={key:result.get(key) for key in (
            "sources","search_query","search_status","search_error",
            "web_approval","web_requested","change_proposal",
            "voice_review","voice_review_result","utterance_intent",
            "conflict_detected","retrieval_trace","knowledge_evidence",
            "evidence_source","proposals","missing","intent",
            "parse_warning","integrity_warning","normalized_input",
            "external_origin","tool_trace","approval_id","approval_source","speaker","requested_action"
        ) if result.get(key) is not None}
        provenance=_json(provenance)[:60000]
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            cur=con.execute("""INSERT INTO research_turns(
                session_id,actor_id,question,answer,experiment,profile,ablation,web_mode,
                route,model,tokens,prompt_tokens,total_ms,provenance
              ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid,actor_id or "",question,answer,experiment,profile,ablation,web_mode,
                 str(result.get("route") or "")[:90],str(result.get("model") or "")[:90],
                 int(result.get("tokens") or 0),int(result.get("prompt_tokens") or 0),
                 float(result.get("total_ms") or 0),provenance))
            con.execute("UPDATE research_sessions SET updated_at=CURRENT_TIMESTAMP WHERE id=?",(sid,))
            con.execute("""UPDATE research_sessions SET title=? WHERE id=?
                           AND title='Новый диалог'""",(question[:90],sid))
        return {"turn_id":cur.lastrowid,"session_id":sid,"saved":True}

    def grade(self,sid,turn_id,rating,note=""):
        if rating not in ("good","bad"):
            raise ValueError("Некорректная оценка")
        with self.engine.connect() as con:
            row=con.execute("SELECT id FROM research_turns WHERE id=? AND session_id=?",
                            (turn_id,sid)).fetchone()
            if row is None:
                raise ValueError("Ответ не найден в указанном диалоге")
            con.execute("UPDATE research_turns SET rating=?,teacher_note=? WHERE id=?",
                        (rating,str(note)[:1200],turn_id))
            con.execute("""INSERT INTO research_events(session_id,turn_id,event_type,data)
                           VALUES(?,?,'grade',?)""",
                        (sid,turn_id,_json({"rating":rating,"note":str(note)[:1200]})))
        return {"saved":True,"turn_id":turn_id,"rating":rating}

    def event(self,sid,event_type,data,turn_id=None):
        if not self.exists(sid):
            raise ValueError("Диалог не найден")
        if not event_type or not re.fullmatch(r"[a-z][a-z0-9_.]{0,50}",event_type):
            raise ValueError("Некорректный тип события")
        if turn_id is not None:
            with self.engine.connect() as con:
                if not con.execute("SELECT 1 FROM research_turns WHERE id=? AND session_id=?",
                                   (turn_id,sid)).fetchone():
                    raise ValueError("Ответ не принадлежит диалогу")
        if not isinstance(data,dict):
            raise ValueError("Событие должно быть объектом")
        raw=_json(data)
        if len(raw)>15000:
            raise ValueError("Слишком большое событие")
        with self.engine.connect() as con:
            cur=con.execute("""INSERT INTO research_events(session_id,turn_id,event_type,data)
                               VALUES(?,?,?,?)""",(sid,turn_id,event_type,raw))
            con.execute("UPDATE research_sessions SET updated_at=CURRENT_TIMESTAMP WHERE id=?",(sid,))
        return {"event_id":cur.lastrowid,"saved":True}
