"""Durable, explicit operation approvals; model text never grants tool access.

Only public web search is wired to an executor. Other actions may be described
but MUST NOT be executed merely because a language model proposed them.
A request has a stable ID, exact arguments, a session, and a five-minute lease.
"""
import json
import re
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_operations (
 id TEXT PRIMARY KEY,
 session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
 actor_id TEXT NOT NULL DEFAULT '',
 kind TEXT NOT NULL,
 payload TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected','expired','superseded')),
 created_epoch REAL NOT NULL,
 resolved_epoch REAL
);
CREATE INDEX IF NOT EXISTS idx_pending_operations_session
 ON pending_operations(session_id,status,created_epoch);
CREATE TABLE IF NOT EXISTS operation_audit (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 operation_id TEXT NOT NULL,
 session_id TEXT NOT NULL,
 decision TEXT NOT NULL,
 source TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS operation_executions (
 operation_id TEXT PRIMARY KEY REFERENCES pending_operations(id),
 session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
 actor_id TEXT NOT NULL DEFAULT '',
 query TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('running','succeeded','failed')),
 answer TEXT NOT NULL DEFAULT '',
 sources TEXT NOT NULL DEFAULT '[]',
 error TEXT NOT NULL DEFAULT '',
 started_epoch REAL NOT NULL,
 finished_epoch REAL
);
CREATE INDEX IF NOT EXISTS idx_operation_executions_lookup
 ON operation_executions(session_id,actor_id,started_epoch);
"""
TTL = 300
# Full phrases only, not substrings of a conversation. A generic 'yes'
# MUST NOT approve internet access, knowledge edits, or device operations.
ALLOW = re.compile(r"(?iu)^(?:(?:я\s+)?разрешаю(?:\s+(?:этот\s+)?поиск)?|"
                   r"да\s*,?\s*разрешаю(?:\s+поиск)?|"
                   r"подтверждаю\s+поиск|"
                   r"разрешить\s+поиск)\s*[.!]*$")
DENY = re.compile(r"(?iu)^(?:(?:я\s+)?не\s+разрешаю(?:\s+поиск)?|"
                  r"запрещаю\s+поиск|отмена\s+поиска|"
                  r"отмени\s+поиск)\s*[.!]*$")


def spoken_search_decision(phrase):
    from .utterance import classify
    body=classify(phrase)["normalized"].strip()
    if ALLOW.fullmatch(body):
        return "approved"
    if DENY.fullmatch(body):
        return "rejected"
    return None


class OperationApprovals:
    def __init__(self,engine,sessions,behavior=None):
        self.engine=engine
        self.sessions=sessions
        self.behavior=behavior
        with engine.connect() as con:
            con.executescript(SCHEMA)
            if "actor_id" not in {r["name"] for r in con.execute("PRAGMA table_info(pending_operations)")}:
                con.execute("ALTER TABLE pending_operations ADD COLUMN actor_id TEXT NOT NULL DEFAULT ''")

    def propose_search(self,session_id,query,question,actor_id=""):
        if not self.sessions.exists(session_id):
            raise ValueError("Для согласия на поиск нужна сохранённая сессия")
        query=str(query or "").strip()[:240]
        if len(query)<3:
            raise ValueError("Пустой поисковый запрос")
        payload={"query":query,"question":str(question or "")[:3000]}
        operation_id=str(uuid.uuid4())
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("""UPDATE pending_operations SET status='superseded',
                resolved_epoch=? WHERE session_id=? AND status='pending'""",
                        (time.time(),session_id))
            con.execute("""INSERT INTO pending_operations
                (id,session_id,actor_id,kind,payload,status,created_epoch)
                VALUES(?,?,?, 'web.search',?, 'pending',?)""",
                (operation_id,session_id,actor_id or "",
                 json.dumps(payload,ensure_ascii=False),time.time()))
        return {"id":operation_id,"kind":"web.search","query":query,
                "question":payload["question"],"risk":"external_network",
                "expires_seconds":TTL,"status":"pending"}

    def clear(self,session_id):
        if not self.sessions.exists(session_id):
            return
        with self.engine.connect() as con:
            con.execute("""UPDATE pending_operations SET status='superseded',
                 resolved_epoch=? WHERE session_id=? AND status='pending'""",
                (time.time(),session_id))

    def resolve(self,session_id,phrase="",operation_id="",source="voice",actor_id=""):
        decision = spoken_search_decision(phrase) if source=="voice" else None
        if not decision and source=="voice" and self.behavior is not None:
            from .adaptive import WEB_CONTEXT
            learned=self.behavior.decision(phrase, WEB_CONTEXT, actor_id)
            decision={"approve":"approved","deny":"rejected"}.get(learned)
        if source=="button":
            decision="approved" if operation_id else None
        if not decision:
            return {"handled":False}
        return self._apply_decision(session_id,decision,operation_id,source,actor_id)

    def resolve_scoped(self,session_id,operation_id,decision,actor_id=""):
        """Internal dialogue-coordinator entry: exact operation + server policy.

        This is not exposed as a client-selectable decision override. The
        caller must have verified latest-offer, actor, intent and risk.
        """
        if decision not in ("approved","rejected") or not operation_id:
            raise ValueError("Invalid scoped decision")
        return self._apply_decision(session_id,decision,operation_id,
                                    "contextual_dialogue",actor_id)

    def _apply_decision(self,session_id,decision,operation_id,source,actor_id):
        if not self.sessions.exists(session_id):
            return {"handled":True,"status":"no_session","executed":False,
                    "answer":"Для разрешения поиска нужна активная сессия."}
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if operation_id:
                row=con.execute("""SELECT * FROM pending_operations
                      WHERE id=? AND session_id=?""",
                      (operation_id,session_id)).fetchone()
            else:
                row=con.execute("""SELECT * FROM pending_operations
                      WHERE session_id=? AND status='pending'
                      ORDER BY created_epoch DESC LIMIT 1""",(session_id,)).fetchone()
            if not row or row["status"]!="pending":
                return {"handled":True,"status":"no_pending_search","executed":False,
                        "answer":"Нет ожидающего разрешения на поиск. Сначала попроси найти информацию."}
            if row["actor_id"]!=(actor_id or ""):
                return {"handled":True,"status":"wrong_speaker","executed":False,
                        "answer":"Разрешение запрашивал другой собеседник. Повтори поиск от своего профиля."}
            if row["kind"]!="web.search":
                return {"handled":True,"status":"unsupported_operation","executed":False,
                        "answer":"Для этой операции нет разрешённого исполнителя."}
            if time.time()-row["created_epoch"]>TTL:
                con.execute("""UPDATE pending_operations SET status='expired',
                    resolved_epoch=? WHERE id=?""",(time.time(),row["id"]))
                return {"handled":True,"status":"expired","executed":False,
                        "answer":"Согласие на поиск устарело. Повтори запрос."}
            con.execute("""UPDATE pending_operations SET status=?,resolved_epoch=?
                       WHERE id=? AND status='pending'""",
                        (decision,time.time(),row["id"]))
            con.execute("""INSERT INTO operation_audit(operation_id,session_id,decision,source)
                       VALUES(?,?,?,?)""",(row["id"],session_id,decision,source))
        result={"handled":True,"status":decision,"operation_id":row["id"],
                "kind":"web.search","executed":False,
                "query":json.loads(row["payload"])["query"],
                "question":json.loads(row["payload"])["question"]}
        if decision=="rejected":
            result["answer"]="Поиск отменён. Запрос во внешний сервис не отправлен."
        return result

    def execution_start(self,session_id,operation_id,actor_id=""):
        """Atomically claim exactly one execution for an approved operation."""
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            approval=con.execute("""SELECT * FROM pending_operations
                WHERE id=? AND session_id=? AND actor_id=?
                  AND kind='web.search' AND status='approved'""",
                (operation_id,session_id,actor_id or "")).fetchone()
            if not approval:
                return {"started":False,"status":"not_approved","executed":False}
            existing=con.execute("""SELECT status FROM operation_executions
                WHERE operation_id=?""",(operation_id,)).fetchone()
            if existing:
                return {"started":False,"status":existing["status"],
                        "executed":existing["status"]=="succeeded"}
            query=json.loads(approval["payload"])["query"]
            con.execute("""INSERT INTO operation_executions
                (operation_id,session_id,actor_id,query,status,started_epoch)
                VALUES(?,?,?,?, 'running', ?)""",
                (operation_id,session_id,actor_id or "",query,time.time()))
        return {"started":True,"status":"running","operation_id":operation_id,
                "query":query,"executed":False}

    def execution_finish(self,session_id,operation_id,actor_id="",status="failed",
                         answer="",sources=None,error=""):
        """Persist the real executor result, not Qwen's assertion of progress."""
        if status not in ("succeeded","failed"):
            raise ValueError("Invalid operation execution status")
        sanitized=[]
        for source in (sources or [])[:5]:
            if isinstance(source,dict):
                sanitized.append({"title":str(source.get("title") or "")[:180],
                                  "url":str(source.get("url") or "")[:500],
                                  "status":str(source.get("status") or "")[:50]})
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            cur=con.execute("""UPDATE operation_executions
                SET status=?,answer=?,sources=?,error=?,finished_epoch=?
                WHERE operation_id=? AND session_id=? AND actor_id=?
                AND status='running'""",
                (status,str(answer)[:5000],json.dumps(sanitized,ensure_ascii=False),
                 str(error)[:400],time.time(),operation_id,session_id,actor_id or ""))
        return {"saved":cur.rowcount==1,"status":status,
                "operation_id":operation_id,"executed":status=="succeeded"}

    def execution_status(self,session_id,actor_id=""):
        """Read the real state of the most recent search and its last result."""
        if not self.sessions.exists(session_id):
            return {"current":None,"last_result":None}
        actor_id=actor_id or ""
        def expose(row):
            if not row:
                return None
            d=dict(row)
            try:
                d["sources"]=json.loads(d.get("sources") or "[]")
            except (TypeError,ValueError):
                d["sources"]=[]
            return d
        with self.engine.connect() as con:
            current=con.execute("""SELECT p.id AS operation_id,p.status AS approval_status,
                p.created_epoch,e.status AS execution_status,e.started_epoch,
                e.query,e.answer,e.sources,e.error
                FROM pending_operations p LEFT JOIN operation_executions e
                ON p.id=e.operation_id
                WHERE p.session_id=? AND p.actor_id=?
                ORDER BY p.created_epoch DESC LIMIT 1""",
                (session_id,actor_id)).fetchone()
            completed=con.execute("""SELECT e.operation_id,e.status AS execution_status,
                e.query,e.answer,e.sources,e.error,e.finished_epoch
                FROM operation_executions e WHERE e.session_id=? AND e.actor_id=?
                AND e.status IN ('succeeded','failed')
                ORDER BY e.started_epoch DESC LIMIT 1""",
                (session_id,actor_id)).fetchone()
        current_state=expose(current)
        if (current_state and current_state.get("execution_status")=="running"
                and time.time()-float(current_state.get("started_epoch") or 0)>900):
            current_state["execution_status"]="interrupted"
            current_state["error"]="No completion recorded within 15 minutes"
        return {"current":current_state,"last_result":expose(completed)}
