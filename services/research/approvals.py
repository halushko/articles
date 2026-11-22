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
"""
TTL = 300
# Full phrases only, not substrings of a conversation. A generic 'yes'
# MUST NOT approve internet access, knowledge edits, or device operations.
ALLOW = re.compile(r"(?iu)^(?:разрешаю(?:\s+(?:этот\s+)?поиск)?|"
                   r"да\s+разрешаю(?:\s+поиск)?|"
                   r"подтверждаю\s+поиск|"
                   r"разрешить\s+поиск)\s*[.!]*$")
DENY = re.compile(r"(?iu)^(?:не\s+разрешаю(?:\s+поиск)?|"
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
    def __init__(self,engine,sessions):
        self.engine=engine
        self.sessions=sessions
        with engine.connect() as con:
            con.executescript(SCHEMA)

    def propose_search(self,session_id,query,question):
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
                (id,session_id,kind,payload,status,created_epoch)
                VALUES(?,?, 'web.search',?, 'pending',?)""",
                (operation_id,session_id,json.dumps(payload,ensure_ascii=False),time.time()))
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

    def resolve(self,session_id,phrase="",operation_id="",source="voice"):
        decision = spoken_search_decision(phrase) if source=="voice" else None
        if source=="button":
            decision="approved" if operation_id else None
        if not decision:
            return {"handled":False}
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
