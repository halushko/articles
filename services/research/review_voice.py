"""Short, session-scoped spoken confirmations for teacher knowledge reviews.

A spoken "да" is *never* a generic permission. It is accepted only when a
specific review was shown in the same existing research session within five
minutes. Multi-candidate queues require an ordinal or explicit id.
Voice is not speaker authentication; the gateway must authenticate the client.
Only knowledge proposals/renames are supported, never device operations.
"""
import json
import re
import time
from .utterance import classify

SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_review_queue (
 session_id TEXT PRIMARY KEY,
 reviews TEXT NOT NULL,
 created_epoch REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS voice_review_audit (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 session_id TEXT NOT NULL,
 review_kind TEXT NOT NULL,
 review_id INTEGER NOT NULL,
 decision TEXT NOT NULL,
 outcome TEXT NOT NULL,
 source_text TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
TTL_SECONDS = 300
VOICE_YES = re.compile(
    r"(?iu)^(?:да|ага|угу|подтверждаю|"
    r"да\s+подтверждаю|да\s+применяй|применяй|"
    r"согласен|согласна|подтверди|подтвердить|"
    r"подтверждаю\s+объединение|подтверди\s+объединение|"
    r"да\s+объединяй|"
    r"сохрани(?:\s+это)?)\s*[.!]*$"
)
VOICE_NO = re.compile(
    r"(?iu)^(?:нет|неа|нет\s+не\s+надо|"
    r"отклоняю|отклони|отменяю|отмени|"
    r"не\s+подтверждаю|отмена)\s*[.!]*$"
)
VOICE_INDEX = re.compile(
    r"(?iu)^(?P<act>подтверди|подтверждаю|применяй|"
    r"отклони|отклоняю|отмена)\s+"
    r"(?:(?:изменение|предложение|знание|факт)\s*)?"
    r"(?:номер\s*)?(?:№|#)?\s*(?P<index>\d{1,9}|"
    r"первое|второе|третье|четв[её]ртое|пятое)"
    r"(?:\s+(?:изменение|предложение))?\s*[.!]*$"
)
WORD_INDEX = {"первое":1,"второе":2,"третье":3,
              "четвертое":4,"четвёртое":4,"пятое":5}


def speech_intent(text):
    body=classify(text)["normalized"].casefold().strip()
    if VOICE_YES.fullmatch(body):
        return "confirm",None
    if VOICE_NO.fullmatch(body):
        return "reject",None
    match=VOICE_INDEX.fullmatch(body)
    if match:
        action=match["act"]
        decision=("reject" if action.startswith(("отклон","отмен")) else "confirm")
        raw=match["index"]
        index=-WORD_INDEX[raw] if raw in WORD_INDEX else int(raw)
        return decision,index
    return None,None


class VoiceReviews:
    def __init__(self,engine,apprentice,sessions):
        self.engine=engine
        self.apprentice=apprentice
        self.sessions=sessions
        with engine.connect() as con:
            con.executescript(SCHEMA)

    def queue(self,session_id,changes=(),proposals=()):
        if not self.sessions.exists(session_id):
            raise ValueError("Сессия для голосового подтверждения не найдена")
        reviews=[]
        with self.engine.connect() as con:
            for item in list(changes)[:4]:
                if not isinstance(item,dict) or not isinstance(item.get("id"),int):
                    continue
                row=con.execute("SELECT status FROM knowledge_changes WHERE id=?",
                                (item["id"],)).fetchone()
                if row and row["status"]=="pending":
                    reviews.append({"kind":"change","id":item["id"]})
            for item in list(proposals)[:4]:
                if not isinstance(item,dict) or not isinstance(item.get("id"),int):
                    continue
                row=con.execute("SELECT status FROM apprentice_proposals WHERE id=?",
                                (item["id"],)).fetchone()
                if row and row["status"]=="pending":
                    reviews.append({"kind":"proposal","id":item["id"]})
            # Deduplicate while keeping visible UI order.
            reviews=list({(x["kind"],x["id"]):x for x in reviews}.values())
            if reviews:
                con.execute("""INSERT INTO voice_review_queue
                         (session_id,reviews,created_epoch) VALUES(?,?,?)
                         ON CONFLICT(session_id) DO UPDATE SET
                         reviews=excluded.reviews,created_epoch=excluded.created_epoch""",
                         (session_id,json.dumps(reviews),time.time()))
            else:
                con.execute("DELETE FROM voice_review_queue WHERE session_id=?",
                            (session_id,))
        return {"ready":bool(reviews),"count":len(reviews),"expires_in":TTL_SECONDS,
                "instruction":(("Найден конфликт с другой сущностью. "
                  "Скажи «Подтверждаю объединение» или «Отклоняю»."
                  if changes and changes[0].get("requires_merge_approval") else
                  "Скажи «Подтверждаю» или «Отклоняю».")
                 if len(reviews)==1 else
                 "Несколько предложений. Скажи «Подтверди первое» или «Отклони второе»."
                 if reviews else "")}

    def clear(self,session_id):
        if not self.sessions.exists(session_id):
            return
        with self.engine.connect() as con:
            con.execute("DELETE FROM voice_review_queue WHERE session_id=?",
                        (session_id,))

    def resolve(self,session_id,text):
        decision,index=speech_intent(text)
        if not decision:
            self.clear(session_id)
            return {"handled":False}
        if not self.sessions.exists(session_id):
            return {"handled":True,"status":"no_session","executed":False,
                    "answer":"Голосовое подтверждение работает только внутри открытого диалога."}
        with self.engine.connect() as con:
            row=con.execute("SELECT * FROM voice_review_queue WHERE session_id=?",
                            (session_id,)).fetchone()
        if not row or time.time()-row["created_epoch"]>TTL_SECONDS:
            self.clear(session_id)
            return {"handled":True,"status":"no_active_review","executed":False,
                    "answer":"Сейчас нет свежего предложения для голосового подтверждения. Повтори просьбу."}
        reviews=json.loads(row["reviews"])
        # The teacher may have clicked a button for one item. Remove already
        # reviewed records without extending the original voice-review TTL.
        with self.engine.connect() as con:
            valid=[]
            for item in reviews:
                table="knowledge_changes" if item["kind"]=="change" else "apprentice_proposals"
                found=con.execute(f"SELECT status FROM {table} WHERE id=?",
                                  (item["id"],)).fetchone()
                if found and found["status"]=="pending":
                    valid.append(item)
            if valid!=reviews:
                if valid:
                    con.execute("UPDATE voice_review_queue SET reviews=? WHERE session_id=?",
                                (json.dumps(valid),session_id))
                else:
                    con.execute("DELETE FROM voice_review_queue WHERE session_id=?",
                                (session_id,))
            reviews=valid
        if not reviews:
            return {"handled":True,"status":"no_active_review","executed":False,
                    "answer":"Эти предложения уже проверены. Ожидающих изменений нет."}
        if index is None and len(reviews)!=1:
            return {"handled":True,"status":"ambiguous_review","executed":False,
                    "answer":f"Есть {len(reviews)} предложений. Укажи номер: например, «Подтверди первое»."}
        if index is None:
            selected=0
        else:
            # Negative indexes are unambiguous *spoken ordinals*.
            if index<0:
                selected=-index-1
            else:
                ord_idx=index-1
                matched=[i for i,item in enumerate(reviews) if item["id"]==index]
                if matched and ord_idx in range(len(reviews)) and ord_idx not in matched:
                    return {"handled":True,"status":"ambiguous_review","executed":False,
                        "answer":"Назови порядковый номер словами: «Подтверди первое»."}
                selected=matched[0] if matched else ord_idx
        if not 0<=selected<len(reviews):
            return {"handled":True,"status":"unknown_review","executed":False,
                    "answer":"Такого номера нет среди предложений в текущем диалоге."}
        picked=reviews[selected]
        try:
            if picked["kind"]=="change":
                with self.engine.connect() as con:
                    pending=con.execute(
                        "SELECT conflict_node_id FROM knowledge_changes WHERE id=?",
                        (picked["id"],)).fetchone()
                conflict=bool(pending and pending["conflict_node_id"] is not None)
                allow_merge=bool(re.search(r"(?iu)\bобъединен(?:ие|ия)|объединяй\b",text))
                if conflict and decision=="confirm" and not allow_merge:
                    return {"handled":True,"status":"merge_confirmation_needed",
                            "executed":False,
                            "answer":"Это имя уже есть в другом узле графа. "
                            "Чтобы объединить идентичности, скажи отдельно "
                            "«Подтверждаю объединение». Обычного «да» недостаточно."}
                result=self.apprentice.changes.review(
                    picked["id"],decision,resolve_conflict=allow_merge)
            else:
                result=self.apprentice.review(picked["id"],decision)
        except (ValueError,KeyError) as error:
            self.clear(session_id)
            return {"handled":True,"status":"stale_review","executed":False,
                    "answer":"Предложение устарело или уже обработано: "+str(error)[:150]}
        # A confirmed role name or entity name remains independent of tool
        # permissions; the action below is a SQLite teacher review ONLY.
        remaining=reviews[:selected]+reviews[selected+1:]
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if remaining:
                con.execute("UPDATE voice_review_queue SET reviews=? WHERE session_id=?",
                            (json.dumps(remaining),session_id))
            else:
                con.execute("DELETE FROM voice_review_queue WHERE session_id=?",
                            (session_id,))
            con.execute("""INSERT INTO voice_review_audit
                     (session_id,review_kind,review_id,decision,outcome,source_text)
                     VALUES(?,?,?,?,?,?)""",
                        (session_id,picked["kind"],picked["id"],decision,
                         result["status"],str(text)[:300]))
        self.sessions.event(session_id,"voice_review",{
            "kind":picked["kind"],"item_id":picked["id"],
            "decision":decision,"status":result["status"],"spoken":str(text)[:250]})
        if result["status"]=="confirmed":
            if picked["kind"]=="change":
                answer=("Подтверждено. Имя изменено в SQLite: "
                        f"«{result['expected_value']}» — «{result['current_value']}».")
            else:
                answer=f"Подтверждено. Знание «{result['name']}» сохранено в SQLite."
        elif result["status"]=="rejected":
            answer="Отклонено. Изменение не применено."
        else:
            answer="Изменение устарело. Память не перезаписана."
        if remaining:
            answer+=f" Осталось предложений: {len(remaining)}."
        return {"handled":True,"status":result["status"],"answer":answer,
                "review":{"kind":picked["kind"],"id":picked["id"],"decision":decision},
                "remaining":len(remaining),"executed":False}
