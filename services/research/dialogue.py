"""Authoritative dialogue state and contextual response routing.

The language model understands meaning; it is never an authorization source.
Only an existing, actor-scoped, unexpired, *last offered* operation may use a
contextual acknowledgement. Unknown or physical operations are never executed.
A single router arbitrates between tool operations and knowledge reviews.
"""
import json
import re
import time

from .normalize import normalize
from .approvals import TTL as OPERATION_TTL, spoken_search_decision
from .review_voice import TTL_SECONDS as REVIEW_TTL, speech_intent
from .adaptive import WEB_CONTEXT

SCHEMA = """
CREATE TABLE IF NOT EXISTS dialogue_frames (
 session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
 actor_id TEXT NOT NULL DEFAULT '',
 topic TEXT NOT NULL DEFAULT '',
 goal TEXT NOT NULL DEFAULT '',
 assistant_question TEXT NOT NULL DEFAULT '',
 last_route TEXT NOT NULL DEFAULT '',
 last_intent TEXT NOT NULL DEFAULT '',
 last_turn_id INTEGER,
 updated_epoch REAL NOT NULL,
 PRIMARY KEY(session_id,actor_id)
);
"""

# These are basic conversational yes/no meanings, NOT programmable commands
# or permissions. A bare acknowledgement is considered for low-risk operations
# only if the exact proposal is the last assistant offer in this session.
YES = frozenset(("да", "ага", "угу", "так", "yes", "yeah"))
NO = frozenset(("нет", "неа", "ні", "no", "nope"))
ACK = YES | NO

POLICIES = {
    "web.search": {"risk": "low", "executor": True, "contextual_ack": True},
    "knowledge.review": {"risk": "memory_write", "executor": True,
                         "contextual_ack": True},
    "device.control": {"risk": "physical", "executor": False,
                       "contextual_ack": False},
    "data.delete": {"risk": "destructive", "executor": False,
                    "contextual_ack": False},
}


def _decode(value):
    try:
        result = json.loads(value or "{}")
    except (ValueError, TypeError):
        return {}
    return result if isinstance(result, dict) else {}


def _kind_of_reply(text):
    # Whole utterance only, never substrings such as "да" in a lesson.
    value = normalize(text)
    if value in YES:
        return "approve"
    if value in NO:
        return "deny"
    return None


class DialogueCoordinator:
    def __init__(self, engine, sessions, approvals, voice_reviews, behavior):
        self.engine = engine
        self.sessions = sessions
        self.approvals = approvals
        self.voice_reviews = voice_reviews
        self.behavior = behavior
        with engine.connect() as con:
            con.executescript(SCHEMA)

    def record(self, session_id, turn_id, actor_id, question, answer, result):
        """Persist the model's *descriptive* state, never permissions.

        Called after a successful journal append. A new subject/goal is stored
        only when the model or a real tool offer provided it. Short answers keep
        the established goal. All fields are length bounded.
        """
        if not self.sessions.exists(session_id):
            return
        result = result if isinstance(result, dict) else {}
        frame = result.get("dialogue_frame")
        frame = frame if isinstance(frame, dict) else {}
        offer = result.get("web_approval")
        offer = offer if isinstance(offer, dict) else {}
        topic = str(frame.get("topic") or offer.get("query") or "").strip()[:160]
        goal = str(frame.get("goal") or offer.get("question") or "").strip()[:260]
        assistant_question = str(frame.get("assistant_question") or "").strip()[:260]
        if offer and not assistant_question:
            assistant_question = "Разрешить поиск: " + str(offer.get("query") or "")[:190]
        route = str(result.get("route") or "")[:80]
        intent = str(result.get("intent") or (result.get("utterance_intent") or {}).get("intent") or "")[:40]
        actor_id = str(actor_id or "")[:36]
        with self.engine.connect() as con:
            con.execute("""INSERT INTO dialogue_frames
                (session_id,actor_id,topic,goal,assistant_question,last_route,
                 last_intent,last_turn_id,updated_epoch)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id,actor_id) DO UPDATE SET
                topic=CASE WHEN excluded.topic<>'' THEN excluded.topic ELSE dialogue_frames.topic END,
                goal=CASE WHEN excluded.goal<>'' THEN excluded.goal ELSE dialogue_frames.goal END,
                assistant_question=excluded.assistant_question,
                last_route=excluded.last_route,
                last_intent=excluded.last_intent,
                last_turn_id=excluded.last_turn_id,
                updated_epoch=excluded.updated_epoch""",
                (session_id,actor_id,topic,goal,assistant_question,route,intent,
                 turn_id,time.time()))

    def context(self, session_id, actor_id="", limit=6):
        """Trusted persisted history, isolated by selected speaking profile."""
        if not self.sessions.exists(session_id):
            return {"state":{}, "history":[], "pending":[]}
        actor_id=str(actor_id or "")[:36]
        with self.engine.connect() as con:
            frame=con.execute("""SELECT topic,goal,assistant_question,last_route,
                last_intent,last_turn_id FROM dialogue_frames
                WHERE session_id=? AND actor_id=?""",(session_id,actor_id)).fetchone()
            turns=con.execute("""SELECT question,answer FROM research_turns
                WHERE session_id=? AND actor_id=?
                ORDER BY id DESC LIMIT ?""",
                (session_id,actor_id,min(8,max(1,int(limit))))).fetchall()
        history=[]
        for row in reversed(turns):
            history.extend(({"role":"user","content":row["question"][:800]},
                            {"role":"assistant","content":row["answer"][:1200]}))
        return {"state":dict(frame) if frame else {}, "history":history,
                "pending":self.pending_summary(session_id,actor_id)}

    def _pending(self, session_id):
        now=time.time()
        with self.engine.connect() as con:
            operations=[dict(row) for row in con.execute(
                """SELECT * FROM pending_operations WHERE session_id=?
                   AND status='pending' ORDER BY created_epoch DESC""",
                (session_id,))]
            queue=con.execute("SELECT * FROM voice_review_queue WHERE session_id=?",
                              (session_id,)).fetchone()
        active=[]
        stale=[]
        for row in operations:
            item={"kind":row["kind"],"id":row["id"],"actor_id":row["actor_id"],
                  "created_epoch":row["created_epoch"],"payload":_decode(row["payload"])}
            (active if now-row["created_epoch"]<=OPERATION_TTL else stale).append(item)
        if queue:
            item={"kind":"knowledge.review", "id":"review:"+session_id,
                  "actor_id":queue["actor_id"],"created_epoch":queue["created_epoch"],
                  "payload":{"reviews":json.loads(queue["reviews"])}}
            (active if now-queue["created_epoch"]<=REVIEW_TTL else stale).append(item)
        return active,stale

    def pending_summary(self, session_id, actor_id=""):
        active,_=self._pending(session_id)
        return [{"kind":item["kind"],"risk":POLICIES.get(item["kind"],{}).get("risk","unknown"),
                 "id":item["id"]}
                for item in active if item["actor_id"]==str(actor_id or "")]

    def _last_offer(self, session_id):
        with self.engine.connect() as con:
            row=con.execute("""SELECT id,actor_id,route,provenance FROM research_turns
                WHERE session_id=? ORDER BY id DESC LIMIT 1""",(session_id,)).fetchone()
        if row is None:
            return None
        result=dict(row)
        result["provenance"]=_decode(result["provenance"])
        return result

    @staticmethod
    def _was_offered(pending, last):
        if not last or last["actor_id"]!=pending["actor_id"]:
            return False
        provenance=last["provenance"]
        if pending["kind"]=="web.search":
            offer=provenance.get("web_approval") or {}
            return (last["route"] in ("apprentice_web_permission","apprentice_tutor")
                    and isinstance(offer,dict) and offer.get("id")==pending["id"])
        if pending["kind"]=="knowledge.review":
            offer=provenance.get("voice_review") or {}
            change=provenance.get("change_proposal") or {}
            return (isinstance(offer,dict) and offer.get("ready") is True
                    or isinstance(change,dict) and change.get("id") is not None)
        return False

    def resolve(self, session_id, text, actor_id="", web_enabled=True):
        """Arbitrate a conversational reply; never authorizes a physical action.

        A last-offer check prevents "да" after a topic change from activating an
        older proposal. A missing review is *not* returned as a review error.
        """
        if not self.sessions.exists(session_id):
            return {"handled":False}
        actor_id=str(actor_id or "")[:36]
        text=str(text or "").strip()
        if len(text)>250:
            return {"handled":False}
        basic=_kind_of_reply(text)
        search=spoken_search_decision(text)
        learned=self.behavior.decision(text, WEB_CONTEXT, actor_id)
        review_decision,review_index=speech_intent(text)
        if not any((basic,search,learned,review_decision)):
            return {"handled":False}
        active,stale=self._pending(session_id)
        last=self._last_offer(session_id)
        speaker_pending=[item for item in active if item["actor_id"]==actor_id]
        # An acknowledgement to a physical/destructive proposal is never
        # sufficient to authorize anything, even if another low-risk tool is
        # also pending. The executor must require a separate exact grant.
        if basic and any(
                POLICIES.get(item["kind"],{}).get("risk") in ("physical","destructive","unknown")
                for item in speaker_pending):
            return {"handled":True,"status":"not_authorized","executed":False,
                    "answer":"Одного «да» недостаточно для управления устройствами "
                             "или другой опасной операции. Ничего не выполняю."}
        # Do not silently choose the newest operation when the speaker has
        # more than one pending item (e.g. a lesson and a search).
        if basic and len(speaker_pending)>1:
            return {"handled":True,"status":"ambiguous","executed":False,
                    "answer":"Сейчас ожидается несколько разных подтверждений. "
                             "Укажи конкретное действие: «Разрешаю поиск» "
                             "или «Подтверди первое знание»."}

        # A plain answer must refer to the latest prompt. A specific learned
        # search phrase is still bound to the originally offered operation.
        eligible=[item for item in active
                  if item["actor_id"]==actor_id and self._was_offered(item,last)]
        different=[item for item in active if item["actor_id"]!=actor_id]
        if not eligible:
            if different and (basic or search or learned or review_decision):
                return {"handled":True,"status":"wrong_speaker","executed":False,
                        "answer":"Предложение ожидает ответа другого собеседника. "
                                 "В твоём профиле ничего не выполняю."}
            if stale and (basic or search or learned or review_decision):
                return {"handled":True,"status":"expired","executed":False,
                        "answer":"Предыдущее предложение устарело. Повтори запрос."}
            return {"handled":False}

        # Semantic meaning only proposes candidates. Operation policy decides
        # which can accept a contextual acknowledgement.
        candidate=[]
        for item in eligible:
            kind=item["kind"]
            if kind=="web.search" and (search or learned or basic):
                candidate.append(item)
            elif kind=="knowledge.review" and review_decision:
                candidate.append(item)
            elif kind not in POLICIES and basic:
                candidate.append(item)
            elif kind in POLICIES and basic and not POLICIES[kind]["contextual_ack"]:
                candidate.append(item)

        if not candidate:
            return {"handled":False}
        # Explicitly named search / specific numbered review disambiguates;
        # a bare yes/no never arbitrarily selects among multiple operations.
        specifically_search=bool(search or learned)
        specifically_review=review_index is not None
        if specifically_search:
            candidate=[x for x in candidate if x["kind"]=="web.search"]
        elif specifically_review:
            candidate=[x for x in candidate if x["kind"]=="knowledge.review"]
        if not candidate:
            return {"handled":False}
        if len(candidate)!=1:
            return {"handled":True,"status":"ambiguous","executed":False,
                    "answer":"Есть несколько ожидающих предложений. "
                             "Уточни, какое именно подтверждаешь: "
                             "например, «Разрешаю поиск» или «Подтверди первое знание»."}
        choice=candidate[0]
        policy=POLICIES.get(choice["kind"],{"executor":False,"risk":"unknown",
                                            "contextual_ack":False})
        if not policy["executor"] or (basic and not policy["contextual_ack"]):
            return {"handled":True,"status":"not_authorized","executed":False,
                    "answer":"Одного «да» недостаточно для этой операции. "
                             "К тому же доступного разрешённого исполнителя нет. "
                             "Ничего не выполняю."}
        if choice["kind"]=="web.search":
            if not web_enabled:
                return {"handled":True,"status":"web_disabled","executed":False,
                        "answer":"Интернет отключён. Поиск не выполнялся."}
            decision=("approved" if search=="approved" or learned=="approve"
                      or basic=="approve" else "rejected")
            if search=="rejected" or learned=="deny" or basic=="deny":
                decision="rejected"
            result=self.approvals.resolve_scoped(session_id,choice["id"],decision,
                                                 actor_id=actor_id)
            result["category"]="web.search"
            result["contextual"]=basic is not None
            return result
        if choice["kind"]=="knowledge.review":
            result=self.voice_reviews.resolve(session_id,text,actor_id)
            result["category"]="knowledge.review"
            result["contextual"]=basic is not None
            return result
        return {"handled":True,"status":"unsupported_operation","executed":False,
                "answer":"Для операции нет проверенного исполнителя."}
