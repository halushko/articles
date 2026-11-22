"""v0.9 conversation state, contextual consent and authorization separation.

No live LLM, internet or physical devices are used by these tests.
"""
import json
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from services.research.engine import Engine
from services.research.settings import load
from services.research.apprentice import ApprenticeMemory
from services.research.sessions import ConversationStore
from services.research.review_voice import VoiceReviews
from services.research.approvals import OperationApprovals
from services.research.adaptive import BehaviorMemory, WEB_CONTEXT
from services.research.dialogue import DialogueCoordinator
from services.research.api import app as research_app
from services.gateway import api as gateway


class DialogueCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.engine=Engine(str(Path(self.temp.name)/"dialogue.sqlite3"),load())
        self.memory=ApprenticeMemory(self.engine)
        self.sessions=ConversationStore(self.engine)
        self.behavior=BehaviorMemory(self.engine)
        self.voice=VoiceReviews(self.engine,self.memory,self.sessions)
        self.approvals=OperationApprovals(self.engine,self.sessions,self.behavior)
        self.dialogue=DialogueCoordinator(self.engine,self.sessions,
                                          self.approvals,self.voice,self.behavior)
        self.sid=self.sessions.create("Картошка")["id"]
        self.actor="actor-dima"

    def tearDown(self):
        self.temp.cleanup()

    def save(self, question, answer, route, actor=None, **data):
        actor=self.actor if actor is None else actor
        result={"route":route,**data}
        saved=self.sessions.save_turn(self.sid,question,answer,"apprentice",
                                      "fast","combined","ask",result,actor_id=actor)
        self.dialogue.record(self.sid,saved["turn_id"],actor,question,answer,result)
        return saved

    def web_offer(self,actor=None):
        actor=self.actor if actor is None else actor
        p=self.approvals.propose_search(self.sid,"рецепт картофельного пюре",
                                        "Найди в интернете рецепт пюре",actor)
        self.save("Найди в интернете рецепт пюре",
                  "Могу выполнить поиск рецепта. Разрешить?",
                  "apprentice_web_permission",actor,web_approval=p,
                  utterance_intent={"intent":"command"})
        return p

    def memory_offer(self):
        change=self.memory.context("Переименуй себя в Домик")["change_proposal"]
        queued=self.voice.queue(self.sid,changes=[change],actor_id=self.actor)
        self.save("Переименуй себя в Домик","Предлагаю переименование. Подтверди.",
                  "apprentice_confirmed_context",voice_review=queued,
                  change_proposal=change)
        return change

    def test_plain_yes_without_pending_is_ordinary_dialogue(self):
        self.save("Хочешь рецепт картошки?","Да, хочешь рецепт?",
                  "apprentice_tutor",dialogue_frame={
                      "topic":"картошка","goal":"приготовить пюре",
                      "assistant_question":"Хочешь рецепт?"})
        resolved=self.dialogue.resolve(self.sid,"да",self.actor)
        self.assertFalse(resolved["handled"])
        state=self.dialogue.context(self.sid,self.actor)
        self.assertEqual(state["state"]["goal"],"приготовить пюре")
        self.assertEqual(state["state"]["assistant_question"],"Хочешь рецепт?")
        self.assertEqual(state["history"][-1]["content"],"Да, хочешь рецепт?")

    def test_search_offer_yes_approves_only_saved_query_once(self):
        p=self.web_offer()
        result=self.dialogue.resolve(self.sid,"да",self.actor)
        self.assertTrue(result["handled"])
        self.assertEqual(result["status"],"approved")
        self.assertEqual(result["category"],"web.search")
        self.assertEqual(result["query"],"рецепт картофельного пюре")
        self.assertEqual(result["operation_id"],p["id"])
        self.assertFalse(self.dialogue.resolve(self.sid,"да",self.actor)["handled"])
        with self.engine.connect() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM operation_audit").fetchone()[0],1)

    def test_search_offer_no_rejects_without_search(self):
        self.web_offer()
        result=self.dialogue.resolve(self.sid,"нет",self.actor)
        self.assertEqual(result["status"],"rejected")
        self.assertFalse(result["executed"])

    def test_search_disabled_does_not_consume_approval(self):
        self.web_offer()
        blocked=self.dialogue.resolve(self.sid,"да",self.actor,web_enabled=False)
        self.assertEqual(blocked["status"],"web_disabled")
        self.assertEqual(self.dialogue.resolve(self.sid,"да",self.actor)["status"],"approved")

    def test_yes_does_not_authorize_after_unrelated_turn(self):
        self.web_offer()
        self.save("Расскажи анекдот","Колобок пошёл гулять","apprentice_tutor")
        answer=self.dialogue.resolve(self.sid,"да",self.actor)
        self.assertFalse(answer["handled"])
        with self.engine.connect() as con:
            self.assertEqual(con.execute(
                "SELECT status FROM pending_operations").fetchone()["status"],"pending")

    def test_wrong_profile_cannot_confirm_another_persons_offer(self):
        self.web_offer()
        resolved=self.dialogue.resolve(self.sid,"да","actor-nadya")
        self.assertEqual(resolved["status"],"wrong_speaker")
        self.assertEqual(self.dialogue.resolve(self.sid,"да",self.actor)["status"],"approved")

    def test_memory_review_yes_applies_only_known_change(self):
        change=self.memory_offer()
        answer=self.dialogue.resolve(self.sid,"да",self.actor)
        self.assertEqual(answer["category"],"knowledge.review")
        self.assertEqual(answer["status"],"confirmed")
        self.assertEqual(answer["review"]["id"],change["id"])
        self.assertEqual(self.memory.dialogue.state()["roles"]["assistant"],"Домик")

    def test_both_memory_and_web_pending_require_disambiguation(self):
        self.memory_offer()
        self.web_offer()
        result=self.dialogue.resolve(self.sid,"да",self.actor)
        self.assertEqual(result["status"],"ambiguous")
        self.assertFalse(result["executed"])
        with self.engine.connect() as con:
            self.assertEqual(con.execute("SELECT status FROM pending_operations").fetchone()[0],
                             "pending")
            self.assertIsNotNone(con.execute("SELECT * FROM voice_review_queue").fetchone())

    def test_physical_action_never_authorized_by_yes(self):
        self.web_offer()
        with self.engine.connect() as con:
            con.execute("""INSERT INTO pending_operations
                (id,session_id,actor_id,kind,payload,status,created_epoch)
                VALUES(?,?,?,'device.control',?,'pending',?)""",
                (str(uuid.uuid4()),self.sid,self.actor,
                 json.dumps({"device":"oven","power":"on"}),time.time()))
        blocked=self.dialogue.resolve(self.sid,"да",self.actor)
        self.assertEqual(blocked["status"],"not_authorized")
        self.assertFalse(blocked["executed"])
        with self.engine.connect() as con:
            self.assertEqual(con.execute(
                "SELECT COUNT(*) FROM operation_audit").fetchone()[0],0)

    def test_expired_search_cannot_be_revived_by_yes(self):
        op=self.web_offer()
        with self.engine.connect() as con:
            con.execute("UPDATE pending_operations SET created_epoch=? WHERE id=?",
                        (time.time()-400,op["id"]))
        self.assertEqual(self.dialogue.resolve(self.sid,"да",self.actor)["status"],"expired")
        self.assertEqual(self.approvals.resolve(self.sid,"разрешаю")["status"],"expired")

    def test_learned_phrase_is_still_scoped_and_not_a_tool(self):
        text="Когда разрешаешь поиск фраза Можешь приступать значит согласие"
        self.behavior.learn(text,[{"kind":"interpretation","context":WEB_CONTEXT,
            "trigger":"Можешь приступать","meaning":"approve",
            "evidence":"Можешь приступать"}],self.actor)
        self.assertFalse(self.dialogue.resolve(self.sid,"Можешь приступать",self.actor)["handled"])
        self.web_offer()
        self.assertEqual(self.dialogue.resolve(self.sid,"Можешь приступать",self.actor)["status"],
                         "approved")

    def test_actor_scoped_history_and_state_survives_reopening(self):
        self.save("У меня домашняя картошка","Хорошо","apprentice_tutor",
                  dialogue_frame={"topic":"еда","goal":"приготовить пюре"})
        self.save("Секрет Нади","Не повторяй","apprentice_tutor",actor="actor-nadya")
        fresh=DialogueCoordinator(self.engine,self.sessions,self.approvals,
                                  self.voice,self.behavior)
        dima=fresh.context(self.sid,self.actor)
        nadya=fresh.context(self.sid,"actor-nadya")
        self.assertEqual(dima["state"]["goal"],"приготовить пюре")
        self.assertNotIn("Секрет Нади",str(dima["history"]))
        self.assertNotIn("У меня домашняя картошка",str(nadya["history"]))


class DialogueHttpTests(unittest.TestCase):
    def test_real_research_endpoint_and_turn_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ,{"DATABASE_PATH":str(Path(tmp)/"research.sqlite3")}):
                with TestClient(research_app) as client:
                    actor=client.post("/research/actors",json={"display_name":"Дима"}).json()["id"]
                    sid=client.post("/research/sessions",json={}).json()["id"]
                    op=client.post("/research/operations/web/propose",json={
                        "session_id":sid,"actor_id":actor,"query":"рецепт картофельного пюре",
                        "question":"найди рецепт пюре"}).json()
                    self.assertEqual(client.post("/research/sessions/"+sid+"/turns",json={
                        "question":"найди рецепт пюре","answer":"Разрешить поиск?",
                        "experiment":"apprentice","profile":"fast","ablation":"combined",
                        "web_mode":"ask","actor_id":actor,
                        "result":{"route":"apprentice_web_permission",
                                  "web_approval":op,
                                  "dialogue_frame":{"topic":"пюре","goal":"найти рецепт"}}
                    }).status_code,200)
                    context=client.get("/research/dialogue/context/"+sid,
                                       params={"actor_id":actor}).json()
                    self.assertEqual(context["state"]["goal"],"найти рецепт")
                    result=client.post("/research/dialogue/resolve",json={
                        "session_id":sid,"actor_id":actor,"text":"Да",
                        "web_enabled":True}).json()
                    self.assertEqual(result["status"],"approved")
                    self.assertEqual(result["query"],"рецепт картофельного пюре")
                    self.assertFalse(client.post("/research/dialogue/resolve",json={
                        "session_id":sid,"actor_id":actor,"text":"Да"}).json()["handled"])


class GatewayCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_yes_routes_to_saved_web_query_without_qwen_or_device_tools(self):
        calls=[]
        async def forward(url,method="GET",payload=None):
            calls.append(url)
            if url.endswith("/research/dialogue/resolve"):
                self.assertEqual(payload["text"],"да")
                return {"handled":True,"status":"approved","category":"web.search",
                        "operation_id":"stored-id","question":"найди пюре",
                        "query":"рецепт картофельного пюре"}
            if url.endswith("/research/web/search"):
                self.assertEqual(payload["query"],"рецепт картофельного пюре")
                return {"available":False,"status":"unavailable",
                        "reason":"offline","sources":[]}
            raise AssertionError(url)
        with patch.object(gateway,"forward",forward):
            result=await gateway.apprentice_chat(gateway.ChatIn(
                text="да",experiment="apprentice",web_mode="ask",
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),0)
        self.assertEqual(result["route"],"apprentice_web_unavailable")
        self.assertEqual(result["approval_id"],"stored-id")
        self.assertEqual(len(calls),2)

    async def test_no_pending_yes_uses_persisted_history_for_normal_chat(self):
        calls=[]
        async def forward(url,method="GET",payload=None):
            calls.append(url)
            if url.endswith("/research/dialogue/resolve"):
                return {"handled":False}
            if url.endswith("/apprentice/context"):
                return {"normalized_text":"да",
                        "utterance_intent":{"intent":"unknown"},
                        "dialogue":{"state":{"goal":"обсудить рецепт"},
                                    "history":[{"role":"assistant",
                                                "content":"Хочешь рецепт пюре?"}]},
                        "capabilities":[]}
            if url.endswith("/research/operations/web/clear"):
                return {"cleared":True}
            if url.endswith("/analyze"):
                self.assertIn("обсудить рецепт",str(payload["memory"]))
                return {"reply":"Тогда вот рецепт пюре","intent":"chat",
                        "needs_web":False,"proposals":[],"learning":[],
                        "tokens":12,"model":"test"}
            raise AssertionError(url)
        with patch.object(gateway,"forward",forward):
            result=await gateway.apprentice_chat(gateway.ChatIn(
                text="да",experiment="apprentice",web_mode="ask",
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),0)
        self.assertEqual(result["route"],"apprentice_tutor")
        self.assertEqual(result["answer"],"Тогда вот рецепт пюре")
        self.assertFalse(any("/apprentice/voice/resolve" in x for x in calls))

    async def test_physical_reply_is_blocked_without_any_tool(self):
        async def forward(url,method="GET",payload=None):
            if url.endswith("/research/dialogue/resolve"):
                return {"handled":True,"status":"not_authorized",
                        "executed":False,"answer":"Опасное действие не разрешено."}
            raise AssertionError(url)
        with patch.object(gateway,"forward",forward):
            result=await gateway.apprentice_chat(gateway.ChatIn(
                text="да",experiment="apprentice",web_mode="ask",
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),0)
        self.assertFalse(result["executed"])
        self.assertIn("не разрешено",result["answer"])


if __name__=="__main__":
    unittest.main()
