"""World-memory invariants: experience is reusable, not silently promoted to truth."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.research.engine import Engine
from services.research.settings import load
from services.research.apprentice import ApprenticeMemory
from services.research.sessions import ConversationStore
from services.research.adaptive import BehaviorMemory
from services.research.approvals import OperationApprovals
from services.research.review_voice import VoiceReviews
from services.research.dialogue import DialogueCoordinator
from services.research.actors import ActorRegistry
from services.research.world import WorldMemory
from services.gateway import api as gateway


class WorldMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.engine=Engine(str(Path(self.tmp.name)/"world.sqlite3"),load())
        self.apprentice=ApprenticeMemory(self.engine)
        self.sessions=ConversationStore(self.engine)
        self.behavior=BehaviorMemory(self.engine)
        self.approvals=OperationApprovals(self.engine,self.sessions,self.behavior)
        self.reviews=VoiceReviews(self.engine,self.apprentice,self.sessions)
        self.dialogue=DialogueCoordinator(self.engine,self.sessions,
                                         self.approvals,self.reviews,self.behavior)
        self.actors=ActorRegistry(self.engine)
        self.world=WorldMemory(self.engine,self.apprentice,self.dialogue,self.behavior)
        self.a=self.actors.create("Дима")
        self.b=self.actors.create("Надя")
        self.first=self.sessions.create("Первый")["id"]
        self.next=self.sessions.create("Второй")["id"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_recalls_cross_session_user_statement_with_provenance_not_assistant_facts(self):
        self.sessions.save_turn(
            self.first,"У меня на балконе стоит аккумулятор",
            "Ассистент выдумал пять аккумуляторов","apprentice","fast",
            "combined","off",{"route":"llm_fast"},actor_id=self.a["id"])
        recalled=self.world.recall("Что ты знаешь об аккумуляторе на балконе?",
                                  self.a["id"],self.next)
        self.assertEqual(len(recalled),1)
        self.assertIn("аккумулятор",recalled[0]["text"])
        self.assertNotIn("выдумал",str(recalled))
        self.assertFalse(recalled[0]["verified"])
        self.assertEqual(recalled[0]["source"],"user_statement_in_dialogue")
        self.assertFalse(self.world.recall("аккумулятор на балконе",
                                          self.b["id"],self.next))
        self.assertFalse(self.world.recall("аккумулятор на балконе",
                                          "",self.next))
        ctx=self.world.context("Аккумулятор на балконе",speaker=self.a,
                               actor_id=self.a["id"],session_id=self.next)
        self.assertEqual(ctx["identity"]["user"],"Дима")
        self.assertEqual(len(ctx["experiences"]),1)
        self.assertTrue(ctx["world_model"]["episodes_are_unverified"])
        self.assertNotIn("аккумулятор",str(ctx["graph"]))

    def test_recent_source_is_from_server_not_model_guess(self):
        saved=self.sessions.save_turn(
            self.next,"Какие инструменты нужны для замены люстры?",
            "Достаточно молотка","hybrid","fast","combined","ask",
            {"route":"world_grounded_chat","model":"qwen2.5:3b"},
            actor_id=self.a["id"])
        self.dialogue.record(self.next,saved["turn_id"],self.a["id"],
                             "Какие инструменты нужны для замены люстры?",
                             "Достаточно молотка",{"route":"world_grounded_chat"})
        ctx=self.dialogue.context(self.next,self.a["id"])
        evidence=ctx["last_response_evidence"]
        self.assertEqual(evidence["source_kind"],"unverified_model_answer")
        self.assertEqual(evidence["model"],"qwen2.5:3b")
        self.assertIn("люстры",ctx["history"][-2]["content"])
        self.assertEqual(self.dialogue.context(self.next,self.b["id"])["history"],[])


class WorldSearchPolicyTests(unittest.TestCase):
    def test_implicit_search_reuses_previous_user_question_not_model_reply(self):
        previous={"dialogue":{"history":[
            {"role":"user","content":"Какие инструменты нужны для замены люстры"},
            {"role":"assistant","content":"Непроверенный ответ про молоток"},
            {"role":"user","content":"Откуда такая информация"},
            {"role":"assistant","content":"Это был мой непроверенный ответ"}
        ],"state":{}}}
        query=gateway.proposed_web_query("поищи в интернете",{})
        self.assertEqual(gateway.contextual_search_query(
            "поищи в интернете",query,previous),
            "Какие инструменты нужны для замены люстры")
        self.assertEqual(gateway.contextual_search_query(
            "поищи в интернете",query,{"dialogue":{"history":[],"state":{}}}),"")
        self.assertEqual(gateway.contextual_search_query(
            "поищи в интернете рецепт пюре","рецепт пюре",previous),"рецепт пюре")


class WorldHybridGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_uses_same_confirmed_identity_without_qwen(self):
        called=[]
        async def mock(url,method="GET",payload=None):
            called.append(url)
            if url == gateway.RESEARCH_URL+"/resolve":
                return {"is_command":False,"status":"not_command"}
            if url.endswith("/apprentice/context"):
                self.assertEqual(payload["actor_id"],"dima-selected-actor")
                return {"direct_reply":"Тебя зовут Дима.",
                        "utterance_intent":{"intent":"question"},
                        "speaker":{"display_name":"Дима"}}
            raise AssertionError("Unexpected call: "+url)
        with patch.object(gateway,"forward",mock):
            result=await gateway.chat_core(gateway.ChatIn(
                text="Как меня зовут?",experiment="hybrid",
                actor_id="dima-selected-actor"))
        self.assertEqual(result["answer"],"Тебя зовут Дима.")
        self.assertEqual(result["route"],"apprentice_confirmed_context")
        self.assertFalse(any(x.endswith("/chat") or x.endswith("/analyze")
                             for x in called))


if __name__=="__main__":
    unittest.main()
