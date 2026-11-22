"""Behavioral learning regressions: quoted phrases, persistence and approvals."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.research.engine import Engine
from services.research.settings import load
from services.research.adaptive import BehaviorMemory, WEB_CONTEXT
from services.research.apprentice import ApprenticeMemory
from services.research.sessions import ConversationStore
from services.research.approvals import OperationApprovals
from services.gateway import api as gateway


class BehaviorRulesTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.engine=Engine(str(Path(self.temp.name)/"adaptive.sqlite3"),load())
        self.rules=BehaviorMemory(self.engine)

    def tearDown(self):
        self.temp.cleanup()

    def teach(self, text="Когда я отвечаю Да поищи на предложение поиска это согласие"):
        return self.rules.learn(text,[{
            "kind":"interpretation","context":WEB_CONTEXT,
            "trigger":"Да поищи","meaning":"approve",
            "instruction":"","evidence":"Да поищи"
        }],"actor-a")

    def test_grounded_learning_persists_without_hardcoded_phrase(self):
        saved=self.teach()
        self.assertEqual(len(saved),1)
        self.assertEqual(self.rules.decision("ДА, ПОИЩИ!",WEB_CONTEXT,"actor-a"),"approve")
        self.assertIsNone(self.rules.decision("ДА, ПОИЩИ!",WEB_CONTEXT,"actor-b"))
        self.assertEqual(BehaviorMemory(self.engine).decision("да поищи",WEB_CONTEXT,"actor-a"),"approve")
        self.assertEqual(saved[0]["kind"],"interpretation")

    def test_cannot_invent_command_or_learn_from_ungrounded_source(self):
        self.assertFalse(self.rules.learn("Привет",[{
            "kind":"interpretation","context":WEB_CONTEXT,
            "trigger":"Да поищи","meaning":"approve",
            "evidence":"Да поищи"}],"actor-a"))
        self.assertFalse(self.rules.learn("Когда говорю да разрешаю поиск",[{
            "kind":"interpretation","context":WEB_CONTEXT,
            "trigger":"да","meaning":"approve","evidence":"да разрешаю поиск"
        }],"actor-a"))
        self.assertIsNone(self.rules.decision("да",WEB_CONTEXT,"actor-a"))

    def test_update_and_forget_with_actor_scope(self):
        first=self.teach()[0]
        changed=self.rules.learn("Теперь Да поищи означает отказ",[{
            "kind":"interpretation","context":WEB_CONTEXT,
            "trigger":"Да поищи","meaning":"deny",
            "instruction":"","evidence":"Да поищи"}],"actor-a")
        self.assertEqual(first["id"],changed[0]["id"])
        self.assertEqual(self.rules.decision("да поищи",WEB_CONTEXT,"actor-a"),"deny")
        with self.assertRaises(ValueError):
            self.rules.forget(first["id"],"actor-b")
        self.rules.forget(first["id"],"actor-a")
        self.assertIsNone(self.rules.decision("да поищи",WEB_CONTEXT,"actor-a"))

    def test_preference_in_retrieval_and_no_approval_effect(self):
        text="Пожалуйста отвечай мне кратко и по существу"
        stored=self.rules.learn(text,[{
            "kind":"preference","context":"dialogue","trigger":"",
            "meaning":"","instruction":"Отвечать кратко и по существу",
            "evidence":"отвечай мне кратко и по существу"}],"actor-a")
        self.assertEqual(stored[0]["kind"],"preference")
        self.assertIn("кратко",self.rules.context("actor-a")[0]["instruction"])
        self.assertIsNone(self.rules.decision("Отвечать кратко",WEB_CONTEXT,"actor-a"))

    def test_feedback_corrects_future_answers_without_crossing_people(self):
        apprentice=ApprenticeMemory(self.engine)
        apprentice.grade("Как разрешить интернет поиск",
                         "Попроси ещё раз", "bad",
                         "После подтверждения выполняй ранее предложенный поиск",
                         actor_id="actor-a")
        relevant=apprentice.context("Как разрешить интернет поиск",
                                   speaker={"id":"actor-a","display_name":"Дима"})
        self.assertEqual(relevant["teacher_feedback"][0]["rating"],"bad")
        self.assertIn("ранее предложенный",relevant["teacher_feedback"][0]["correction"])
        other=apprentice.context("Как разрешить интернет поиск",
                                speaker={"id":"actor-b","display_name":"Надя"})
        self.assertFalse(other["teacher_feedback"])
        self.assertEqual(ApprenticeMemory(self.engine).context(
            "Как разрешить интернет поиск",speaker={"id":"actor-a"})["teacher_feedback"][0]["rating"],"bad")

    def test_spoken_approval_scoped_to_pending_operation(self):
        self.teach()
        sessions=ConversationStore(self.engine)
        sid=sessions.create("test")["id"]
        approvals=OperationApprovals(self.engine,sessions,self.rules)
        no_pending=approvals.resolve(sid,"Да поищи",actor_id="actor-a")
        self.assertEqual(no_pending["status"],"no_pending_search")
        proposed=approvals.propose_search(sid,"weather tomorrow","What is weather?",actor_id="actor-a")
        wrong=approvals.resolve(sid,"Да поищи",actor_id="actor-b")
        self.assertFalse(wrong["handled"])
        approved=approvals.resolve(sid,"Да поищи",actor_id="actor-a")
        self.assertEqual(approved["status"],"approved")
        self.assertEqual(approved["query"],"weather tomorrow")
        again=approvals.resolve(sid,"Да поищи",actor_id="actor-a")
        self.assertEqual(again["status"],"no_pending_search")
        self.assertEqual(approved["operation_id"],proposed["id"])


class GatewayLearningTests(unittest.IsolatedAsyncioTestCase):
    async def test_teaching_does_not_trigger_search(self):
        calls=[]
        async def fake_forward(url,method="GET",payload=None):
            calls.append(url)
            if url.endswith("/apprentice/context"):
                return {"normalized_text":"Когда предлагаешь поиск, Да поищи значит согласие",
                        "utterance_intent":{"intent":"command"},
                        "behavior_rules":[],"capabilities":[]}
            if url.endswith("/analyze"):
                return {"intent":"teach","reply":"Понял","needs_web":False,
                        "learning":[{"kind":"interpretation","context":WEB_CONTEXT,
                                     "trigger":"Да поищи","meaning":"approve",
                                     "instruction":"","evidence":"Да поищи"}],
                        "tokens":12,"model":"qwen-test"}
            if url.endswith("/apprentice/behavior/learn"):
                return {"rules":[{"kind":"interpretation","context":WEB_CONTEXT,
                                  "trigger":"Да поищи","meaning":"approve"}]}
            raise AssertionError("Unexpected request: "+url)
        with patch.object(gateway,"forward",fake_forward):
            response=await gateway.apprentice_chat(gateway.ChatIn(
                text="Когда предлагаешь поиск, Да поищи значит согласие",
                experiment="apprentice",web_mode="ask"),0)
        self.assertEqual(response["route"],"apprentice_behavior_learned")
        self.assertFalse(any(url.endswith("/research/web/search") for url in calls))
        self.assertFalse(any(url.endswith("/research/operations/web/propose") for url in calls))
        self.assertFalse(response["executed"])

    def test_quoted_search_verb_is_not_a_direct_request(self):
        self.assertEqual(gateway.proposed_web_query(
            "Когда спрашиваешь разрешение, фраза Да поищи означает согласие",{}),"")
        self.assertTrue(gateway.proposed_web_query("Поищи документацию",{}))


if __name__=="__main__":
    unittest.main()
