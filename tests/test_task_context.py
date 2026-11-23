"""Task continuity: user-only topic tracking, ASR search and approved query."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.research.task_context import plan, merge_focus, search_clause
from services.research.engine import Engine
from services.research.settings import load
from services.research.sessions import ConversationStore
from services.research.approvals import OperationApprovals
from services.research.dialogue import DialogueCoordinator
from services.research.review_voice import VoiceReviews
from services.research.apprentice import ApprenticeMemory
from services.research.adaptive import BehaviorMemory
from services.gateway import api as gateway


class TaskParserTests(unittest.TestCase):
    def test_search_typo_still_resolves_intent(self):
        found=plan([],"поища рецепт борща")
        self.assertTrue(found["explicit_search"])
        self.assertEqual(found["search_query"],"рецепт борща")
        self.assertFalse(found["context_used"])

    def test_followup_fills_missing_subject_without_model(self):
        previous=["дай рецепт котлет","поища рецепт борща"]
        found=plan(previous,"мне нужен классический рецепт, поищи в интернете")
        self.assertEqual(found["search_query"],"классический рецепт борща")
        self.assertTrue(found["context_used"])
        again=plan(previous+["мне нужен классический рецепт, поищи в интернете"],
                   "мне нужен классический рецепт, поищи в интернете")
        self.assertEqual(again["search_query"],"классический рецепт борща")

    def test_generic_subject_resolution_not_hardcoded_to_recipes(self):
        found=plan(["найди документацию Home Assistant REST API"],
                   "мне нужна официальная документация, поищи в интернете")
        self.assertEqual(found["search_query"],
                         "официальная документация Home Assistant REST API")

    def test_new_subject_overrides_old_subject(self):
        found=plan(["найди рецепт котлет"],"поища рецепт борща")
        self.assertEqual(found["search_query"],"рецепт борща")
        self.assertFalse(found["context_used"])
        found=plan(["найди инструкцию принтера"],"найди документацию роутера")
        self.assertEqual(found["search_query"],"документацию роутера")

    def test_subjectless_search_without_context_asks_clarification(self):
        found=plan([],"поищи в интернете")
        self.assertEqual(found["status"],"missing_subject")
        self.assertEqual(found["search_query"],"")

    def test_subjectless_search_uses_last_explicit_topic(self):
        found=plan(["найди инструкцию принтера","разрешаю поиск"],
                   "поищи в интернете")
        self.assertEqual(found["search_query"],"инструкцию принтера")
        self.assertTrue(found["context_used"])

    def test_quoted_teaching_is_not_web_action(self):
        found=plan([],"Когда ты спрашиваешь, фраза поищи в интернете "
                       "значит согласие")
        self.assertFalse(found["explicit_search"])

    def test_unrelated_questions_do_not_replace_topic(self):
        found=plan(["найди рецепт борща","как меня зовут",
                    "какая погода","кто ты"],"поищи в интернете")
        self.assertEqual(found["status"],"missing_subject")
        self.assertEqual(found["search_query"],"")


class TaskIsolationTests(unittest.TestCase):
    def test_context_read_is_actor_and_session_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine=Engine(str(Path(tmp)/"research.db"),load())
            store=ConversationStore(engine)
            apprentice=ApprenticeMemory(engine)
            approvals=OperationApprovals(engine,store)
            dialogue=DialogueCoordinator(
                engine,store,approvals,VoiceReviews(engine,apprentice,store),
                BehaviorMemory(engine))
            first=store.create()["id"]
            second=store.create()["id"]
            dima="11111111-1111-4111-8111-111111111111"
            nadya="22222222-2222-4222-8222-222222222222"
            store.save_turn(first,"поища рецепт борща","Разреши поиск.",
                            "apprentice","fast","combined","ask",{},actor_id=dima)
            self.assertEqual(dialogue.task_context(first,dima,
                "мне нужен классический рецепт, поищи в интернете")["search_query"],
                "классический рецепт борща")
            self.assertEqual(dialogue.task_context(first,nadya,
                "поищи в интернете")["status"],"missing_subject")
            self.assertEqual(dialogue.task_context(second,dima,
                "поищи в интернете")["status"],"missing_subject")
            store.clear(first)
            self.assertEqual(dialogue.task_context(first,dima,
                "поищи в интернете")["status"],"missing_subject")


class GatewayTaskContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_followup_uses_completed_query_before_asking_permission(self):
        calls=[]
        async def forward(url,method="GET",payload=None):
            calls.append((url,payload))
            if url.endswith("/apprentice/context"):
                return {
                    "normalized_text":"мне нужен классический рецепт, поищи в интернете",
                    "utterance_intent":{"intent":"command"},
                    "dialogue":{"history":[
                        {"role":"user","content":"поища рецепт борща"}]},
                    "task_context":{
                        "subject":"классический рецепт борща",
                        "search_query":"классический рецепт борща",
                        "explicit_search":True,"context_used":True,"status":"ready"}}
            if url.endswith("/research/operations/web/propose"):
                self.assertEqual(payload["query"],"классический рецепт борща")
                return {"id":"approval-123","query":payload["query"],
                        "question":payload["question"]}
            raise AssertionError("Wrong route: "+url)
        with patch.object(gateway,"forward",forward):
            result=await gateway.apprentice_chat(gateway.ChatIn(
                text="мне нужен классический рецепт, поищи в интернете",
                experiment="apprentice",web_mode="ask",
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),
                time.perf_counter())
        self.assertEqual(result["route"],"apprentice_web_permission")
        self.assertEqual(len(calls),2)

    async def test_typo_is_search_command_not_qwen_chitchat(self):
        calls=[]
        async def forward(url,method="GET",payload=None):
            calls.append(url)
            if url.endswith("/research/dialogue/resolve"):
                return {"handled":False}
            if url.endswith("/apprentice/context"):
                return {
                    "normalized_text":"поища рецепт борща",
                    "utterance_intent":{"intent":"unknown"},
                    "dialogue":{},
                    "task_context":{
                        "subject":"рецепт борща","search_query":"рецепт борща",
                        "explicit_search":True,"status":"ready"}}
            if url.endswith("/research/operations/web/propose"):
                return {"id":"approval","query":payload["query"]}
            raise AssertionError("Must not call Qwen: "+url)
        with patch.object(gateway,"forward",forward):
            result=await gateway.apprentice_chat(gateway.ChatIn(
                text="поища рецепт борща",web_mode="ask",
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),
                time.perf_counter())
        self.assertEqual(result["route"],"apprentice_web_permission")
        self.assertEqual(len(calls),3)
        self.assertNotIn("/analyze",str(calls))
        self.assertNotIn("/grounded-chat",str(calls))

    async def test_executor_uses_approved_resolved_topic_for_sources_and_llm(self):
        calls=[]
        async def forward(url,method="GET",payload=None):
            calls.append((url,payload))
            if url.endswith("/research/operations/web/start"):
                return {"started":True,"status":"running"}
            if url.endswith("/research/web/search"):
                self.assertEqual(payload["query"],"классический рецепт борща")
                return {"available":True,"status":"ok","sources":[
                    {"title":"Классический борщ",
                     "url":"https://example.org/borscht",
                     "status":"read","snippet":"Свекла и капуста"}]}
            if url.endswith("/web-answer"):
                self.assertEqual(payload["question"],"классический рецепт борща")
                return {"answer":"В борщ добавляют свеклу.",
                        "route":"apprentice_web_evidence","model":"qwen2.5:3b"}
            if url.endswith("/research/operations/web/finish"):
                return {"saved":True,"status":"succeeded"}
            raise AssertionError(url)
        with patch.object(gateway,"forward",forward):
            result=await gateway.execute_approved_search(gateway.ChatIn(
                text="мне нужен классический рецепт, поищи в интернете",
                experiment="apprentice",web_mode="ask",
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),
                {"operation_id":"123","query":"классический рецепт борща",
                 "question":"мне нужен классический рецепт, поищи в интернете"},
                time.perf_counter())
        self.assertEqual(result["route"],"apprentice_web_evidence")
        self.assertEqual(result["search_query"],"классический рецепт борща")
        self.assertEqual(len(calls),4)


if __name__=="__main__":
    unittest.main()
