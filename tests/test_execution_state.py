"""Real-user regression suite: voice consent, tool state, topic memory, model loops.

No Ollama and no network: an executor can be claimed only after consent and
the user's latest result/previous context come from SQLite, never model prose.
"""
import os
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from services.research.api import app as research_app

from services.research.engine import Engine
from services.research.settings import load
from services.research.sessions import ConversationStore
from services.research.approvals import OperationApprovals, spoken_search_decision
from services.research.dialogue import DialogueCoordinator
from services.research.review_voice import VoiceReviews
from services.research.adaptive import BehaviorMemory
from services.research.apprentice import ApprenticeMemory
from services.gateway import api as gateway
from services.gateway.grounding import response_from_records,looks_like_loop,claims_unobserved_search
from services.llm.api import compact_world_context


class ExecutionStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.engine=Engine(str(Path(self.tmp.name)/"ops.sqlite3"),load())
        self.sessions=ConversationStore(self.engine)
        self.sid=self.sessions.create("test")["id"]
        self.actor="dima-actor"
        self.ops=OperationApprovals(self.engine,self.sessions)

    def tearDown(self):
        self.tmp.cleanup()

    def test_natural_spoken_consent_and_no_substring_escalation(self):
        for phrase in ("я разрешаю","я разрешаю поиск","Разрешаю поиск",
                       "Да, разрешаю поиск","команда я разрешаю этот поиск"):
            self.assertEqual(spoken_search_decision(phrase),"approved",phrase)
            self.assertTrue(gateway.is_explicit_web_decision(phrase),phrase)
        for phrase in ("я не разрешаю","не разрешаю поиск","отмени поиск"):
            self.assertEqual(spoken_search_decision(phrase),"rejected",phrase)
        for phrase in ("да","я разрешаю поиск и удали все данные",
                       "мой сосед говорит я разрешаю поиск"):
            self.assertIsNone(spoken_search_decision(phrase),phrase)
            self.assertFalse(gateway.is_explicit_web_decision(phrase),phrase)

    def test_exactly_once_claim_and_real_finished_result(self):
        offer=self.ops.propose_search(self.sid,"как заменить колесо",
                                      "Найди инструкцию",self.actor)
        not_yet=self.ops.execution_start(self.sid,offer["id"],self.actor)
        self.assertFalse(not_yet["started"])
        permission=self.ops.resolve(self.sid,"я разрешаю поиск",actor_id=self.actor)
        self.assertEqual(permission["status"],"approved")
        self.assertFalse(self.ops.execution_start(self.sid,offer["id"],"other")["started"])
        self.assertTrue(self.ops.execution_start(self.sid,offer["id"],self.actor)["started"])
        self.assertFalse(self.ops.execution_start(self.sid,offer["id"],self.actor)["started"])
        self.assertEqual(self.ops.execution_status(self.sid,self.actor)["current"]["execution_status"],
                         "running")
        saved=self.ops.execution_finish(self.sid,offer["id"],self.actor,
               "succeeded","Реальный результат",
               [{"title":"Мануал","url":"https://example.org/manual","status":"read"}])
        self.assertTrue(saved["saved"])
        self.assertFalse(self.ops.execution_finish(self.sid,offer["id"],self.actor,
                           "failed","ложный результат")["saved"])
        state=self.ops.execution_status(self.sid,self.actor)
        self.assertEqual(state["current"]["execution_status"],"succeeded")
        self.assertEqual(state["last_result"]["answer"],"Реальный результат")
        self.assertEqual(state["last_result"]["sources"][0]["url"],
                         "https://example.org/manual")
        self.assertIsNone(self.ops.execution_status(self.sid,"other")["last_result"])
        self.assertEqual(self.ops.resolve(self.sid,"я разрешаю",
                          actor_id=self.actor)["status"],"no_pending_search")

    def test_failure_is_persisted_and_not_invented_as_running(self):
        offer=self.ops.propose_search(self.sid,"документация","Вопрос",self.actor)
        self.ops.resolve(self.sid,"разрешаю поиск",actor_id=self.actor)
        self.ops.execution_start(self.sid,offer["id"],self.actor)
        self.ops.execution_finish(self.sid,offer["id"],self.actor,"failed",
                                  "Сервис не работает",[],"connection_error")
        state=self.ops.execution_status(self.sid,self.actor)
        reply=response_from_records("Где результат поиска?",{"operations":state,
                  "dialogue":{}})
        self.assertEqual(reply[1],"sqlite_operation_status")
        self.assertIn("ошибкой",reply[0])
        self.assertNotIn("ищу",reply[0])


class DialogueGroundingTests(unittest.TestCase):
    def test_records_answer_provenance(self):
        memory={"speaker":{"display_name":"Дима","origin":"selected_profile"},
                "dialogue":{"last_response_evidence":{
                    "route":"world_grounded_chat","source_kind":"unverified_model_answer",
                    "model":"qwen2.5:3b","sources":[]}}}
        response=response_from_records("Откуда ты узнал моё имя?",memory)
        self.assertEqual(response[1],"sqlite_answer_provenance")
        self.assertIn("SQLite",response[0])
        response=response_from_records("Я спрашивал про колесо. Откуда ты взял эти сведения?",memory)
        self.assertIn("без проверенного источника",response[0])

    def test_remembers_prior_topic_from_actor_scoped_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=Engine(str(Path(folder)/"dialogue.sqlite3"),load())
            sessions=ConversationStore(engine)
            behavior=BehaviorMemory(engine)
            apprentice=ApprenticeMemory(engine)
            reviews=VoiceReviews(engine,apprentice,sessions)
            approvals=OperationApprovals(engine,sessions,behavior)
            coordinator=DialogueCoordinator(engine,sessions,approvals,reviews,behavior)
            sid=sessions.create("Проверка")["id"]
            for question in ("Найди инструкцию по замене колеса автомобиля",
                             "Какие инструменты нужны для замены колеса автомобиля?",
                             "Забудем про автомобиль. Что такое Home Assistant?",
                             "Для чего он используется?"):
                sessions.save_turn(sid,question,"тест","hybrid","fast","combined",
                                   "ask",{"route":"world_grounded_chat",
                                          "model":"qwen2.5:3b"},actor_id="actor-A")
            context=coordinator.context(sid,"actor-A")
            previous=context["previous_topic"]
            self.assertIn("автомобил",previous["subject"])
            self.assertIn("колеса",previous["reference"])
            found=response_from_records("О чём мы говорили перед Home Assistant?",
                                        {"dialogue":context})
            self.assertEqual(found[1],"sqlite_dialogue_recall")
            self.assertIn("колеса",found[0])
            self.assertFalse(coordinator.context(sid,"actor-B")["history"])

    def test_no_unobserved_search_or_repeated_output(self):
        self.assertTrue(looks_like_loop("Чтобы заменить колесо, "+("шайбы для шарнира "*12)))
        self.assertFalse(looks_like_loop("Домкрат, баллонный ключ, запасное колесо и упоры."))
        self.assertTrue(claims_unobserved_search(
            "Я ищу информацию в интернете, скоро отвечу."))


class ExecutionHttpTests(unittest.TestCase):
    def test_real_api_permission_execution_status_and_world_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ,{"DATABASE_PATH":str(Path(folder)/"live.sqlite3")}):
                with TestClient(research_app) as client:
                    actor=client.post("/research/actors",json={
                        "display_name":"Дима"}).json()["id"]
                    sid=client.post("/research/sessions",json={}).json()["id"]
                    common={"session_id":sid,"actor_id":actor}
                    offer=client.post("/research/operations/web/propose",json={
                        **common,"query":"замена колеса","question":"Найди инструкцию"}).json()
                    resolved=client.post("/research/operations/web/resolve",json={
                        **common,"phrase":"я разрешаю поиск"}).json()
                    self.assertEqual(resolved["status"],"approved")
                    op=offer["id"]
                    started=client.post("/research/operations/web/start",json={
                        **common,"operation_id":op}).json()
                    self.assertTrue(started["started"])
                    again=client.post("/research/operations/web/start",json={
                        **common,"operation_id":op}).json()
                    self.assertFalse(again["started"])
                    finished=client.post("/research/operations/web/finish",json={
                        **common,"operation_id":op,"status":"succeeded",
                        "answer":"Проверенный ответ","sources":[{
                            "title":"Руководство","url":"https://example.org",
                            "status":"read"}]}).json()
                    self.assertTrue(finished["saved"])
                    status=client.post("/research/operations/web/status",json=common).json()
                    self.assertEqual(status["current"]["execution_status"],"succeeded")
                    context=client.post("/apprentice/context",json={
                        "query":"Где результат поиска?",**common}).json()
                    self.assertEqual(context["operations"]["last_result"]["answer"],
                                     "Проверенный ответ")
                    denied=client.post("/research/operations/web/status",json={
                        "session_id":sid,"actor_id":""}).json()
                    self.assertIsNone(denied["last_result"])


class PromptBudgetTests(unittest.TestCase):
    def test_important_memory_is_valid_json_even_with_large_graph(self):
        memory={"identity":{"user":"Дима","assistant":"Джарвис"},
                "speaker":{"display_name":"Дима"},
                "dialogue":{"state":{"goal":"заменить колесо"},
                            "last_response_evidence":{"source_kind":"external_web",
                                                     "route":"apprentice_web_evidence"},
                            "previous_topic":{"subject":"автомобиль"}},
                "operations":{"current":{"approval_status":"approved",
                                         "execution_status":"succeeded"}},
                "graph":{"nodes":[{"name":"Сущность "+str(i),
                                    "description":"очень длинное описание "*80}
                                   for i in range(20)],
                         "edges":[]},
                "knowledge":[{"fact":"много текста "*150}]}
        raw=compact_world_context(memory)
        compact=json.loads(raw)
        self.assertLessEqual(len(raw),4600)
        self.assertEqual(compact["identity"]["user"],"Дима")
        self.assertEqual(compact["dialogue_state"]["goal"],"заменить колесо")
        self.assertEqual(compact["last_response_evidence"]["source_kind"],
                         "external_web")
        self.assertEqual(compact["operation_state"]["execution"],"succeeded")


class GatewayExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_says_ya_razreshayu_and_real_executor_is_called_once(self):
        calls=[]
        async def stub(url,method="GET",payload=None):
            calls.append(url)
            if url.endswith("/research/operations/web/resolve"):
                self.assertEqual(payload["phrase"],"я разрешаю поиск")
                return {"handled":True,"status":"approved",
                        "operation_id":"a"*36,"query":"замена колеса",
                        "question":"Найди инструкцию"}
            if url.endswith("/research/operations/web/start"):
                return {"started":True,"status":"running"}
            if url.endswith("/research/web/search"):
                self.assertEqual(payload["query"],"замена колеса")
                return {"available":False,"status":"offline",
                        "reason":"searxng offline","sources":[]}
            if url.endswith("/research/operations/web/finish"):
                self.assertEqual(payload["status"],"failed")
                return {"saved":True,"status":"failed"}
            raise AssertionError("Unexpected call: "+url)
        with patch.object(gateway,"forward",stub):
            result=await gateway.apprentice_chat(gateway.ChatIn(
                text="я разрешаю поиск",experiment="hybrid",web_mode="ask",
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),0)
        self.assertEqual(result["route"],"apprentice_web_unavailable")
        self.assertEqual(result["operation_state"]["status"],"failed")
        self.assertEqual(len(calls),4)
        self.assertFalse(any(url.endswith("/grounded-chat") for url in calls))


if __name__=="__main__":
    unittest.main()
