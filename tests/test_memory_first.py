"""Memory-first JARVIS and bounded dialogue/export regressions (offline)."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from services.research.engine import Engine
from services.research.settings import load
from services.research.sessions import ConversationStore
from services.research.findings import ResearchFindings
from services.research.api import app as research_app
from services.gateway import api as gateway


URL = "https://example.org/kotlety"
EVIDENCE = [{"title": "Котлеты — рецепт", "url": URL,
             "snippet": "Котлеты: фарш, лук, соль.", "status": "read"}]
RECIPE = "Проверенный на сайте [1] рецепт котлет: фарш и лук, поджарить."
TWO = "Рецепт борща из публичного источника."


def web_result(query="рецепт котлет", answer=RECIPE):
    return {
        "answer": answer, "sources": EVIDENCE,
        "route": "apprentice_web_evidence", "model": "qwen2.5:3b",
        "search_query": query, "search_status": "ok",
        "search_executed": True, "tokens": 24
    }


class FindingPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = Engine(str(Path(self.tmp.name)/"data.db"), load())
        self.store = ConversationStore(self.engine)
        self.findings = ResearchFindings(self.engine)
        self.one = self.store.create()["id"]
        self.two = self.store.create()["id"]
        self.actor = "11111111-1111-4111-8111-111111111111"

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, session, actor="", query="рецепт котлет", answer=RECIPE):
        result = web_result(query, answer)
        self.store.save_turn(session, "Поищи в интернете "+query, answer,
                             "apprentice","fast","combined","auto",result,actor)
        return self.findings.remember(session, actor, "Поищи в интернете "+query,
                                      answer,result)

    def test_remembers_search_answer_not_just_topic(self):
        self.assertTrue(self.record(self.one, actor=self.actor))
        recalled = self.findings.recall("Дай рецепт котлет",self.two,self.actor)
        self.assertTrue(recalled["found"])
        self.assertEqual(recalled["answer"], RECIPE)
        self.assertEqual(recalled["sources"][0]["url"], URL)
        self.assertEqual(recalled["evidence_source"],
                         "saved_public_web_research_unverified")
        self.assertFalse(self.findings.recall(
            "дай рецепт борща",self.two,self.actor)["found"])
        self.assertFalse(self.findings.recall(
            "предложи другой рецепт котлет",self.two,self.actor)["found"])

    def test_never_crosses_speaker_or_anonymous_session(self):
        self.assertTrue(self.record(self.one, actor=self.actor))
        self.assertFalse(self.findings.recall("дай рецепт котлет", self.two,
            "22222222-2222-4222-8222-222222222222")["found"])
        self.assertFalse(self.findings.recall("дай рецепт котлет",self.two)["found"])
        self.assertTrue(self.record(self.two))
        self.assertTrue(self.findings.recall("дай рецепт котлет",self.two)["found"])
        self.assertFalse(self.findings.recall("дай рецепт котлет",self.one)["found"])

    def test_only_real_search_with_provenance_is_stored(self):
        fake = {**web_result(), "route": "gemini_general"}
        self.assertFalse(self.findings.remember(self.one,self.actor,
                                                "рецепт котлет","галлюцинация",fake))
        fake = {**web_result(), "search_status": "unverified"}
        self.assertFalse(self.findings.remember(self.one,self.actor,
                                                "рецепт котлет","галлюцинация",fake))
        fake = {**web_result(), "sources": [
            {"url": "https://127.0.0.1/private", "title": "local"}]}
        self.assertFalse(self.findings.remember(self.one,self.actor,
                                                "рецепт котлет","галлюцинация",fake))

    def test_clear_deletes_session_traces_and_pending_state_but_retains_actor_findings(self):
        from services.research.approvals import OperationApprovals
        from services.research.dialogue import DialogueCoordinator
        from services.research.review_voice import VoiceReviews
        from services.research.adaptive import BehaviorMemory
        from services.research.apprentice import ApprenticeMemory
        self.assertTrue(self.record(self.one, actor=self.actor))
        self.assertTrue(self.record(self.one))
        self.store.event(self.one, "test_event", {"trace": "past"})
        approvals = OperationApprovals(self.engine,self.store)
        voice = VoiceReviews(self.engine,ApprenticeMemory(self.engine),self.store)
        dlg = DialogueCoordinator(self.engine,self.store,approvals,voice,
                                  BehaviorMemory(self.engine))
        approvals.propose_search(self.one,"рецепт борща","найди рецепт")
        self.assertTrue(dlg.pending_summary(self.one) or
                        approvals.execution_status(self.one).get("current"))
        result=self.store.clear(self.one)
        self.assertTrue(result["cleared"])
        self.assertGreaterEqual(result["turns_deleted"],2)
        self.assertFalse(self.store.read(self.one)["turns"])
        self.assertFalse(self.store.read(self.one)["events"])
        self.assertEqual(dlg.context(self.one)["history"], [])
        self.assertEqual(dlg.pending_summary(self.one), [])
        self.assertTrue(self.findings.recall("дай рецепт котлет",self.two,
                                             self.actor)["found"])
        self.assertFalse(self.findings.recall("дай рецепт котлет",self.one)["found"])
        self.assertEqual(self.store.read(self.one)["title"], "Новый диалог")


class ResearchHttpTests(unittest.TestCase):
    def test_clear_and_compact_export_contract(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"DATABASE_PATH": str(Path(tmp)/"research.sqlite3")}):
            with TestClient(research_app) as client:
                sid=client.post("/research/sessions",json={}).json()["id"]
                for i in range(12):
                    turn=client.post(f"/research/sessions/{sid}/turns",json={
                        "question": f"Вопрос {i}", "answer":f"Ответ {i}",
                        "experiment":"apprentice","profile":"fast",
                        "ablation":"combined","web_mode":"off",
                        "result":{"route":"test",
                                  "debug_trace":{"trace_id":i}}})
                    self.assertEqual(turn.status_code,200)
                compact=client.get(f"/research/sessions/{sid}/export?last=3")
                self.assertEqual(compact.status_code,200)
                doc=compact.json()["markdown"]
                self.assertNotIn("Вопрос 0",doc)
                self.assertIn("Вопрос 11",doc)
                self.assertIn('"trace_id": 11',doc)
                self.assertNotIn('"trace_id": 1,',doc)
                self.assertEqual(len(compact.json()["session"]["turns"]),3)
                full=client.get(f"/research/sessions/{sid}/export").json()
                self.assertEqual(len(full["session"]["turns"]),12)
                self.assertEqual(client.post(f"/research/sessions/{sid}/clear",
                                            json={}).status_code,422)
                result=client.post(f"/research/sessions/{sid}/clear",
                                   json={"confirm":True})
                self.assertEqual(result.status_code,200)
                self.assertEqual(result.json()["turns_deleted"],12)
                exported=client.get(f"/research/sessions/{sid}/export").json()
                self.assertEqual(exported["session"]["turns"],[])
                self.assertNotIn("Вопрос 0",exported["markdown"])
                self.assertEqual(client.post("/research/sessions/bad/clear",
                                            json={"confirm":True}).status_code,404)


class GatewayMemoryFirstTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_recipe_is_recalled_without_llm_or_network(self):
        calls=[]
        async def forward(url,method="GET",payload=None):
            calls.append(url)
            if url.endswith("/apprentice/context"):
                return {"normalized_text":"дай рецепт котлет",
                        "utterance_intent":{"intent":"question"},"dialogue":{}}
            if url.endswith("/research/findings/recall"):
                self.assertEqual(payload["query"],"дай рецепт котлет")
                return {"found":True,"id":34,"search_query":"рецепт котлет",
                        "answer":RECIPE,"sources":EVIDENCE,
                        "created_at":"2026-11-22 13:00:00","match_score":1.0,
                        "evidence_source":"saved_public_web_research_unverified"}
            raise AssertionError("Unnecessary model/search call: "+url)
        with patch.dict(os.environ,{"GEMINI_ENABLED":"true"}), \
             patch.object(gateway,"forward",forward):
            reply=await gateway.apprentice_chat(gateway.ChatIn(
                text="дай рецепт котлет",web_mode="off"),time.perf_counter())
        self.assertEqual(reply["route"],"memory_public_research")
        self.assertEqual(reply["answer"], RECIPE)
        self.assertEqual(reply["sources"][0]["url"], URL)
        self.assertEqual(reply["model"],"SQLite")
        self.assertFalse(reply["search_executed"])
        self.assertEqual(len(calls),2)

    async def test_explicit_fresh_web_request_does_not_replay_cached_recipe(self):
        calls=[]
        async def forward(url,method="GET",payload=None):
            calls.append(url)
            if url.endswith("/apprentice/context"):
                return {"normalized_text":"поищи в интернете рецепт котлет",
                        "utterance_intent":{"intent":"question"},"dialogue":{}}
            if url.endswith("/research/operations/web/propose"):
                return {"query":payload["query"],"id":"approval"}
            raise AssertionError("Unexpected call: "+url)
        with patch.dict(os.environ,{"GEMINI_ENABLED":"true"}), \
             patch.object(gateway,"forward",forward):
            reply=await gateway.apprentice_chat(gateway.ChatIn(
                text="поищи в интернете рецепт котлет",web_mode="ask",
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),
                time.perf_counter())
        self.assertEqual(reply["route"],"apprentice_web_permission")
        self.assertNotIn("/research/findings/recall",str(calls))

    async def test_search_uses_local_qwen_even_when_gemini_enabled(self):
        calls=[]
        async def forward(url,method="GET",payload=None):
            calls.append(url)
            if url.endswith("/research/web/search"):
                return {"available":True,"status":"ok","sources":EVIDENCE}
            if url.endswith("/web-answer"):
                return {"route":"apprentice_web_evidence","model":"qwen2.5:3b",
                        "answer":"С сайта [1] фарш и лук."}
            raise AssertionError("Unnecessary Gemini call: "+url)
        with patch.dict(os.environ,{"GEMINI_ENABLED":"true"}), \
             patch.object(gateway,"forward",forward):
            reply=await gateway.answer_from_public_search(
                gateway.ChatIn(text="рецепт котлет"),"рецепт котлет",{},
                "рецепт котлет",time.perf_counter())
        self.assertEqual(reply["route"],"apprentice_web_evidence")
        self.assertEqual(reply["model"],"qwen2.5:3b")
        self.assertNotIn("/gemini-answer",str(calls))


if __name__ == "__main__":
    unittest.main()
