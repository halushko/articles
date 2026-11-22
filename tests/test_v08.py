"""v0.8 integration regressions: intent -> explicit operation -> audit.

No tests require Ollama or a live web server.
"""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.research.engine import Engine
from services.research.settings import load
from services.research.sessions import ConversationStore
from services.research.approvals import OperationApprovals,spoken_search_decision
from services.gateway import api as gateway


class DurableOperationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.engine=Engine(str(Path(self.temp.name)/"v08.sqlite3"),load())
        self.sessions=ConversationStore(self.engine)
        self.approvals=OperationApprovals(self.engine,self.sessions)
        self.sid=self.sessions.create("разговор")["id"]
        self.second=self.sessions.create("другой")["id"]

    def tearDown(self):
        self.temp.cleanup()

    def test_speech_requires_explicit_named_permission(self):
        for phrase in ("разрешаю","Разрешаю поиск","команда разрешаю поиск",
                       "да разрешаю","я разрешаю","я разрешаю поиск","подтверждаю поиск"):
            self.assertEqual(spoken_search_decision(phrase),"approved")
        for phrase in ("не разрешаю","отмени поиск"):
            self.assertEqual(spoken_search_decision(phrase),"rejected")
        for phrase in ("да","подтверждаю","Как искать в интернете",
                       "разрешаю поиск и удали память"):
            self.assertIsNone(spoken_search_decision(phrase),phrase)

    def test_exact_session_one_time_and_no_replay(self):
        created=self.approvals.propose_search(self.sid,"рецепт голубцов",
                                               "Найди рецепт голубцов")
        self.assertEqual(created["kind"],"web.search")
        self.assertEqual(self.approvals.resolve(self.sid,"да")["handled"],False)
        self.assertEqual(self.approvals.resolve(self.second,"разрешаю поиск")["status"],
                         "no_pending_search")
        first=self.approvals.resolve(self.sid,"разрешаю")
        self.assertEqual(first["status"],"approved")
        self.assertEqual(first["query"],"рецепт голубцов")
        self.assertEqual(first["question"],"Найди рецепт голубцов")
        self.assertEqual(first["operation_id"],created["id"])
        second=self.approvals.resolve(self.sid,"разрешаю поиск",
                                      operation_id=created["id"])
        self.assertEqual(second["status"],"no_pending_search")
        with self.engine.connect() as con:
            history=con.execute("SELECT decision FROM operation_audit").fetchall()
        self.assertEqual([x["decision"] for x in history],["approved"])

    def test_new_request_supersedes_old_and_ttl(self):
        old=self.approvals.propose_search(self.sid,"первая страница","первый поиск")
        new=self.approvals.propose_search(self.sid,"вторая страница","второй поиск")
        self.assertEqual(self.approvals.resolve(self.sid,operation_id=old["id"],
                      source="button")["status"],"no_pending_search")
        with self.engine.connect() as con:
            con.execute("UPDATE pending_operations SET created_epoch=? WHERE id=?",
                        (time.time()-400,new["id"]))
        self.assertEqual(self.approvals.resolve(self.sid,"разрешаю поиск")["status"],"expired")

    def test_explicit_reject_and_clear(self):
        self.approvals.propose_search(self.sid,"что такое API","что такое API")
        self.assertEqual(self.approvals.resolve(self.sid,"не разрешаю")["status"],"rejected")
        self.approvals.propose_search(self.sid,"какая погода","какая погода")
        self.approvals.clear(self.sid)
        self.assertEqual(self.approvals.resolve(self.sid,"разрешаю")["status"],
                         "no_pending_search")


class GatewayWebApprovalTests(unittest.IsolatedAsyncioTestCase):
    async def test_spoken_permission_uses_saved_query_without_qwen(self):
        calls=[]
        async def forward(url,method="GET",payload=None):
            calls.append((url,payload))
            if url.endswith("/research/operations/web/resolve"):
                return {"handled":True,"status":"approved",
                        "operation_id":"permit1","question":"Найди рецепт",
                        "query":"рецепт голубцов"}
            if url.endswith("/research/operations/web/start"):
                return {"started":True,"status":"running"}
            if url.endswith("/research/operations/web/finish"):
                self.assertEqual(payload["status"],"failed")
                return {"saved":True,"status":"failed"}
            if url.endswith("/research/web/search"):
                self.assertEqual(payload["query"],"рецепт голубцов")
                return {"available":False,"status":"unavailable",
                        "reason":"search offline","sources":[]}
            raise AssertionError("Unexpected call: "+url)
        with patch.object(gateway,"forward",forward):
            result=await gateway.apprentice_chat(gateway.ChatIn(
                text="разрешаю поиск",experiment="apprentice",web_mode="ask",
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),0)
        self.assertEqual(result["approval_id"],"permit1")
        self.assertEqual(result["route"],"apprentice_web_unavailable")
        self.assertEqual(len(calls),4)

    async def test_explicit_search_skips_qwen_and_queues_approval(self):
        calls=[]
        async def forward(url,method="GET",payload=None):
            calls.append((url,payload))
            if url.endswith("/apprentice/context"):
                return {"normalized_text":"Найди в интернете рецепт голубцов",
                        "utterance_intent":{"intent":"command","confidence":"high"}}
            if url.endswith("/research/operations/web/propose"):
                return {"id":"approved-id","query":payload["query"],"question":payload["question"]}
            raise AssertionError("Unexpected call: "+url)
        with patch.object(gateway,"forward",forward):
            result=await gateway.apprentice_chat(gateway.ChatIn(
                text="Найди в интернете рецепт голубцов",experiment="apprentice",
                web_mode="ask",session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),0)
        self.assertEqual(result["route"],"apprentice_web_permission")
        self.assertIn("Разрешаю поиск",result["answer"])
        self.assertEqual(len(calls),2)
        self.assertEqual(result["web_approval"]["id"],"approved-id")

    async def test_legacy_boolean_permission_cannot_skip_approval(self):
        async def forward(url,method="GET",payload=None):
            if url.endswith("/research/operations/web/resolve"):
                self.assertEqual(payload["source"],"voice")
                return {"handled":True,"status":"no_pending_search","executed":False,
                        "answer":"Нет ожидающего разрешения."}
            raise AssertionError(url)
        with patch.object(gateway,"forward",forward):
            result=await gateway.apprentice_chat(gateway.ChatIn(
                text="Найди в интернете рецепт",experiment="apprentice",
                web_mode="ask",allow_web_search=True,
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),0)
        self.assertEqual(result["route"],"web_permission_no_pending_search")
        self.assertFalse(result["executed"])


if __name__=="__main__":
    unittest.main()
