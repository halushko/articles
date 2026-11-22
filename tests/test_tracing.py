"""Cross-mode diagnostics: independent of model, search and database connectivity."""
import tempfile
import httpx
import unittest
from pathlib import Path
from unittest.mock import patch

from services.gateway import api as gateway
from services.gateway.tracing import (ChatTrace, CURRENT_TRACE,
                                      output_summary, input_summary)
from services.research.engine import Engine
from services.research.settings import load
from services.research.sessions import ConversationStore, markdown
from services.llm.api import prompt_diagnostics


class TracePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_modes_trace_and_persist_with_same_md_export(self):
        saved = []
        calls = []

        async def fake_forward(url, method="GET", payload=None):
            calls.append(url)
            if url.endswith("/turns"):
                saved.append(payload)
                return {"turn_id": len(saved)}
            if url == gateway.RESEARCH_URL + "/resolve":
                return {"status": "not_command", "is_command": False}
            if url.endswith("/chat"):
                return {"answer": "Ответ модели", "route": "llm_fast",
                        "model": "qwen2.5:3b", "tokens": 20,
                        "prompt_trace": {"origin": "isolated_llm_baseline",
                                         "memory_injected": False}}
            if url.endswith("/apprentice/context"):
                return {"direct_reply": "", "normalized_text": payload["query"],
                        "identity": {"user": "Дима"},
                        "utterance_intent": {"intent": "question"},
                        "dialogue": {"history": []}, "knowledge": [],
                        "graph": {}, "capabilities": []}
            if url.endswith("/research/operations/web/clear"):
                return {"cleared": True}
            if url.endswith("/analyze"):
                return {"reply": "Ответ ученика", "intent": "ask",
                        "proposals": [], "model": "qwen2.5:3b",
                        "tokens": 22, "prompt_tokens": 99,
                        "prompt_trace": {"origin": "apprentice_analyze",
                                         "history_messages": 0}}
            if url.endswith("/research/dialogue/resolve"):
                return {"handled": False}
            raise AssertionError(url)

        with patch.object(gateway, "forward", fake_forward):
            for mode in ("llm_only", "hybrid", "research_only", "apprentice"):
                before = len(calls)
                data = gateway.ChatIn(
                    text="Как меня зовут?", experiment=mode,
                    session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae",
                    request_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae")
                answer = await gateway.chat(data)
                trace = answer["debug_trace"]
                self.assertEqual(trace["selected_mode"], mode)
                self.assertEqual(trace["outcome"]["route"], answer["route"])
                self.assertEqual(saved[-1]["result"]["debug_trace"]["selected_mode"], mode)
                self.assertTrue(any(s["stage"] == "gateway.route_selection"
                                    for s in trace["steps"]))
                urls = calls[before:]
                self.assertEqual(any(u == gateway.RESEARCH_URL + "/resolve" for u in urls),
                                 mode in ("hybrid", "research_only"))
                self.assertEqual(any(u.endswith("/chat") for u in urls),
                                 mode in ("hybrid", "llm_only"))
                self.assertEqual(any(u.endswith("/apprentice/context") for u in urls),
                                 mode == "apprentice")
        self.assertEqual(len(saved), 4)

    async def test_forward_records_real_http_metadata_without_body(self):
        original=httpx.AsyncClient
        seen=[]
        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200,json={
                "status":"needs_confirmation", "is_command":False,
                "candidates":[{"target_name":"PRIVATE"}],
                "answer":"SECRET ANSWER"
            })
        def fake_client(**kwargs):
            return original(transport=httpx.MockTransport(handler))

        trace=ChatTrace(gateway.ChatIn(text="SECRET QUERY",experiment="hybrid"))
        token=CURRENT_TRACE.set(trace)
        try:
            with patch.object(gateway.httpx,"AsyncClient",side_effect=fake_client):
                result=await gateway.forward("http://research:8000/resolve",
                                             "POST",{"text":"SECRET QUERY","ablation":"combined"})
        finally:
            CURRENT_TRACE.reset(token)
        self.assertEqual(result["status"],"needs_confirmation")
        stages=[s["stage"] for s in trace.steps]
        self.assertIn("service.request",stages)
        self.assertIn("service.response",stages)
        self.assertIn("/resolve",seen[0])
        self.assertIn("needs_confirmation",str(trace.result()))
        self.assertNotIn("SECRET QUERY",str(trace.result()))
        self.assertNotIn("SECRET ANSWER",str(trace.result()))
        self.assertNotIn("PRIVATE",str(trace.result()))

    async def test_upstream_failure_still_persists_trace(self):
        saved = []
        async def fake_forward(url, method="GET", payload=None):
            if url.endswith("/turns"):
                saved.append(payload)
                return {"turn_id": 1}
            raise gateway.HTTPException(503, "Model offline")
        with patch.object(gateway, "forward", fake_forward):
            with self.assertRaises(gateway.HTTPException):
                await gateway.chat(gateway.ChatIn(
                    text="Как меня зовут?", experiment="hybrid",
                    session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"))
        self.assertEqual(saved[0]["result"]["debug_trace"]["outcome"]["error"], "HTTP 503")
        self.assertEqual(saved[0]["result"]["route"], "chat_error")

    def test_metadata_not_full_request_or_web_page(self):
        self.assertNotIn("SECRET", str(input_summary("/analyze", {
            "text": "SECRET", "memory": {"secret": "SECRET"}, "profile": "fast"
        })))
        result = output_summary("/chat", {
            "answer": "SECRET",
            "prompt_trace": {"origin": "isolated_llm_baseline"},
            "model": "test-model"
        })
        self.assertNotIn("SECRET", str(result))
        info=prompt_diagnostics([{"role":"system","content":"PRIVATE SYSTEM"},
                                 {"role":"user","content":"PRIVATE USER"}],
                                 "test", memory_injected=False)
        self.assertNotIn("PRIVATE", str(info))
        self.assertEqual(info["messages"][1]["chars"], 12)


class TraceMarkdownTests(unittest.TestCase):
    def test_export_persists_trace_for_index_llm_and_previous_turns(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=Engine(str(Path(folder)/"trace.sqlite3"), load())
            store=ConversationStore(engine)
            sid=store.create("Трассировка")["id"]
            store.save_turn(sid, "Старый вопрос", "Старый ответ",
                            "hybrid", "fast", "combined", "off",
                            {"route":"llm_fast"})
            debug={"schema":"jarvis.debug.v1","selected_mode":"hybrid",
                   "request_id":"example","steps":[
                       {"stage":"service.request","details":{
                           "service":"research","endpoint":"/resolve"}},
                       {"stage":"service.response","details":{
                           "output":{"is_command":False}}},
                       {"stage":"service.response","details":{
                           "output":{"prompt_trace":{
                               "origin":"isolated_llm_baseline",
                               "memory_injected":False}}}}]}
            store.save_turn(sid,"Как меня зовут?","Не знаю","hybrid",
                            "fast","combined","off",
                            {"route":"llm_fast","debug_trace":debug},
                            actor_id="selected-profile")
            record=store.read(sid)
            export=markdown(record)
            self.assertEqual(len(record["turns"]),2)
            self.assertIn("Старый вопрос",export)
            self.assertIn("Трассировка выполнения",export)
            self.assertIn('"selected_mode": "hybrid"',export)
            self.assertIn('"memory_injected": false',export)
            self.assertIn("selected-profile",export)
            self.assertEqual(record["turns"][-1]["provenance"]["debug_trace"],debug)


if __name__=="__main__":
    unittest.main()
