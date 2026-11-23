"""Offline Gemini tests: free generation, public SearXNG evidence, privacy, fallback.

Synthetic API responses test routing; deployment must still be checked against
the user's real (secret) Gemini key and Docker network.
"""
import os
import time
import unittest
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from services.llm import gemini
from services.llm.api import GeminiAnswerIn
from services.gateway import api as gateway
from services.gateway.cloud import public_only, recipe_request, public_information_request


class FakeGoogle:
    def __init__(self, response, status=200):
        self.response = response
        self.status = status
        self.request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def post(self, url, headers, json):
        self.request = (url, headers, json)
        return httpx.Response(self.status, json=self.response,
                              request=httpx.Request("POST", url))


SAMPLE = {
    "candidates": [{
        "content": {"parts": [{"text": "Рецепт: фарш, лук, соль."}]},
        "finishReason": "STOP",
        "groundingMetadata": {
            "webSearchQueries": ["котлеты классический рецепт"],
            "groundingChunks": [
                {"web": {"title": "Котлеты", "uri": "https://example.org/kotlety"}},
                {"web": {"title": "Duplicated", "uri": "https://example.org/kotlety"}},
            ],
        },
    }],
    "usageMetadata": {"promptTokenCount": 36, "candidatesTokenCount": 47},
}
PLAIN = {
    "candidates": [{"content": {"parts": [{"text": "Рецепт: фарш, лук, соль."}]},
                    "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 18, "candidatesTokenCount": 23},
}
EVIDENCE = [{"title": "Рецепт котлет", "url": "https://example.org/kotlety",
             "status": "read", "page_text": "Фарш, лук, соль. Обжарить до готовности.",
             "memory": "DO NOT SEND PRIVATE MEMORY"}]


class GeminiUnitTests(unittest.TestCase):
    def test_free_generation_is_default_not_google_search(self):
        self.assertFalse(GeminiAnswerIn(question="Что такое Python").search)
        payload = gemini.request_payload("рецепт котлет")
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "рецепт котлет")
        self.assertEqual(gemini.DEFAULT_MODEL, "gemini-3.5-flash-lite")

    def test_google_grounding_remains_explicit_only(self):
        request = gemini.request_payload("рецепт котлет", True)
        self.assertEqual(request["tools"], [{"google_search": {}}])
        for sensitive in ("memory", "actor_id", "session_id", "history", "HA_TOKEN"):
            self.assertNotIn(sensitive, str(request))

    def test_supplied_sources_are_data_not_google_tool_or_private_memory(self):
        request = gemini.request_payload("рецепт котлет", False, EVIDENCE)
        self.assertNotIn("tools", request)
        self.assertIn("Фарш, лук", str(request))
        self.assertIn("https://example.org/kotlety", str(request))
        self.assertNotIn("DO NOT SEND PRIVATE MEMORY", str(request))
        self.assertNotIn("actor_id", str(request))
        with self.assertRaises(ValueError):
            gemini.request_payload("рецепт котлет", True, EVIDENCE)

    def test_grounding_sources_distinct_from_model_claims(self):
        result = gemini.parse_response(SAMPLE, True, "gemini-test", 1.0)
        self.assertEqual(result["route"], "gemini_google_grounded")
        self.assertTrue(result["search_verified"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["url"], "https://example.org/kotlety")

    def test_no_grounding_means_no_verified_search_answer(self):
        result = gemini.parse_response({"candidates": [{
            "content": {"parts": [{"text": "В интернете нашёл котлеты с мёдом!"}]},
            "finishReason": "STOP"}]}, True, "gemini-test", 1)
        self.assertEqual(result["route"], "gemini_search_unverified")
        self.assertFalse(result["search_verified"])
        self.assertNotIn("мёдом", result["answer"])

    def test_truncated_and_empty_response(self):
        bad = {"candidates": [{"content": {"parts": [{"text": "Первый шаг"}]},
                               "finishReason": "MAX_TOKENS"}]}
        result = gemini.parse_response(bad, False, "gemini-test", 1)
        self.assertFalse(result["response_complete"])
        self.assertIn("неполный", result["answer"])
        self.assertFalse(gemini.parse_response({"candidates": []},
                         False, "gemini-test", 1)["response_usable"])


class GeminiAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_free_generation_uses_key_only_in_header(self):
        fake = FakeGoogle(PLAIN)
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true",
                                      "GEMINI_API_KEY": "test-secret",
                                      "GEMINI_MODEL": "gemini-3.5-flash-lite"}), \
             patch.object(gemini.httpx, "AsyncClient", return_value=fake):
            result = await gemini.generate("рецепт котлет", False)
        url, headers, payload = fake.request
        self.assertEqual(headers["x-goog-api-key"], "test-secret")
        self.assertNotIn("test-secret", str(payload))
        self.assertNotIn("tools", payload)
        self.assertEqual(result["route"], "gemini_general")
        self.assertIn("gemini-3.5-flash-lite:generateContent", url)

    async def test_sourced_answer_has_provenance_without_google_grounding(self):
        fake = FakeGoogle(PLAIN)
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true",
                                      "GEMINI_API_KEY": "test-secret",
                                      "GEMINI_MODEL": "gemini-3.5-flash-lite"}), \
             patch.object(gemini.httpx, "AsyncClient", return_value=fake):
            result = await gemini.generate("рецепт котлет", sources=EVIDENCE)
        self.assertEqual(result["route"], "gemini_searxng_evidence")
        self.assertEqual(result["prompt_trace"]["source_count"], 1)
        self.assertFalse(result["search_executed"])
        self.assertNotIn("tools", fake.request[2])
        self.assertNotIn("DO NOT SEND PRIVATE MEMORY", str(fake.request[2]))

    async def test_quota_error_is_not_disguised_as_bad_api_key(self):
        fake = FakeGoogle({"error": {"status": "RESOURCE_EXHAUSTED"}}, status=429)
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true",
                                      "GEMINI_API_KEY": "test-secret"}), \
             patch.object(gemini.httpx, "AsyncClient", return_value=fake):
            with self.assertRaises(HTTPException) as error:
                await gemini.generate("рецепт котлет")
        self.assertEqual(error.exception.status_code, 429)
        self.assertIn("квота", error.exception.detail)

    async def test_missing_key_must_not_call_google(self):
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true", "GEMINI_API_KEY": ""}):
            with self.assertRaises(HTTPException) as error:
                await gemini.generate("рецепт котлет")
        self.assertEqual(error.exception.status_code, 503)


class GatewayGeminiTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_search_ask_mode_requires_consent_not_google_bypass(self):
        calls = []
        async def forward(url, method="GET", payload=None):
            calls.append((url, payload))
            if url.endswith("/apprentice/context"):
                return {"normalized_text": "поищи в интернете рецепт котлет",
                        "utterance_intent": {"intent": "question"}, "dialogue": {}}
            if url.endswith("/research/operations/web/propose"):
                return {"query": payload["query"], "id": "approval"}
            raise AssertionError("unexpected unapproved external call: " + url)
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true"}), \
             patch.object(gateway, "forward", forward):
            result = await gateway.apprentice_chat(gateway.ChatIn(
                text="поищи в интернете рецепт котлет", experiment="apprentice",
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae",
                web_mode="ask"), time.perf_counter())
        self.assertEqual(result["route"], "apprentice_web_permission")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("/gemini-answer", str(calls))
        self.assertNotIn("/research/web/search", str(calls))

    async def test_auto_search_uses_searxng_then_free_gemini_without_tools(self):
        calls = []
        async def forward(url, method="GET", payload=None):
            calls.append((url, payload))
            if url.endswith("/apprentice/context"):
                return {"normalized_text": "поищи в интернете рецепт котлет",
                        "utterance_intent": {"intent": "question"}, "dialogue": {}}
            if url.endswith("/research/web/search"):
                return {"available": True, "status": "ok", "sources": EVIDENCE}
            if url.endswith("/gemini-answer"):
                self.assertFalse(payload["search"])
                self.assertEqual(payload["sources"], EVIDENCE)
                return dict(gemini.parse_response(PLAIN, False, "gemini-test", .3),
                            route="gemini_searxng_evidence")
            raise AssertionError(url)
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true"}), \
             patch.object(gateway, "forward", forward):
            result = await gateway.apprentice_chat(gateway.ChatIn(
                text="поищи в интернете рецепт котлет", experiment="apprentice",
                web_mode="auto"), time.perf_counter())
        self.assertEqual(result["route"], "gemini_searxng_evidence")
        self.assertTrue(result["search_executed"])
        self.assertEqual(result["search_status"], "ok")
        self.assertEqual(result["sources"][0]["url"], EVIDENCE[0]["url"])
        self.assertEqual(len(calls), 3)
        self.assertNotIn("/web-answer", str(calls))

    async def test_public_general_question_uses_gemini_even_when_web_off(self):
        calls = []
        async def forward(url, method="GET", payload=None):
            calls.append((url, payload))
            if url.endswith("/apprentice/context"):
                return {"normalized_text": "Что такое Python",
                        "utterance_intent": {"intent": "question"}, "dialogue": {}}
            if url.endswith("/gemini-answer"):
                self.assertEqual(payload, {"question": "Что такое Python",
                                           "search": False})
                return gemini.parse_response(PLAIN, False, "gemini-test", .3)
            raise AssertionError(url)
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true"}), \
             patch.object(gateway, "forward", forward):
            result = await gateway.apprentice_chat(gateway.ChatIn(
                text="Что такое Python", web_mode="off"), time.perf_counter())
        self.assertEqual(result["route"], "gemini_general")
        self.assertFalse(result["search_executed"])
        self.assertEqual(len(calls), 2)

    async def test_gemini_quota_error_falls_back_to_qwen_for_general_question(self):
        calls = []
        async def forward(url, method="GET", payload=None):
            calls.append(url)
            if url.endswith("/apprentice/context"):
                return {"normalized_text": "Почему небо синее",
                        "utterance_intent": {"intent": "question"}, "dialogue": {}}
            if url.endswith("/gemini-answer"):
                raise HTTPException(429, "Gemini: квота исчерпана (429)")
            if url.endswith("/grounded-chat"):
                return {"answer": "Рассеяние Рэлея.", "model": "qwen2.5:3b",
                        "route": "world_conversation"}
            raise AssertionError(url)
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true"}), \
             patch.object(gateway, "forward", forward):
            result = await gateway.apprentice_chat(gateway.ChatIn(
                text="Почему небо синее"), time.perf_counter())
        self.assertEqual(result["answer"], "Рассеяние Рэлея.")
        self.assertEqual(result["fallback_from"], "gemini")
        self.assertIn("квота", result["gemini_error"])
        self.assertEqual(len(calls), 3)

    async def test_gemini_quota_error_falls_back_to_qwen_for_real_search(self):
        calls = []
        async def forward(url, method="GET", payload=None):
            calls.append(url)
            if url.endswith("/research/web/search"):
                return {"available": True, "status": "ok", "sources": EVIDENCE}
            if url.endswith("/gemini-answer"):
                raise HTTPException(429, "Gemini: квота исчерпана (429)")
            if url.endswith("/web-answer"):
                return {"answer": "По источнику [1] фарш и лук.",
                        "route": "apprentice_web_evidence", "model": "qwen2.5:3b"}
            raise AssertionError(url)
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true"}), \
             patch.object(gateway, "forward", forward):
            result = await gateway.answer_from_public_search(
                gateway.ChatIn(text="рецепт котлет"), "рецепт котлет", {},
                "рецепт котлет", time.perf_counter())
        self.assertEqual(result["route"], "apprentice_web_evidence")
        self.assertEqual(result["fallback_from"], "gemini")
        self.assertEqual(result["search_status"], "ok")
        self.assertEqual(len(calls), 3)

    async def test_private_queries_never_leave_gateway(self):
        calls = []
        async def forward(url, method="GET", payload=None):
            calls.append(url)
            if url.endswith("/apprentice/context"):
                return {"normalized_text": "поищи в интернете заряд моего Наскулила",
                        "utterance_intent": {"intent": "command"}, "dialogue": {}}
            raise AssertionError("Sent private data to cloud/search: " + url)
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true"}), \
             patch.object(gateway, "forward", forward):
            result = await gateway.apprentice_chat(gateway.ChatIn(
                text="поищи в интернете заряд моего Наскулила",
                experiment="apprentice", web_mode="ask"), time.perf_counter())
        self.assertEqual(result["route"], "gemini_private_data_blocked")
        self.assertEqual(len(calls), 1)

    def test_cloud_boundary(self):
        self.assertTrue(public_only("рецепт котлет"))
        self.assertTrue(public_only("дай мне рецепт котлет"))
        self.assertTrue(public_information_request("Что такое Python"))
        self.assertTrue(public_information_request("Почему небо синее"))
        self.assertFalse(public_information_request("Привет, Джарвис"))
        for phrase in ("где мой телефон", "покажи 192.168.1.22",
                       "моя почта alice@example.org",
                       "поищи мой пароль", "какая температура у Наскулила",
                       "Я вчера сломал телефон", "Что мы обсуждали"):
            self.assertFalse(public_only(phrase), phrase)
        self.assertTrue(recipe_request("дай мне рецепт котлет"))
        self.assertFalse(recipe_request("напомни, как я готовил котлеты"))


if __name__ == "__main__":
    unittest.main()
