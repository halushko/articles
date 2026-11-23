"""Gemini fallback: public-only payload, grounding verification, no side effects.

All tests use synthetic Gemini responses: a passing test is not proof that
the configured Google key works on the deployed server.
"""
import os
import time
import unittest
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from services.llm import gemini
from services.gateway import api as gateway
from services.gateway.cloud import public_only, recipe_request


class FakeGoogle:
    def __init__(self, response):
        self.response = response
        self.request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def post(self, url, headers, json):
        self.request = (url, headers, json)
        return httpx.Response(200, json=self.response,
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


class GeminiUnitTests(unittest.TestCase):
    def test_search_tool_is_explicit_and_never_receives_memory(self):
        request = gemini.request_payload("рецепт котлет", True)
        self.assertEqual(request["tools"], [{"google_search": {}}])
        self.assertEqual(request["contents"][0]["parts"][0]["text"], "рецепт котлет")
        for sensitive in ("memory", "actor_id", "session_id", "history", "HA_TOKEN"):
            self.assertNotIn(sensitive, str(request))

    def test_grounding_source_is_distinct_from_model_claims(self):
        result = gemini.parse_response(SAMPLE, True, "gemini-2.5-flash-lite", 1.0)
        self.assertEqual(result["route"], "gemini_google_grounded")
        self.assertTrue(result["search_verified"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["url"], "https://example.org/kotlety")

    def test_no_grounding_means_no_verified_search_answer(self):
        result = gemini.parse_response({"candidates": [{
            "content": {"parts": [{"text": "В интернете я нашёл котлеты с мёдом!"}]},
            "finishReason": "STOP"}]}, True, "gemini-test", 1)
        self.assertEqual(result["route"], "gemini_search_unverified")
        self.assertFalse(result["search_verified"])
        self.assertNotIn("мёдом", result["answer"])

    def test_truncated_response_marked_incomplete(self):
        bad = {"candidates": [{
            "content": {"parts": [{"text": "Первый шаг"}]},
            "finishReason": "MAX_TOKENS",
            "groundingMetadata": {"groundingChunks": [
                {"web": {"uri": "https://example.org/recipe"}}]},
        }]}
        result = gemini.parse_response(bad, True, "gemini-test", 1)
        self.assertFalse(result["response_complete"])
        self.assertIn("неполный", result["answer"])


class GeminiAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_google_called_without_memory_or_key_in_body(self):
        fake = FakeGoogle(SAMPLE)
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true",
                                      "GEMINI_API_KEY": "test-secret",
                                      "GEMINI_MODEL": "gemini-2.5-flash-lite"}), \
             patch.object(gemini.httpx, "AsyncClient", return_value=fake):
            result = await gemini.generate("рецепт котлет", True)
        url, headers, payload = fake.request
        self.assertEqual(headers["x-goog-api-key"], "test-secret")
        self.assertNotIn("test-secret", str(payload))
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "рецепт котлет")
        self.assertTrue(result["search_verified"])
        self.assertIn("gemini-2.5-flash-lite:generateContent", url)

    async def test_missing_key_must_not_call_google(self):
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true", "GEMINI_API_KEY": ""}):
            with self.assertRaises(HTTPException) as exc:
                await gemini.generate("рецепт котлет", True)
        self.assertEqual(exc.exception.status_code, 503)


class GatewayGeminiTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_auto_even_if_old_web_setting_asks(self):
        calls = []
        async def forward(url, method="GET", payload=None):
            calls.append((url, payload))
            if url.endswith("/apprentice/context"):
                return {"normalized_text": "поищи в интернете рецепт котлет",
                        "utterance_intent": {"intent": "question"},
                        "dialogue": {}}
            if url.endswith("/gemini-answer"):
                self.assertEqual(payload, {"question": "рецепт котлет", "search": True})
                return dict(gemini.parse_response(SAMPLE, True, "gemini-test", 0.3))
            raise AssertionError(url)
        with patch.dict(os.environ, {"GEMINI_ENABLED": "true"}), \
             patch.object(gateway, "forward", forward):
            result = await gateway.apprentice_chat(gateway.ChatIn(
                text="поищи в интернете рецепт котлет", experiment="apprentice",
                web_mode="ask"), time.perf_counter())
        self.assertEqual(result["route"], "gemini_google_grounded")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("/analyze", str(calls))
        self.assertNotIn("/research/web/search", str(calls))

    async def test_private_queries_never_leave_gateway(self):
        calls = []
        async def forward(url, method="GET", payload=None):
            calls.append(url)
            if url.endswith("/apprentice/context"):
                return {"normalized_text": "поищи в интернете заряд моего Наскулила",
                        "utterance_intent": {"intent": "command"},
                        "dialogue": {}}
            raise AssertionError("Sent private data to cloud or other service: "+url)
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
        for phrase in ("где мой телефон", "покажи 192.168.1.22",
                       "моя почта alice@example.org",
                       "поищи мой пароль", "какая температура у Наскулила"):
            self.assertFalse(public_only(phrase), phrase)
        self.assertTrue(recipe_request("дай мне рецепт котлет"))
        self.assertFalse(recipe_request("напомни, как я готовил котлеты"))
