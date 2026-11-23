"""Core API orchestration tests; mock service calls, not Qwen's semantics."""
import unittest
from unittest.mock import patch

from services.core import api


class CoreAPIFlow(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = []
        self.session = "123e4567-e89b-12d3-a456-426614174000"
        self.actor = "123e4567-e89b-12d3-a456-426614174111"
        self.web_plan = {
            "intent": "research", "needs_clarification": False,
            "clarifying_question": "", "explanation": "Research required",
            "steps": [{
                "id": "find", "capability": "web.search",
                "arguments": {"query": "рецепт котлет"},
                "depends_on": []
            }]
        }

    async def fake(self, service, path, payload):
        self.calls.append((service, path, payload))
        if path == "/research/operations/web/resolve":
            return {"handled": False}
        if path == "/research/planner/context":
            return {"task": {}, "history": [], "knowledge": [],
                    "private_names": [], "experiences": [],
                    "capabilities": [{
                        "name": "web.search", "status": "available",
                        "executor": "search.search"
                    }]}
        if path == "/plan":
            return {"plan": self.web_plan, "model": "qwen2.5:3b"}
        if path == "/research/operations/web/propose":
            return {"id": self.session, "query": payload["query"],
                    "question": payload["question"], "status": "pending"}
        raise AssertionError("Unexpected call: " + service + path)

    async def test_full_new_turn_stops_at_permission(self):
        request = api.ExecuteIn(text="Найди рецепт котлет", session_id=self.session,
                                actor_id=self.actor, web_mode="ask")
        with patch.object(api, "call", self.fake):
            outcome = await api.execute(request)
        self.assertEqual(outcome["route"], "core_search_approval_needed")
        self.assertFalse(outcome["search_executed"])
        self.assertEqual([name for name,_,_ in self.calls],
                         ["research", "research", "llm", "research"])
        self.assertEqual(self.calls[-1][2]["query"], "рецепт котлет")

    async def test_preview_has_no_execution(self):
        request = api.PlanIn(text="Найди рецепт котлет", session_id=self.session,
                             actor_id=self.actor, web_mode="ask")
        with patch.object(api, "call", self.fake):
            outcome = await api.propose_plan(request)
        self.assertEqual(outcome["status"], "proposed")
        self.assertFalse(outcome["execution_allowed"])
        self.assertEqual([name for name,_,_ in self.calls], ["research", "llm"])

    async def test_approval_uses_original_query_without_replanning(self):
        async def approved(service, path, payload):
            self.calls.append((service, path, payload))
            if path == "/research/operations/web/resolve":
                return {"handled": True, "status": "approved",
                        "operation_id": "op123", "query": "рецепт котлет",
                        "question": "Найди рецепт котлет"}
            if path == "/research/planner/context":
                return {"task": {}, "knowledge": [], "private_names": []}
            if path.endswith("/start"):
                return {"started": True}
            if path == "/v1/search":
                return {"available": False, "status": "no_results",
                        "reason": "nothing", "sources": [], "executed": True}
            if path.endswith("/finish"):
                return {"saved": True}
            raise AssertionError("Unexpected call: " + service + path)
        request = api.ExecuteIn(text="Разрешаю поиск", session_id=self.session,
                                actor_id=self.actor, web_mode="ask",
                                approval_id="op123", allow_web_search=True)
        with patch.object(api, "call", approved):
            outcome = await api.execute(request)
        self.assertEqual(outcome["route"], "core_search_unavailable")
        self.assertEqual([name for name,_,_ in self.calls],
                         ["research", "research", "research", "search", "research"])
        self.assertEqual(self.calls[3][2]["query"], "рецепт котлет")
        self.assertEqual(self.calls[0][2]["source"], "button")
        self.assertNotIn("llm", [name for name,_,_ in self.calls])


if __name__ == "__main__":
    unittest.main()
