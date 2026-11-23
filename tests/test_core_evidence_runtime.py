"""Real Core contract tests with fake executors, not hand-coded intent words."""
import unittest

from services.core.evidence import cited_synthesis, query_eligible
from services.core.runtime import research_answer, run_plan

QUERY = "Найди рецепт котлет"
PAGE = "Котлеты готовят из фарша, лука и хлеба. " * 6
SRC = {"title": "Тестовый источник", "url": "https://example.org/recipe",
       "page_text": PAGE, "status": "read", "snippet": "Котлеты из фарша"}


def plan(*steps, intent="research", status="proposed"):
    return {"intent": intent, "steps": list(steps), "status": status}


def step(name, capability, **arguments):
    return {"id": name, "capability": capability, "arguments": arguments,
            "depends_on": []}


class EvidencePolicyTests(unittest.TestCase):
    def test_cotlets_honey_does_not_get_searched(self):
        ok, reason = query_eligible(QUERY, "рецепт котлет с мёдом")
        self.assertFalse(ok)
        self.assertEqual(reason, "ungrounded_query_expansion")
        self.assertTrue(query_eligible(QUERY, "рецепт котлет")[0])

    def test_contextual_followup_can_keep_previous_goal(self):
        self.assertTrue(query_eligible(
            "А теперь чебурека", "рецепт чебурека",
            {"subject": "рецепт омлета"})[0])

    def test_private_and_new_topic_are_blocked(self):
        self.assertFalse(query_eligible("Проверь Наскулила",
                                        "Наскулил", private_names=["Наскулил"])[0])
        self.assertFalse(query_eligible("Поищи", "у меня дома")[0])
        self.assertFalse(query_eligible("Найди котлеты",
                                        "почему котлеты с мёдом")[0])

    def test_fabricated_citations_and_urls_rejected(self):
        self.assertTrue(cited_synthesis("Котлеты готовят из фарша [1]", [SRC]))
        self.assertFalse(cited_synthesis("Котлеты с мёдом [1]", [SRC]))
        self.assertFalse(cited_synthesis("Факт [2]", [SRC]))
        self.assertFalse(cited_synthesis(
            "Факт [1] https://wrong.example/recipe", [SRC]))


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.req = {"text": QUERY, "actor_id": "actor-1",
                    "session_id": "sess-1", "web_mode": "auto"}
        self.ctx = {"knowledge": [], "task": {}, "private_names": []}
        self.calls = []

    async def fake_call(self, service, path, payload):
        self.calls.append((service, path, payload))
        if service == "search":
            return {"status": "ok", "available": True,
                    "sources": [SRC], "executed": True}
        if service == "gemini":
            return {"answer": "Котлеты готовят из фарша. [1]",
                    "response_usable": True, "model": "Gemini"}
        if path.endswith("/propose"):
            return {"id": "op1", "query": payload["query"], "status": "pending"}
        if path.endswith("/start"):
            return {"started": True}
        if path.endswith("/finish"):
            return {"saved": True}
        raise AssertionError("Unexpected internal tool: " + service + path)

    async def test_research_then_synthesis_order(self):
        p = plan(step("memory", "knowledge.lookup"),
                 step("web", "web.search", query="рецепт котлет"),
                 step("response", "dialogue.respond"))
        out = await run_plan(self.req, self.ctx, p, self.fake_call)
        self.assertEqual(out["route"], "gemini_searxng_evidence")
        self.assertEqual([v[0] for v in self.calls], ["search", "gemini"])
        self.assertTrue(out["search_executed"])

    async def test_ask_mode_never_searches_without_approval(self):
        self.req["web_mode"] = "ask"
        p = plan(step("find", "web.search", query="рецепт котлет"))
        out = await run_plan(self.req, self.ctx, p, self.fake_call)
        self.assertEqual(out["route"], "core_search_approval_needed")
        self.assertEqual([v[0] for v in self.calls], ["research"])
        self.assertFalse(out["search_executed"])

    async def test_invalid_expansion_blocks_tool(self):
        p = plan(step("web", "web.search", query="рецепт котлет с мёдом"))
        out = await run_plan(self.req, self.ctx, p, self.fake_call)
        self.assertEqual(out["route"], "core_query_clarification")
        self.assertEqual(self.calls, [])

    async def test_no_web_without_evidence_when_disabled(self):
        self.req["web_mode"] = "off"
        p = plan(step("reply", "dialogue.respond"), intent="question")
        out = await run_plan(self.req, self.ctx, p, self.fake_call)
        self.assertEqual(out["route"], "core_research_disabled")
        self.assertEqual(self.calls, [])

    async def test_missing_executor_aborts_entire_plan(self):
        p = {"status": "unavailable", "missing_capability": "ha.turn_on",
             "intent": "mixed", "steps": []}
        out = await run_plan(self.req, self.ctx, p, self.fake_call)
        self.assertEqual(out["route"], "core_capability_unavailable")
        self.assertEqual(self.calls, [])

    async def test_memory_is_verbatim_and_no_model_answer(self):
        self.ctx["knowledge"] = [{
            "type": "fact", "subject": "Наскулил",
            "predicate": "модель", "object": "EcoFlow DELTA Pro", "verified": True}]
        p = plan(step("memory", "knowledge.lookup"), intent="recall")
        out = await run_plan(self.req, self.ctx, p, self.fake_call)
        self.assertEqual(out["route"], "core_confirmed_memory")
        self.assertIn("EcoFlow DELTA Pro", out["answer"])
        self.assertEqual(self.calls, [])

    async def test_no_pages_means_no_gemini(self):
        async def only_snippets(service, path, payload):
            self.calls.append((service, path, payload))
            if service == "search":
                return {"status": "ok", "available": True,
                        "sources": [{**SRC, "status": "search_snippet",
                                     "page_text": ""}], "executed": True}
            raise AssertionError("No ungrounded synthesis permitted")
        result = await research_answer(self.req, self.ctx, "рецепт котлет",
                                       only_snippets)
        self.assertEqual(result["route"], "core_search_evidence_excerpt")
        self.assertEqual(len(self.calls), 1)

    async def test_approved_search_is_claimed_once_and_finished(self):
        a = {"operation_id": "op1", "query": "рецепт котлет",
             "question": QUERY}
        result = await research_answer(self.req, self.ctx, a["query"],
                                       self.fake_call, approval=a)
        paths = [x[1] for x in self.calls]
        self.assertEqual(paths[0], "/research/operations/web/start")
        self.assertEqual(paths[-1], "/research/operations/web/finish")
        self.assertTrue(result["operation_state"]["saved"])
        self.assertEqual(result["approval_id"], "op1")

    async def test_failed_approval_claim_does_not_search(self):
        async def denied(service, path, payload):
            self.calls.append((service, path, payload))
            return {"started": False, "status": "succeeded"}
        a = {"operation_id": "op1", "query": "рецепт котлет",
             "question": QUERY}
        out = await research_answer(self.req, self.ctx, a["query"], denied, approval=a)
        self.assertEqual(out["route"], "core_search_already_processed")
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
