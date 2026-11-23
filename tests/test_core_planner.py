"""Decision-planning contract tests; no Ollama, DB or internet required."""
import unittest

from services.core.planner import catalog, validate


CAPS = [
    {"name": "knowledge.lookup", "status": "available", "executor": "research.context"},
    {"name": "knowledge.propose", "status": "available", "executor": "research.propose"},
    {"name": "web.search", "status": "available", "executor": "search.search"},
    {"name": "external.action", "status": "unavailable", "executor": None},
]


class PlannerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.available = catalog(CAPS, "ask")

    def plan(self, steps, **kwargs):
        return {"intent": kwargs.get("intent", "mixed"), "steps": steps,
                "needs_clarification": kwargs.get("needs_clarification", False),
                "clarifying_question": kwargs.get("clarifying_question", ""),
                "explanation": "Proposed plan"}

    def step(self, ident, capability, depends_on=None, arguments=None):
        return {"id": ident, "capability": capability,
                "depends_on": depends_on or [],
                "arguments": arguments or {}}

    def test_dynamic_registry_rejects_unknown_and_unavailable_executors(self):
        self.assertNotIn("external.action", self.available)
        self.assertNotIn("home_assistant.set_state", self.available)
        self.assertNotIn("web.search", catalog(CAPS, "off"))
        altered = [{"name": "web.search", "status": "available",
                    "executor": "attacker.remote"}]
        self.assertNotIn("web.search", catalog(altered, "ask"))

    def test_composite_plan_and_derived_approval(self):
        raw = self.plan([
            self.step("retrieve", "knowledge.lookup", arguments={"entity": "Наскулил"}),
            self.step("research", "web.search", ["retrieve"],
                      {"query": "EcoFlow DELTA Pro specs"}),
            self.step("reply", "dialogue.respond", ["research"])
        ])
        result = validate(raw, self.available)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual([x["id"] for x in result["steps"]],
                         ["retrieve", "research", "reply"])
        self.assertTrue(result["steps"][1]["requires_approval"])
        self.assertFalse(result["steps"][0]["requires_approval"])
        self.assertFalse(result["execution_allowed"])
        self.assertFalse(result["executed"])

    def test_cannot_invent_tool_or_bypass_policy(self):
        raw = self.plan([
            self.step("first", "knowledge.lookup"),
            self.step("second", "home_assistant.turn_on", ["first"])
        ])
        result = validate(raw, self.available)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["steps"], [])
        # "auto" is a user setting, not permission to run any unknown action.
        self.assertEqual(validate(raw, catalog(CAPS, "auto"))["steps"], [])

    def test_failed_dependencies_fail_entire_plan(self):
        for deps in (["missing"], ["two"], [123]):
            with self.subTest(deps=deps):
                self.assertEqual(validate(self.plan([
                    self.step("one", "knowledge.lookup", depends_on=deps)
                ]), self.available)["status"], "invalid")
        self.assertEqual(validate(self.plan([
            self.step("one", "knowledge.lookup"),
            self.step("one", "dialogue.respond")
        ]), self.available)["status"], "invalid")

    def test_clarification_is_not_execution(self):
        raw = self.plan([], intent="unclear", needs_clarification=True,
                        clarifying_question="Какой инвертор ты имеешь в виду?")
        result = validate(raw, self.available)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["steps"], [])
        self.assertFalse(result["execution_allowed"])

    def test_invalid_or_empty_output_abstains(self):
        self.assertEqual(validate(None, self.available)["status"], "invalid")
        self.assertEqual(validate(self.plan([]), self.available)["status"], "invalid")
        self.assertEqual(validate(self.plan([self.step("x", "knowledge.lookup")] * 6),
                                  self.available)["status"], "invalid")

    def test_web_policy_from_user_setting_not_model(self):
        raw = self.plan([self.step("s", "web.search", arguments={"query": "sun"})])
        self.assertEqual(validate(raw, catalog(CAPS, "off"))["status"], "unavailable")
        self.assertTrue(validate(raw, catalog(CAPS, "ask"))["steps"][0]["requires_approval"])
        self.assertFalse(validate(raw, catalog(CAPS, "auto"))["steps"][0]["requires_approval"])


if __name__ == "__main__":
    unittest.main()
