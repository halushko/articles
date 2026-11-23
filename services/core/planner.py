"""Validate untrusted LLM plans against real server-advertised capabilities.

This module has no natural-language phrase routing. All hard conditions are
authorization/structural constraints, not interpretations of human sentences.
"""
import json
import re

# The first slice is deliberately READ/PROPOSE only. This is a contract
# allowlist, not a keyword table and not an executor registry.
CONTRACTS = {
    "dialogue.respond": {
        "description": "Compose a conversational answer using the supplied context.",
        "side_effects": False, "approval": False, "executor": "llm.grounded_chat"
    },
    "knowledge.lookup": {
        "description": "Read previously acquired knowledge, without writing.",
        "side_effects": False, "approval": False, "executor": "research.context"
    },
    "knowledge.propose": {
        "description": "Propose new knowledge for separate teacher review; do not confirm it.",
        "side_effects": True, "approval": True, "executor": "research.propose"
    },
    "web.search": {
        "description": "Request a public web search using the existing search approval policy.",
        "side_effects": True, "approval": True, "executor": "search.search"
    },
}
INTENTS = {"question", "teaching", "action", "research", "recall", "mixed", "unclear"}
IDENTIFIER = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,31}$")


def catalog(runtime_capabilities, web_mode):
    """Only expose routes verified by Research, not LLM-invented tools."""
    advertised = {
        item.get("name"): item for item in runtime_capabilities
        if isinstance(item, dict) and item.get("status") == "available"
    }
    result = {}
    for name, contract in CONTRACTS.items():
        if name == "dialogue.respond":
            result[name] = {"name": name, **contract}
            continue
        runtime = advertised.get(name)
        if not runtime or runtime.get("executor") != contract["executor"]:
            continue
        if name == "web.search" and web_mode == "off":
            continue
        result[name] = {"name": name, **contract}
        if name == "web.search":
            # Network permission is scoped to an actual operation; "auto" is
            # a user policy setting, not a grant from the model.
            result[name]["approval"] = web_mode != "auto"
    return result


def validate(raw, available):
    """Fail closed: an invalid step invalidates the entire proposed plan.

    This does NOT authorize a valid plan. All steps remain unexecuted, and
    the executor must independently check actor, policy and parameters.
    """
    base = {"executed": False, "execution_allowed": False,
            "steps": [], "model_suggestion_only": True}
    if not isinstance(raw, dict):
        return {**base, "status": "invalid", "reason": "Planner did not return a JSON object"}
    intent = raw.get("intent")
    if intent not in INTENTS:
        return {**base, "status": "invalid", "reason": "Unknown intent category"}
    steps = raw.get("steps")
    if not isinstance(steps, list) or len(steps) > 5:
        return {**base, "status": "invalid", "reason": "Invalid step list"}
    result = []
    seen = set()
    for step in steps:
        if not isinstance(step, dict):
            return {**base, "status": "invalid", "reason": "Non-object step"}
        sid = step.get("id")
        cap = step.get("capability")
        args = step.get("arguments")
        deps = step.get("depends_on")
        if not isinstance(sid, str) or not IDENTIFIER.fullmatch(sid) or sid in seen:
            return {**base, "status": "invalid", "reason": "Duplicate or invalid step identifier"}
        if not isinstance(cap, str) or cap not in available:
            return {**base, "status": "unavailable",
                    "reason": "Запрошенная возможность недоступна: " + str(cap)[:80],
                    "missing_capability": str(cap)[:80], "intent": intent}
        if not isinstance(args, dict) or not isinstance(deps, list):
            return {**base, "status": "invalid", "reason": "Invalid step arguments or dependencies"}
        if any(not isinstance(dep, str) or dep not in seen for dep in deps):
            return {**base, "status": "invalid", "reason": "Invalid or forward dependency"}
        if len(json.dumps(args, ensure_ascii=False, default=str)) > 2000:
            return {**base, "status": "invalid", "reason": "Oversized tool arguments"}
        contract = available[cap]
        result.append({
            "id": sid, "capability": cap, "arguments": args, "depends_on": deps,
            "requires_approval": contract["approval"],
            "side_effects": contract["side_effects"],
            "executor": contract["executor"], "status": "proposed"
        })
        seen.add(sid)
    clarification = raw.get("clarifying_question")
    needs_clarification = raw.get("needs_clarification")
    if not isinstance(needs_clarification, bool) or not isinstance(clarification, str):
        return {**base, "status": "invalid", "reason": "Invalid clarification fields"}
    if needs_clarification and not clarification.strip():
        return {**base, "status": "invalid", "reason": "Clarification requested without question"}
    if not result and not needs_clarification:
        return {**base, "status": "invalid", "reason": "Empty actionable plan"}
    return {
        **base, "status": "needs_clarification" if needs_clarification else "proposed",
        "intent": intent, "steps": result,
        "clarifying_question": clarification.strip()[:400] if needs_clarification else "",
        "explanation": str(raw.get("explanation") or "")[:600],
    }
