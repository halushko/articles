"""Bounded, privacy-aware diagnostics for every gateway chat route.

The export contains control flow and prompt *metadata*, never raw system prompts,
credentials, external page bodies, or full model payloads. Conversation text is
already saved separately in the journal for the owner's explicit export.
"""
import time
from contextvars import ContextVar
from urllib.parse import urlsplit

CURRENT_TRACE = ContextVar("jarvis_chat_trace", default=None)
MAX_STEPS = 80


def _count(value):
    return len(value) if isinstance(value, (dict, list)) else 0


def _llm_summary(data):
    trace = data.get("prompt_trace")
    return trace if isinstance(trace, dict) else None


def input_summary(endpoint, payload):
    """Allowlisted telemetry only: never record arbitrary request values."""
    p = payload if isinstance(payload, dict) else {}
    summary = {}
    if "profile" in p:
        summary["profile"] = str(p["profile"])[:20]
    if "ablation" in p:
        summary["ablation"] = str(p["ablation"])[:20]
    if "actor_id" in p:
        summary["actor_selected"] = bool(p["actor_id"])
    if "session_id" in p:
        summary["session_supplied"] = bool(p["session_id"])
    if "history" in p:
        summary["history_supplied"] = _count(p["history"])
    if "memory" in p:
        memory = p.get("memory") if isinstance(p["memory"], dict) else {}
        summary["memory_keys"] = sorted(memory.keys())[:25]
        dialogue = memory.get("dialogue") if isinstance(memory.get("dialogue"), dict) else {}
        summary["trusted_history_supplied"] = _count(dialogue.get("history"))
    if "query" in p or "question" in p:
        summary["query_chars"] = len(str(p.get("query") or p.get("question") or ""))
    if endpoint == "/chat":
        summary["baseline_no_memory_by_design"] = True
    if endpoint == "/web-answer":
        summary["source_count"] = _count(p.get("sources"))
        summary["origin"] = str(p.get("origin") or "")[:32]
    if endpoint in ("/research/operations/web/resolve", "/research/dialogue/resolve"):
        summary["button_approval"] = bool(p.get("approval_id"))
    return summary


def output_summary(endpoint, data):
    if not isinstance(data, dict):
        return {"response_type": type(data).__name__}
    out = {}
    for key in ("route", "status", "search_status", "source", "evidence_source",
                "model", "intent", "handled", "matched", "is_command", "executed",
                "available", "needs_web", "web_requested", "normalized_text",
                "confidence", "semantic_status", "score", "threshold"):
        if key in data and isinstance(data[key], (str, bool, int, float)):
            if key == "normalized_text":
                out["normalization_applied"] = True
            else:
                out[key] = data[key] if not isinstance(data[key], str) else data[key][:80]
    if endpoint == "/apprentice/context":
        out["direct_reply_found"] = bool(data.get("direct_reply"))
        out["identity_roles"] = {
            k: str(v)[:80] for k, v in (data.get("identity") or {}).items()
            if k in ("user", "assistant") and isinstance(v, str)
        }
        task = data.get("task_context") or {}
        if isinstance(task, dict) and task:
            out["task_resolution"] = {
                "status": str(task.get("status") or "")[:40],
                "explicit_search": bool(task.get("explicit_search")),
                "context_used": bool(task.get("context_used")),
                "subject_chars": len(str(task.get("subject") or "")),
                "query_chars": len(str(task.get("search_query") or "")),
                "history_turns_examined": int(task.get("history_turns_examined") or 0),
                "source": str(task.get("source") or "")[:60]
            }
        out["knowledge_items"] = _count(data.get("knowledge"))
        out["graph_items"] = _count(data.get("graph"))
        out["feedback_items"] = _count(data.get("teacher_feedback"))
        out["episodic_user_turns"] = _count(data.get("experiences"))
        out["capability_count"] = _count(data.get("capabilities"))
        out["operation_blocked"] = bool(data.get("unavailable_operation"))
        out["utterance_intent"] = (data.get("utterance_intent") or {}).get("intent")
        dialogue = data.get("dialogue") or {}
        if isinstance(dialogue, dict):
            out["persisted_history_turns"] = _count(dialogue.get("history"))
            last=dialogue.get("last_response_evidence") or {}
            if isinstance(last,dict):
                out["last_answer_source_kind"] = str(last.get("source_kind") or "")[:50]
                out["last_answer_route"] = str(last.get("route") or "")[:80]
            state = dialogue.get("state") or {}
            if isinstance(state, dict):
                out["dialogue_state"] = {
                    k: str(state[k])[:140] for k in ("topic", "goal", "assistant_question")
                    if state.get(k)
                }
    if endpoint == "/memory":
        out["index_counts"] = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    if endpoint == "/resolve":
        out["index_status"] = data.get("status")
        out["target_found"] = bool(data.get("target_id"))
        out["candidate_count"] = _count(data.get("candidates"))
        out["semantic_candidate_count"] = _count(data.get("semantic_candidates"))
        out["selected_target"] = str(data.get("target_id") or "")[:100]
        out["index_route"] = str(data.get("route") or "")[:60]
    if endpoint == "/research/dialogue/resolve":
        out["operation_category"] = data.get("category")
    if endpoint == "/research/web/search":
        out["source_count"] = _count(data.get("sources"))
    if endpoint in ("/chat", "/analyze", "/web-answer"):
        out["prompt_trace"] = _llm_summary(data)
        out["model_intent"] = data.get("intent") if endpoint == "/analyze" else None
        out["model_parse_warning"] = bool(data.get("parse_warning"))
        if endpoint == "/analyze":
            out["model_proposed_web_query"] = bool(data.get("web_query"))
            out["model_proposals_count"] = _count(data.get("proposals"))
            out["model_changes_count"] = _count(data.get("changes"))
            frame=data.get("dialogue_frame") or {}
            if isinstance(frame,dict):
                out["model_dialogue_frame"] = {
                    k: str(frame[k])[:120] for k in ("topic","goal","assistant_question")
                    if frame.get(k)
                }
    return out


class ChatTrace:
    def __init__(self, data):
        self.started = time.perf_counter()
        self.request_id = data.request_id
        self.mode = data.experiment
        self.profile = data.profile
        self.ablation = data.ablation
        self.web_mode = data.web_mode
        self.session_id = data.session_id or ""
        self.actor_id = data.actor_id or ""
        self.steps = []
        self.add("gateway.received", selected_mode=self.mode,
                 profile=self.profile, ablation=self.ablation,
                 web_mode=self.web_mode, session_present=bool(self.session_id),
                 actor_selected=bool(self.actor_id),
                 client_history_turns=_count(data.history),
                 note=("llm_only is an isolated baseline" if self.mode == "llm_only"
                       else "legacy index then LLM" if self.mode == "hybrid"
                       else "index only" if self.mode == "research_only"
                       else "apprentice with dialogue state"))

    def add(self, stage, status="ok", **details):
        if len(self.steps) >= MAX_STEPS:
            return
        self.steps.append({
            "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 2),
            "stage": stage,
            "status": status,
            "details": details
        })

    def result(self, response=None, error=None):
        output = response if isinstance(response, dict) else {}
        return {
            "schema": "jarvis.debug.v1",
            "request_id": self.request_id,
            "selected_mode": self.mode,
            "profile": self.profile,
            "ablation": self.ablation,
            "web_mode": self.web_mode,
            "actor_id": self.actor_id,
            "session_id": self.session_id,
            "steps": self.steps,
            "outcome": {
                "route": output.get("route", "chat_error" if error else "unknown"),
                "model": output.get("model", ""),
                "tokens": output.get("tokens", 0),
                "prompt_tokens": output.get("prompt_tokens", 0),
                "elapsed_ms": round((time.perf_counter()-self.started)*1000, 2),
                "error": error or ""
            }
        }


def trace_forward_endpoint(url):
    """Stable service+path; do not expose arbitrary URLs/hostnames."""
    p = urlsplit(url)
    path = p.path or "/"
    if p.hostname == urlsplit(__import__("os").getenv("RESEARCH_URL", "http://research:8000")).hostname:
        return "research", path
    if p.hostname == urlsplit(__import__("os").getenv("LLM_URL", "http://llm:8000")).hostname:
        return "llm", path
    return "unknown", "<redacted>"
