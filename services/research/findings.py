"""Persistent, source-attributed research findings used BEFORE cloud LLMs.

This is NOT confirmed world knowledge: every reused answer retains provenance.
No model answer without a successful real public search can enter this cache.
Speaker profiles are selector hints, not authenticated identity; only public
research is cached, never personal or HA state.
"""
import json
import re
from urllib.parse import urlsplit

from .normalize import normalize
from .web_research import public_https

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_findings (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 actor_id TEXT NOT NULL DEFAULT '',
 session_id TEXT NOT NULL,
 search_query TEXT NOT NULL,
 question TEXT NOT NULL,
 answer TEXT NOT NULL,
 sources TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_research_findings_actor
 ON research_findings(actor_id,id);
CREATE INDEX IF NOT EXISTS idx_research_findings_session
 ON research_findings(session_id,id);
"""

# This is a retrieval index, NOT a whitelist of allowed topics or actions.
# Common request words add no topical evidence.
COMMON = frozenset(
    "дай дайте мне мені покажи покажите расскажи расскажите подскажи "
    "найди найдите поищи поищите ищи искать интернет интернете сети "
    "веб web search please find give show tell explain "
    "могу можно нужно хочу как где что какой какая какие "
    "тот тотже такой это этот этих ранее прошлый предыдущий "
    "рецепт рецепты рецепта рецепту рецептом "
    "документация документацию документ документы "
    "the a an and or is are for in on of to me"
    .split()
)
# User explicitly wants new research, not a replay of a cached answer.
FRESH = re.compile(
    r"(?iu)\b(?:нов(?:ый|ую|ые|ого|ее)|друг(?:ой|ие|ую|ого)|"
    r"свеж\w*|актуальн\w*|сегодн\w*|сейчас|"
    r"обнови\w*|перепроверь\w*|проверь\w*|"
    r"последн\w*|альтернатив\w*|еще|ещё|another|latest|current|update)\b"
)


def words(text):
    """Small multilingual stem normalization. Ambiguity => cache miss."""
    result = set()
    for word in normalize(text).split():
        if len(word) < 3 or word in COMMON:
            continue
        if word.isdigit():
            result.add(word)
            continue
        stem = word
        if len(word) >= 6:
            for ending in ("ами", "ями", "ого", "ему", "ыми", "ими", "иях",
                           "ах", "ях", "ов", "ев", "ам", "ям",
                           "ой", "ей", "ый", "ий", "ое", "ая",
                           "ою", "ую", "ы", "и", "а", "я", "у", "е"):
                if word.endswith(ending) and len(word) - len(ending) >= 4:
                    stem = word[:-len(ending)]
                    break
        result.add(stem)
    return result


def safe_sources(raw):
    """Only previously observed real public search results, bounded for storage."""
    clean = []
    if not isinstance(raw, list):
        return clean
    for item in raw[:5]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        if not public_https(url):
            continue
        clean.append({
            "title": str(item.get("title") or urlsplit(url).hostname)[:180],
            "url": url[:1600],
            "status": str(item.get("status") or "search_snippet")[:40],
            "snippet": str(item.get("snippet") or "")[:500],
        })
    return clean


class ResearchFindings:
    def __init__(self, engine):
        self.engine = engine
        with engine.connect() as con:
            con.executescript(SCHEMA)

    def remember(self, session_id, actor_id, question, answer, result):
        """Save ONLY completed real-search responses; no invented LLM 'facts'."""
        if not isinstance(result, dict):
            return False
        if (result.get("route") not in ("apprentice_web_evidence",
                                        "gemini_searxng_evidence")
                or result.get("search_status") != "ok"
                or not result.get("search_executed", True)):
            return False
        query = str(result.get("search_query") or question).strip()[:240]
        excerpts = safe_sources(result.get("sources"))
        answer = str(answer or "").strip()[:6000]
        if not answer or not excerpts or not words(query):
            return False
        actor_id = str(actor_id or "")[:36]
        with self.engine.connect() as con:
            con.execute("INSERT INTO research_findings "
                        "(actor_id,session_id,search_query,question,answer,sources) "
                        "VALUES(?,?,?,?,?,?)",
                        (actor_id, session_id, query, str(question)[:500],
                         answer, json.dumps(excerpts, ensure_ascii=False)))
            # Keep a finite number of findings per person / anonymous session.
            if actor_id:
                con.execute("""DELETE FROM research_findings WHERE actor_id=?
                    AND id NOT IN (SELECT id FROM research_findings
                                   WHERE actor_id=? ORDER BY id DESC LIMIT 150)""",
                            (actor_id, actor_id))
            else:
                con.execute("""DELETE FROM research_findings
                    WHERE actor_id='' AND session_id=?
                    AND id NOT IN (SELECT id FROM research_findings
                                   WHERE actor_id='' AND session_id=?
                                   ORDER BY id DESC LIMIT 60)""",
                            (session_id, session_id))
        return True

    def recall(self, query, session_id="", actor_id=""):
        """Exact-enough topical recall, scoped to actor or anonymous session.

        Non-overlapping topical words mean NOT FOUND. Abstain rather than
        silently substitute one recipe/document for a different one.
        """
        if FRESH.search(query):
            return {"found": False, "reason": "fresh_requested"}
        desired = words(query)
        if not desired:
            return {"found": False, "reason": "no_topic"}
        actor_id = str(actor_id or "")[:36]
        with self.engine.connect() as con:
            if actor_id:
                rows = con.execute(
                    """SELECT * FROM research_findings WHERE actor_id=?
                       ORDER BY id DESC LIMIT 150""", (actor_id,)).fetchall()
            elif session_id:
                rows = con.execute(
                    """SELECT * FROM research_findings WHERE actor_id=''
                       AND session_id=? ORDER BY id DESC LIMIT 60""",
                    (session_id,)).fetchall()
            else:
                return {"found": False, "reason": "no_scope"}
        ranked = []
        for item in rows:
            topical = words(item["search_query"])
            overlap = desired & topical
            # All requested specific terms must match, at least one term.
            if not overlap or len(overlap) != len(desired):
                continue
            # Ambiguous general terms like "рецепт" are excluded by words().
            score = len(overlap) / max(1, len(topical))
            ranked.append((score, item))
        if not ranked:
            return {"found": False, "reason": "no_matching_public_research"}
        ranked.sort(key=lambda pair: (-pair[0], -pair[1]["id"]))
        # Equal topical matches are equivalent query variants; newest wins.
        score, row = ranked[0]
        return {"found": True, "id": row["id"], "actor_id": row["actor_id"],
                "session_id": row["session_id"], "search_query": row["search_query"],
                "answer": row["answer"], "sources": json.loads(row["sources"]),
                "created_at": row["created_at"],
                "evidence_source": "saved_public_web_research_unverified",
                "match_score": round(score, 3),
                "note": "Повторно использован прошлый публичный поиск, без новой проверки."}

    def forget_anonymous_session(self, con, session_id):
        con.execute("DELETE FROM research_findings WHERE actor_id='' AND session_id=?",
                    (session_id,))
