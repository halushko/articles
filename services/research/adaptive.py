"""Learning from explicit behavioral instructions, separate from world facts.

The language model proposes interpretations; this module checks grounding and
persists reversible, per-speaker rules. A learned rule never grants a tool or
bypasses operation approvals. Only the executor may interpret a scoped phrase
as a decision on an already pending operation.
"""
import sqlite3

from .normalize import normalize


SCHEMA = """
CREATE TABLE IF NOT EXISTS behavior_rules (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 actor_id TEXT NOT NULL DEFAULT '',
 kind TEXT NOT NULL CHECK(kind IN ('interpretation','preference','correction')),
 context TEXT NOT NULL,
 trigger TEXT NOT NULL DEFAULT '',
 trigger_key TEXT NOT NULL DEFAULT '',
 meaning TEXT NOT NULL DEFAULT '',
 instruction TEXT NOT NULL DEFAULT '',
 evidence TEXT NOT NULL,
 source_text TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'active',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(actor_id,kind,context,trigger_key)
);
CREATE INDEX IF NOT EXISTS idx_behavior_rules_active
 ON behavior_rules(actor_id,status,context,trigger_key);
CREATE TABLE IF NOT EXISTS behavior_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 rule_id INTEGER NOT NULL REFERENCES behavior_rules(id),
 event TEXT NOT NULL,
 previous TEXT NOT NULL DEFAULT '',
 source_text TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
KINDS = {"interpretation", "preference", "correction"}
# An exact context/effect pair is an executor capability boundary, not a list
# of taught phrases. New supported executors must explicitly implement their
# own approvals before any learned interpretation can invoke them.
WEB_CONTEXT = "pending:web.search"
EFFECTS = {"approve", "deny"}


class BehaviorMemory:
    def __init__(self, engine):
        self.engine = engine
        with engine.connect() as con:
            con.executescript(SCHEMA)

    def learn(self, source_text, learning, actor_id=""):
        """Accept explicit user teachings, not generated knowledge or questions.

        `evidence` must be a literal substring of the user's own utterance.
        An interpretation trigger must also appear literally in that utterance.
        Thus a hallucinated phrase/teaching cannot be installed by the model.
        """
        source_text = str(source_text or "").strip()[:3000]
        actor_id = str(actor_id or "")[:36]
        if not source_text or not isinstance(learning, list):
            return []
        saved = []
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            for proposal in learning[:3]:
                if not isinstance(proposal, dict):
                    continue
                kind = str(proposal.get("kind") or "").strip()
                context = str(proposal.get("context") or "dialogue").strip()[:100]
                trigger = str(proposal.get("trigger") or "").strip()[:120]
                meaning = str(proposal.get("meaning") or "").strip()[:200]
                instruction = str(proposal.get("instruction") or "").strip()[:500]
                evidence = str(proposal.get("evidence") or "").strip()[:600]
                if kind not in KINDS or not context or len(evidence) < 8:
                    continue
                if normalize(evidence) not in normalize(source_text):
                    continue
                key = normalize(trigger)
                if kind == "interpretation":
                    if not key or key not in normalize(source_text):
                        continue
                    # Only the exact web approval context supports an action,
                    # and single generic acknowledgment words cannot authorize.
                    if context == WEB_CONTEXT:
                        if meaning not in EFFECTS or len(key.split()) < 2:
                            continue
                    elif meaning in EFFECTS:
                        continue
                else:
                    if not instruction or len(normalize(instruction)) < 8:
                        continue
                    context = "dialogue"
                    trigger = ""
                    key = "instruction:" + normalize(instruction)[:100]
                    meaning = ""
                previous = con.execute(
                    """SELECT * FROM behavior_rules WHERE
                       actor_id=? AND kind=? AND context=? AND trigger_key=?""",
                    (actor_id, kind, context, key),
                ).fetchone()
                if previous and (previous["meaning"], previous["instruction"]) == (meaning, instruction) and previous["status"] == "active":
                    saved.append(self._public(previous))
                    continue
                if previous:
                    con.execute("""UPDATE behavior_rules SET trigger=?,meaning=?,
                        instruction=?,evidence=?,source_text=?,status='active',
                        updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                        (trigger, meaning, instruction, evidence, source_text, previous["id"]))
                    rule_id = previous["id"]
                    old = previous["meaning"] or previous["instruction"]
                    event = "revised"
                else:
                    cur = con.execute("""INSERT INTO behavior_rules
                        (actor_id,kind,context,trigger,trigger_key,meaning,
                         instruction,evidence,source_text)
                        VALUES(?,?,?,?,?,?,?,?,?)""",
                        (actor_id, kind, context, trigger, key, meaning,
                         instruction, evidence, source_text))
                    rule_id, old, event = cur.lastrowid, "", "learned"
                con.execute("""INSERT INTO behavior_events
                    (rule_id,event,previous,source_text) VALUES(?,?,?,?)""",
                    (rule_id, event, old, source_text))
                row = con.execute("SELECT * FROM behavior_rules WHERE id=?", (rule_id,)).fetchone()
                saved.append(self._public(row))
        return saved

    @staticmethod
    def _public(row):
        return {field: row[field] for field in (
            "id", "actor_id", "kind", "context", "trigger",
            "meaning", "instruction", "evidence", "status",
        )}

    def context(self, actor_id="", limit=16):
        """Personal rules never leak to other selected profiles."""
        with self.engine.connect() as con:
            rows = con.execute("""SELECT * FROM behavior_rules
                WHERE actor_id=? AND status='active'
                ORDER BY updated_at DESC,id DESC LIMIT ?""",
                (str(actor_id or "")[:36], max(1, min(int(limit), 40)))).fetchall()
        return [self._public(row) for row in rows]

    def decision(self, phrase, context, actor_id=""):
        """Whole-utterance exact match; no fuzzy approvals, no implicit consent."""
        if context != WEB_CONTEXT:
            return None
        key = normalize(phrase)
        if not key:
            return None
        with self.engine.connect() as con:
            row = con.execute("""SELECT meaning FROM behavior_rules
                WHERE actor_id=? AND kind='interpretation'
                  AND context=? AND trigger_key=? AND status='active'""",
                (str(actor_id or "")[:36], context, key)).fetchone()
        return row["meaning"] if row and row["meaning"] in EFFECTS else None

    def forget(self, rule_id, actor_id=""):
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("""SELECT * FROM behavior_rules
                WHERE id=? AND actor_id=? AND status='active'""",
                (rule_id, str(actor_id or "")[:36])).fetchone()
            if row is None:
                raise ValueError("Active rule not found for this speaker")
            con.execute("UPDATE behavior_rules SET status='retracted',updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        (rule_id,))
            con.execute("""INSERT INTO behavior_events(rule_id,event,previous)
                VALUES(?,'retracted',?)""", (rule_id, row["meaning"] or row["instruction"]))
        return {"forgotten": True, "rule_id": rule_id, "executed": False}

    def snapshot(self, actor_id=""):
        with self.engine.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM behavior_rules WHERE status='active' AND actor_id=?",
                                (str(actor_id or "")[:36],)).fetchone()[0]
            events = [dict(row) for row in con.execute(
                "SELECT * FROM behavior_events ORDER BY id DESC LIMIT 20")]
        return {"active": count, "rules": self.context(actor_id, 40),
                "recent_events": events,
                "note": "Learned interpretations are not executable tools."}
