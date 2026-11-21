"""Research module: append-only correction history, current mappings and online ranker.

This module never imports an LLM or invokes a chat model. Exact and online
ablations can be evaluated offline with the same evidence snapshots.
"""
import json
import math
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager
from contextlib import contextmanager
from .normalize import DEMO_TARGETS, features, normalize, parse_command

SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phrase TEXT NOT NULL,
    phrase_key TEXT NOT NULL,
    action TEXT NOT NULL,
    target_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_evidence_key ON evidence(phrase_key);
CREATE TABLE IF NOT EXISTS active_mappings (
    phrase_key TEXT PRIMARY KEY,
    target_id TEXT NOT NULL,
    evidence_id INTEGER NOT NULL REFERENCES evidence(id),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS online_weights (
    target_id TEXT NOT NULL,
    feature TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY(target_id, feature)
);
CREATE TABLE IF NOT EXISTS concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    domain TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS aliases (
    concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    alias_key TEXT NOT NULL,
    UNIQUE(concept_id, alias_key)
);
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    left_id INTEGER NOT NULL REFERENCES concepts(id),
    relation TEXT NOT NULL,
    right_id INTEGER NOT NULL REFERENCES concepts(id),
    UNIQUE(left_id, relation, right_id)
);
CREATE TABLE IF NOT EXISTS semantic_examples (
    evidence_id INTEGER PRIMARY KEY REFERENCES evidence(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vector TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phrase TEXT NOT NULL,
    ablation TEXT NOT NULL,
    route TEXT NOT NULL,
    target_id TEXT,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER REFERENCES decisions(id),
    rating TEXT NOT NULL CHECK(rating IN ('good','bad')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class Engine:
    def __init__(self, path: str, settings: dict):
        self.path = path
        self.settings = settings
        self.targets = {t["id"]: t["name"] for t in DEMO_TARGETS}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA foreign_keys=ON")
            con.execute("PRAGMA busy_timeout=10000")
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def learn(self, phrase: str, target_id: str, negative_ids=None):
        parsed = parse_command(phrase)
        if not parsed["is_command"] or not parsed["safe"] or not parsed["target_text"]:
            raise ValueError("Teach requires an affirmative, unambiguous demo command")
        if target_id not in self.targets:
            raise ValueError("Unknown target ID; no real Home Assistant actions supported")
        key = normalize(parsed["target_text"])
        vector = features(key)
        negatives = set(negative_ids or [])
        if not negatives.issubset(self.targets) or target_id in negatives:
            raise ValueError("Invalid negative targets")
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            previous = con.execute("SELECT target_id FROM active_mappings WHERE phrase_key=?", (key,)).fetchone()
            old_target = previous["target_id"] if previous else None
            if old_target and old_target != target_id:
                negatives.add(old_target)
            cur = con.execute(
                "INSERT INTO evidence(phrase,phrase_key,action,target_id) VALUES(?,?,?,?)",
                (phrase.strip(), key, parsed["action"], target_id),
            )
            evidence_id = cur.lastrowid
            con.execute("""INSERT INTO active_mappings(phrase_key,target_id,evidence_id)
                VALUES(?,?,?) ON CONFLICT(phrase_key) DO UPDATE SET
                target_id=excluded.target_id,evidence_id=excluded.evidence_id,updated_at=CURRENT_TIMESTAMP""",
                (key, target_id, evidence_id))
            rate = float(self.settings["learning_rate"])
            # One active phrase = one contribution. Repeating the same teaching
            # adds audit evidence but must not artificially multiply a weight.
            # A correction subtracts the superseded phrase contribution.
            if old_target != target_id:
                for feature, value in vector.items():
                    self._weight_update(con, target_id, feature, rate*value)
                    if old_target:
                        self._weight_update(con, old_target, feature, -rate*value)
            # Additional explicit negatives are distinct from superseding the old mapping.
            for bad in negatives - ({old_target} if old_target else set()):
                for feature, value in vector.items():
                    self._weight_update(con, bad, feature, -rate*value)
            count = con.execute("SELECT COUNT(*) AS n FROM evidence WHERE phrase_key=?", (key,)).fetchone()["n"]
        return {"saved": True, "evidence_id": evidence_id, "phrase_key": key,
                "current_target": target_id, "previous_target": old_target,
                "history_entries_for_phrase": count, "executed": False}

    @staticmethod
    def _weight_update(con, target_id: str, feature: str, delta: float):
        con.execute("""INSERT INTO online_weights(target_id,feature,value) VALUES(?,?,?)
            ON CONFLICT(target_id,feature) DO UPDATE SET value=value+excluded.value""",
            (target_id, feature, delta))

    def _rank_online(self, con, target_text: str):
        query = features(target_text)
        if not query:
            return []
        scores = {}
        norms = {}
        for row in con.execute("SELECT target_id,feature,value FROM online_weights"):
            target = row["target_id"]
            value = float(row["value"])
            norms[target] = norms.get(target, 0) + value*value
            scores[target] = scores.get(target, 0) + query.get(row["feature"], 0.0)*value
        counts = {r["target_id"]: r["n"] for r in con.execute(
            "SELECT target_id,COUNT(DISTINCT phrase_key) AS n FROM active_mappings GROUP BY target_id")}
        result = []
        for target_id, dot in scores.items():
            if target_id not in self.targets:
                continue
            score = dot / math.sqrt(norms[target_id]) if norms[target_id] else 0.0
            result.append({"target_id": target_id, "target_name": self.targets[target_id],
                           "score": round(score, 5), "distinct_examples": counts.get(target_id, 0)})
        return sorted(result, key=lambda item: item["score"], reverse=True)

    def predict(self, phrase: str, ablation: str = "combined", record: bool = False):
        if ablation not in {"exact", "online", "combined"}:
            raise ValueError("ablation must be exact, online or combined")
        start = time.perf_counter()
        parsed = parse_command(phrase)
        result = {"is_command": parsed["is_command"], "action": parsed["action"],
                  "safe": parsed["safe"], "target_id": None, "target_name": None,
                  "candidates": [], "ablation": ablation, "executed": False}
        if not parsed["is_command"]:
            result.update({"status": "not_command", "route": "not_command"})
        elif not parsed["safe"]:
            result.update({"status": "blocked", "route": "safety", "reason": parsed.get("reason")})
        elif not parsed["target_text"]:
            result.update({"status": "needs_confirmation", "route": "missing_target"})
        else:
            key = normalize(parsed["target_text"])
            with self.connect() as con:
                row = con.execute("SELECT target_id FROM active_mappings WHERE phrase_key=?", (key,)).fetchone()
                if row and row["target_id"] in self.targets:
                    target = row["target_id"]
                    result.update({"status": "resolved", "route": "exact_confirmed",
                                   "target_id": target, "target_name": self.targets[target]})
                elif ablation == "exact":
                    result.update({"status": "needs_confirmation", "route": "exact_miss"})
                else:
                    ranked = self._rank_online(con, key)
                    result["candidates"] = ranked[:int(self.settings["max_candidates"])]
                    if ranked:
                        top = ranked[0]
                        margin = top["score"] - (ranked[1]["score"] if len(ranked)>1 else 0.0)
                        result["margin"] = round(margin, 5)
                        accepted = (top["score"] >= float(self.settings["online_min_similarity"])
                                    and margin >= float(self.settings["online_min_margin"])
                                    and top["distinct_examples"] >= int(self.settings["min_distinct_examples"]))
                        if accepted:
                            result.update({"status": "resolved", "route": "online_reranker",
                                           "target_id": top["target_id"], "target_name": top["target_name"]})
                        else:
                            result.update({"status": "needs_confirmation", "route": "online_abstain"})
                    else:
                        result.update({"status": "needs_confirmation", "route": "no_evidence"})
        result["latency_ms"] = round((time.perf_counter()-start)*1000, 2)
        if record:
            with self.connect() as con:
                row = con.execute("INSERT INTO decisions(phrase,ablation,route,target_id,latency_ms) VALUES(?,?,?,?,?)",
                                  (phrase[:500], ablation, result["route"], result["target_id"], result["latency_ms"]))
                result["decision_id"] = row.lastrowid
        return result

    def log_decision(self, result: dict, phrase: str, ablation: str):
        with self.connect() as con:
            row = con.execute("INSERT INTO decisions(phrase,ablation,route,target_id,latency_ms) VALUES(?,?,?,?,?)",
                              (phrase[:500], ablation, result["route"], result["target_id"], result["latency_ms"]))
            return row.lastrowid

    def store_embedding(self, evidence_id: int, model: str, vector: list[float]):
        with self.connect() as con:
            con.execute("INSERT OR REPLACE INTO semantic_examples(evidence_id,model,vector) VALUES(?,?,?)",
                        (evidence_id, model, json.dumps(vector)))

    def semantic_rank(self, query_vector: list[float], model: str):
        if not query_vector:
            return []
        with self.connect() as con:
            active = {r["phrase_key"]: r["target_id"] for r in con.execute("SELECT phrase_key,target_id FROM active_mappings")}
            rows = con.execute("""SELECT s.vector,e.phrase_key,e.target_id FROM semantic_examples s
                JOIN evidence e ON e.id=s.evidence_id WHERE s.model=?""", (model,)).fetchall()
        scores = {}
        for row in rows:
            if active.get(row["phrase_key"]) != row["target_id"]:
                continue  # Superseded evidence must not vote for the old target.
            v = json.loads(row["vector"])
            if len(v) != len(query_vector):
                continue
            denominator = math.sqrt(sum(a*a for a in v)*sum(a*a for a in query_vector))
            score = sum(a*b for a,b in zip(v,query_vector))/denominator if denominator else 0.0
            scores[row["target_id"]] = max(scores.get(row["target_id"], -1), score)
        with self.connect() as con:
            counts = {r["target_id"]: r["n"] for r in con.execute(
                "SELECT target_id,COUNT(*) AS n FROM active_mappings GROUP BY target_id")}
        return sorted(({"target_id": tid, "target_name": self.targets.get(tid,tid),
                        "score": round(s,5), "distinct_examples": counts.get(tid,0)}
                       for tid,s in scores.items() if tid in self.targets),
                      key=lambda item:item["score"], reverse=True)

    def remember_concept(self, name: str, kind: str = "other", domain: str = "general", description: str = "", aliases=None):
        key = normalize(name)
        if not key:
            raise ValueError("Empty concept name")
        with self.connect() as con:
            con.execute("INSERT INTO concepts(name,name_key,kind,domain,description) VALUES(?,?,?,?,?)",
                        (name.strip(), key, kind.strip(), domain.strip(), description.strip()))
            cid = con.execute("SELECT last_insert_rowid()").fetchone()[0]
            for alias in aliases or []:
                if normalize(alias):
                    con.execute("INSERT OR IGNORE INTO aliases(concept_id,alias,alias_key) VALUES(?,?,?)",
                                (cid, alias.strip(), normalize(alias)))
        return {"id": cid, "name": name, "kind": kind, "domain": domain, "aliases": aliases or []}

    def memory_snapshot(self):
        with self.connect() as con:
            tables = {label: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                      for label, table in (("corrections","evidence"), ("active_mappings","active_mappings"),
                                           ("weight_features","online_weights"), ("concepts","concepts"),
                                           ("aliases","aliases"), ("relations","relations"),
                                           ("semantic_vectors","semantic_examples"), ("decisions","decisions"))}
            latest = [dict(r) for r in con.execute(
                "SELECT phrase,target_id,created_at,id FROM evidence ORDER BY id DESC LIMIT 12")]
            current = [dict(r) for r in con.execute(
                "SELECT a.phrase_key,a.target_id,a.updated_at FROM active_mappings a ORDER BY a.updated_at DESC,a.phrase_key LIMIT 50")]
            concepts = [dict(r) for r in con.execute(
                "SELECT id,name,kind,domain,created_at FROM concepts ORDER BY id DESC LIMIT 20")]
        return {"counts": tables, "recent_corrections": latest, "active_mappings": current,
                "concepts": concepts, "note": "All counts from SQLite; no LLM involved."}

    def record_feedback(self, decision_id: int, rating: str):
        if rating not in {"good", "bad"}:
            raise ValueError("Invalid feedback rating")
        with self.connect() as con:
            if not con.execute("SELECT 1 FROM decisions WHERE id=?", (decision_id,)).fetchone():
                raise ValueError("Unknown decision ID")
            con.execute("INSERT INTO feedback(decision_id,rating) VALUES(?,?)", (decision_id, rating))
        return {"saved": True, "note": "Feedback logged; a rating is not a ground-truth label."}
