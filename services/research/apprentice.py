"""Apprentice memory: generic, domain-independent learning from a human teacher.

No connection to physical devices, code execution or external service is granted
by recording an entity/action/lesson. Model extractions remain proposals until
explicitly confirmed by the teacher.
"""
import json
import sqlite3
from difflib import SequenceMatcher
from .normalize import normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS apprentice_proposals (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 kind TEXT NOT NULL,
 name TEXT NOT NULL,
 name_key TEXT NOT NULL,
 description TEXT NOT NULL,
 domain TEXT NOT NULL,
 details TEXT NOT NULL,
 source_text TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_apprentice_proposals_status ON apprentice_proposals(status,id);
CREATE TABLE IF NOT EXISTS world_entities (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 name TEXT NOT NULL,
 name_key TEXT NOT NULL,
 entity_type TEXT NOT NULL DEFAULT 'unknown',
 domain TEXT NOT NULL DEFAULT 'general',
 description TEXT NOT NULL DEFAULT '',
 source_proposal_id INTEGER REFERENCES apprentice_proposals(id),
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(name_key,domain)
);
CREATE TABLE IF NOT EXISTS world_aliases (
 entity_id INTEGER NOT NULL REFERENCES world_entities(id) ON DELETE CASCADE,
 alias TEXT NOT NULL,
 alias_key TEXT NOT NULL,
 PRIMARY KEY(entity_id,alias_key)
);
CREATE TABLE IF NOT EXISTS world_actions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 name TEXT NOT NULL,
 name_key TEXT NOT NULL,
 domain TEXT NOT NULL,
 description TEXT NOT NULL,
 parameters TEXT NOT NULL DEFAULT '{}',
 executor TEXT DEFAULT NULL,
 source_proposal_id INTEGER REFERENCES apprentice_proposals(id),
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(name_key,domain)
);
CREATE TABLE IF NOT EXISTS world_facts (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 subject TEXT NOT NULL,
 predicate TEXT NOT NULL,
 object TEXT NOT NULL,
 domain TEXT NOT NULL,
 source_proposal_id INTEGER REFERENCES apprentice_proposals(id),
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS world_lessons (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 title TEXT NOT NULL,
 title_key TEXT NOT NULL,
 domain TEXT NOT NULL,
 explanation TEXT NOT NULL,
 steps TEXT NOT NULL DEFAULT '[]',
 examples TEXT NOT NULL DEFAULT '[]',
 source_proposal_id INTEGER REFERENCES apprentice_proposals(id),
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(title_key,domain)
);
CREATE TABLE IF NOT EXISTS capability_ideas (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 name TEXT NOT NULL,
 name_key TEXT NOT NULL,
 domain TEXT NOT NULL,
 description TEXT NOT NULL,
 source_proposal_id INTEGER REFERENCES apprentice_proposals(id),
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(name_key,domain)
);
CREATE TABLE IF NOT EXISTS apprentice_grades (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 question TEXT NOT NULL,
 answer TEXT NOT NULL,
 rating TEXT NOT NULL CHECK(rating IN ('good','bad')),
 teacher_note TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS apprentice_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 event_type TEXT NOT NULL,
 proposal_id INTEGER,
 note TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
KINDS = {"entity", "action", "fact", "lesson", "capability"}
TABLES = {"entity": "world_entities", "action": "world_actions",
          "fact": "world_facts", "lesson": "world_lessons",
          "capability": "capability_ideas"}


class ApprenticeMemory:
    def __init__(self, engine):
        self.engine = engine
        with engine.connect() as con:
            con.executescript(SCHEMA)

    def propose(self, source_text, proposals):
        if not isinstance(proposals, list):
            raise ValueError("proposals must be a list")
        saved = []
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            for item in proposals[:5]:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "").strip().lower()
                name = str(item.get("name") or "").strip()[:120]
                description = str(item.get("description") or "").strip()[:1200]
                domain = str(item.get("domain") or "general").strip()[:60] or "general"
                if kind not in KINDS or not normalize(name) or not description:
                    continue
                # A model can suggest a capability, but cannot mark it runnable.
                details = item.get("details") or {}
                if not isinstance(details, dict):
                    details = {}
                details = {k:v for k,v in details.items()
                           if k in {"entity_type","aliases","subject","predicate","object",
                                    "steps","examples","parameters"}}
                if kind == "fact" and not all(str(details.get(k) or "").strip()
                        for k in ("subject","predicate","object")):
                    continue
                details_text = json.dumps(details, ensure_ascii=False)[:5000]
                # Identical extraction from the same lesson should not create duplicates.
                previous = con.execute("""SELECT id,status FROM apprentice_proposals
                      WHERE kind=? AND name_key=? AND description=? AND source_text=?
                      ORDER BY id DESC LIMIT 1""",
                      (kind,normalize(name),description,source_text[:3000])).fetchone()
                if previous:
                    saved.append({"id":previous["id"],"kind":kind,"name":name,"status":previous["status"]})
                    continue
                cur = con.execute("""INSERT INTO apprentice_proposals
                  (kind,name,name_key,description,domain,details,source_text)
                  VALUES(?,?,?,?,?,?,?)""",
                  (kind,name,normalize(name),description,domain,details_text,source_text[:3000]))
                saved.append({"id":cur.lastrowid,"kind":kind,"name":name,"status":"pending"})
                con.execute("INSERT INTO apprentice_events(event_type,proposal_id,note) VALUES('proposed',?,?)",
                            (cur.lastrowid, f"{kind}: {name}"[:300]))
        return saved

    @staticmethod
    def _details(row):
        try:
            value = json.loads(row["details"])
        except (ValueError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}

    def review(self, proposal_id, decision, name=None, description=None):
        if decision not in {"confirm","reject"}:
            raise ValueError("decision must be confirm or reject")
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM apprentice_proposals WHERE id=?", (proposal_id,)).fetchone()
            if not row:
                raise ValueError("Unknown proposal")
            if row["status"] != "pending":
                raise ValueError("Proposal has already been reviewed")
            new_name = (name.strip()[:120] if isinstance(name,str) and name.strip() else row["name"])
            new_description = (description.strip()[:1200] if isinstance(description,str) and description.strip()
                               else row["description"])
            if not normalize(new_name):
                raise ValueError("Empty name")
            status = "confirmed" if decision == "confirm" else "rejected"
            con.execute("""UPDATE apprentice_proposals
                SET name=?,name_key=?,description=?,status=?,reviewed_at=CURRENT_TIMESTAMP WHERE id=?""",
                (new_name,normalize(new_name),new_description,status,proposal_id))
            if decision == "confirm":
                data = self._details(row)
                kind = row["kind"]
                domain = row["domain"]
                if kind == "entity":
                    con.execute("""INSERT INTO world_entities
                        (name,name_key,entity_type,domain,description,source_proposal_id)
                        VALUES(?,?,?,?,?,?) ON CONFLICT(name_key,domain) DO UPDATE SET
                        description=excluded.description,entity_type=excluded.entity_type,
                        source_proposal_id=excluded.source_proposal_id""",
                        (new_name,normalize(new_name),str(data.get("entity_type") or "unknown")[:80],
                         domain,new_description,proposal_id))
                    entity_id = con.execute("SELECT id FROM world_entities WHERE name_key=? AND domain=?",
                                            (normalize(new_name),domain)).fetchone()["id"]
                    for alias in data.get("aliases",[]) if isinstance(data.get("aliases"),list) else []:
                        if isinstance(alias,str) and normalize(alias):
                            con.execute("INSERT OR IGNORE INTO world_aliases VALUES(?,?,?)",
                                        (entity_id,alias[:120],normalize(alias)))
                elif kind == "action":
                    con.execute("""INSERT INTO world_actions
                        (name,name_key,domain,description,parameters,source_proposal_id)
                        VALUES(?,?,?,?,?,?) ON CONFLICT(name_key,domain) DO UPDATE SET
                        description=excluded.description,parameters=excluded.parameters,
                        source_proposal_id=excluded.source_proposal_id""",
                        (new_name,normalize(new_name),domain,new_description,
                         json.dumps(data.get("parameters") or {},ensure_ascii=False)[:2000],proposal_id))
                elif kind == "fact":
                    con.execute("""INSERT INTO world_facts(subject,predicate,object,domain,source_proposal_id)
                        VALUES(?,?,?,?,?)""",
                        (str(data["subject"])[:120],str(data["predicate"])[:100],
                         str(data["object"])[:300],domain,proposal_id))
                elif kind == "lesson":
                    con.execute("""INSERT INTO world_lessons
                        (title,title_key,domain,explanation,steps,examples,source_proposal_id)
                        VALUES(?,?,?,?,?,?,?) ON CONFLICT(title_key,domain) DO UPDATE SET
                        explanation=excluded.explanation,steps=excluded.steps,
                        examples=excluded.examples,source_proposal_id=excluded.source_proposal_id""",
                        (new_name,normalize(new_name),domain,new_description,
                         json.dumps(data.get("steps") or [],ensure_ascii=False)[:3000],
                         json.dumps(data.get("examples") or [],ensure_ascii=False)[:3000],proposal_id))
                elif kind == "capability":
                    con.execute("""INSERT INTO capability_ideas
                        (name,name_key,domain,description,source_proposal_id)
                        VALUES(?,?,?,?,?) ON CONFLICT(name_key,domain) DO UPDATE SET
                        description=excluded.description,
                        source_proposal_id=excluded.source_proposal_id""",
                        (new_name,normalize(new_name),domain,new_description,proposal_id))
            con.execute("INSERT INTO apprentice_events(event_type,proposal_id,note) VALUES(?,?,?)",
                        (status,proposal_id,f"{row['kind']}: {new_name}"[:300]))
        return {"id":proposal_id,"status":status,"kind":row["kind"],"name":new_name,
                "executed":False}

    def grade(self, question, answer, rating, teacher_note=""):
        """Teacher-scored answer: evaluation evidence, NOT a knowledge update."""
        if rating not in {"good","bad"} or not question.strip() or not answer.strip():
            raise ValueError("Invalid graded answer")
        with self.engine.connect() as con:
            cur = con.execute("""INSERT INTO apprentice_grades
                (question,answer,rating,teacher_note) VALUES(?,?,?,?)""",
                (question[:3000],answer[:3000],rating,teacher_note[:1200]))
        return {"saved":True,"id":cur.lastrowid,"rating":rating,
                "note":"Feedback recorded for evaluation, not used to train Qwen",
                "executed":False}

    def forget(self, kind, item_id):
        """Retract a confirmed item without destroying the teaching audit."""
        if kind not in TABLES:
            raise ValueError("Unknown knowledge type")
        table = TABLES[kind]  # SQL identifier is from an internal allowlist.
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(f"SELECT * FROM {table} WHERE id=?", (item_id,)).fetchone()
            if not row:
                raise ValueError("Unknown knowledge item")
            proposal_id = row["source_proposal_id"]
            con.execute(f"DELETE FROM {table} WHERE id=?", (item_id,))
            if proposal_id is not None:
                con.execute("""UPDATE apprentice_proposals SET status='retracted',
                    reviewed_at=CURRENT_TIMESTAMP WHERE id=? AND status='confirmed'""",
                    (proposal_id,))
            con.execute("INSERT INTO apprentice_events(event_type,proposal_id,note) VALUES(?,?,?)",
                        ("retracted",proposal_id,f"{kind} #{item_id}"[:300]))
        return {"forgotten":True,"kind":kind,"id":item_id,"executed":False}

    @staticmethod
    def _row(row):
        item = dict(row)
        if "details" in item:
            try:
                item["details"] = json.loads(item["details"])
            except (TypeError, ValueError):
                item["details"] = {}
        for field in ("parameters","steps","examples"):
            if field in item:
                try:
                    item[field] = json.loads(item[field])
                except (ValueError, TypeError):
                    pass
        return item

    def snapshot(self):
        with self.engine.connect() as con:
            counts = {k:con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                      for k,table in TABLES.items()}
            counts["pending"] = con.execute(
                "SELECT COUNT(*) FROM apprentice_proposals WHERE status='pending'").fetchone()[0]
            counts["graded"] = con.execute(
                "SELECT COUNT(*) FROM apprentice_grades").fetchone()[0]
            counts["graded_good"] = con.execute(
                "SELECT COUNT(*) FROM apprentice_grades WHERE rating='good'").fetchone()[0]
            counts["rejected"] = con.execute(
                "SELECT COUNT(*) FROM apprentice_proposals WHERE status='rejected'").fetchone()[0]
            proposals = [self._row(r) for r in con.execute(
                """SELECT * FROM apprentice_proposals WHERE status='pending'
                   ORDER BY id DESC LIMIT 30""")]
            nodes = {}
            for kind,table in TABLES.items():
                nodes[kind] = [self._row(r) for r in con.execute(
                    f"SELECT * FROM {table} ORDER BY id DESC LIMIT 35")]
            events = [dict(r) for r in con.execute(
                "SELECT * FROM apprentice_events ORDER BY id DESC LIMIT 20")]
        return {"counts":counts,"pending":proposals,"knowledge":nodes,
                "events":events,"note":"Only confirmed teachings are in knowledge. Actions are descriptions, not runnable tools."}

    def context(self, query):
        """Retrieve confirmed knowledge only. Similarity is a fast *hint*, not proof."""
        text = normalize(query)
        tokens = set(text.split())
        with self.engine.connect() as con:
            rows = []
            for kind,table in TABLES.items():
                data = con.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT 500").fetchall()
                for row in data:
                    obj = self._row(row)
                    title = obj.get("name") or obj.get("title") or obj.get("subject","")
                    search = " ".join(str(obj.get(k) or "") for k in
                                      ("name","title","subject","predicate","object","domain","description","explanation"))
                    if kind == "entity":
                        alias = con.execute("SELECT alias FROM world_aliases WHERE entity_id=?",
                                            (obj["id"],)).fetchall()
                        search += " " + " ".join(r["alias"] for r in alias)
                        obj["aliases"] = [r["alias"] for r in alias]
                    hay = normalize(search)
                    overlap = len(tokens.intersection(set(hay.split()))) / max(1,len(tokens))
                    near = max((SequenceMatcher(None,word,normalize(title)).ratio()
                                for word in tokens), default=0.0)
                    score = max(overlap, 0.7*near)
                    if text and score >= 0.28:
                        obj["kind"] = kind
                        obj["relevance"] = round(score,3)
                        rows.append(obj)
            counts = {k:con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                      for k,table in TABLES.items()}
            pending_count = con.execute(
                "SELECT COUNT(*) FROM apprentice_proposals WHERE status='pending'").fetchone()[0]
        rows.sort(key=lambda item:item["relevance"],reverse=True)
        # "What have I taught you?" is an overview request, not a lexical
        # match against a particular concept. Include recent confirmed items.
        overview = any(word in text for word in
                       ("научил", "научился", "знаешь", "знаєш", "умеешь",
                        "вмієш", "запомнил", "памят", "знан", "remember",
                        "learned", "knowledge"))
        if overview:
            with self.engine.connect() as con:
                recent = []
                for kind, table in TABLES.items():
                    for row in con.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT 3"):
                        item = self._row(row)
                        item["kind"] = kind
                        recent.append(item)
            rows = recent[:12]
        # Do not overwhelm a small model with irrelevant facts.
        return {"knowledge":rows[:12] if overview else rows[:9],"counts":counts,"pending_count":pending_count,
                "execution_tools":[],"note":"Knowledge is teacher-confirmed. No external execution tools are connected."}
