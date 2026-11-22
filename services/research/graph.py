"""Teacher-confirmed knowledge graph projected from the existing Apprentice tables.

Materialized node IDs are stable across domains/restarts. Edges derive ONLY from
confirmed world_facts, never from model replies or pending proposals. Older raw
SQLite records remain unchanged and available for provenance/undo.
"""
import re
from .normalize import normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_nodes (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 name_key TEXT NOT NULL UNIQUE,
 name TEXT NOT NULL,
 active INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS graph_edges (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 subject_id INTEGER NOT NULL REFERENCES graph_nodes(id),
 predicate TEXT NOT NULL,
 object_id INTEGER NOT NULL REFERENCES graph_nodes(id),
 source_fact_id INTEGER NOT NULL UNIQUE REFERENCES world_facts(id) ON DELETE CASCADE,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_subject ON graph_edges(subject_id);
CREATE TABLE IF NOT EXISTS graph_equivalences (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 alias_key TEXT NOT NULL UNIQUE,
 alias TEXT NOT NULL,
 canonical_key TEXT NOT NULL,
 canonical_name TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

# Transparent, limited intent parsing. For general questions Qwen gets the
# verified graph as context instead of this parser pretending to understand all language.
ENTITY_WORDS = ("сущност", "сутност", "entity", "entities", "объект", "обьект")
RELATION_WORDS = ("связ", "отношен", "relation", "факт", "relation")
KNOWLEDGE_WORDS = ("памят", "помниш", "помнишь", "знани", "научил", "сохранил", "запомнил")
ASK_WORDS = ("перечис", "покаж", "назови", "какие", "каких", "сколько",
             "что", "все", "список", "перелік", "які", "що", "що ти")
HAS_PREDICATE = ("имеет","есть","владеет","владелец","owns","has","possesses",
                 "належить","має","имеется","обладает","принадлежит")
HOME_WORDS = ("дома","домой","в доме","у меня дома","домашн","моем доме",
              "моём доме","квартир","осел","оселі","вдома","хаты")
LIST_LIMIT = 80


def key(value):
    return normalize(str(value or "").strip())[:160]


def user_referent(value, roles):
    v = key(value)
    if v in ("user","пользователь","собеседник","я","мне","у меня","мой",
             "владелец","учитель","користувач"):
        return roles.get("user") or "пользователь"
    if v in ("assistant","ассистент","ученик","ты","джарвис"):
        return roles.get("assistant") or "JARVIS"
    return value


def inspection_kind(query):
    text = key(query)
    question = any(x in text for x in ASK_WORDS) or text.endswith("?")
    if not question:
        return None
    if any(w in text for w in ENTITY_WORDS):
        return "entities"
    if any(w in text for w in RELATION_WORDS):
        return "relations"
    if any(w in text for w in ("действи", "операци", "урок", "навык", "способност")):
        return "skills"
    if any(w in text for w in HOME_WORDS) and any(x in text for x in
            ("у меня", "моем", "моём", "мої", "мои", "моих", "что", "які", "есть")):
        return "personal_home"
    if any(w in text for w in KNOWLEDGE_WORDS):
        return "knowledge"
    return None


class KnowledgeGraph:
    def __init__(self, engine):
        self.engine = engine
        with engine.connect() as con:
            con.executescript(SCHEMA)
        self.rebuild()

    def equivalences(self):
        with self.engine.connect() as con:
            return [dict(r) for r in con.execute(
                "SELECT id,alias_key,alias,canonical_key,canonical_name FROM graph_equivalences ORDER BY id")]

    def add_equivalence(self, alias, canonical):
        """Explicit teacher decision; never infer identity from model similarity."""
        alias = str(alias or "").strip()
        canonical = str(canonical or "").strip()
        a,b=key(alias),key(canonical)
        if not (2 <= len(alias) <= 120 and 2 <= len(canonical) <= 120) or a==b:
            raise ValueError("Укажи два разных названия, не длиннее 120 символов")
        # Only alias an actual confirmed node, not an imaginary capability.
        graph=self.snapshot()
        target=next((n for n in graph["nodes"] if
                     key(n["name"])==b or b in (key(x) for x in n["aliases"])),None)
        if target is None:
            raise ValueError("Каноническая сущность должна быть подтверждена")
        final_key=key(target["name"])
        if a==final_key:
            raise ValueError("Нельзя направить синоним на самого себя")
        with self.engine.connect() as con:
            con.execute("""INSERT INTO graph_equivalences(alias_key,alias,canonical_key,canonical_name)
                VALUES(?,?,?,?) ON CONFLICT(alias_key) DO UPDATE SET
                alias=excluded.alias,canonical_key=excluded.canonical_key,
                canonical_name=excluded.canonical_name""",
                (a,alias,final_key,target["name"]))
            con.execute("""INSERT INTO apprentice_events(event_type,proposal_id,note)
                VALUES('graph_alias_confirmed',NULL,?)""",
                (f"{alias} → {target['name']}"[:300],))
        self.rebuild()
        return {"saved":True,"alias":alias,"canonical":target["name"],"executed":False}

    def delete_equivalence(self, item_id):
        with self.engine.connect() as con:
            row=con.execute("DELETE FROM graph_equivalences WHERE id=? RETURNING id,alias,canonical_name",
                            (item_id,)).fetchone()
            if row:
                con.execute("""INSERT INTO apprentice_events(event_type,proposal_id,note)
                    VALUES('graph_alias_retracted',NULL,?)""",
                    (f"{row['alias']} → {row['canonical_name']}"[:300],))
        if not row:
            raise ValueError("Синоним для объединения не найден")
        self.rebuild()
        return {"deleted":True,"id":item_id,"executed":False}

    @staticmethod
    def _resolve(value, mapping):
        k=key(value)
        label=str(value)
        seen=set()
        while k in mapping and k not in seen:
            seen.add(k)
            canonical_key,canonical_name=mapping[k]
            k,label=canonical_key,canonical_name
        return k,label

    def rebuild(self):
        """Idempotent, teacher-audited projection.

        Same normalized text merges across legacy domains. Different spellings
        merge only when the teacher explicitly approves an equivalence.
        """
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            mapping={r["alias_key"]:(r["canonical_key"],r["canonical_name"])
                     for r in con.execute(
                      "SELECT alias_key,canonical_key,canonical_name FROM graph_equivalences")}
            con.execute("UPDATE graph_nodes SET active=0")
            con.execute("DELETE FROM graph_edges")
            cache = {}
            def node(value):
                label = str(value or "").strip()[:160]
                k,label = self._resolve(label,mapping)
                if not k:
                    return None
                if k not in cache:
                    con.execute("""INSERT INTO graph_nodes(name_key,name,active)
                        VALUES(?,?,1) ON CONFLICT(name_key)
                        DO UPDATE SET active=1""", (k,label))
                    cache[k] = con.execute(
                        "SELECT id FROM graph_nodes WHERE name_key=?", (k,)).fetchone()["id"]
                return cache[k]
            entities = con.execute(
                "SELECT id,name FROM world_entities ORDER BY id").fetchall()
            for row in entities:
                node(row["name"])
            roles = {row["role"]:row["display_name"] for row in
                     con.execute("SELECT role,display_name FROM dialogue_roles")}
            facts = con.execute(
                "SELECT id,subject,predicate,object FROM world_facts ORDER BY id").fetchall()
            for row in facts:
                subj = user_referent(row["subject"],roles)
                obj = user_referent(row["object"],roles)
                a,b = node(subj),node(obj)
                if a is not None and b is not None:
                    con.execute("""INSERT INTO graph_edges
                        (subject_id,predicate,object_id,source_fact_id)
                        VALUES(?,?,?,?)""",
                        (a,row["predicate"],b,row["id"]))

    def snapshot(self):
        self.rebuild()
        with self.engine.connect() as con:
            mapping={r["alias_key"]:(r["canonical_key"],r["canonical_name"])
                     for r in con.execute(
                      "SELECT alias_key,canonical_key,canonical_name FROM graph_equivalences")}
            nodes = [dict(row) for row in con.execute(
                """SELECT id,name,name_key FROM graph_nodes WHERE active=1 ORDER BY id""")]
            edges = [dict(row) for row in con.execute(
                """SELECT e.id,e.predicate,e.source_fact_id,
                      s.id AS subject_id,s.name AS subject,
                      o.id AS object_id,o.name AS object
                   FROM graph_edges e
                   JOIN graph_nodes s ON s.id=e.subject_id
                   JOIN graph_nodes o ON o.id=e.object_id
                   ORDER BY e.id""")]
            evidence = {}
            for row in con.execute("SELECT id,name,domain,description FROM world_entities"):
                k=self._resolve(row["name"],mapping)[0]
                evidence.setdefault(k,[]).append(
                    {"table":"world_entities","id":row["id"],"domain":row["domain"],
                     "description":row["description"]})
            aliases = {}
            for row in con.execute("""SELECT e.name,a.alias FROM world_aliases a
                                      JOIN world_entities e ON e.id=a.entity_id"""):
                k=self._resolve(row["name"],mapping)[0]
                aliases.setdefault(k,set()).add(row["alias"])
            equivalences=[dict(r) for r in con.execute(
                "SELECT id,alias_key,alias,canonical_key,canonical_name FROM graph_equivalences ORDER BY id")]
            for row in equivalences:
                target_key=self._resolve(row["canonical_name"],mapping)[0]
                aliases.setdefault(target_key,set()).add(row["alias"])
            # The original label is also a useful approved alias when it was merged.
            for row in con.execute("SELECT name FROM world_entities"):
                target_key=self._resolve(row["name"],mapping)[0]
                if target_key != key(row["name"]):
                    aliases.setdefault(target_key,set()).add(row["name"])
        for n in nodes:
            n["evidence"]=evidence.get(n["name_key"],[])
            n["aliases"]=sorted(aliases.get(n["name_key"],set()))
            n.pop("name_key",None)
        return {"nodes":nodes, "edges":edges,
                "equivalences":equivalences,
                "counts":{"canonical_entities":len(nodes),"confirmed_relations":len(edges)},
                "identity_resolution":"normalized names plus teacher-approved equivalences",
                "provenance":"confirmed world_entities/world_facts and explicit teacher aliases"}

    def lookup(self, query, graph=None, limit=12):
        graph = graph or self.snapshot()
        text = key(query)
        tokens = set(text.split())
        if not tokens:
            return {"nodes":[],"edges":[]}
        scores=[]
        for n in graph["nodes"]:
            hay=key(n["name"]+" "+ " ".join(n["aliases"])+" "+
                    " ".join((x["domain"]+" "+x["description"]) for x in n["evidence"]))
            words=set(hay.split())
            match=len(tokens & words)
            score=match/max(1,len(tokens))
            if key(n["name"]) in text:
                score=max(score,1.0)
            if score >= 0.17:
                scores.append((score,n))
        scores.sort(key=lambda pair:(-pair[0],pair[1]["id"]))
        found=[x[1] for x in scores[:limit]]
        ids={n["id"] for n in found}
        edges=[edge for edge in graph["edges"] if edge["subject_id"] in ids or edge["object_id"] in ids]
        # Bring immediate neighbors of relevant concepts, even if their words
        # are not present in the question.
        related_ids=ids|{edge["subject_id"] for edge in edges}|{edge["object_id"] for edge in edges}
        full=[n for n in graph["nodes"] if n["id"] in related_ids]
        return {"nodes":full[:limit*2],"edges":edges[:limit*2]}

    def inspect(self, query, roles=None, graph=None):
        kind=inspection_kind(query)
        if not kind:
            return None
        graph=graph or self.snapshot()
        nodes,edges=graph["nodes"],graph["edges"]
        counts=graph["counts"]
        if kind=="entities":
            if not nodes:
                answer="В подтверждённой памяти пока нет сущностей."
            else:
                shown=nodes[:LIST_LIMIT]
                answer=(f"В подтверждённой памяти {len(nodes)} уникальных сущностей "
                        f"(по нормализованным именам и одобренным синонимам): "+
                        "; ".join(f"{n['name']} (#{n['id']})" for n in shown)+".")
                if len(nodes)>len(shown):
                    answer+=f" Показаны первые {len(shown)}; полный список смотри в карте знаний."
            trace={"query_type":"entities","count":len(nodes)}
        elif kind=="relations":
            if not edges:
                answer="Подтверждённых отношений между сущностями пока нет."
            else:
                shown=edges[:LIST_LIMIT]
                answer=f"Подтверждённых связей: {len(edges)}. "+ "; ".join(
                    f"{e['subject']} — {e['predicate']} → {e['object']} [факт #{e['source_fact_id']}]"
                    for e in shown)
                if len(edges)>len(shown): answer+=" (список сокращён)"
            trace={"query_type":"relations","count":len(edges)}
        elif kind=="personal_home":
            roles=roles or {}
            username=key(roles.get("user") or "пользователь")
            teacher_aliases={a["alias_key"]:(a["canonical_key"],a["canonical_name"])
                             for a in graph.get("equivalences",[])}
            username=self._resolve(username,teacher_aliases)[0]
            owned=[e for e in edges
                   if key(e["subject"])==username and
                   key(e["predicate"]) in HAS_PREDICATE]
            if owned:
                items=[e["object"] for e in owned]
                answer="В подтверждённых связях указано, что у тебя есть: "+", ".join(items)+"."
                neighbors=[e for e in edges if any(key(e["subject"])==key(i) for i in items)
                           and e not in owned]
                if neighbors:
                    answer+=" Дополнительные связи: "+"; ".join(
                        f"{e['subject']} — {e['predicate']} → {e['object']}"
                        for e in neighbors[:8])+"."
            else:
                answer=("Пока нет подтверждённых связей о том, что именно тебе "
                        "принадлежит дома. Я не буду угадывать по одному упоминанию.")
                names=[n["name"] for n in nodes if any(
                    x in key(" ".join([e["domain"],e["description"]]))
                    for e in n["evidence"] for x in ("дом","house","home","smart","смарт"))]
                if names: answer+=" В памяти упомянуты: "+", ".join(names[:8])+"."
            trace={"query_type":"personal_home","verified_owned_edges":len(owned)}
        elif kind=="skills":
            # Actions/lessons/capability ideas are separately reported by the
            # research index; do not label all named graph nodes as skills.
            return None
        else:
            answer=(f"В подтверждённой памяти {counts['canonical_entities']} "
                    f"сущностей и {counts['confirmed_relations']} связей.")
            if nodes: answer+=" Сущности: "+", ".join(n["name"] for n in nodes[:16])+"."
            if edges: answer+=" Связи: "+"; ".join(
                f"{e['subject']} — {e['predicate']} → {e['object']}" for e in edges[:12])+"."
            trace={"query_type":"knowledge","nodes":len(nodes),"edges":len(edges)}
        return {"answer":answer,"kind":kind,"trace":trace,
                "counts":counts, "executed":False,"source":"confirmed_sqlite_graph"}
