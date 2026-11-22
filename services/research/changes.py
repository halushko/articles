"""Reviewed, typed updates to existing knowledge. Never infer a new identity
from model prose, a graded answer or a web source.

Updates carry a stable target (role key or entity id), old value, proposed new
value, actor and revision evidence. Confirm is an atomic compare-and-swap:
stale proposals cannot silently overwrite a more recent correction.
"""
import re
import sqlite3
from .normalize import normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_changes (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 target_kind TEXT NOT NULL CHECK(target_kind IN ('role','entity')),
 target_id TEXT NOT NULL,
 field TEXT NOT NULL DEFAULT 'name',
 expected_value TEXT NOT NULL,
 proposed_value TEXT NOT NULL,
 source_text TEXT NOT NULL,
 source_kind TEXT NOT NULL DEFAULT 'teacher_request',
 conflict_node_id INTEGER,
 status TEXT NOT NULL DEFAULT 'pending',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_changes_pending
 ON knowledge_changes(status,target_kind,target_id);
CREATE TABLE IF NOT EXISTS knowledge_change_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 change_id INTEGER NOT NULL REFERENCES knowledge_changes(id),
 event TEXT NOT NULL,
 before_value TEXT NOT NULL,
 after_value TEXT NOT NULL,
 note TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS dialogue_role_aliases (
 role TEXT NOT NULL,
 name TEXT NOT NULL,
 name_key TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(role,name_key)
);
"""

# Parsing only detects explicit mutation intent. It does not guess what an
# unspecified object refers to, or treat a question as a write request.
SELF_RENAME = (
    re.compile(r"(?iu)^\s*я\s+просил\s+(?:тебя\s+переименовать|"
               r"переименовать\s+тебя)\s+(?:в|на)\s+(.+?)\s*[.!]*$"),
    re.compile(r"(?iu)^\s*(?:переименуй\s+себя|"
               r"поменяй\s+сво[её]\s+имя|"
               r"измени\s+сво[её]\s+имя|"
               r"назови\s+себя)\s+(?:в|на)\s+(.+?)\s*[.!]*$"),
    re.compile(r"(?iu)^\s*(?:теперь\s+)?(?:я\s+)?"
               r"(?:тебя\s+буду\s+звать|буду\s+звать\s+тебя|"
               r"тебя\s+зову|называйся)\s+(.+?)\s*[.!]*$"),
    re.compile(r"(?iu)^\s*(?:давайте\s+)?теперь\s+"
               r"(?:тебя\s+буду\s+звать|буду\s+тебя\s+звать)\s+(.+?)\s*[.!]*$"),
)
ENTITY_RENAME = (
    re.compile(r"(?iu)^\s*(?:переименуй|переименовать|поменять|"
               r"измени\s+название|поменяй\s+(?:имя|название))\s+"
               r"(?:(?:сущность|объект)\s+)?(.+?)\s+(?:в|на)\s+(.+?)\s*[.!]*$"),
    re.compile(r"(?iu)^\s*(.+?)\s+(?:теперь\s+называется|"
               r"теперь\s+зов[её]тся|переименован[ао]?\s+в)\s+(.+?)\s*[.!]*$"),
)
CHANGE_QUESTION = re.compile(
    r"(?iu)(?:почему\s+ты\s+(?:продолжаешь|говоришь)|"
    r"ты\s+же\s+(?:сказал|говорил)|"
    r"я\s+(?:же\s+)?просил\s+тебя\s+переименовать|"
    r"ты\s+(?:не\s+)?переименовал(?:ся)?|"
    r"опять\s+(?:старое|прежнее)\s+имя)"
)

def name_key(value):
    return normalize(str(value or "").strip())

def valid_name(value):
    candidate = str(value or "").strip().strip(" «»\"'„“”.,!\n\t")
    if not (2 <= len(candidate) <= 100) or len(candidate.split()) > 8:
        raise ValueError("Имя должно содержать от 2 до 100 символов и не более 8 слов")
    if not re.fullmatch(r"[\w\s\-.'А-Яа-яЁёІіЇїЄєҐґ]+", candidate, re.UNICODE):
        raise ValueError("Имя содержит неподходящие символы")
    if re.search(r"(?iu)\b(?:token|токен|пароль|password|bearer|secret)\b",candidate):
        raise ValueError("Секреты нельзя сохранять как имена")
    return candidate


class KnowledgeChanges:
    def __init__(self, engine, graph):
        self.engine=engine
        self.graph=graph
        with self.engine.connect() as con:
            con.executescript(SCHEMA)
            columns={row["name"] for row in con.execute(
                "PRAGMA table_info(knowledge_changes)")}
            if "conflict_node_id" not in columns:
                con.execute("ALTER TABLE knowledge_changes ADD COLUMN conflict_node_id INTEGER")

    @staticmethod
    def view(row):
        return dict(row) if row else None

    def _role_name(self,role):
        with self.engine.connect() as con:
            row=con.execute("SELECT display_name FROM dialogue_roles WHERE role=?",
                            (role,)).fetchone()
        return row["display_name"] if row else ""

    def _current(self, con, kind, tid):
        if kind=="role" and tid in ("assistant","user"):
            row=con.execute("SELECT display_name FROM dialogue_roles WHERE role=?",
                            (tid,)).fetchone()
            return row["display_name"] if row else None
        if kind=="entity" and str(tid).isdigit():
            row=con.execute("SELECT name FROM world_entities WHERE id=?",
                            (int(tid),)).fetchone()
            return row["name"] if row else None
        return None

    @staticmethod
    def _present_change(row):
        if row:
            row["conflict_detected"]=row.get("conflict_node_id") is not None
            row["requires_merge_approval"]=row["conflict_detected"]
        return row

    def snapshot(self, limit=20):
        with self.engine.connect() as con:
            pending=[self._present_change(dict(x)) for x in con.execute(
                "SELECT * FROM knowledge_changes WHERE status='pending' ORDER BY id DESC LIMIT ?",(limit,))]
            recent=[self._present_change(dict(x)) for x in con.execute(
                "SELECT * FROM knowledge_changes ORDER BY id DESC LIMIT ?",(limit,))]
            events=[dict(x) for x in con.execute(
                "SELECT * FROM knowledge_change_events ORDER BY id DESC LIMIT ?",(limit,))]
            aliases=[dict(x) for x in con.execute(
                "SELECT role,name FROM dialogue_role_aliases ORDER BY created_at")]
        return {"pending":pending,"recent":recent,"events":events,"role_aliases":aliases,
                "note":"Identity changes require explicit review; grades do not update attributes."}

    def _resolve_entity(self, con, label):
        target_key=name_key(label)
        rows=con.execute("""SELECT e.id,e.name,e.domain,
                GROUP_CONCAT(a.alias,'|') AS aliases
                FROM world_entities e LEFT JOIN world_aliases a ON a.entity_id=e.id
                GROUP BY e.id ORDER BY e.id""").fetchall()
        matches=[row for row in rows
                 if name_key(row["name"])==target_key or target_key in {
                    name_key(x) for x in (row["aliases"] or "").split("|") if x}]
        return matches

    def detect_request(self,text):
        phrase=str(text or "").strip()
        if not phrase or "?" in phrase or len(phrase)>300:
            return None
        # The teacher can add context and ask to restore an older name:
        # "Тебя раньше звали Джарвис ... переименуй себя в Джарвис обратно".
        # The final imperative is the instruction; the preface is evidence,
        # not a separate identity or an implicit confirmation.
        nested=re.search(
            r"(?iu)\bпереименуй\s+себя\s+(?:в|на)\s+"
            r"(?P<name>[\w.'-]{2,80})(?:\s+обратно)?\s*[.!]*$",phrase)
        if nested and not re.search(r"(?iu)\bне\s+переименуй\b",phrase):
            try:
                return {"target_kind":"role","target_id":"assistant",
                        "proposed_value":valid_name(nested["name"])}
            except ValueError:
                return None
        restore=re.fullmatch(
            r"(?iu)\s*верни\s+себе\s+(?:старое|прежнее)\s+имя\s*[.!]*",phrase)
        if restore:
            with self.engine.connect() as con:
                names=[row["name"] for row in con.execute(
                    "SELECT name FROM dialogue_role_aliases WHERE role='assistant'")]
            names=[n for n in names if name_key(n)!=name_key(
                self._role_name("assistant"))]
            if len(names)==1:
                return {"target_kind":"role","target_id":"assistant",
                        "proposed_value":names[0]}
            return {"error":"Уточни прежнее имя: "+", ".join(names)}
        for pat in SELF_RENAME:
            m=pat.match(phrase)
            if m:
                try:
                    return {"target_kind":"role","target_id":"assistant",
                            "proposed_value":valid_name(m.group(1))}
                except ValueError:
                    return None
        for pat in ENTITY_RENAME:
            m=pat.match(phrase)
            if m:
                try:
                    old=valid_name(m.group(1))
                    new=valid_name(m.group(2))
                except ValueError:
                    return None
                # Catch role self-renaming expressed by name rather than pronoun,
                # but only after exact matching to a confirmed assistant identity.
                return {"target_kind":"entity","target_name":old,
                        "proposed_value":new}
        return None

    def propose(self,source_text,target_kind,target_id=None,target_name=None,
                proposed_value="",source_kind="teacher_request"):
        if target_kind not in ("role","entity"):
            raise ValueError("Можно менять только атрибут существующей сущности")
        new=valid_name(proposed_value)
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if target_kind=="role":
                if target_id not in ("assistant","user"):
                    raise ValueError("Неизвестная роль")
                tid=str(target_id)
            else:
                if target_id is not None and str(target_id).isdigit():
                    tid=str(int(target_id))
                    if self._current(con,"entity",tid) is None:
                        raise ValueError("Целевая сущность не существует")
                else:
                    if not target_name:
                        raise ValueError("Не указана существующая сущность для переименования")
                    candidates=self._resolve_entity(con,target_name)
                    if len(candidates)!=1:
                        raise ValueError("Найдено "+str(len(candidates))+
                                         " совпадений для «"+str(target_name)[:90]+
                                         "». Укажи однозначную существующую сущность")
                    tid=str(candidates[0]["id"])
            old=self._current(con,target_kind,tid)
            if old is None:
                raise ValueError("Изменяемый объект ещё не существует")
            if name_key(old)==name_key(new):
                return {"changed":False,"already_current":True,
                        "target_kind":target_kind,"target_id":tid,
                        "expected_value":old,"proposed_value":old,
                        "message":"Это имя уже установлено."}
            # Explicit changes may not steal another distinct entity's name.
            conflict=con.execute("SELECT id FROM graph_nodes WHERE name_key=? AND active=1",
                                 (name_key(new),)).fetchone()
            old_node=con.execute("SELECT id FROM graph_nodes WHERE name_key=? AND active=1",
                                 (name_key(old),)).fetchone()
            conflict_id=(conflict["id"] if conflict
                         and (not old_node or conflict["id"]!=old_node["id"])
                         else None)
            # A conflicting graph name is a reviewable *merge proposal*, never
            # silently treated as a new entity or an unconditional error.
            duplicate=con.execute("""SELECT * FROM knowledge_changes
                  WHERE target_kind=? AND target_id=? AND status='pending'
                  AND expected_value=? AND proposed_value=?
                  ORDER BY id DESC LIMIT 1""",(target_kind,tid,old,new)).fetchone()
            if duplicate:
                return self._present_change(self.view(duplicate))
            cur=con.execute("""INSERT INTO knowledge_changes
                (target_kind,target_id,expected_value,proposed_value,
                 source_text,source_kind,conflict_node_id) VALUES(?,?,?,?,?,?,?)""",
                (target_kind,tid,old,new,str(source_text)[:3000],source_kind,conflict_id))
            row=con.execute("SELECT * FROM knowledge_changes WHERE id=?",
                            (cur.lastrowid,)).fetchone()
        return self._present_change(self.view(row))

    def from_text(self,text):
        cmd=self.detect_request(text)
        if not cmd:
            return None
        if cmd.get("error"):
            return {"error":cmd["error"],"intent":"change_knowledge"}
        try:
            return self.propose(text,**cmd)
        except ValueError as error:
            return {"error":str(error),"source_text":text,"intent":"change_knowledge"}

    def model_proposals(self,source_text,changes):
        """Guard an LLM extraction with a teacher's *explicit* update request
        and anchored target/new value. This cannot be invoked by web results.
        """
        if not isinstance(changes,list) or not changes:
            return []
        phrase=str(source_text or "").strip()
        if not phrase or "?" in phrase or not re.search(
            r"(?iu)\b(?:переимен|переназ|поменя|измени|назов|"
            r"теперь\s+зов|теперь\s+назыв|буду\s+звать|rename)\w*",phrase):
            return []
        saved=[]
        for raw in changes[:2]:
            if not isinstance(raw,dict):
                continue
            new=raw.get("new_name")
            if not isinstance(new,str) or name_key(new) not in name_key(phrase):
                continue
            target=raw.get("target")
            old=raw.get("old_name")
            kind="role" if target in ("assistant","user") else "entity"
            try:
                change=self.propose(phrase,kind,
                                    target_id=target if kind=="role" else None,
                                    target_name=old if kind=="entity" else None,
                                    proposed_value=new,source_kind="teacher_request_model_extracted")
                if change.get("id"):
                    saved.append(change)
            except ValueError:
                continue
        return saved

    def explain(self,text,roles):
        """A repeated contradiction is evidence of a pending/unapplied update.
        Never claim successful mutation if verified state is unchanged.
        """
        if not CHANGE_QUESTION.search(str(text or "")):
            return None
        with self.engine.connect() as con:
            row=con.execute("""SELECT * FROM knowledge_changes
                WHERE target_kind='role' AND target_id='assistant'
                ORDER BY id DESC LIMIT 1""").fetchone()
        if not row:
            return None
        current=roles.get("assistant","JARVIS")
        if row["status"]=="pending":
            return (f"Ты прав: я говорил о переименовании, но оно не было выполнено. "
                    f"Моё подтверждённое имя пока «{current}». "
                    f"Есть предложение #{row['id']} изменить его на "
                    f"«{row['proposed_value']}» — требуется твоё подтверждение.")
        if row["status"]=="confirmed":
            return f"Моё подтверждённое имя сейчас «{current}». Оно изменено и записано в SQLite."
        return f"Моё подтверждённое имя «{current}». Последнее предложение было отклонено или устарело."

    def _retarget_graph(self,con,old,new,merge_approved=False):
        oldkey,newkey=name_key(old),name_key(new)
        other=con.execute("SELECT id FROM graph_nodes WHERE name_key=?",(newkey,)).fetchone()
        previous=con.execute("SELECT id FROM graph_nodes WHERE name_key=?",(oldkey,)).fetchone()
        if other and (not previous or other["id"]!=previous["id"]):
            if not merge_approved:
                raise ValueError("Имя уже принадлежит другому графовому узлу. "
                                 "Необходимо явное подтверждение объединения.")
            if previous:
                # Teacher-approved merge: preserve previous node id and source
                # facts; graph edges are regenerated from audited world_facts.
                con.execute("DELETE FROM graph_edges WHERE subject_id=? OR object_id=?",
                            (other["id"],other["id"]))
                con.execute("DELETE FROM graph_nodes WHERE id=?",(other["id"],))
        if previous:
            con.execute("UPDATE graph_nodes SET name_key=?,name=? WHERE id=?",
                        (newkey,new,previous["id"]))
        # Retarget already-confirmed aliases from older historical names.
        con.execute("""UPDATE graph_equivalences
                       SET canonical_key=?,canonical_name=? WHERE canonical_key=?""",
                    (newkey,new,oldkey))
        # The new canonical name must not remain an alias of itself.
        con.execute("DELETE FROM graph_equivalences WHERE alias_key=?",(newkey,))
        # The formerly canonical name becomes a historical alias.
        con.execute("""INSERT INTO graph_equivalences
                     (alias_key,alias,canonical_key,canonical_name) VALUES(?,?,?,?)
                     ON CONFLICT(alias_key) DO UPDATE SET
                     alias=excluded.alias,canonical_key=excluded.canonical_key,
                     canonical_name=excluded.canonical_name""",
                    (oldkey,old,newkey,new))

    def review(self,change_id,decision,new_name=None,resolve_conflict=False):
        if decision not in ("confirm","reject"):
            raise ValueError("Неизвестное решение")
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row=con.execute("SELECT * FROM knowledge_changes WHERE id=?",
                            (change_id,)).fetchone()
            if not row or row["status"]!="pending":
                raise ValueError("Предложение не найдено или уже проверено")
            kind,tid=row["target_kind"],row["target_id"]
            old=row["expected_value"]
            current=self._current(con,kind,tid)
            changed_to=valid_name(new_name) if new_name else row["proposed_value"]
            if decision=="reject":
                status="rejected"
            elif current is None or name_key(current)!=name_key(old):
                status="stale"
            elif name_key(old)==name_key(changed_to):
                status="rejected"
            else:
                existing=con.execute("SELECT id FROM graph_nodes WHERE name_key=?",
                                     (name_key(changed_to),)).fetchone()
                old_graph=con.execute("SELECT id FROM graph_nodes WHERE name_key=?",
                                      (name_key(old),)).fetchone()
                collision=bool(existing and
                    (not old_graph or existing["id"]!=old_graph["id"]))
                if collision and not resolve_conflict:
                    raise ValueError("Имя уже используется другой сущностью в графе. "
                         "Подтверди отдельно объединение идентичностей.")
                self._retarget_graph(con,old,changed_to,
                                     merge_approved=resolve_conflict)
                if kind=="role":
                    con.execute("""INSERT INTO dialogue_role_aliases(role,name,name_key)
                         VALUES(?,?,?) ON CONFLICT(role,name_key) DO NOTHING""",
                         (tid,old,name_key(old)))
                    con.execute("DELETE FROM dialogue_role_aliases WHERE role=? AND name_key=?",
                                (tid,name_key(changed_to)))
                    con.execute("""UPDATE dialogue_roles
                        SET display_name=?,source='teacher_confirmed_change',
                            updated_at=CURRENT_TIMESTAMP WHERE role=?""",
                        (changed_to,tid))
                else:
                    # Preserve original stable row id and all historical aliases.
                    con.execute("UPDATE world_entities SET name=?,name_key=? WHERE id=?",
                                (changed_to,name_key(changed_to),int(tid)))
                    con.execute("""INSERT OR IGNORE INTO world_aliases(entity_id,alias,alias_key)
                                   VALUES(?,?,?)""",
                                (int(tid),old,name_key(old)))
                status="confirmed"
            con.execute("""UPDATE knowledge_changes SET status=?,reviewed_at=CURRENT_TIMESTAMP,
                           proposed_value=? WHERE id=?""",
                        (status,changed_to,change_id))
            con.execute("""INSERT INTO knowledge_change_events
                         (change_id,event,before_value,after_value,note)
                         VALUES(?,?,?,?,?)""",
                        (change_id,status,old,changed_to,
                         "Observed="+str(current)[:100]))
            if status=="confirmed":
                con.execute("""UPDATE knowledge_changes SET status='superseded',
                               reviewed_at=CURRENT_TIMESTAMP
                               WHERE target_kind=? AND target_id=? AND status='pending'
                                 AND id<>?""",(kind,tid,change_id))
            con.execute("""INSERT INTO apprentice_events(event_type,proposal_id,note)
                          VALUES(?,NULL,?)""",
                        ("knowledge_change_"+status,kind+":"+str(tid)+" "+old+" → "+changed_to))
        if status=="confirmed":
            self.graph.rebuild()
        result={"id":change_id,"status":status,"target_kind":kind,"target_id":tid,
                "merged_identity":bool(resolve_conflict and status=="confirmed"),
                "expected_value":old,"proposed_value":changed_to,
                "current_value":changed_to if status=="confirmed" else current,
                "executed":False}
        if status=="stale":
            result["message"]="Исходное значение изменилось. Сначала создай новое предложение."
        return result
