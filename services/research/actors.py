"""Stable people and device hints. Neither device ownership nor voice is authentication.

An actor ID represents a person, not their current display name. A device may
offer several profiles; each *turn* names its speaker explicitly. The existing
confirmed world graph remains authoritative and is never rewritten by a profile
selection. No private data is exposed on the strength of this selector.
"""
import re
import uuid
from .normalize import normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS actors (
 id TEXT PRIMARY KEY,
 display_name TEXT NOT NULL,
 name_key TEXT NOT NULL UNIQUE,
 graph_node_id INTEGER REFERENCES graph_nodes(id) ON DELETE SET NULL,
 origin TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS actor_devices (
 device_id TEXT PRIMARY KEY,
 label TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS actor_device_profiles (
 device_id TEXT NOT NULL REFERENCES actor_devices(device_id) ON DELETE CASCADE,
 actor_id TEXT NOT NULL REFERENCES actors(id) ON DELETE CASCADE,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(device_id,actor_id)
);
CREATE TABLE IF NOT EXISTS actor_session_history (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
 actor_id TEXT REFERENCES actors(id) ON DELETE SET NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
NAME = re.compile(r"^[\w\s.'’\-]{2,70}$",re.UNICODE)

class ActorRegistry:
    def __init__(self, engine):
        self.engine=engine
        with engine.connect() as con:
            con.executescript(SCHEMA)
            # Upgrade one-person legacy data when a confirmed introduction
            # restored the user's role. Do not infer Nadya or assign a shared
            # device just because it exists in the dialogue history.
            count=con.execute("SELECT COUNT(*) FROM actors").fetchone()[0]
            if not count:
                legacy=con.execute("""SELECT display_name,source FROM dialogue_roles
                                       WHERE role='user'""").fetchone()
                if legacy:
                    name=str(legacy["display_name"]).strip()
                    if NAME.fullmatch(name) and normalize(name) not in ("и","не","как"):
                        node=self._node(con,name)
                        con.execute("""INSERT INTO actors
                            (id,display_name,name_key,graph_node_id,origin)
                            VALUES(?,?,?,?,?)""",
                            (str(uuid.uuid4()),name,normalize(name),node,
                             "legacy_confirmed" if legacy["source"].startswith(
                              ("confirmed_","recovered_")) else "legacy_unverified"))

    @staticmethod
    def _node(con,name):
        found=con.execute("""SELECT id FROM graph_nodes WHERE name_key=?
                              AND active=1""",(normalize(name),)).fetchone()
        return found["id"] if found else None

    def list(self,device_id=""):
        with self.engine.connect() as con:
            people=[dict(row) for row in con.execute("""
                SELECT id,display_name,graph_node_id,origin,created_at
                FROM actors ORDER BY created_at,id""")]
            available=[]
            if device_id:
                available=[row["actor_id"] for row in con.execute(
                    "SELECT actor_id FROM actor_device_profiles WHERE device_id=?",
                    (device_id,))]
                label=con.execute("SELECT label FROM actor_devices WHERE device_id=?",
                                  (device_id,)).fetchone()
            else:
                label=None
        return {"people":people,"device":{"id":device_id,
                "label":label["label"] if label else "",
                "suggested_profiles":available},
                "identity_note":"Профиль или устройство — подсказка, не аутентификация."}

    def find(self,actor_id):
        if not actor_id:
            return None
        try:
            actor_id=str(uuid.UUID(str(actor_id)))
        except ValueError:
            return None
        with self.engine.connect() as con:
            row=con.execute("""SELECT id,display_name,graph_node_id,origin
                                FROM actors WHERE id=?""",(actor_id,)).fetchone()
        return dict(row) if row else None

    def create(self,name):
        name=" ".join(str(name or "").split()).strip()
        if not NAME.fullmatch(name) or name[0] in "-'.":
            raise ValueError("Имя должно быть от 2 до 70 символов, без команд и URL")
        key=normalize(name)
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            exists=con.execute("SELECT id FROM actors WHERE name_key=?",(key,)).fetchone()
            if exists:
                return dict(con.execute("SELECT id,display_name,graph_node_id,origin FROM actors WHERE id=?",(exists["id"],)).fetchone())
            # A claimed profile does not automatically become a confirmed
            # fact or a verified user account, even if a graph node exists.
            uid=str(uuid.uuid4())
            con.execute("""INSERT INTO actors
                (id,display_name,name_key,graph_node_id,origin)
                VALUES(?,?,?,NULL,'manually_selected_unverified')""",(uid,name,key))
        return self.find(uid)

    def link_existing_entity(self,actor_id,node_id,confirmed=False):
        """Never mark a claim as authenticated based on a client-supplied flag.

        The mapping can help answer explicitly contextual questions, but is
        labelled *unverified*. It is not a substitute for person login.
        """
        if not self.find(actor_id):
            raise ValueError("Профиль человека не найден")
        with self.engine.connect() as con:
            node=con.execute("SELECT id FROM graph_nodes WHERE id=? AND active=1",
                             (node_id,)).fetchone()
            if not node:
                raise ValueError("Подтверждённая сущность не найдена")
            con.execute("""UPDATE actors SET graph_node_id=?,
                  origin='manual_graph_link_unverified' WHERE id=?""",
                  (node_id,actor_id))
        return self.find(actor_id)

    def add_device(self,device_id,actor_id,label):
        if not self.find(actor_id):
            raise ValueError("Профиль не найден")
        try:
            device_id=str(uuid.UUID(str(device_id)))
        except ValueError as error:
            raise ValueError("Нужен корректный идентификатор устройства") from error
        label=" ".join(str(label or "").split())[:70]
        if not label:
            label="Устройство"
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("""INSERT INTO actor_devices(device_id,label)
                        VALUES(?,?) ON CONFLICT(device_id) DO UPDATE SET
                        label=excluded.label""",(device_id,label))
            con.execute("""INSERT OR IGNORE INTO actor_device_profiles(device_id,actor_id)
                        VALUES(?,?)""",(device_id,actor_id))
        return {"device_id":device_id,"actor_id":actor_id,"label":label,
                "authenticated":False,"hint_only":True}

    def record_turn_speaker(self,session_id,actor_id):
        actor=self.find(actor_id)
        if not actor:
            return None
        with self.engine.connect() as con:
            con.execute("""INSERT INTO actor_session_history(session_id,actor_id)
                           VALUES(?,?)""",(session_id,actor["id"]))
        return actor
