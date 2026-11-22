"""Dialogue context: speaker identity, teacher-approved speech corrections, runtime capabilities.

The research service records explicit user teaching in SQLite, not in model weights.
A candidate skill never becomes a runnable tool simply because the model says so.
"""
import os
import re

# Only these routes actually exist in the current gateway/research services.
# Never infer extra execution powers from descriptions in world_actions.
REAL_CAPABILITIES = (
    {"name": "knowledge.lookup", "status": "available",
     "description": "Читать подтверждённые знания через индекс SQLite.", "executor": "research.context"},
    {"name": "knowledge.propose", "status": "available",
     "description": "Предложить учителю знание на проверку; не исполняет действия.", "executor": "research.propose"},
    {"name": "knowledge.confirm", "status": "teacher_only",
     "description": "Подтверждать знания может только учитель.", "executor": None},
    {"name": "web.search", "status": "available",
     "description": "Поиск публичных источников через бесплатный SearXNG; только при разрешении в настройке.", "executor": "research.web.search"},
    {"name": "web.read_public", "status": "available",
     "description": "Ограниченное чтение публичных HTTPS страниц из результатов SearXNG; не даёт доступ к локальным сетям.", "executor": "research.web.search"},
    {"name": "google.ai_mode", "status": "teacher_only",
     "description": "Пользователь открывает Google AI Mode вручную в браузере и вставляет ответ.", "executor": None},
    {"name": "network.http", "status": "unavailable",
     "description": "Нет доступного модели клиента для внешних HTTP-запросов.", "executor": None},
    {"name": "code.execute", "status": "unavailable",
     "description": "Нет среды для запуска произвольного кода.", "executor": None},
    {"name": "external.action", "status": "unavailable",
     "description": "Нет коннектора для управления внешними системами.", "executor": None},
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS dialogue_roles (
 role TEXT PRIMARY KEY CHECK(role IN ('assistant','user')),
 display_name TEXT NOT NULL,
 source TEXT NOT NULL,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS speech_corrections (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 heard TEXT NOT NULL,
 heard_key TEXT NOT NULL UNIQUE,
 correct TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS runtime_capabilities (
 name TEXT PRIMARY KEY,
 status TEXT NOT NULL,
 description TEXT NOT NULL,
 executor TEXT
);
CREATE TABLE IF NOT EXISTS dialogue_role_aliases (
 role TEXT NOT NULL,
 name TEXT NOT NULL,
 name_key TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(role,name_key)
);
"""

NAME_CHARS = r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ'-]{0,35}"
INTRODUCTION = re.compile(
    rf"(?iu)\b(?:меня\s+зовут|мо[её]\s+имя|зови\s+меня|називай\s+мене|"
    rf"мене\s+звуть)\s+({NAME_CHARS})"
)
SELF_NAME_QUERY = re.compile(
    r"(?iu)\b(?:как\s+(?:меня\s+зовут|зовут\s+меня)|какое\s+у\s+меня\s+имя|"
    r"як\s+мене\s+звати|як\s+моє\s+ім.?я)\b"
)
ASSISTANT_NAME_QUERY = re.compile(
    r"(?iu)\b(?:как\s+(?:тебя\s+зовут|зовут\s+тебя)|какое\s+у\s+тебя\s+имя|"
    r"як\s+тебе\s+звати|як\s+твоє\s+ім.?я)\b"
)
ASSISTANT_NAME_CHOICE = re.compile(
    r"(?iu)\b(?:тебя\s+зовут\s+[^?.!]{2,100}\s+или\s+[^?.!]{2,100}|"
    r"(?:а\s+)?тебя\s+зовут\s+как|"
    r"как\s+тебя\s+теперь\s+зовут)\b"
)
TEACH_CORRECTION = re.compile(
    r"(?isu)\bкогда\s+я\s+говорю\s+(.+?)\s*,?\s*"
    r"(?:я\s+имею\s+в\s+виду|подразумеваю|это\s+значит)\s+(.+?)\s*[.!?]*$"
)


def extract_name(text):
    """Only an explicit *first-person* introduction updates the speaker role."""
    match = INTRODUCTION.search(text[:250])
    return match.group(1).capitalize() if match else None


def clean_phrase(value):
    return str(value).strip().strip(" «»\"„“”'.,!?\n\r\t").strip()


def key_for(value):
    return " ".join(value.casefold().split())


CAPABILITY_QUERY = re.compile(
    r"(?iu)\b(?:что\s+ты\s+(?:умеешь|можешь\s+делать)|"
    r"какие\s+действия\s+(?:ты\s+)?можешь\s+выполнять|"
    r"какие\s+у\s+тебя\s+(?:возможности|инструменты)|"
    r"на\s+что\s+ты\s+способен|what\s+can\s+you\s+do)\b"
)


class DialogueMemory:
    def __init__(self, engine):
        self.engine = engine
        with self.engine.connect() as con:
            con.executescript(SCHEMA)
            con.execute("""INSERT OR IGNORE INTO dialogue_roles(role,display_name,source)
                VALUES('assistant','JARVIS','app_identity')""")
            # This describes actual available routes. A saved lesson cannot enable one.
            for tool in REAL_CAPABILITIES:
                status=tool["status"]
                if (tool["name"].startswith("web.")
                        and os.getenv("WEB_SEARCH_ENABLED","true").lower()
                        not in ("1","true","yes")):
                    status="unavailable"
                con.execute("""INSERT INTO runtime_capabilities(name,status,description,executor)
                    VALUES(?,?,?,?) ON CONFLICT(name) DO UPDATE SET
                    status=excluded.status, description=excluded.description,
                    executor=excluded.executor""",
                    (tool["name"], status, tool["description"],
                     tool["executor"] if status=="available" else None))
            # Preserve correctly confirmed introductions from earlier versions.
            exists = con.execute("SELECT 1 FROM dialogue_roles WHERE role='user'").fetchone()
            if not exists:
                confirmed = con.execute("""SELECT name,source_text FROM apprentice_proposals
                    WHERE kind='entity' AND status='confirmed' ORDER BY id DESC LIMIT 150""")
                for item in confirmed:
                    parsed = extract_name(item["source_text"])
                    if parsed and key_for(parsed) == key_for(item["name"]):
                        con.execute("""INSERT INTO dialogue_roles(role,display_name,source)
                            VALUES('user',?,'confirmed_previous_introduction')""", (parsed,))
                        break

    def add_correction(self, heard, correct):
        heard, correct = clean_phrase(heard), clean_phrase(correct)
        if not (2 <= len(heard) <= 80 and 2 <= len(correct) <= 100):
            raise ValueError("Произношение и правильное название: от 2 до 80/100 символов")
        if key_for(heard) == key_for(correct):
            raise ValueError("Произношение совпадает с правильным названием")
        if not re.fullmatch(r"[\w\s\-']+", heard, re.UNICODE) or "\n" in heard:
            raise ValueError("Произношение должно содержать слова, а не команды или URL")
        if (re.search(r"(?iu)\b(?:token|токен|парол|password|secret|bearer|api.?key)\b",
                      heard + " " + correct) or "://" in correct or "@" in correct):
            raise ValueError("Не сохраняем токены, пароли и адреса как словарные замены")
        with self.engine.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if con.execute("SELECT COUNT(*) FROM speech_corrections").fetchone()[0] >= 100:
                existing = con.execute("SELECT 1 FROM speech_corrections WHERE heard_key=?",
                                       (key_for(heard),)).fetchone()
                if not existing:
                    raise ValueError("Лимит словаря: 100 исправлений")
            con.execute("""INSERT INTO speech_corrections(heard,heard_key,correct)
                VALUES(?,?,?) ON CONFLICT(heard_key) DO UPDATE SET
                heard=excluded.heard,correct=excluded.correct""",
                (heard,key_for(heard),correct))
        return {"heard":heard,"correct":correct,"saved":True,"executed":False}

    def delete_correction(self, item_id):
        with self.engine.connect() as con:
            row = con.execute("DELETE FROM speech_corrections WHERE id=? RETURNING id", (item_id,)).fetchone()
        if not row:
            raise ValueError("Исправление не найдено")
        return {"deleted":True,"id":item_id,"executed":False}

    def state(self):
        with self.engine.connect() as con:
            roles = {r["role"]:r["display_name"] for r in con.execute(
                "SELECT role,display_name FROM dialogue_roles")}
            aliases = [dict(r) for r in con.execute(
                "SELECT id,heard,correct FROM speech_corrections ORDER BY id DESC LIMIT 100")]
            tools = [dict(r) for r in con.execute(
                "SELECT name,status,description,executor FROM runtime_capabilities ORDER BY name")]
            role_aliases = [dict(r) for r in con.execute(
                "SELECT role,name FROM dialogue_role_aliases ORDER BY created_at")]
        return {"roles":roles,"speech_corrections":aliases,"capabilities":tools,
                "role_aliases":role_aliases}

    def normalize_speech(self, text):
        with self.engine.connect() as con:
            rows = con.execute("SELECT heard,correct FROM speech_corrections").fetchall()
        if not rows:
            return text
        # One pass: longest phrase first; replacements cannot cascade.
        mapping = {key_for(r["heard"]):r["correct"] for r in rows}
        parts = sorted((r["heard"] for r in rows), key=len, reverse=True)
        pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(p).replace(r"\ ",r"\s+") for p in parts) +
            r")(?!\w)", re.IGNORECASE
        )
        return pattern.sub(lambda match: mapping.get(key_for(match.group(0)), match.group(0)), text)

    def prepare(self, raw):
        taught = None
        match = TEACH_CORRECTION.search(raw.strip())
        if match:
            heard,correct = clean_phrase(match.group(1)),clean_phrase(match.group(2))
            try:
                taught = self.add_correction(heard,correct)
            except ValueError:
                pass  # Invalid natural-language mention is handled by ordinary chat.
            if taught:
                state = self.state()
                return {"text":raw,"direct_reply":f"Запомнил произношение: «{heard}» → «{correct}».",
                        "roles":state["roles"],"taught":taught}
        text = self.normalize_speech(raw)
        new_name = extract_name(text)
        if new_name:
            with self.engine.connect() as con:
                con.execute("""INSERT INTO dialogue_roles(role,display_name,source)
                    VALUES('user',?,'user_explicit_statement') ON CONFLICT(role)
                    DO UPDATE SET display_name=excluded.display_name,
                                  source=excluded.source,updated_at=CURRENT_TIMESTAMP""",
                    (new_name,))
            return {"text":text,"direct_reply":(
                        f"Приятно познакомиться, {new_name}. "
                        f"Меня зовут {self.state()['roles'].get('assistant','JARVIS')}."),
                    "roles":self.state()["roles"],"taught_name":new_name}
        roles = self.state()["roles"]
        compact = text.strip()
        # Identity questions are resolved from roles, not similarity-ranked nouns.
        if len(compact) < 100 and SELF_NAME_QUERY.search(compact):
            name = roles.get("user")
            return {"text":text, "roles":roles,
                    "direct_reply":f"Тебя зовут {name}." if name else
                    "Ты ещё не представился. Как тебя зовут?"}
        if len(compact) < 160 and (
                ASSISTANT_NAME_QUERY.search(compact)
                or ASSISTANT_NAME_CHOICE.search(compact)):
            return {"text":text,"roles":roles,
                    "direct_reply":f"Меня зовут {roles.get('assistant','JARVIS')}."}
        if len(compact)<180 and CAPABILITY_QUERY.search(compact):
            tools=self.state()["capabilities"]
            active=[t["description"] for t in tools if t["status"]=="available"]
            restricted=[t["name"] for t in tools if t["status"]=="unavailable"]
            answer="Мои доступные инструменты: "+(
                "; ".join(active) if active else "пока нет доступных инструментов"
            )+"."
            answer+=(" Не умею выполнять произвольные команды или управлять "
                     "устройствами: "+", ".join(restricted)+".")
            answer+=" Подтверждать новые знания может только учитель."
            return {"text":text,"roles":roles,"direct_reply":answer}
        return {"text":text,"roles":roles,"direct_reply":None}
