"""Safety boundary for requested operations.

A recognized imperative is NOT an executable tool call. The world graph can
describe a capability but only runtime executors may act. This function
separates device and destructive requests from normal informational questions.
The gate never performs a side effect, so future executors can require their
own typed plan and purpose-bound approval.
"""
import re

from .utterance import classify

# Conjugations explicitly selected from verbs describing external actions,
# not free-form substring matches (to avoid blocking a question about an API).
PHYSICAL = re.compile(
    r"(?iu)\b(?:включи|выключи|отключи|переключи|"
    r"открой|закрой|запусти|останови|разблокируй|"
    r"увімкни|вимкни|відкрий|закрий|запусти|зупини)\b"
)
TARGET = re.compile(
    r"(?iu)\b(?:свет|світло|лампу|лампы|светильник|"
    r"розетк\w*|бойлер|котел|газ|генератор|"
    r"инвертор|інвертор|аккумулятор|акумулятор|"
    r"ворота|дверь|двери|двері|замок|"
    r"насос|обогревател\w*|опалення|отопление|"
    r"home\s*assistant|ha)\b"
)
DESTRUCTIVE = re.compile(
    r"(?iu)\b(?:удали|сотри|очисти|уничтожь|забудь|"
    r"знищ|видали|очисть)\b"
)
DATA = re.compile(
    r"(?iu)\b(?:всю|все|всі|весь|память|пам'ять|"
    r"знания|знання|историю|історію|данные|дані|"
    r"базу|таблиц\w*|файл\w*|запис\w*|сущност\w*|"
    r"архив|архів)\b"
)

def request_without_executor(text):
    parsed=classify(text)
    if parsed["intent"]!="command":
        return None
    body=parsed["normalized"]
    if PHYSICAL.search(body) and TARGET.search(body):
        return {"kind":"device.control","risk":"high",
                "reason":"no_device_executor",
                "requires_approval":True,"executable":False}
    if DESTRUCTIVE.search(body) and DATA.search(body):
        return {"kind":"data.delete","risk":"high",
                "reason":"no_destructive_executor",
                "requires_approval":True,"executable":False}
    return None
