import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from services.research.engine import Engine
from services.research.settings import load
from services.research.apprentice import ApprenticeMemory
from services.research.api import app as research_app
from services.gateway import api as gateway


class ApprenticeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = Engine(str(Path(self.tmp.name) / "test.sqlite3"), load())
        self.memory = ApprenticeMemory(self.engine)

    def tearDown(self):
        self.tmp.cleanup()

    def test_proposals_are_not_facts_until_confirmed(self):
        p = self.memory.propose("Мелкий это аккумулятор", [{
            "kind":"entity", "name":"Мелкий", "domain":"дом",
            "description":"Аккумулятор, стоящий в гараже",
            "details":{"entity_type":"аккумулятор","aliases":["Мелкого","Млкий"]}}])[0]
        self.assertEqual(self.memory.snapshot()["counts"]["pending"],1)
        self.assertFalse(self.memory.context("Мелкого")["knowledge"])
        self.memory.review(p["id"],"confirm")
        result = self.memory.context("Мелкого")["knowledge"]
        self.assertTrue(result)
        self.assertEqual(result[0]["name"],"Мелкий")
        self.assertIn("Млкий",result[0]["aliases"])
        self.assertEqual(self.memory.snapshot()["counts"]["entity"],1)

    def test_reject_and_duplicate_are_safe(self):
        p={"kind":"entity","name":"А","description":"Учитель показал А"}
        first=self.memory.propose("А это объект",[p])[0]
        self.assertEqual(first["id"],self.memory.propose("А это объект",[p])[0]["id"])
        self.memory.review(first["id"],"reject")
        self.assertEqual(self.memory.snapshot()["counts"]["entity"],0)
        with self.assertRaises(ValueError):
            self.memory.review(first["id"],"confirm")

    def test_lessons_can_be_revised_and_queried(self):
        lesson={"kind":"lesson","name":"Сортировка массива","domain":"программирование",
                "description":"Обменивать соседние элементы до завершения",
                "details":{"steps":["Сравнить соседние","Обменять при необходимости"],
                           "examples":["[3,1,2] → [1,2,3]"]}}
        p=self.memory.propose("Я учу тебя сортировке",[lesson])[0]
        self.memory.review(p["id"],"confirm")
        results=self.memory.context("Как отсортировать массив?")["knowledge"]
        self.assertTrue(any(x["kind"]=="lesson" for x in results))
        lesson_rows=self.memory.snapshot()["knowledge"]["lesson"]
        self.assertEqual(lesson_rows[0]["steps"][0],"Сравнить соседние")
        self.assertEqual(lesson_rows[0]["examples"][0],"[3,1,2] → [1,2,3]")

    def test_action_is_not_executable(self):
        p=self.memory.propose("Научи операции",[
            {"kind":"action","name":"Сделать HTTP-запрос",
             "description":"Запрос по сети","details":{"parameters":{"url":"string"}}},
            {"kind":"capability","name":"Управлять домом",
             "description":"Гипотетический навык будущего"},
        ])
        for item in p:
            self.memory.review(item["id"],"confirm")
        state=self.memory.snapshot()
        self.assertIsNone(state["knowledge"]["action"][0]["executor"])
        self.assertEqual(state["counts"]["capability"],1)
        self.assertEqual(self.memory.context("HTTP запрос")["execution_tools"],[])

    def test_invalid_facts_not_inserted(self):
        r=self.memory.propose("Какая погода?",[
            {"kind":"fact","name":"Погода","description":"Дождь"},
            {"kind":"bad","name":"F","description":"not a type"}])
        self.assertEqual(r,[])

    def test_teacher_grades_are_evidence_not_training(self):
        result=self.memory.grade("Сортируй [2,1]","[1,2]","good")
        self.assertTrue(result["saved"])
        self.assertEqual(self.memory.snapshot()["counts"]["graded"],1)
        self.assertEqual(self.memory.snapshot()["counts"]["graded_good"],1)
        self.assertEqual(self.memory.snapshot()["counts"]["lesson"],0)
        with self.assertRaises(ValueError):
            self.memory.grade("q","a","incorrect")

    def test_explicit_retraction_preserves_audit(self):
        p=self.memory.propose("Z объект",[
            {"kind":"entity","name":"Z","description":"Объект для забвения"}])[0]
        self.memory.review(p["id"],"confirm")
        self.assertEqual(self.memory.snapshot()["counts"]["entity"],1)
        self.memory.forget("entity",1)
        state=self.memory.snapshot()
        self.assertEqual(state["counts"]["entity"],0)
        self.assertFalse(self.memory.context("Z")["knowledge"])
        self.assertEqual(state["events"][0]["event_type"],"retracted")
        with self.engine.connect() as con:
            status=con.execute("SELECT status FROM apprentice_proposals WHERE id=?",
                               (p["id"],)).fetchone()["status"]
        self.assertEqual(status,"retracted")

    def test_dialogue_roles_use_explicit_teacher_introduction(self):
        self.assertEqual(self.memory.context("Как тебя зовут?")["direct_reply"],
                         "Меня зовут JARVIS.")
        self.assertIn("не представился",
                      self.memory.context("А как зовут меня?")["direct_reply"])
        first=self.memory.context("Привет, меня зовут Дима, а тебя?")
        self.assertIn("Дима",first["direct_reply"])
        self.assertEqual(self.memory.context("А как зовут меня?")["direct_reply"],
                         "Тебя зовут Дима.")
        self.assertEqual(self.memory.snapshot()["dialogue"]["roles"]["user"],"Дима")
        self.assertEqual(self.memory.snapshot()["counts"]["entity"],0)

    def test_learn_pronunciation_with_explicit_correction(self):
        phrase="Когда я говорю «хомо сестранте», я имею в виду Home Assistant"
        lesson=self.memory.context(phrase)
        self.assertIn("Запомнил произношение",lesson["direct_reply"])
        corrected=self.memory.context("У меня система на хомо сестранте")
        self.assertEqual(corrected["normalized_text"],
                         "У меня система на Home Assistant")
        self.assertEqual(self.memory.snapshot()["dialogue"]["speech_corrections"][0]["correct"],
                         "Home Assistant")
        with self.assertRaises(ValueError):
            self.memory.dialogue.add_correction("секрет","Bearer token")
        item=self.memory.snapshot()["dialogue"]["speech_corrections"][0]
        self.memory.dialogue.delete_correction(item["id"])
        self.assertEqual(self.memory.context("хомо сестранте")["normalized_text"],
                         "хомо сестранте")

    def test_tool_registry_distinguishes_ideas_from_execution(self):
        tools={x["name"]:x for x in self.memory.context("Подключись к API")["capabilities"]}
        self.assertEqual(tools["knowledge.lookup"]["status"],"available")
        self.assertEqual(tools["knowledge.confirm"]["status"],"teacher_only")
        self.assertEqual(tools["network.http"]["status"],"unavailable")
        self.assertIsNone(tools["network.http"]["executor"])
        self.assertEqual(self.memory.context("Подключись к API")["execution_tools"],[])

    def test_teacher_can_edit_model_extraction(self):
        p=self.memory.propose("Млкий",[{"kind":"entity","name":"Млкий",
                                    "description":"Неудачная запись"}])[0]
        self.memory.review(p["id"],"confirm",name="Мелкий",description="Аккумулятор")
        entity=self.memory.snapshot()["knowledge"]["entity"][0]
        self.assertEqual(entity["name"],"Мелкий")
        self.assertEqual(entity["description"],"Аккумулятор")


class ApprenticeApiTests(unittest.TestCase):
    def test_http_pronunciation_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ,{"DATABASE_PATH":str(Path(tmp)/"speech.sqlite3")}):
                with TestClient(research_app) as client:
                    r=client.post("/apprentice/speech-corrections",json={
                        "heard":"хоум ассистент","correct":"Home Assistant"})
                    self.assertEqual(r.status_code,200)
                    state=client.get("/apprentice/state").json()
                    self.assertEqual(len(state["dialogue"]["speech_corrections"]),1)
                    self.assertEqual(
                        client.post("/apprentice/context",json={
                          "query":"Система хоум ассистент"}).json()["normalized_text"],
                        "Система Home Assistant")
                    item=state["dialogue"]["speech_corrections"][0]
                    r=client.post("/apprentice/speech-corrections/delete",
                                  json={"item_id":item["id"]})
                    self.assertEqual(r.status_code,200)

    def test_http_review_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ,{"DATABASE_PATH":str(Path(tmp)/"research.sqlite3")}):
                with TestClient(research_app) as client:
                    proposal=client.post("/apprentice/propose",json={
                        "source_text":"Есть объект Z","proposals":[
                            {"kind":"entity","name":"Z","description":"Объект Z"}]
                    }).json()["proposals"][0]
                    self.assertEqual(client.get("/apprentice/state").json()["counts"]["entity"],0)
                    response=client.post("/apprentice/review",json={
                        "proposal_id":proposal["id"],"decision":"confirm"})
                    self.assertEqual(response.status_code,200)
                    self.assertEqual(client.get("/apprentice/state").json()["counts"]["entity"],1)


class ApprenticeGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_identity_faq_bypasses_llm(self):
        calls=[]
        async def fake_forward(url,method="GET",payload=None):
            calls.append(url)
            if "/context" in url:
                return {"normalized_text":"Как меня зовут",
                        "direct_reply":"Тебя зовут Дима."}
            raise AssertionError("No model call should be needed for role identity")
        with patch.object(gateway,"forward",fake_forward):
            result=await gateway.chat(gateway.ChatIn(
                text="Как меня зовут",experiment="apprentice"))
        self.assertEqual(result["route"],"apprentice_confirmed_context")
        self.assertEqual(result["answer"],"Тебя зовут Дима.")
        self.assertEqual(result["tokens"],0)
        self.assertEqual(len(calls),1)

    async def test_connection_request_reports_missing_tool_without_llm(self):
        calls=[]
        async def fake_forward(url,method="GET",payload=None):
            calls.append(url)
            if "/context" in url:
                return {"normalized_text":"Я хочу чтобы ты подключился к Home Assistant",
                        "capabilities":[
                            {"name":"network.http","status":"unavailable"},
                            {"name":"external.action","status":"unavailable"}]}
            raise AssertionError("No model call needed when HTTP tool is unavailable")
        with patch.object(gateway,"forward",fake_forward):
            r=await gateway.chat(gateway.ChatIn(
                text="Я хочу чтобы ты подключился к Home Assistant",
                experiment="apprentice"))
        self.assertEqual(r["route"],"apprentice_capability_boundary")
        self.assertIn("нет HTTP-инструмента",r["answer"])
        self.assertEqual(len(calls),1)
        self.assertFalse(r["executed"])

    async def test_fake_api_claim_is_corrected_and_proposal_not_saved(self):
        calls=[]
        async def fake_forward(url,method="GET",payload=None):
            calls.append(url)
            if "/context" in url:
                return {"knowledge":[],"execution_tools":[]}
            if "/analyze" in url:
                return {"reply":"Я запросил доступ к API вашего SmartHome по URL.",
                        "proposals":[{"kind":"lesson","name":"Подключение к API",
                                      "description":"Я это сделал"}],
                        "missing":"","tokens":22}
            raise AssertionError("A hallucinated lesson must never be saved")
        with patch.object(gateway,"forward",fake_forward):
            r=await gateway.chat(gateway.ChatIn(
                text="https://example.org",experiment="apprentice"))
        self.assertIn("не подключался",r["answer"])
        self.assertEqual(r["proposals"],[])
        self.assertFalse(r["executed"])
        self.assertEqual(len(calls),2)
        self.assertIsNotNone(r["integrity_warning"])

    async def test_pipeline_passes_only_confirmed_memory_and_never_executes(self):
        calls=[]
        async def fake_forward(url,method="GET",payload=None):
            calls.append((url,method,payload))
            if "/context" in url:
                return {"knowledge":[],"execution_tools":[]}
            if "/analyze" in url:
                self.assertEqual(payload["text"],"Это новый язык")
                return {"reply":"Что это за язык?","intent":"teach","missing":"описание",
                        "model":"qwen2.5:3b","tokens":44,
                        "proposals":[{"kind":"entity","name":"новый язык",
                                      "description":"Язык из сообщения"}]}
            if "/propose" in url:
                return {"proposals":[{"id":7,"kind":"entity","name":"новый язык",
                                      "status":"pending"}]}
            raise AssertionError(url)
        with patch.object(gateway,"forward",fake_forward):
            r=await gateway.chat(gateway.ChatIn(
                text="Это новый язык",experiment="apprentice"))
        self.assertEqual(r["route"],"apprentice_tutor")
        self.assertEqual(r["proposals"][0]["id"],7)
        self.assertFalse(r["executed"])
        self.assertEqual(len(calls),3)
        self.assertEqual(calls[0][0].endswith("/context"),True)


if __name__=="__main__":
    unittest.main()
