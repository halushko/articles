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

    def test_teacher_can_edit_model_extraction(self):
        p=self.memory.propose("Млкий",[{"kind":"entity","name":"Млкий",
                                    "description":"Неудачная запись"}])[0]
        self.memory.review(p["id"],"confirm",name="Мелкий",description="Аккумулятор")
        entity=self.memory.snapshot()["knowledge"]["entity"][0]
        self.assertEqual(entity["name"],"Мелкий")
        self.assertEqual(entity["description"],"Аккумулятор")


class ApprenticeApiTests(unittest.TestCase):
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
