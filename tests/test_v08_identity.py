"""v0.8 regression: durable identity, shared devices, semantic safety.

Tests also exercise recovery of real-world legacy teaching ("Привет ...
Дима а тебя") without overwriting source proposals or model weights.
"""
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from services.research.engine import Engine
from services.research.settings import load
from services.research.apprentice import ApprenticeMemory
from services.research.actors import ActorRegistry
from services.research.sessions import ConversationStore
from services.research.review_voice import VoiceReviews
from services.research.safety import request_without_executor
from services.research.api import app as research_app
from services.gateway import api as gateway


class SharedSpeakerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.engine=Engine(str(Path(self.temp.name)/"speakers.sqlite3"),load())
        self.memory=ApprenticeMemory(self.engine)
        original=self.memory.propose("Привет Меня зовут Дима а тебя",[
            {"kind":"entity","name":"Дима","description":"человек, с которым я общаюсь",
             "domain":"общение"}])
        self.memory.review(original[0]["id"],"confirm")
        # Old v0.7 deleted a corrupt singleton name "И" and never restored it.
        with self.engine.connect() as con:
            con.execute("DELETE FROM dialogue_roles WHERE role='user'")
        self.memory=ApprenticeMemory(self.engine)
        self.sessions=ConversationStore(self.engine)
        self.people=ActorRegistry(self.engine)

    def tearDown(self):
        self.temp.cleanup()

    def test_legacy_confirmed_identity_and_graph_link_recovered(self):
        self.assertEqual(self.memory.dialogue.state()["roles"]["user"],"Дима")
        people=self.people.list()["people"]
        self.assertEqual(len(people),1)
        self.assertEqual(people[0]["display_name"],"Дима")
        self.assertIsNotNone(people[0]["graph_node_id"])
        self.assertIn("Дима",self.memory.context("Как меня зовут",
                       speaker=self.people.find(people[0]["id"]))["direct_reply"])
        self.assertEqual(self.memory.dialogue.state()["roles"]["user"],"Дима")

    def test_two_people_same_device_distinct_per_turn_identities(self):
        dima=self.people.list()["people"][0]
        nadya=self.people.create("Надя")
        sid=self.sessions.create("Дима и Надя")["id"]
        device_id=str(uuid.uuid4())
        self.people.add_device(device_id,dima["id"],"Планшет")
        self.people.add_device(device_id,nadya["id"],"Планшет")
        self.assertEqual(set(self.people.list(device_id)["device"]["suggested_profiles"]),
                         {dima["id"],nadya["id"]})
        self.assertEqual(self.memory.context("Как меня зовут",
                           speaker=self.people.find(dima["id"]))["direct_reply"],"Тебя зовут Дима.")
        self.assertEqual(self.memory.context("Как меня зовут",
                           speaker=self.people.find(nadya["id"]))["direct_reply"],"Тебя зовут Надя.")
        self.assertIn("не выбран собеседник",self.memory.context("Как меня зовут",
                           speaker={"display_name":None})["direct_reply"])
        # A selected person cannot overwrite the old singleton by simply saying
        # "Меня зовут Надя" into a shared microphone.
        self.memory.context("Меня зовут Надя",speaker=self.people.find(nadya["id"]))
        self.assertEqual(self.memory.dialogue.state()["roles"]["user"],"Дима")
        for uid in (dima["id"],nadya["id"]):
            self.sessions.save_turn(sid,"Привет","Здравствуйте","apprentice","fast",
                                     "combined","ask",{"route":"sqlite"},actor_id=uid)
            self.people.record_turn_speaker(sid,uid)
        self.assertEqual([t["actor_id"] for t in self.sessions.read(sid)["turns"]],
                         [dima["id"],nadya["id"]])
        self.assertEqual(self.people.create("Надя")["id"],nadya["id"])

    def test_voice_memory_review_rejects_wrong_speaker(self):
        dima=self.people.list()["people"][0]["id"]
        nadya=self.people.create("Надя")["id"]
        sid=self.sessions.create("Один планшет")["id"]
        voice=VoiceReviews(self.engine,self.memory,self.sessions)
        proposal=self.memory.context("Переименуй себя в Домик")["change_proposal"]
        voice.queue(sid,changes=[proposal],actor_id=dima)
        denied=voice.resolve(sid,"Подтверждаю",actor_id=nadya)
        self.assertEqual(denied["status"],"wrong_speaker")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],
                         "Меня зовут JARVIS.")
        permitted=voice.resolve(sid,"Подтверждаю",actor_id=dima)
        self.assertEqual(permitted["status"],"confirmed")

    def test_safe_operation_classification(self):
        for phrase,kind in (
            ("выключи свет","device.control"),
            ("открой дверь","device.control"),
            ("Удали всю память","data.delete"),
        ):
            operation=request_without_executor(phrase)
            self.assertIsNotNone(operation,phrase)
            self.assertEqual(operation["kind"],kind)
            self.assertFalse(operation["executable"])
        for phrase in ("Как включить свет","Вопрос как удалить память",
                       "Расскажи о бойлере"):
            self.assertIsNone(request_without_executor(phrase),phrase)


class ActorApiTests(unittest.TestCase):
    def test_identity_endpoints_preserve_legacy_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ",{"DATABASE_PATH":str(Path(tmp)/"api.sqlite3")}):
                with TestClient(research_app) as client:
                    dima=client.post("/research/actors",json={"display_name":"Дима"})
                    self.assertEqual(dima.status_code,200)
                    person=dima.json()
                    self.assertEqual(client.get("/research/actors").json()["people"][0]["id"],
                                     person["id"])
                    device=str(uuid.uuid4())
                    response=client.post("/research/actors/device",json={
                        "device_id":device,"actor_id":person["id"],"label":"Часы"})
                    self.assertEqual(response.status_code,200)
                    self.assertFalse(response.json()["authenticated"])
                    sid=client.post("/research/sessions",json={}).json()["id"]
                    response=client.post("/research/sessions/"+sid+"/turns",json={
                        "question":"Как меня зовут","answer":"Тебя зовут Дима.",
                        "experiment":"apprentice","profile":"fast","ablation":"combined",
                        "web_mode":"off","actor_id":person["id"],
                        "result":{"route":"test"}})
                    self.assertEqual(response.status_code,200)
                    self.assertEqual(client.get("/research/sessions/"+sid).json()
                                     ["turns"][0]["actor_id"],person["id"])
                    intent=client.post("/apprentice/intent",json={
                        "text":"вопрос как меня зовут"}).json()
                    self.assertEqual(intent["intent"],"question")


class SafetyGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_dangerous_command_is_never_a_qwen_tool_call(self):
        seen=[]
        async def forward(url,method="GET",payload=None):
            seen.append(url)
            if url.endswith("/apprentice/context"):
                return {"normalized_text":"выключи свет",
                        "utterance_intent":{"intent":"command","confidence":"high"},
                        "unavailable_operation":{
                            "kind":"device.control","reason":"no_device_executor",
                            "risk":"high","executable":False}}
            raise AssertionError(url)
        with patch.object(gateway,"forward",forward):
            response=await gateway.apprentice_chat(gateway.ChatIn(
                text="выключи свет",experiment="apprentice",web_mode="ask"),0)
        self.assertEqual(response["route"],"apprentice_action_unavailable")
        self.assertFalse(response["executed"])
        self.assertEqual(len(seen),1)


if __name__=="__main__":
    unittest.main()
