"""JARVIS v0.7 regression tests: speech intent and session-scoped reviews."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from services.research.engine import Engine
from services.research.settings import load
from services.research.apprentice import ApprenticeMemory
from services.research.sessions import ConversationStore
from services.research.review_voice import VoiceReviews, speech_intent
from services.research.utterance import classify
from services.research.api import app as research_app
from services.gateway import api as gateway


class UtteranceRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.engine=Engine(str(Path(self.temp.name)/"research.sqlite3"),load())
        self.memory=ApprenticeMemory(self.engine)

    def tearDown(self):
        self.temp.cleanup()

    def test_asr_questions_and_inversions_never_become_user_names(self):
        self.memory.context("Меня зовут Дима")
        utterances=(
            "Как меня зовут и тебя зовут",
            "вопрос как меня зовут и Как тебя зовут",
            "Как тебя зовут и Как меня зовут",
            "тебя зовут и Как меня зовут",
            "меня зовут и Как тебя зовут",
            "Как меня завут и тебя завут",
            "Вопрос: как тебя завут",
            "Как тебя зовут",
        )
        for phrase in utterances:
            result=self.memory.context(phrase)
            self.assertIn("дом",result["direct_reply"].casefold()
                          if False else result["direct_reply"].replace("JARVIS","дом"))
            self.assertNotIn("Приятно познакомиться",result["direct_reply"])
            self.assertEqual(self.memory.dialogue.state()["roles"]["user"],"Дима")
            self.assertEqual(result["utterance_intent"]["intent"],"question")
        for phrase in utterances[:6]:
            reply=self.memory.context(phrase)["direct_reply"]
            self.assertIn("Тебя зовут Дима.",reply)
            self.assertIn("Меня зовут JARVIS.",reply)

    def test_no_question_mark_does_not_grant_teaching_permission(self):
        for phrase in ("Сколько у меня устройств",
                       "Зачем нужны API",
                       "Вопрос: Меня зовут Дима",
                       "Как меня завут",
                       "ты можешь ли включить свет"):
            observed=classify(phrase)
            self.assertEqual(observed["intent"],"question",phrase)
            self.assertFalse(observed["allow_memory_proposal"])
        for phrase in ("Переименуй себя в Домик",
                       "Команда: переименуй себя в Домик",
                       "Тебя раньше звали Джарвис сейчас Домик переименуй себя в Джарвис обратно"):
            self.assertEqual(classify(phrase)["intent"],"command")
        self.assertEqual(classify("Утверждение: меня зовут Дима")["intent"],"statement")

    def test_invalid_question_and_corrections_do_not_poison_roles(self):
        self.assertIsNone(self.memory.dialogue.prepare("Вопрос: меня зовут И") .get("taught_name"))
        self.assertNotIn("user",self.memory.dialogue.state()["roles"])
        self.assertEqual(self.memory.dialogue.prepare("Утверждение: Меня зовут Надя")["taught_name"],"Надя")
        self.assertEqual(self.memory.context("Вопрос: Как меня завут")["direct_reply"],"Тебя зовут Надя.")
        proposals=self.memory.propose("Вопрос как работает API",[
            {"kind":"lesson","name":"Неверный урок","description":"Вопрос не урок"}])
        self.assertFalse(proposals)

    def test_legacy_bad_role_is_removed_on_restart(self):
        with self.engine.connect() as con:
            con.execute("""INSERT INTO dialogue_roles(role,display_name,source)
                          VALUES('user','И','user_explicit_statement')""")
        reopened=ApprenticeMemory(self.engine)
        self.assertNotIn("user",reopened.dialogue.state()["roles"])
        self.assertEqual(reopened.context("Как меня зовут")["direct_reply"],
                         "Ты ещё не представился.")


class NameCollisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.engine=Engine(str(Path(self.tmp.name)/"graph.sqlite3"),load())
        self.memory=ApprenticeMemory(self.engine)

    def tearDown(self):
        self.tmp.cleanup()

    def add_entity(self,name):
        proposal=self.memory.propose("Имя "+name,[{
            "kind":"entity","name":name,"description":"Подтверждённое имя",
            "domain":"person"}])[0]
        self.memory.review(proposal["id"],"confirm")

    def test_rename_back_to_historical_name_after_explicit_review(self):
        self.add_entity("JARVIS")
        self.add_entity("Джарвис")
        graph=self.memory.graph.snapshot()
        old_id=next(x["id"] for x in graph["nodes"] if x["name"]=="JARVIS")
        first=self.memory.context("Переименуй себя в Домик")["change_proposal"]
        self.assertEqual(self.memory.changes.review(first["id"],"confirm")["status"],"confirmed")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],"Меня зовут Домик.")
        result=self.memory.context("Переименуй себя в Джарвис")
        proposed=result["change_proposal"]
        self.assertTrue(proposed["requires_merge_approval"])
        self.assertIn("отдельной сущности",result["direct_reply"])
        with self.assertRaisesRegex(ValueError,"объединен"):
            self.memory.changes.review(proposed["id"],"confirm")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],"Меня зовут Домик.")
        merged=self.memory.changes.review(proposed["id"],"confirm",
                                          resolve_conflict=True)
        self.assertEqual(merged["status"],"confirmed")
        self.assertTrue(merged["merged_identity"])
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],"Меня зовут Джарвис.")
        snapshot=self.memory.graph.snapshot()
        node=next(x for x in snapshot["nodes"] if x["name"]=="Джарвис")
        self.assertEqual(node["id"],old_id)
        self.assertIn("JARVIS",node["aliases"])
        self.assertIn("Домик",node["aliases"])
        self.assertEqual(len(snapshot["nodes"]),1)
        self.assertEqual(self.memory.snapshot()["counts"]["entity"],2)

    def test_contextual_rename_command_is_not_a_new_entity(self):
        self.add_entity("JARVIS")
        initial=self.memory.context("Переименуй себя в Домик")["change_proposal"]
        self.memory.changes.review(initial["id"],"confirm")
        proposal=self.memory.context(
            "тебя раньше звали Джарвис сейчас Домик переименуй себя в JARVIS обратно"
        )["change_proposal"]
        self.assertEqual(proposal["target_kind"],"role")
        self.assertEqual(proposal["proposed_value"],"JARVIS")
        self.assertEqual(proposal["status"],"pending")
        self.memory.changes.review(proposal["id"],"confirm")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],
                         "Меня зовут JARVIS.")


class VoiceReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.engine=Engine(str(Path(self.temp.name)/"speech.sqlite3"),load())
        self.memory=ApprenticeMemory(self.engine)
        self.sessions=ConversationStore(self.engine)
        self.voice=VoiceReviews(self.engine,self.memory,self.sessions)
        self.sid=self.sessions.create("голосовые проверки")["id"]
        self.other=self.sessions.create("другой диалог")["id"]

    def tearDown(self):
        self.temp.cleanup()

    def test_short_confirmation_is_bound_to_session_and_proposal(self):
        proposed=self.memory.context("Переименуй себя в Домик")["change_proposal"]
        self.assertEqual(self.voice.resolve(self.sid,"Да")["status"],"no_active_review")
        self.voice.queue(self.sid,changes=[proposed])
        self.assertEqual(self.voice.resolve(self.other,"Да")["status"],"no_active_review")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],"Меня зовут JARVIS.")
        applied=self.voice.resolve(self.sid,"Да подтверждаю")
        self.assertEqual(applied["status"],"confirmed")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],"Меня зовут Домик.")
        self.assertEqual(self.voice.resolve(self.sid,"Да")["status"],"no_active_review")
        history=self.sessions.read(self.sid)["events"]
        self.assertEqual(history[-1]["event_type"],"voice_review")
        self.assertEqual(history[-1]["data"]["decision"],"confirm")

    def test_unrelated_question_invalidates_old_confirmation(self):
        proposed=self.memory.context("Переименуй себя в Домик")["change_proposal"]
        self.voice.queue(self.sid,changes=[proposed])
        self.assertFalse(self.voice.resolve(self.sid,"Как тебя зовут")["handled"])
        self.assertEqual(self.voice.resolve(self.sid,"Да")["status"],"no_active_review")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],"Меня зовут JARVIS.")

    def test_voice_rejection_and_two_candidates_require_selection(self):
        one=next(p for p in self.memory.propose("У меня есть часы",[
            {"kind":"entity","name":"часы","description":"Наручные часы"}])
            if p["kind"]=="entity")
        two=next(p for p in self.memory.propose("У меня есть планшет",[
            {"kind":"entity","name":"планшет","description":"Планшет"}])
            if p["kind"]=="entity")
        self.voice.queue(self.sid,proposals=[one,two])
        ambiguous=self.voice.resolve(self.sid,"Да")
        self.assertEqual(ambiguous["status"],"ambiguous_review")
        answer=self.voice.resolve(self.sid,"Подтверди второе")
        self.assertEqual(answer["status"],"confirmed")
        self.assertEqual(self.memory.snapshot()["counts"]["entity"],1)
        reject=self.voice.resolve(self.sid,"Отклоняю")
        self.assertEqual(reject["status"],"rejected")
        self.assertEqual(self.memory.snapshot()["counts"]["entity"],1)

    def test_merge_conflict_needs_explicit_voice_authorization(self):
        p=self.memory.propose("Есть другое имя",[
            {"kind":"entity","name":"Джарвис","description":"Отдельный узел"}])[0]
        self.memory.review(p["id"],"confirm")
        first=self.memory.context("Переименуй себя в Домик")["change_proposal"]
        self.memory.changes.review(first["id"],"confirm")
        pending=self.memory.context("Переименуй себя в Джарвис")["change_proposal"]
        self.assertTrue(pending["requires_merge_approval"])
        self.voice.queue(self.sid,changes=[pending])
        refused=self.voice.resolve(self.sid,"Да")
        self.assertEqual(refused["status"],"merge_confirmation_needed")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],"Меня зовут Домик.")
        applied=self.voice.resolve(self.sid,"Подтверждаю объединение")
        self.assertEqual(applied["status"],"confirmed")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],"Меня зовут Джарвис.")

    def test_voice_confirmation_only_short_utterances(self):
        self.assertEqual(speech_intent("У меня дома есть кошка"),(None,None))
        self.assertEqual(speech_intent("Подтверди первое"),("confirm",-1))
        self.assertEqual(speech_intent("Отклони второе"),("reject",-2))


class VoiceHttpTests(unittest.TestCase):
    def test_end_to_end_state_and_events(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ,{"DATABASE_PATH":str(Path(temp)/"api.sqlite3")}):
                with TestClient(research_app) as client:
                    sid=client.post("/research/sessions",json={"title":"speech"}).json()["id"]
                    first=client.post("/apprentice/context",json={
                        "query":"Переименуй себя в Домик","session_id":sid}).json()
                    self.assertEqual(first["voice_review"]["count"],1)
                    wrong=client.post("/apprentice/voice/resolve",json={
                        "session_id":sid,"text":"Вопрос как тебя зовут"}).json()
                    self.assertFalse(wrong["handled"])
                    no=client.post("/apprentice/voice/resolve",json={
                        "session_id":sid,"text":"Да"}).json()
                    self.assertEqual(no["status"],"no_active_review")
                    client.post("/apprentice/context",json={
                        "query":"Переименуй себя в Домик","session_id":sid})
                    reply=client.post("/apprentice/voice/resolve",json={
                        "session_id":sid,"text":"Команда: Да"}).json()
                    self.assertEqual(reply["status"],"confirmed")
                    observed=client.post("/apprentice/context",json={
                        "query":"Как тебя завут","session_id":sid}).json()
                    self.assertEqual(observed["direct_reply"],"Меня зовут Домик.")
                    session=client.get("/research/sessions/"+sid).json()
                    self.assertTrue(any(event["event_type"]=="voice_review" for event in session["events"]))


class GatewayVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_spoken_confirmation_does_not_call_qwen(self):
        calls=[]
        async def forward(url,method="GET",payload=None):
            calls.append(url)
            if url.endswith("/research/dialogue/resolve"):
                return {"handled":True,"category":"knowledge.review",
                        "answer":"Подтверждено.",
                        "status":"confirmed","executed":False,
                        "review":{"kind":"change","id":11,"decision":"confirm"}}
            raise AssertionError(url)
        with patch.object(gateway,"forward",forward):
            response=await gateway.apprentice_chat(gateway.ChatIn(
                text="Да",experiment="apprentice",
                session_id="b312050f-5ae1-4b20-86bc-4ea463dcc4ae"),0)
        self.assertEqual(response["route"],"apprentice_voice_review")
        self.assertEqual(len(calls),1)
        self.assertFalse(response["executed"])


if __name__=="__main__":
    unittest.main()
