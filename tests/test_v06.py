"""Regression experiments for JARVIS v0.6: typed mutations + truthfulness.

The test suite models the observed 'JARVIS -> Домик' failure as a state
transition, not a canned answer; the same test is repeated on an arbitrary
confirmed object. A grade is never a write permission.
"""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from services.research.settings import load
from services.research.engine import Engine
from services.research.apprentice import ApprenticeMemory
from services.research.api import app as research_app
from services.gateway import api as gateway


class KnowledgeChangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.engine=Engine(str(Path(self.tmp.name)/"test.sqlite3"),load())
        self.memory=ApprenticeMemory(self.engine)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rename_is_pending_until_confirmed_and_survives_restart(self):
        first=self.memory.context("Переименуй себя в Домик")
        item=first["change_proposal"]
        self.assertEqual(item["target_kind"],"role")
        self.assertEqual(item["status"],"pending")
        self.assertIn("НЕ применено",first["direct_reply"])
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],
                         "Меня зовут JARVIS.")
        # A model-graded answer is evidence, not a name change.
        self.memory.grade("Переименуй себя","Я переименован в Домик","good")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],
                         "Меня зовут JARVIS.")
        explanation=self.memory.context("Почему ты продолжаешь говорить старое имя?")
        self.assertIn("не было выполнено",explanation["direct_reply"])
        reviewed=self.memory.changes.review(item["id"],"confirm")
        self.assertEqual(reviewed["status"],"confirmed")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],
                         "Меня зовут Домик.")
        reopened=ApprenticeMemory(self.engine)
        self.assertEqual(reopened.context("А тебя зовут как")["direct_reply"],
                         "Меня зовут Домик.")
        self.assertEqual(reopened.context("Тебя зовут Домик или Джарвис?")["direct_reply"],
                         "Меня зовут Домик.")
        aliases=reopened.snapshot()["dialogue"]["role_aliases"]
        self.assertIn(("assistant","JARVIS"),
                      [(x["role"],x["name"]) for x in aliases])

    def test_repeated_request_and_rename_back(self):
        proposal=self.memory.context("Я просил переименовать тебя в Домик")["change_proposal"]
        self.assertEqual(proposal["proposed_value"],"Домик")
        self.memory.changes.review(proposal["id"],"confirm")
        another=self.memory.context("Переименуй себя в JARVIS")["change_proposal"]
        self.memory.changes.review(another["id"],"confirm")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],
                         "Меня зовут JARVIS.")
        self.assertFalse(any(e["name"]=="JARVIS" for e in
                             self.memory.changes.snapshot()["role_aliases"]))
        self.assertTrue(any(e["name"]=="Домик" for e in
                            self.memory.changes.snapshot()["role_aliases"]))

    def test_reject_does_not_mutate_and_duplicate_is_deduplicated(self):
        a=self.memory.context("Давайте теперь тебя буду звать Домик")["change_proposal"]
        b=self.memory.context("Давайте теперь тебя буду звать Домик")["change_proposal"]
        self.assertEqual(a["id"],b["id"])
        result=self.memory.changes.review(a["id"],"reject")
        self.assertEqual(result["status"],"rejected")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],
                         "Меня зовут JARVIS.")
        again=self.memory.context("Переименуй себя в Домик")["change_proposal"]
        self.assertGreater(again["id"],a["id"])
        self.assertEqual(self.memory.changes.review(again["id"],"confirm")["status"],
                         "confirmed")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],
                         "Меня зовут Домик.")

    def test_old_proposal_cannot_override_new_name(self):
        a=self.memory.context("Переименуй себя в Домик")["change_proposal"]
        b=self.memory.context("Переименуй себя в Помощник")["change_proposal"]
        self.memory.changes.review(b["id"],"confirm")
        with self.assertRaises(ValueError):
            self.memory.changes.review(a["id"],"confirm")
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],
                         "Меня зовут Помощник.")
        # Second rename keeps all historical names.
        again=self.memory.context("Переименуй себя в Домик")["change_proposal"]
        self.memory.changes.review(again["id"],"confirm")
        aliases=self.memory.snapshot()["dialogue"]["role_aliases"]
        self.assertTrue(any(x["name"]=="JARVIS" for x in aliases))
        self.assertTrue(any(x["name"]=="Помощник" for x in aliases))
        self.assertEqual(self.memory.context("Как тебя зовут")["direct_reply"],
                         "Меня зовут Домик.")

    def test_entity_rename_preserves_entity_and_graph_identity(self):
        candidate=self.memory.propose("Кот Барсик — питомец",[
            {"kind":"entity","name":"Барсик","description":"Домашний кот",
             "domain":"pets","details":{"entity_type":"cat"}}])[0]
        self.memory.review(candidate["id"],"confirm")
        fact=self.memory.propose("У меня есть Барсик",[
            {"kind":"fact","name":"Питомец","description":"Владелец",
             "details":{"subject":"user","predicate":"имеет","object":"Барсик"}}])[0]
        self.memory.review(fact["id"],"confirm")
        before=self.memory.snapshot()
        entity_id=before["knowledge"]["entity"][0]["id"]
        graph_id=next(x["id"] for x in before["graph"]["nodes"]
                      if x["name"]=="Барсик")
        proposal=self.memory.context("Переименуй Барсик в Рыжик")["change_proposal"]
        self.assertEqual(proposal["target_kind"],"entity")
        self.assertEqual(int(proposal["target_id"]),entity_id)
        self.assertEqual(self.memory.snapshot()["knowledge"]["entity"][0]["name"],"Барсик")
        self.memory.changes.review(proposal["id"],"confirm")
        after=self.memory.snapshot()
        self.assertEqual(after["knowledge"]["entity"][0]["id"],entity_id)
        self.assertEqual(after["knowledge"]["entity"][0]["name"],"Рыжик")
        node=next(x for x in after["graph"]["nodes"] if x["name"]=="Рыжик")
        self.assertEqual(node["id"],graph_id)
        self.assertIn("Барсик",node["aliases"])
        self.assertTrue(any(edge["object"]=="Рыжик" for edge in after["graph"]["edges"]))
        self.assertEqual(after["counts"]["fact"],1)

    def test_ambiguity_refused_and_question_cannot_mutate(self):
        for domain in ("home","work"):
            p=self.memory.propose("Система X",[
                {"kind":"entity","name":"Система","description":"Одна из систем",
                 "domain":domain}])[0]
            self.memory.review(p["id"],"confirm")
        msg=self.memory.context("Переименуй Система в Другая")
        self.assertIn("Найдено 2 совпадений",msg["direct_reply"])
        self.assertFalse(msg["change_proposal"].get("id"))
        self.assertIsNone(self.memory.context("Ты переименовал себя в Домик?")["change_proposal"])

    def test_model_suggestion_requires_teacher_source_and_real_target(self):
        p=self.memory.propose("Есть чайник",[
            {"kind":"entity","name":"Чайник","description":"Нагревает воду"}])[0]
        self.memory.review(p["id"],"confirm")
        changes=[{"target":"entity","old_name":"Чайник","new_name":"Термопот"}]
        self.assertEqual(self.memory.changes.model_proposals(
            "Сколько чайников у меня?",changes),[])
        self.assertEqual(self.memory.changes.model_proposals(
            "Переименуй Чайник в что-то",changes),[])
        proposed=self.memory.changes.model_proposals(
            "Переименуй Чайник в Термопот",changes)
        self.assertEqual(proposed[0]["status"],"pending")
        self.assertEqual(self.memory.snapshot()["knowledge"]["entity"][0]["name"],"Чайник")
        self.memory.changes.review(proposed[0]["id"],"confirm")
        self.assertEqual(self.memory.snapshot()["knowledge"]["entity"][0]["name"],"Термопот")

    def test_web_search_question_never_becomes_a_lesson(self):
        outcomes=self.memory.propose(
            "Джарвис, найди официальную документацию Home Assistant REST API",[
                {"kind":"lesson","name":"Расчет Home Assistant REST API",
                 "description":"Искать документацию","details":{"steps":["Искать"]}}])
        self.assertEqual(outcomes,[])
        self.assertEqual(self.memory.snapshot()["counts"]["pending"],0)


class ChangeHttpTests(unittest.TestCase):
    def test_review_endpoint_updates_atomic_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ,{"DATABASE_PATH":str(Path(tmp)/"http.sqlite3")}):
                with TestClient(research_app) as client:
                    context=client.post("/apprentice/context",json={
                        "query":"Переименуй себя в Домик"}).json()
                    self.assertIn("НЕ применено",context["direct_reply"])
                    item=context["change_proposal"]
                    self.assertEqual(client.post("/apprentice/changes/review",json={
                        "change_id":item["id"],"decision":"confirm"}).status_code,200)
                    state=client.get("/apprentice/state").json()
                    self.assertEqual(state["dialogue"]["roles"]["assistant"],"Домик")
                    self.assertFalse(state["changes"]["pending"])
                    self.assertEqual(client.post("/apprentice/context",json={
                        "query":"Как тебя зовут"}).json()["direct_reply"],"Меня зовут Домик.")


class GatewayChangeTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_mutation_offer_is_never_claimed_as_executed(self):
        calls=[]
        async def fake(url,method="GET",payload=None):
            calls.append(url)
            if url.endswith("/apprentice/context"):
                return {"normalized_text":payload["query"],
                        "direct_reply":"Предлагаю сменить имя, не применено.",
                        "change_proposal":{"id":44,"status":"pending",
                           "target_kind":"role","target_id":"assistant",
                           "expected_value":"JARVIS","proposed_value":"Домик"}}
            raise AssertionError(url)
        with patch.object(gateway,"forward",fake):
            result=await gateway.chat(gateway.ChatIn(
                text="Переименуй себя в Домик",experiment="apprentice"))
        self.assertEqual(result["model"],"SQLite")
        self.assertEqual(result["change_proposal"]["proposed_value"],"Домик")
        self.assertFalse(result["executed"])
        self.assertEqual(len(calls),1)

    async def test_status_progress_records_real_phases_and_finishes(self):
        request_id="12c7ad0a-7636-4bb2-bd37-75936e664492"
        async def fake(url,method="GET",payload=None):
            if url.endswith("/apprentice/context"):
                state=await gateway.chat_status(request_id)
                self.assertEqual(state["phase"],"memory")
                return {"direct_reply":"Меня зовут JARVIS."}
            raise AssertionError(url)
        with patch.object(gateway,"forward",fake):
            result=await gateway.chat(gateway.ChatIn(
                text="Как тебя зовут",experiment="apprentice",request_id=request_id))
        self.assertIn("JARVIS",result["answer"])
        state=await gateway.chat_status(request_id)
        self.assertEqual(state["phase"],"done")
        self.assertFalse(state["running"])
        self.assertGreaterEqual(state["elapsed_seconds"],0)
        with self.assertRaises(Exception):
            await gateway.chat_status("invalid")


if __name__=="__main__":
    unittest.main()
