"""Evidence-grounded checks for the apprentice memory graph.

The assertions distinguish confirmed teacher evidence, pending hypotheses,
raw duplicate records and the deliberately isolated LLM-only baseline.
"""
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


class CanonicalGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.engine=Engine(str(Path(self.tmp.name)/"knowledge.sqlite3"),load())
        self.memory=ApprenticeMemory(self.engine)

    def tearDown(self):
        self.tmp.cleanup()

    def _confirm(self, text, data):
        proposals=self.memory.propose(text,data)
        for proposal in proposals:
            self.memory.review(proposal["id"],"confirm")
        return proposals

    def test_legacy_duplicate_names_deduplicate_without_deleting_evidence(self):
        self._confirm("Дима – человек в проекте",[
            {"kind":"entity","name":"Дима","domain":"личное","description":"Человек"},
            {"kind":"entity","name":"Дима","domain":"диалог","description":"Собеседник"}])
        state=self.memory.snapshot()
        self.assertEqual(state["counts"]["entity"],2)   # raw provenance preserved
        self.assertEqual(state["graph"]["counts"]["canonical_entities"],1)
        node=state["graph"]["nodes"][0]
        self.assertEqual(node["name"],"Дима")
        self.assertEqual(len(node["evidence"]),2)
        canonical_id=node["id"]
        # Startup migration is idempotent and preserves graph IDs and old rows.
        restored=ApprenticeMemory(self.engine)
        self.assertEqual(restored.snapshot()["graph"]["nodes"][0]["id"],canonical_id)
        self.assertEqual(restored.snapshot()["counts"]["entity"],2)

    def test_unconfirmed_or_rejected_facts_never_enter_graph(self):
        proposal=self.memory.propose("У меня есть кот",[])[0]
        self.assertEqual(self.memory.snapshot()["graph"]["counts"]["confirmed_relations"],0)
        self.assertFalse(self.memory.context("Что у меня есть дома?")["inspection"]["trace"]["verified_owned_edges"])
        self.memory.review(proposal["id"],"reject")
        self.assertEqual(self.memory.snapshot()["graph"]["counts"]["confirmed_relations"],0)
        self.assertEqual(len(self.memory.snapshot()["pending"]),0)

    def test_teaching_relationships_then_query_paraphrases(self):
        self.memory.context("Меня зовут Дима")
        candidates=self.memory.propose("У меня смартхаус на Home Assistant",[])
        triples=[p for p in candidates if p["kind"]=="fact"]
        self.assertEqual(len(triples),2)
        # Never pre-confirm. Plain mention is NOT evidence of actual ownership.
        self.assertEqual(self.memory.snapshot()["graph"]["counts"]["confirmed_relations"],0)
        for p in triples:
            self.memory.review(p["id"],"confirm")
        graph=self.memory.snapshot()["graph"]
        self.assertEqual(len(graph["edges"]),2)
        self.assertTrue(any(e["subject"]=="Дима" and
                            e["predicate"]=="имеет" and
                            e["object"]=="смартхаус" for e in graph["edges"]))
        query=self.memory.context("Что у меня есть дома?")
        self.assertIn("смартхаус",query["direct_reply"])
        self.assertIn("Home Assistant",query["direct_reply"])
        self.assertEqual(query["inspection"]["source"],"confirmed_sqlite_graph")
        # An unrelated question cannot acquire a device as an asserted fact.
        unrelated=self.memory.context("Что ты знаешь о спутниках?")
        self.assertIsNone(unrelated["direct_reply"])

    def test_edit_real_triple_not_only_label(self):
        proposal=self.memory.propose("У меня есть гараж",[
            {"kind":"fact","name":"ошибочная связь",
             "description":"описание","details":{
                 "subject":"user","predicate":"имеет","object":"ошибка"}}])
        derived=next(p for p in proposal if p["name"].startswith("Пользователь имеет"))
        model=next(p for p in proposal if p["name"]=="ошибочная связь")
        self.memory.review(model["id"],"confirm",details={
            "subject":"user","predicate":"имеет","object":"гараж"})
        graph=self.memory.snapshot()["graph"]
        self.assertEqual(len(graph["edges"]),1)
        self.assertEqual(graph["edges"][0]["object"],"гараж")
        self.assertNotIn("ошибка",[n["name"] for n in graph["nodes"]])
        with self.engine.connect() as con:
            stored=con.execute("SELECT details FROM apprentice_proposals WHERE id=?",
                               (model["id"],)).fetchone()["details"]
        self.assertIn('"гараж"',stored)
        # A forgotten link must disappear immediately, but its audit survives.
        with self.engine.connect() as con:
            fact_id=con.execute("SELECT id FROM world_facts WHERE source_proposal_id=?",
                               (model["id"],)).fetchone()["id"]
        self.memory.forget("fact",fact_id)
        self.assertEqual(self.memory.snapshot()["graph"]["counts"]["confirmed_relations"],0)

    def test_graph_inspection_reads_confirmed_evidence_only(self):
        self._confirm("Мелкий это аккумулятор",[
            {"kind":"entity","name":"Мелкий","domain":"дом","description":"Аккумулятор"},
            {"kind":"entity","name":"Дима","domain":"общение","description":"Человек"}])
        for question in ("перечисли мне твои сущности",
                         "а в твоих сущностях что сейчас находится?",
                         "Сколько сущностей у тебя сохранено?"):
            info=self.memory.context(question)["inspection"]
            self.assertEqual(info["kind"],"entities")
            self.assertIn("Мелкий",info["answer"])
            self.assertIn("Дима",info["answer"])
            self.assertEqual(info["counts"]["canonical_entities"],2)
        self.assertIsNone(self.memory.graph.inspect("Расскажи анекдот."))

    def test_retract_source_does_not_erase_other_sources(self):
        props=self._confirm("Дима",[
            {"kind":"entity","name":"Дима","domain":"общение","description":"Человек"},
            {"kind":"entity","name":"Дима","domain":"личное","description":"Собеседник"}])
        legacy=self.memory.snapshot()["knowledge"]["entity"]
        initial_id=self.memory.snapshot()["graph"]["nodes"][0]["id"]
        self.memory.forget("entity",legacy[0]["id"])
        self.assertEqual(self.memory.snapshot()["graph"]["nodes"][0]["id"],initial_id)
        self.assertEqual(self.memory.snapshot()["graph"]["counts"]["canonical_entities"],1)
        self.memory.forget("entity",legacy[1]["id"])
        self.assertEqual(self.memory.snapshot()["graph"]["counts"]["canonical_entities"],0)


class GraphHttpTests(unittest.TestCase):
    def test_pure_inspect_endpoint_and_confirm_then_retract(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ,{"DATABASE_PATH":str(Path(tmp)/"new.sqlite3")}):
                with TestClient(research_app) as client:
                    query={"query":"Перечисли мне твои сущности"}
                    before=client.post("/apprentice/inspect",json=query).json()
                    self.assertTrue(before["matched"])
                    self.assertIn("пока нет",before["inspection"]["answer"])
                    p=client.post("/apprentice/propose",json={
                        "source_text":"Кот – домашнее животное",
                        "proposals":[{"kind":"entity","name":"Кот",
                                      "domain":"дом","description":"Домашнее животное"}]
                    }).json()["proposals"][0]
                    client.post("/apprentice/review",json={"proposal_id":p["id"],"decision":"confirm"})
                    now=client.post("/apprentice/inspect",json=query).json()
                    self.assertIn("Кот",now["inspection"]["answer"])
                    state=client.get("/apprentice/state").json()
                    self.assertEqual(state["graph"]["counts"]["canonical_entities"],1)


class GraphGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_apprentice_inspection_skips_qwen(self):
        async def fake(url,method="GET",payload=None):
            if url.endswith("/apprentice/context"):
                return {"direct_reply":"Подтверждены: Дима, смартхаус.",
                        "inspection":{"trace":{"query_type":"entities"},
                                      "source":"confirmed_sqlite_graph"},
                        "normalized_text":"Перечисли мои сущности"}
            raise AssertionError("LLM not needed for graph inspection")
        with patch.object(gateway,"forward",fake):
            result=await gateway.chat(gateway.ChatIn(
                text="Перечисли мои сущности",experiment="apprentice"))
        self.assertEqual(result["route"],"apprentice_graph_inspection")
        self.assertEqual(result["tokens"],0)
        self.assertEqual(result["evidence_source"],"confirmed_sqlite_graph")

    async def test_hybrid_research_only_share_graph_but_llm_only_isolated(self):
        seen=[]
        async def fake(url,method="GET",payload=None):
            seen.append(url)
            if url.endswith("/apprentice/inspect"):
                return {"matched":True,"inspection":{
                    "answer":"Подтверждено 2 сущности: Дима, кот.",
                    "trace":{"query_type":"entities"},
                    "source":"confirmed_sqlite_graph"}}
            if url.endswith("/chat"):
                return {"answer":"Базовая модель не видит SQLite",
                        "route":"llm_fast","tokens":10}
            raise AssertionError(url)
        with patch.object(gateway,"forward",fake):
            for mode in ("research_only","hybrid"):
                result=await gateway.chat(gateway.ChatIn(
                    text="Перечисли твои сущности",experiment=mode))
                self.assertEqual(result["route"],"sqlite_graph_inspection")
                self.assertIn("кот",result["answer"])
            baseline=await gateway.chat(gateway.ChatIn(
                text="Перечисли твои сущности",experiment="llm_only"))
        self.assertEqual(baseline["experiment"],"llm_only")
        self.assertEqual(len([url for url in seen if "/apprentice/inspect" in url]),2)


if __name__=="__main__":unittest.main()
