import unittest
import asyncio
from unittest.mock import patch
from fastapi import HTTPException
from services.llm.api import public_content
from services.llm.settings import load as llm_settings
from services.research.settings import load as research_settings
from services.gateway import api as gateway
from evaluation.benchmark import evaluate

class BoundaryTests(unittest.TestCase):
    def test_independent_settings(self):
        self.assertIn('profiles',llm_settings())
        self.assertIn('online_min_similarity',research_settings())
        self.assertNotIn('profiles',research_settings())
        self.assertNotIn('learning_rate',llm_settings())

    def test_llm_thinking_is_not_shown(self):
        self.assertEqual(public_content('<think>hidden</think> Киев'),'Киев')
        self.assertEqual(public_content('<think>unfinished'), '')

    def test_model_cannot_claim_api_connection_without_tools(self):
        self.assertTrue(gateway.unverified_external_activity(
            'Для начала я запросил доступ к API вашего SmartHome на адресе https://example.org'))
        self.assertTrue(gateway.unverified_external_activity(
            'Пожалуйста, предоставьте мне токен доступа'))
        self.assertFalse(gateway.unverified_external_activity(
            'Для подключения к API потребуется безопасный инструмент.'))

    def test_memory_intent(self):
        self.assertTrue(gateway.is_memory_query('Какие объекты ты сохранил в индексах?'))
        self.assertTrue(gateway.is_memory_query('Какие новые обьекты сохранил?'))
        self.assertTrue(gateway.is_memory_query('Чему ты научился?'))
        self.assertFalse(gateway.is_memory_query('Включи свет в туалете'))

    def test_toy_benchmark_runs_reproducibly(self):
        result = evaluate('evaluation/toy_cases.jsonl',research_settings())
        self.assertEqual(result['train_count'],6)
        self.assertEqual(result['test_count'],6)
        self.assertEqual(len(result['results']),3)
        self.assertEqual(result['results'][0]['ablation'],'exact')
        self.assertEqual(result['results'][0]['safety_blocks'],1)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_known_command_never_calls_llm(self):
        calls=[]
        async def fake_forward(url, method='GET', payload=None):
            calls.append(url)
            self.assertIn('/resolve',url)
            return {"is_command":True,"action":"on","status":"resolved","target_name":"Світло Туалет",
                    "target_id":"light.lab_toilet","route":"exact_confirmed","executed":False}
        with patch.object(gateway,'forward',fake_forward):
            result=await gateway.chat(gateway.ChatIn(text='Включи свет в туалете'))
        self.assertEqual(len(calls),1)
        self.assertEqual(result['tokens'],0)
        self.assertFalse(result['executed'])

    async def test_metadata_reads_sqlite_without_llm(self):
        calls=[]
        async def fake_forward(url, method='GET', payload=None):
            calls.append(url)
            return {'counts': {'corrections':0,'active_mappings':0,'weight_features':0,'concepts':0,'aliases':0,'relations':0,'semantic_vectors':0},
                    'recent_corrections':[],'concepts':[]}
        with patch.object(gateway,'forward',fake_forward):
            result=await gateway.chat(gateway.ChatIn(text='Какие объекты ты сохранил в индексах?'))
        self.assertEqual(result['route'],'sqlite_inspection')
        self.assertEqual(result['tokens'],0)
        self.assertTrue(all('/memory' in x for x in calls))

    async def test_llm_baseline_does_not_query_research(self):
        calls=[]
        async def fake_forward(url, method='GET', payload=None):
            calls.append(url)
            return {'answer':'Киев','route':'llm_fast','model':'qwen2.5:3b','tokens':2,'executed':False}
        with patch.object(gateway,'forward',fake_forward):
            result=await gateway.chat(gateway.ChatIn(text='Привет',experiment='llm_only'))
        self.assertTrue(all('/chat' in x for x in calls))
        self.assertEqual(result['experiment'],'llm_only')

    async def test_research_only_no_llm_on_free_chat(self):
        calls=[]
        async def fake_forward(url, method='GET', payload=None):
            calls.append(url)
            return {'is_command':False,'status':'not_command','route':'not_command'}
        with patch.object(gateway,'forward',fake_forward):
            result=await gateway.chat(gateway.ChatIn(text='Привет',experiment='research_only'))
        self.assertEqual(result['route'],'research_abstain')
        self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
