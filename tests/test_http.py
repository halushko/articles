import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from services.research.api import app as research_app
from services.gateway.api import app as gateway_app

class ResearchApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.patcher=patch.dict(os.environ,{'DATABASE_PATH':str(Path(self.tmp.name)/'research.sqlite3'),
                                        'SEMANTIC_ENABLED':'false'})
        self.patcher.start()
        self.client=TestClient(research_app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.patcher.stop()
        self.tmp.cleanup()

    def test_health_memory_learn_resolve(self):
        self.assertEqual(self.client.get('/health').status_code,200)
        self.assertEqual(self.client.get('/memory').json()['counts']['corrections'],0)
        r=self.client.post('/resolve',json={'text':'Включи свет в туалете'})
        self.assertEqual(r.json()['status'],'needs_confirmation')
        self.assertIn('decision_id',r.json())
        r=self.client.post('/learn',json={'phrase':'Включи свет в туалете','target_id':'light.lab_toilet'})
        self.assertEqual(r.status_code,200)
        self.assertFalse(r.json()['executed'])
        r=self.client.post('/resolve',json={'text':'Выключи свет в туалете'})
        self.assertEqual(r.json()['target_id'],'light.lab_toilet')
        self.assertEqual(self.client.get('/memory').json()['counts']['corrections'],1)

    def test_reject_invalid_teaching(self):
        r=self.client.post('/learn',json={'phrase':'Привет','target_id':'light.lab_toilet'})
        self.assertEqual(r.status_code,422)
        self.assertEqual(self.client.get('/memory').json()['counts']['corrections'],0)

    def test_concept_index_is_separate(self):
        r=self.client.post('/concepts',json={'name':'Духовка','domain':'техника','kind':'прибор'})
        self.assertEqual(r.status_code,200)
        self.assertEqual(self.client.get('/memory').json()['counts']['concepts'],1)

class WebTests(unittest.TestCase):
    def test_gateway_home_page(self):
        with TestClient(gateway_app) as client:
            r=client.get('/')
            self.assertEqual(r.status_code,200)
            self.assertIn('JARVIS',r.text)
            self.assertIn('id="voice"',r.text)
            self.assertIn('id="voice-language"',r.text)
            self.assertIn('id="voice-status"',r.text)
            js=client.get('/static/app.js')
            self.assertEqual(js.status_code,200)
            self.assertIn('JarvisVoice.initVoice',js.text)
            self.assertIn('voice.speak(r.answer)',js.text)
            voice=client.get('/static/voice.js')
            self.assertEqual(voice.status_code,200)
            self.assertIn('webkitSpeechRecognition',voice.text)
            self.assertIn('recognition.continuous = true',voice.text)
            self.assertIn('form.requestSubmit()',voice.text)
            self.assertIn('speechSynthesis',voice.text)
            self.assertIn('id="voice-output"',r.text)

if __name__=='__main__':unittest.main()
