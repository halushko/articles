import tempfile
import unittest
from pathlib import Path
from services.research.engine import Engine
from services.research.normalize import features, parse_command, normalize
from services.research.settings import load


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name)/'memory.sqlite3')
        self.engine = Engine(self.db, load())

    def tearDown(self):
        self.tmp.cleanup()

    def test_russian_ukrainian_commands(self):
        self.assertEqual(parse_command('Включи свет в туалете')['action'], 'on')
        self.assertEqual(parse_command('Вимкни світло в туалеті')['action'], 'off')
        self.assertFalse(parse_command('Привет')['is_command'])

    def test_negative_command_blocked(self):
        for text in ('Не включай свет в туалете','Не включи свет в туалете','Не выключай свет'):
            res = self.engine.predict(text)
            self.assertEqual(res['status'],'blocked')
            self.assertFalse(res['executed'])

    def test_exact_resolves_immediately(self):
        phrase = 'Включи свет в туалете'
        self.assertEqual(self.engine.predict(phrase)['status'], 'needs_confirmation')
        self.engine.learn(phrase,'light.lab_toilet')
        r = self.engine.predict(phrase)
        self.assertEqual(r['target_id'],'light.lab_toilet')
        self.assertEqual(r['route'],'exact_confirmed')
        self.assertFalse(r['executed'])

    def test_action_invariance(self):
        self.engine.learn('Включи свет в туалете','light.lab_toilet')
        self.assertEqual(self.engine.predict('Выключи свет в туалете')['target_id'],'light.lab_toilet')
        self.assertEqual(self.engine.predict('Выключи свет в туалете')['action'],'off')

    def test_conflict_latest_explicit_wins_history_preserved(self):
        phrase = 'Включи свет в туалете'
        self.engine.learn(phrase,'light.lab_toilet')
        self.engine.learn(phrase,'light.lab_corridor_toilet')
        self.assertEqual(self.engine.predict(phrase)['target_id'],'light.lab_corridor_toilet')
        snapshot = self.engine.memory_snapshot()
        self.assertEqual(snapshot['counts']['corrections'],2)
        self.assertEqual(snapshot['counts']['active_mappings'],1)
        self.assertEqual(snapshot['active_mappings'][0]['target_id'],'light.lab_corridor_toilet')

    def test_repeat_does_not_overweight_one_phrase(self):
        phrase='Включи свет в туалете'
        self.engine.learn(phrase,'light.lab_toilet')
        with self.engine.connect() as con:
            before=con.execute("SELECT SUM(value*value) FROM online_weights WHERE target_id=?",
                               ('light.lab_toilet',)).fetchone()[0]
        for _ in range(6):
            self.engine.learn(phrase,'light.lab_toilet')
        with self.engine.connect() as con:
            after=con.execute("SELECT SUM(value*value) FROM online_weights WHERE target_id=?",
                              ('light.lab_toilet',)).fetchone()[0]
        self.assertAlmostEqual(before,after)
        self.assertEqual(self.engine.memory_snapshot()['counts']['corrections'],7)

    def test_correction_retracts_old_training_contribution(self):
        phrase='Включи свет в туалете'
        for _ in range(5):self.engine.learn(phrase,'light.lab_toilet')
        self.engine.learn(phrase,'light.lab_corridor_toilet')
        with self.engine.connect() as con:
            old=con.execute("SELECT SUM(ABS(value)) FROM online_weights WHERE target_id=?",
                            ('light.lab_toilet',)).fetchone()[0]
        self.assertLess(old or 0.0,1e-7)

    def test_persists_across_engine_restart(self):
        self.engine.learn('Включи свет в туалете','light.lab_toilet')
        new = Engine(self.db,load())
        self.assertEqual(new.predict('Включи свет в туалете')['target_id'],'light.lab_toilet')

    def test_learning_updates_weights(self):
        self.engine.learn('Включи свет в туалете','light.lab_toilet')
        snapshot = self.engine.memory_snapshot()
        self.assertGreater(snapshot['counts']['weight_features'],0)
        self.assertEqual(snapshot['counts']['semantic_vectors'],0)

    def test_unknown_stays_abstained_without_training(self):
        self.assertEqual(self.engine.predict('Включи свет на балконе')['status'],'needs_confirmation')

    def test_online_requires_distinct_confirmed_phrases(self):
        self.engine.learn('Включи свет в туалете','light.lab_toilet')
        self.engine.learn('Включи свет в туалете','light.lab_toilet')
        prediction=self.engine.predict('Погаси свет в туалете',ablation='online')
        self.assertNotEqual(prediction['route'],'online_reranker')

    def test_exact_ablation_does_not_generalize(self):
        self.engine.learn('Включи свет в туалете','light.lab_toilet')
        self.engine.learn('Погаси свет в туалете','light.lab_toilet')
        r=self.engine.predict('Включи свет туалета',ablation='exact')
        self.assertEqual(r['route'],'exact_miss')

    def test_feature_normalization(self):
        f=features('свет в туалете')
        self.assertTrue(f)
        self.assertAlmostEqual(sum(x*x for x in f.values()),1,places=6)
        self.assertEqual(normalize(' ТУАЛЕТЕ! '),'туалете')

    def test_empty_or_invalid_teaching(self):
        for phrase,target in [('Привет','light.lab_toilet'),('Не включи свет','light.lab_toilet'),('Включи свет','bad.target')]:
            with self.assertRaises(ValueError):self.engine.learn(phrase,target)

    def test_concepts_are_explicit(self):
        result=self.engine.remember_concept('Духовка','устройство','техника',aliases=['Печь'])
        self.assertEqual(result['id'],1)
        self.assertEqual(self.engine.memory_snapshot()['counts']['concepts'],1)
        self.assertEqual(self.engine.memory_snapshot()['counts']['aliases'],1)
        with self.assertRaises(Exception):self.engine.remember_concept('духовка')

    def test_embedding_provenance_old_target_ignored(self):
        a=self.engine.learn('Включи свет в туалете','light.lab_toilet')
        self.engine.store_embedding(a['evidence_id'],'bge-m3',[1.0,0.0])
        b=self.engine.learn('Включи свет в туалете','light.lab_corridor_toilet')
        self.engine.store_embedding(b['evidence_id'],'bge-m3',[0.0,1.0])
        ranked=self.engine.semantic_rank([1.0,0.0],'bge-m3')
        self.assertEqual([r['target_id'] for r in ranked],['light.lab_corridor_toilet'])
        self.assertEqual(self.engine.semantic_rank([1.0,0.0],'another-model'),[])

    def test_feedback_not_training_signal(self):
        r=self.engine.predict('Включи свет',record=True)
        self.engine.record_feedback(r['decision_id'],'bad')
        self.assertEqual(self.engine.memory_snapshot()['counts']['active_mappings'],0)

    def test_no_physical_execution(self):
        self.engine.learn('Включи свет в туалете','light.lab_toilet')
        self.assertFalse(self.engine.predict('Включи свет в туалете')['executed'])

if __name__=='__main__':unittest.main()
