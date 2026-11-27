import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import laboratory as lab
from world import load_worlds, transition, cases, dispatch_for

WORLDS = load_worlds(Path(__file__).resolve().parents[1] / "worlds")

class WorldsTest(unittest.TestCase):
    def test_toggle(self):
        w = WORLDS["w01"]
        self.assertEqual(transition(w, "s0", "actions", "A")["y"], 1)
        self.assertEqual(transition(w, "s1", "actions", "A")["y"], 0)
        self.assertEqual(transition(w, "s1", "events", "E")["y"], 0)
        self.assertEqual(len(cases(w)), 6)
    def test_second_world(self):
        self.assertEqual(len(cases(WORLDS["w02"])), 15)
    def test_public_interface(self):
        public = dispatch_for(WORLDS)("POST", "/catalog", {})["worlds"][0]
        self.assertEqual(set(public), {"id","values","actions","events"})

class ExperimentTest(unittest.TestCase):
    def test_persistence_and_no_test_answer_leak(self):
        def fake_post(base, path, data):
            if base == lab.WORLD:
                return dispatch_for(WORLDS)("POST", path, data)
            import json
            p = json.loads(data["prompt"])
            self.assertNotIn("world_audit", p)
            if "cases" in p:
                self.assertTrue(all("expected" not in c for c in p["cases"]))
                answer = {"predictions": {str(c["id"]): 0 for c in p["cases"]}}
            elif "y" in p:
                answer = {"input": p["announced_event"] or "A", "prediction": 0, "knowledge": "hypothesis"}
            else:
                answer = {"knowledge": "learned"}
            return {"answer": answer}
        with tempfile.TemporaryDirectory() as d, patch.object(lab,"DB",str(Path(d)/"db")), patch.object(lab,"post",side_effect=fake_post):
            lab.init_db()
            r = {"id":"test","status":"queued","config":{"world":"w01","profile":"test","steps":4,"seed":42,"event_probability":.25,"reuse":False},"steps":[],"llm_calls":[]}
            lab.run_experiment(r)
            self.assertEqual(r["status"], "done", r.get("error"))
            self.assertEqual(len(lab.get_run("test")["steps"]),4)
            self.assertEqual(lab.load_knowledge("test","w01")["knowledge"],"learned")
            self.assertEqual(r["evaluation"]["total"],6)
            self.assertEqual(r["initial_knowledge"],"")
    def test_schema_and_retry(self):
        with tempfile.TemporaryDirectory() as d, patch.object(lab,"DB",str(Path(d)/"db")):
            lab.init_db()
            r = {"id":"retry","status":"running","phase":"baseline","config":{"seed":42},
                 "interface":{"values":[0,1],"actions":["A"],"events":["E"]},"llm_calls":[]}
            with patch.object(lab,"post",side_effect=[{"answer":{"wrong":0}},{"answer":{"predictions":{"0":1}}}]) as mock:
                answer = lab.ask(r,"test",cases=[{"id":0,"before":0,"input":"A"}])
            self.assertEqual(answer,{"predictions":{"0":1}})
            self.assertEqual(mock.call_count,2)
            traces = lab.get_run("retry")["llm_calls"]
            self.assertFalse(traces[0]["valid"])
            self.assertTrue(traces[1]["valid"])
            self.assertEqual(traces[0]["request"]["schema"]["required"],["predictions"])
            self.assertIn("format_feedback",traces[1]["request"]["prompt"])

    def test_invalid_prediction(self):
        for bad in [True, "1", 2]:
            with self.assertRaises(ValueError):
                lab.prediction(bad,[0,1])

if __name__ == "__main__":
    unittest.main()
