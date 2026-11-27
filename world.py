import json
import os
from pathlib import Path
from common import serve

def load_worlds(directory):
    result = {}
    for path in sorted(Path(directory).glob("*.json")):
        w = json.loads(path.read_text())
        states = set(w["outputs"])
        values = list(w["outputs"].values())
        if w["id"] in result or w["initial"] not in states:
            raise ValueError("Duplicate world or invalid initial state: " + str(path))
        if any(type(v) is not int for v in values) or len(set(values)) != len(values):
            raise ValueError("v1 requires unique integer outputs per state")
        if not w["actions"] or not w["events"] or set(w["actions"]) & set(w["events"]):
            raise ValueError("Nonempty, disjoint action and event names required")
        for table in list(w["actions"].values()) + list(w["events"].values()):
            if set(table) != states or not set(table.values()) <= states:
                raise ValueError("Invalid transition table")
        result[w["id"]] = w
    if not result:
        raise ValueError("No worlds found")
    return result

def describe(w):
    return {"id": w["id"], "values": sorted(w["outputs"].values()),
            "actions": list(w["actions"]), "events": list(w["events"])}

def transition(w, state, kind, name):
    if kind not in ("actions", "events"):
        raise ValueError("Unknown input kind")
    next_state = w[kind][name][state]
    return {"state": next_state, "y": w["outputs"][next_state]}

def cases(w):
    rows = []
    for state, y in w["outputs"].items():
        for kind in ("actions", "events"):
            for name in w[kind]:
                rows.append({"id": len(rows), "before": y, "kind": kind,
                             "input": name, "expected": transition(w, state, kind, name)["y"]})
    return rows

def dispatch_for(worlds):
    def dispatch(method, path, data):
        if method != "POST":
            raise ValueError("POST required")
        if path == "/catalog":
            return {"worlds": [describe(w) for w in worlds.values()]}
        w = worlds[data["world"]]
        if path == "/start":
            return {"state": w["initial"], "y": w["outputs"][w["initial"]]}
        if path == "/step":
            return transition(w, data["state"], data["kind"], data["input"])
        if path == "/cases":
            return {"cases": cases(w)}
        if path == "/audit":
            return w
        raise ValueError("Unknown endpoint")
    return dispatch

if __name__ == "__main__":
    serve(dispatch_for(load_worlds(os.getenv("WORLDS_DIR", "./worlds"))))
