"""Reproducible evidence-only retrieval benchmark for the apprentice graph.

No Qwen calls: this measures confirmation and graph-retrieval invariants, NOT
neural skill learning. Keep these fixtures separate from teaching prompts.
"""
import argparse
import json
import tempfile
from pathlib import Path

from services.research.engine import Engine
from services.research.settings import load
from services.research.apprentice import ApprenticeMemory


def evaluate(dataset):
    rows=[json.loads(x) for x in Path(dataset).read_text(encoding="utf-8").splitlines()
          if x.strip()]
    report=[]
    for case in rows:
        with tempfile.TemporaryDirectory() as tmp:
            engine=Engine(str(Path(tmp)/"graph.sqlite3"),load())
            memory=ApprenticeMemory(engine)
            memory.context("Меня зовут Дима")
            query_items=case["queries"]
            def owned():
                result=[]
                for query in query_items:
                    data=memory.context(query)
                    inspection=data["inspection"]
                    result.append(bool(
                        inspection and inspection["kind"]=="personal_home" and
                        inspection["trace"].get("verified_owned_edges")
                    ))
                return result
            before=owned()
            proposals=memory.propose(case["teach"],[])
            pending=owned()
            for proposed in proposals:
                if proposed["kind"]=="fact":
                    memory.review(proposed["id"],"confirm")
            after=owned()
            graph=memory.snapshot()["graph"]
            ownership=[edge for edge in graph["edges"]
                       if edge["predicate"]=="имеет"
                       and edge["object"].casefold()==case["object"].casefold()]
            for item in ownership:
                memory.forget("fact",item["source_fact_id"])
            revoked=owned()
            passed=(all(not x for x in before+pending+revoked)
                    and len(ownership)==1 and all(after))
            report.append({
                "case":case.get("id",case["object"]),
                "teach":case["teach"],"queries":query_items,
                "false_positive_before":sum(before),
                "false_positive_pending":sum(pending),
                "retrieved_after_confirm":sum(after),
                "retrieved_after_revoke":sum(revoked),
                "expected_heldout_questions":len(query_items),
                "passed":passed
            })
    n=len(report)
    return {"dataset":str(dataset),"cases":n,"passed":sum(x["passed"] for x in report),
            "retrieval_accuracy":(sum(x["retrieved_after_confirm"] for x in report)/
                                  max(1,sum(x["expected_heldout_questions"] for x in report))),
            "examples":report,
            "note":"Deterministic SQLite evaluation only; no claims about model reasoning."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",default="evaluation/graph_cases.jsonl")
    parser.add_argument("--assert-pass",action="store_true")
    args=parser.parse_args()
    results=evaluate(args.dataset)
    print(json.dumps(results,ensure_ascii=False,indent=2))
    if args.assert_pass and results["passed"]!=results["cases"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
