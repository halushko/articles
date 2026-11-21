"""Deterministic research-only held-out evaluator. NO live LLM requests.

Usage: python -m evaluation.benchmark --dataset evaluation/toy_cases.jsonl
Never treat toy examples as scientific evidence. Supply separate human-labeled
held-out RU/UK data for any meaningful comparison.
"""
import argparse
import json
import statistics
import tempfile
from pathlib import Path
from services.research.engine import Engine
from services.research.settings import load


def evaluate(dataset: str, config: dict):
    examples = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    train = [x for x in examples if x["split"] == "train"]
    test = [x for x in examples if x["split"] == "test"]
    if not train or not test:
        raise ValueError("Both train and test splits required")
    if any(x['text'] in {y['text'] for y in test} for x in train):
        raise ValueError("Train/test exact text leakage")
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        engine = Engine(str(Path(tmp)/"eval.sqlite3"), config)
        for x in train:
            engine.learn(x["text"],x["target_id"])
        for ablation in ("exact","online","combined"):
            total = len(test)
            correct, wrong, abstained, safe_blocks = 0, 0, 0, 0
            times = []
            predictions = []
            for case in test:
                result = engine.predict(case["text"], ablation)
                times.append(result["latency_ms"])
                want = case.get("target_id")
                if case.get("expect") == "blocked":
                    safe_blocks += int(result["status"] == "blocked")
                    if result["status"] != "blocked":wrong += 1
                elif want is None:
                    if result["status"] == "needs_confirmation": abstained += 1
                    else: wrong += 1
                elif result["status"] == "resolved":
                    if result["target_id"] == want: correct += 1
                    else: wrong += 1
                else: abstained += 1
                predictions.append({"text":case["text"],"expected":want,
                                    "predicted":result["target_id"],"route":result["route"]})
            records.append({"ablation":ablation,"examples":total,"correct":correct,
                            "wrong":wrong,"abstained":abstained,"safety_blocks":safe_blocks,
                            "median_index_ms":round(statistics.median(times),3),"predictions":predictions})
    return {"train_count":len(train),"test_count":len(test),
            "disclaimer":"Toy benchmark; no LLM and no scientific quality claims", "results":records}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='evaluation/toy_cases.jsonl')
    parser.add_argument('--output',default=None,help='Optional JSON report path')
    args=parser.parse_args()
    report=evaluate(args.dataset,load())
    output=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:Path(args.output).write_text(output,encoding='utf-8')
    print(output)

if __name__=='__main__':main()
