from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from evidence import load_json, save_json


def load_answers(path: Path) -> dict[str, str]:
    rows = load_json(path)
    return {str(item["question_id"]): str(item["answer_choice"]).upper() for item in rows}


def load_categories(input_dir: Path) -> dict[str, str]:
    categories: dict[str, str] = {}
    for path in sorted(input_dir.glob("video_*.json")):
        rows = load_json(path)
        for item in rows:
            qid = str(item.get("question_id"))
            categories[qid] = str(item.get("category") or "Unknown")
    return categories


def main() -> None:
    parser = argparse.ArgumentParser(description="Score agentic predictions after they have been produced.")
    parser.add_argument("--pred", type=Path, default=Path("outputs/agentic_vqa/latest/submission.json"))
    parser.add_argument("--gold", type=Path, default=Path("test_standard_answers.json"))
    parser.add_argument("--input-dir", type=Path, default=Path("test题目合集/inputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/agentic_vqa/latest/score.json"))
    args = parser.parse_args()

    pred = load_answers(args.pred)
    gold = load_answers(args.gold)
    categories = load_categories(args.input_dir)
    common = [qid for qid in gold if qid in pred]
    correct = [qid for qid in common if pred[qid] == gold[qid]]

    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    errors: list[dict[str, Any]] = []
    for qid in common:
        bucket = by_category[categories.get(qid, "Unknown")]
        bucket[1] += 1
        if pred[qid] == gold[qid]:
            bucket[0] += 1
        else:
            errors.append(
                {
                    "question_id": qid,
                    "category": categories.get(qid, "Unknown"),
                    "pred": pred[qid],
                    "gold": gold[qid],
                }
            )

    report = {
        "correct": len(correct),
        "total": len(common),
        "expected_total": len(gold),
        "missing": sorted(set(gold) - set(pred)),
        "extra": sorted(set(pred) - set(gold)),
        "acc": len(correct) / max(len(common), 1),
        "errors": errors,
        "by_category": {
            category: {"correct": ok, "total": total, "acc": ok / total}
            for category, (ok, total) in sorted(by_category.items())
        },
    }
    save_json(args.output, report)
    print(f"score: {len(correct)}/{len(common)} acc={report['acc']:.6f}")
    if report["missing"]:
        print(f"missing: {len(report['missing'])}")
    if errors:
        print(f"errors: {len(errors)}")
        for item in errors[:15]:
            print(f"  {item['question_id']} pred={item['pred']} gold={item['gold']} {item['category']}")


if __name__ == "__main__":
    main()

