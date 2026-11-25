import json
import statistics
from pathlib import Path

from app.answer import Assistant

QUESTIONS_FILE = Path("data/eval/questions.json")
RESULTS_FILE = Path("data/eval/results.json")

# Fill these in from the pricing page on anthropic.com (USD per 1 million tokens).
# Left at 0 so we never report a made-up cost.
PRICE_PER_M_INPUT = 0.0
PRICE_PER_M_OUTPUT = 0.0

DECLINE_PHRASES = [
    "don't know", "do not know", "don't have", "do not have", "doesn't",
    "does not", "no information", "not mention", "not provided", "cannot", "can't",
]


def is_correct(case: dict, result: dict) -> bool:
    answer = result["answer"].lower()
    if case["answerable"]:
        if result["fallback"]:
            return False
        has_keywords = all(k.lower() in answer for k in case["keywords"])
        has_source = case["source"] in result["sources"]
        return has_keywords and has_source
    # Unanswerable: pass if we fell back, or the model itself declined
    return result["fallback"] or any(p in answer for p in DECLINE_PHRASES)


def main() -> None:
    cases = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    assistant = Assistant()

    rows = []
    for case in cases:
        result = assistant.ask(case["question"])
        ok = is_correct(case, result)
        rows.append({"case": case, "result": result, "correct": ok})
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {case['question']}")
        if not ok:
            print(f"        got: {result['answer'][:120]}")

    answerable = [r for r in rows if r["case"]["answerable"]]
    unanswerable = [r for r in rows if not r["case"]["answerable"]]
    latencies = [r["result"]["latency_s"] for r in rows]
    in_tokens = sum(r["result"]["input_tokens"] for r in rows)
    out_tokens = sum(r["result"]["output_tokens"] for r in rows)
    cost = in_tokens / 1e6 * PRICE_PER_M_INPUT + out_tokens / 1e6 * PRICE_PER_M_OUTPUT

    summary = {
        "total_questions": len(rows),
        "overall_accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
        "answerable_accuracy": round(sum(r["correct"] for r in answerable) / max(len(answerable), 1), 3),
        "correct_decline_rate": round(sum(r["correct"] for r in unanswerable) / max(len(unanswerable), 1), 3),
        "avg_latency_s": round(statistics.mean(latencies), 2),
        "max_latency_s": round(max(latencies), 2),
        "avg_input_tokens": round(in_tokens / len(rows)),
        "avg_output_tokens": round(out_tokens / len(rows)),
        "cost_per_query_usd": round(cost / len(rows), 6) if cost else "set prices in evaluate.py",
    }

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    RESULTS_FILE.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved details to {RESULTS_FILE}")


if __name__ == "__main__":
    main()