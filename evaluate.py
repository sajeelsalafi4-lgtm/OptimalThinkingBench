"""Step 4: benchmark adaptive / always-think / never-think Qwen3 on held-out examples."""
import argparse
import json
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots instead of requiring a GUI
import matplotlib.pyplot as plt
import pandas as pd

from llm_backend import Qwen3Backend
from metrics import approximate_correct, summarize


def read_test_data(path, per_subset, seed):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(seed)
    selected = []
    for subset in ["overthinkingbench", "underthinkingbench"]:
        items = [
    r for r in rows
    if r["subset"] == subset
    and str(r.get("answer") or "").strip().lower()
    not in ("", "none", "nan", "null")
]
        rng.shuffle(items)
        selected += items[:min(per_subset, len(items))]
    if not selected or len({r["subset"] for r in selected}) != 2:
        raise ValueError("Test split needs both subsets.")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Compute illustrative proxy OptimalThinkingBench scores.")
    parser.add_argument("--per-subset", type=int, default=3,
                        help="Number of held-out questions from EACH subset")
    parser.add_argument("--strategy", choices=["adaptive", "always_think", "never_think", "all"],
                        default="all")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-max", type=int, default=1000)
    args = parser.parse_args()
    if args.per_subset < 1 or args.max_new_tokens < 1:
        parser.error("--per-subset and --max-new-tokens must both be positive")
    if not Path("data/test.jsonl").exists() or not Path("artifacts/difficulty_router.joblib").exists():
        raise SystemExit("Run prepare_data.py and train_router.py first.")
    examples = read_test_data("data/test.jsonl", args.per_subset, args.seed)
    router = joblib.load("artifacts/difficulty_router.joblib")["model"]
    backend = Qwen3Backend(args.model)
    strategies = (["adaptive", "always_think", "never_think"]
                  if args.strategy == "all" else [args.strategy])
    Path("results").mkdir(exist_ok=True)
    cache_path = Path("results/inference_cache.jsonl")
    cache = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    value = json.loads(line)
                    cache[tuple(value["cache_key"])] = value["output"]
    detail = []
    for i, row in enumerate(examples, start=1):
        probability_hard = float(router.predict_proba([row["question"]])[0, 1])
        for strategy in strategies:
            think = (probability_hard >= 0.5 if strategy == "adaptive"
                     else strategy == "always_think")
            key = (args.model, row["question"], think, args.max_new_tokens)
            if key in cache:
                output = cache[key]
            else:
                print(f"[{i}/{len(examples)}] {strategy}: generating "
                      f"{'THINKING' if think else 'NON-THINKING'} response...", flush=True)
                output = backend.generate(row["question"], think, args.max_new_tokens)
                cache[key] = output
                with cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"cache_key": list(key), "output": output},
                                       ensure_ascii=False) + "\n")
            detail.append({
                "strategy": strategy, "subset": row["subset"],
                "question": row["question"], "expected": row["answer"],
                "prediction": output["answer"], "predicted_probability_hard": probability_hard,
                "selected_thinking": think,
                "correct_proxy": approximate_correct(row["answer"], output["answer"])
                                 if output["complete"] else False,
                **output,
            })
    summary = []
    for strategy in strategies:
        subset_rows = [r for r in detail if r["strategy"] == strategy]
        summary.append({"strategy": strategy, **summarize(subset_rows, args.t_max)})
    result_frame = pd.DataFrame(summary)
    result_frame.to_csv("results/comparison.csv", index=False)
    pd.DataFrame(detail).to_csv("results/detailed_predictions.csv", index=False)
    result_frame.plot(x="strategy", y=["over_auc_oaa_proxy", "under_accuracy_proxy", "f1_otb_proxy"],
                      kind="bar", rot=0, figsize=(10, 5), ylim=(0, 1), legend=True)
    plt.ylabel("Proxy score (0 to 1)")
    plt.title("OptimalThinkingBench-inspired evaluation (NOT official)")
    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=170)
    plt.close()
    print("\n=== BENCHMARK RESULTS (approximate string-match proxy, NOT official) ===")
    print(result_frame.to_string(index=False))
    print("\nSaved results/comparison.csv, detailed_predictions.csv, comparison.png")
    print("Limitations: no official semantic judge, Reasoning Gym verifiers, or math subsets.")


if __name__ == "__main__":
    main()
