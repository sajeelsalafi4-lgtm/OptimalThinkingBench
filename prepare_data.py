"""Step 1: load Meta's public OptimalThinkingBench subset and split without exact-question leakage."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_NAME = "facebook/optimal_thinking_bench"
LABELS = {"overthinkingbench": 0, "underthinkingbench": 1}


def records_from_source(local_jsonl=None):
    if local_jsonl:
        with open(local_jsonl, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(records)} examples from {local_jsonl}")
    else:
        from datasets import load_dataset
        print(f"Downloading dataset {DATASET_NAME} from Hugging Face...")
        records = list(load_dataset(DATASET_NAME, split="train"))
    return records


def normalize_question(text):
    return re.sub(r"\s+", " ", str(text)).casefold().strip()


def make_dataframe(records):
    items = []
    for record in records:
        subset = str(record.get("subset", "")).casefold().strip()
        if subset not in LABELS:
            continue
        question = str(record.get("question", "")).strip()
        if not question:
            continue
        md = record.get("metadata", {})
        if isinstance(md, str):
            try:
                md = json.loads(md)
            except json.JSONDecodeError:
                md = {"raw": md}
        items.append({
            "question": question,
            "answer": str(record.get("answer", "")),
            "subset": subset,
            "label": LABELS[subset],
            "metadata": md,
            "question_key": normalize_question(question),
        })
    df = pd.DataFrame(items)
    if df.empty or set(df["label"].unique()) != {0, 1}:
        raise ValueError("Need both 'overthinkingbench' and 'underthinkingbench' examples.")
    conflict = df.groupby("question_key")["label"].nunique()
    if (conflict > 1).any():
        raise ValueError("Same question exists in both labels. Cannot safely build this split.")
    return df


def split_dataframe(df, seed=42):
    # Split unique questions, not rows: original dataset can contain repeated tasks.
    unique = df.drop_duplicates(subset="question_key")[["question_key", "label"]]
    if unique.groupby("label").size().min() < 7:
        raise ValueError("Too few UNIQUE questions per class for 70/15/15 split; use full dataset.")
    train, temp = train_test_split(
        unique, test_size=0.30, random_state=seed, stratify=unique["label"]
    )
    validation, test = train_test_split(
        temp, test_size=0.50, random_state=seed, stratify=temp["label"]
    )
    splits = {"train": train, "validation": validation, "test": test}
    # Retain ONE record per unique question to avoid duplicate weighting in metrics.
    df_unique = df.drop_duplicates(subset="question_key")
    return {name: df_unique[df_unique["question_key"].isin(part["question_key"])].copy()
            for name, part in splits.items()}


def save_jsonl(frame, path):
    with path.open("w", encoding="utf-8") as f:
        for item in frame.to_dict(orient="records"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare OptimalThinkingBench data.")
    parser.add_argument("--local-jsonl", default=None, help="Path to downloaded otl_bench.jsonl, if available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    df = make_dataframe(records_from_source(args.local_jsonl))
    splits = split_dataframe(df, seed=args.seed)
    directory = Path(args.out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Original rows: {len(df)}; unique questions: {df.question_key.nunique()}")
    for split, part in splits.items():
        save_jsonl(part, directory / f"{split}.jsonl")
        print(f"{split:11s}: {len(part):4d} unique questions  |  "
              f"easy={sum(part.label==0)}, hard={sum(part.label==1)}")
    print("Prepared files saved in data/.")


if __name__ == "__main__":
    main()
