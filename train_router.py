"""Step 2: train TF-IDF + logistic regression router on benchmark EASY/HARD labels."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame(json.loads(line) for line in f if line.strip())


def make_router():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                                  min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)),
    ])


def evaluate(router, rows):
    actual = rows["label"].to_numpy()
    predicted = router.predict(rows["question"].tolist())
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
        "confusion_matrix_easy_hard": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "report": classification_report(actual, predicted, target_names=["easy", "hard"],
                                         labels=[0, 1], zero_division=0, output_dict=True),
    }


def main():
    for split in ["train", "validation", "test"]:
        if not (Path("data") / f"{split}.jsonl").exists():
            raise SystemExit("First run: python prepare_data.py")
    train = read_jsonl("data/train.jsonl")
    validation = read_jsonl("data/validation.jsonl")
    test = read_jsonl("data/test.jsonl")
    router = make_router()
    router.fit(train["question"].tolist(), train["label"].astype(int).tolist())
    results = {"validation": evaluate(router, validation), "test": evaluate(router, test)}
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump({"model": router, "label_names": {0: "easy", 1: "hard"},
                 "decision_threshold_hard": 0.5}, "artifacts/difficulty_router.joblib")
    with open("artifacts/router_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    for split in ["validation", "test"]:
        print(f"{split.title()}  accuracy={results[split]['accuracy']:.3f}  "
              f"balanced_accuracy={results[split]['balanced_accuracy']:.3f}")
        print("Confusion matrix [actual easy/hard x predicted easy/hard]:",
              results[split]["confusion_matrix_easy_hard"])
    print("Saved artifacts/difficulty_router.joblib")
    print("Note: labels indicate benchmark subset, NOT observed model thinking behavior.")


if __name__ == "__main__":
    main()
