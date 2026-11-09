
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load existing model predictions
input_file = Path("results/detailed_predictions.csv")
output_file = Path("results/manual_review.csv")

df = pd.read_csv(input_file)
df["manual_correct"] = float("nan")

print("OPTIMALTHINKINGBENCH - ANSWER REVIEW")
print("=" * 55)

for i, row in df.iterrows():
    print(f"\nQuestion {i + 1} of {len(df)}")
    print("Strategy:", row["strategy"])
    print("Question:", row["question"])
    print("Expected:", row["expected"])
    print("AI Answer:", row["answer"])

    while True:
        score = input(
            "\nIs the answer correct and complete? "
            "(y=yes, n=no, s=skip): "
        ).strip().lower()

        if score in ("y", "n", "s"):
            break
        print("Please enter y, n, or s.")

    if score == "y":
        df.loc[i, "manual_correct"] = 1
    elif score == "n":
        df.loc[i, "manual_correct"] = 0

    # Save progress after each review
    df.to_csv(output_file, index=False)

# Exclude unanswered reviews
reviewed = df.dropna(subset=["manual_correct"]).copy()

if reviewed.empty:
    print("No answers reviewed yet.")
else:
    summary = reviewed.groupby("strategy").agg(
        accuracy=("manual_correct", "mean"),
        reviewed=("manual_correct", "count")
    )

    print("\nMANUALLY REVIEWED ACCURACY")
    print(summary)

    summary.to_csv("results/manual_accuracy.csv")

    # Generate accuracy graph
    plt.figure(figsize=(9, 5))
    plt.bar(
        summary.index,
        summary["accuracy"] * 100
    )

    plt.title("Manually Reviewed Answer Accuracy")
    plt.xlabel("Reasoning Strategy")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.tight_layout()

    plt.savefig(
        "results/manual_accuracy.png",
        dpi=300
    )
    plt.close()

    print("\nAccuracy graph generated successfully!")
    print("Saved: results/manual_accuracy.png")
