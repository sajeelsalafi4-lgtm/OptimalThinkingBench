
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the verified predictions
df = pd.read_csv(
    "results/final_verified_predictions.csv"
)

output = Path("results")
output.mkdir(exist_ok=True)

# Build final summary from the same six responses
summary = df.groupby("strategy", as_index=False).agg(
    total_questions=("verified_correct", "count"),
    correct_answers=("verified_correct", "sum"),
    accuracy=("verified_correct", "mean"),
    avg_time=("seconds", "mean"),
    avg_tokens=("generated_tokens", "mean"),
)

summary["accuracy"] *= 100

order = ["adaptive", "always_think", "never_think"]
summary = summary.set_index("strategy").loc[order].reset_index()

summary.to_csv(
    output / "final_summary.csv",
    index=False
)

# Graph 1: Accuracy
plt.figure(figsize=(9, 5))
plt.bar(summary["strategy"], summary["accuracy"])
plt.title("Verified Answer Accuracy")
plt.xlabel("Reasoning Strategy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(output / "final_accuracy.png", dpi=300)
plt.close()

# Graph 2: Execution time
plt.figure(figsize=(9, 5))
plt.bar(summary["strategy"], summary["avg_time"])
plt.title("Average Inference Time")
plt.xlabel("Reasoning Strategy")
plt.ylabel("Time (seconds)")
plt.tight_layout()
plt.savefig(output / "final_time.png", dpi=300)
plt.close()

# Graph 3: Token consumption
plt.figure(figsize=(9, 5))
plt.bar(summary["strategy"], summary["avg_tokens"])
plt.title("Average Generated Tokens")
plt.xlabel("Reasoning Strategy")
plt.ylabel("Number of Tokens")
plt.tight_layout()
plt.savefig(output / "final_tokens.png", dpi=300)
plt.close()

print("\nFINAL PROJECT RESULTS")
print(summary.to_string(index=False))
print("\nAll three graphs generated successfully!")
