
import os
import pandas as pd
import matplotlib.pyplot as plt

# Read evaluation results
df = pd.read_csv("results/comparison.csv")

os.makedirs("results", exist_ok=True)

# Graph 1: Execution time
plt.figure(figsize=(9, 5))
plt.bar(df["strategy"], df["mean_seconds"])
plt.title("Execution Time Comparison")
plt.xlabel("Reasoning Strategy")
plt.ylabel("Time (seconds)")
plt.tight_layout()
plt.savefig("results/time_comparison.png")
plt.close()

# Graph 2: Token consumption
plt.figure(figsize=(9, 5))
plt.bar(
    df["strategy"],
    df["mean_generated_tokens"]
)
plt.title("Token Consumption Comparison")
plt.xlabel("Reasoning Strategy")
plt.ylabel("Generated Tokens")
plt.tight_layout()
plt.savefig("results/token_comparison.png")
plt.close()

print("Graphs generated successfully!")
