
import re
import pandas as pd

df = pd.read_csv(
    "results/detailed_predictions.csv",
    keep_default_na=False
)

def check_answer(row):
    expected = str(row["expected"]).strip()
    answer = str(row["answer"])

    complete = str(row["complete"]).lower() == "true"

    if not complete:
        return "incorrect_incomplete"

    if expected.lower() in ("", "none", "nan", "null"):
        return "unscorable_missing_reference"

    # Automatically verify hexadecimal answers
    if re.fullmatch(r"0x[0-9a-fA-F]+", expected):
        matches = re.findall(
            r"final\s+answer\s*:\s*[`*\s]*(0x[0-9a-fA-F]+)",
            answer,
            flags=re.IGNORECASE
        )

        if not matches:
            return "needs_manual_review"

        if int(matches[-1], 16) == int(expected, 16):
            return "correct"
        return "incorrect"

    # Other answer types need suitable verification
    return "needs_manual_review"


df["audit_status"] = df.apply(check_answer, axis=1)

print("\nFINAL ANSWER AUDIT")
print("=" * 50)

print(
    df[[
        "strategy",
        "subset",
        "expected",
        "audit_status"
    ]].to_string(index=False)
)

print("\nAUDIT STATUS COUNTS")
print(
    pd.crosstab(
        df["strategy"],
        df["audit_status"]
    )
)

df.to_csv(
    "results/audited_predictions.csv",
    index=False
)

print("\nSaved: results/audited_predictions.csv")
