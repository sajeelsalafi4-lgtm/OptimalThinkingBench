"""Transparent approximate answer grader + paper-inspired OAA/F1 metrics.

IMPORTANT: Exact-string checking is a *proxy*, not the paper's LLM judge or
Reasoning Gym verifier. Valid alternative reasoning-game solutions can be
marked wrong here. Do NOT report these as official benchmark numbers.
"""
import re
import string
from difflib import SequenceMatcher


def normalized(text):
    text = str(text).strip().casefold()
    # The instruction often asks for a boxed final answer.
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(string.whitespace + string.punctuation)


def approximate_correct(expected, actual):
    """Conservative text/numeric matcher for a simple demonstrator.

    Does NOT verify alternative chess-game paths or semantic paraphrases.
    """
    a, b = normalized(expected), normalized(actual)
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        # Numeric answers only: don't extract an arbitrary digit from prose.
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", a.replace(",", "")) and \
           re.fullmatch(r"[-+]?\d+(?:\.\d+)?", b.replace(",", "")):
            return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) <= 1e-8
    except ValueError:
        pass
    # A small allowance for near-identical long-form responses.
    if min(len(a), len(b)) >= 60 and SequenceMatcher(None, a, b).ratio() >= 0.92:
        return True
    return False


def overthinking_adjusted_auc(over_rows, t_max=1000):
    """Normalized integral AUC of OAA over thresholds 0..t_max.

    Each correct example contributes max(0, 1 - thinking_tokens/t_max).
    This is the continuous closed-form integral of the OAA indicator.
    """
    if t_max <= 0:
        raise ValueError("t_max must be > 0")
    if not over_rows:
        return 0.0
    return sum(float(row["correct_proxy"]) * max(
        0.0, 1.0 - float(row["thinking_tokens"]) / t_max
    ) for row in over_rows) / len(over_rows)


def harmonic_mean(a, b):
    return 0.0 if (a + b) == 0 else 2.0 * a * b / (a + b)


def summarize(rows, t_max=1000):
    over = [r for r in rows if r["subset"] == "overthinkingbench"]
    under = [r for r in rows if r["subset"] == "underthinkingbench"]
    if not over or not under:
        raise ValueError("Need both benchmark subsets to report combined score.")
    over_acc = sum(bool(r["correct_proxy"]) for r in over) / len(over)
    over_auc = overthinking_adjusted_auc(over, t_max)
    under_acc = sum(bool(r["correct_proxy"]) for r in under) / len(under)
    return {
        "n_over": len(over), "n_under": len(under),
        "over_accuracy_proxy": round(over_acc, 4),
        "over_auc_oaa_proxy": round(over_auc, 4),
        "under_accuracy_proxy": round(under_acc, 4),
        "f1_otb_proxy": round(harmonic_mean(over_auc, under_acc), 4),
        "mean_thinking_tokens": round(sum(r["thinking_tokens"] for r in rows) / len(rows), 2),
        "mean_generated_tokens": round(sum(r["generated_tokens"] for r in rows) / len(rows), 2),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2),
        "incomplete_answers": sum(not r["complete"] for r in rows),
    }
