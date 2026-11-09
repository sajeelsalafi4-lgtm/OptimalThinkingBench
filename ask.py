"""Step 3: ask a question; a trained router selects Qwen3 thinking mode."""
import argparse
from pathlib import Path

import joblib
from llm_backend import Qwen3Backend


def load_router(path="artifacts/difficulty_router.joblib"):
    if not Path(path).is_file():
        raise SystemExit("Router not trained. Run: python prepare_data.py && python train_router.py")
    return joblib.load(path)["model"]  # Only load your own trusted joblib files.


def predict_mode(router, question):
    probability_hard = float(router.predict_proba([question])[0, 1])
    return probability_hard >= 0.5, probability_hard


def main():
    parser = argparse.ArgumentParser(description="Ask the adaptive Qwen3 reasoning model.")
    parser.add_argument("--question", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--force", choices=["auto", "think", "no_think"], default="auto")
    args = parser.parse_args()
    question = args.question or input("Enter your question: ").strip()
    if not question:
        raise SystemExit("Question cannot be blank.")
    if args.force == "auto":
        router = load_router()
        thinking, prob = predict_mode(router, question)
        print(f"Router predicted hard-question probability: {prob:.1%}")
    else:
        thinking = args.force == "think"
    print(f"Selected mode: {'THINKING' if thinking else 'NON-THINKING'}")
    backend = Qwen3Backend(args.model)
    result = backend.generate(question, thinking, args.max_new_tokens)
    if result["complete"]:
        print("\nAnswer:\n" + result["answer"])
    else:
        print("\nNo complete final answer generated; increase --max-new-tokens.")
    print(f"\nThinking tokens (approx): {result['thinking_tokens']}")
    print(f"Total generated tokens: {result['generated_tokens']}")
    print(f"Inference time: {result['seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
