"""Qwen3 inference with observable thinking-token count; downloads weights on first use."""
import time


class Qwen3Backend:
    def __init__(self, name="Qwen/Qwen3-0.6B"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LLM dependencies missing. Run: python -m pip install -r requirements-llm.txt"
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
        self.model.eval()
        self.name = name
        self.end_think_id = self.tokenizer.convert_tokens_to_ids("</think>")

    def generate(self, question, thinking, max_new_tokens=1024):
        """Returns FINAL answer and count of generated tokens before </think>.

        If the thinking block is cut off before </think>, no final answer is claimed.
        """
        messages = [
            {"role": "system", "content": "Answer the question accurately. Respect any requested final answer format."},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6 if thinking else 0.7,
                top_p=0.95 if thinking else 0.8,
                top_k=20,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        seconds = time.perf_counter() - start
        completion_ids = generation[0][inputs["input_ids"].shape[1]:].tolist()
        if thinking:
            if self.end_think_id not in completion_ids:
                return {"answer": "", "thinking_tokens": len(completion_ids),
                        "generated_tokens": len(completion_ids), "seconds": seconds,
                        "complete": False, "reason": "thinking_block_not_closed"}
            end = completion_ids.index(self.end_think_id)
            thinking_tokens = end  # approximate; includes generated <think> marker if present
            answer = self.tokenizer.decode(completion_ids[end + 1:], skip_special_tokens=True).strip()
        else:
            thinking_tokens = 0
            answer = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return {"answer": answer, "thinking_tokens": thinking_tokens,
                "generated_tokens": len(completion_ids), "seconds": seconds,
                "complete": bool(answer), "reason": "ok" if answer else "empty_answer"}
