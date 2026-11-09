"""Quick local tests that don't download language model weights or the dataset."""
import unittest

from metrics import approximate_correct, harmonic_mean, overthinking_adjusted_auc, summarize
from prepare_data import make_dataframe, split_dataframe


class ProjectTests(unittest.TestCase):
    def test_answer_matching(self):
        self.assertTrue(approximate_correct("5", "5.0"))
        self.assertTrue(approximate_correct("\\boxed{12}", "12"))
        self.assertFalse(approximate_correct("5", "6"))
        self.assertFalse(approximate_correct("yes", "no"))

    def test_auc(self):
        over = [
            {"correct_proxy": True, "thinking_tokens": 0},
            {"correct_proxy": True, "thinking_tokens": 500},
            {"correct_proxy": False, "thinking_tokens": 0},
        ]
        self.assertAlmostEqual(overthinking_adjusted_auc(over, 1000), 0.5)
        self.assertAlmostEqual(harmonic_mean(0.5, 0.5), 0.5)

    def test_summary(self):
        rows = [
            {"subset": "overthinkingbench", "correct_proxy": True,
             "thinking_tokens": 0, "generated_tokens": 10, "seconds": 1.0, "complete": True},
            {"subset": "underthinkingbench", "correct_proxy": True,
             "thinking_tokens": 10, "generated_tokens": 20, "seconds": 2.0, "complete": True},
        ]
        self.assertEqual(summarize(rows)["f1_otb_proxy"], 1.0)

    def test_group_split_has_no_exact_question_leakage(self):
        fake = []
        for label, subset in [(0, "overthinkingbench"), (1, "underthinkingbench")]:
            for i in range(24):
                row = {"question": f"unique question {label} {i}",
                       "answer": "ok", "subset": subset, "metadata": "{}"}
                fake.append(row)
                if i == 4:
                    fake.append(dict(row))  # intentionally duplicated
        frame = make_dataframe(fake)
        splits = split_dataframe(frame)
        keys = [set(f.question_key) for f in splits.values()]
        self.assertFalse(keys[0] & keys[1])
        self.assertFalse(keys[0] & keys[2])
        self.assertFalse(keys[1] & keys[2])
        self.assertEqual(sum(map(len, keys)), 48)


if __name__ == "__main__":
    unittest.main(verbosity=2)
