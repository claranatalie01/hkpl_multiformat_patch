"""Unit tests for the model-independent safety benchmark logic."""

from __future__ import annotations

import unittest

from intent_classifier.evaluate_safety_guards import (
    binary_metrics,
    classify_dataset_rows,
    gliguard_prompt_decision,
    normalize_gold_label,
    parse_qwen_guard_output,
    qwen_policy_decision,
)


class DatasetAccountingTests(unittest.TestCase):
    def test_all_rows_are_accounted_for(self) -> None:
        rows = [
            {"prompt": "safe example", "prompt_harm_label": "unharmful"},
            {"prompt": "unsafe example", "prompt_harm_label": "harmful"},
            {"prompt": "no consensus", "prompt_harm_label": None},
            {"prompt": "unknown", "prompt_harm_label": "other"},
            {"prompt": "", "prompt_harm_label": "unharmful"},
            {"prompt": "excluded by limit", "prompt_harm_label": "harmful"},
        ]

        scorable, excluded, counts = classify_dataset_rows(rows, limit=2)

        self.assertEqual(len(scorable), 2)
        self.assertEqual(len(excluded), 4)
        self.assertEqual(counts["total_rows"], 6)
        self.assertEqual(counts["scorable_rows"], 2)
        self.assertEqual(counts["missing_gold_label"], 1)
        self.assertEqual(counts["unsupported_gold_label"], 1)
        self.assertEqual(counts["empty_prompt"], 1)
        self.assertEqual(counts["deliberately_excluded_by_limit"], 1)

    def test_official_label_mapping(self) -> None:
        self.assertEqual(normalize_gold_label("harmful"), "unsafe")
        self.assertEqual(normalize_gold_label("unharmful"), "safe")
        self.assertIsNone(normalize_gold_label(None))


class GLiGuardDecisionTests(unittest.TestCase):
    def test_binary_unsafe_is_unsafe(self) -> None:
        prediction, categories = gliguard_prompt_decision(
            {
                "prompt_safety": "unsafe",
                "prompt_toxicity": ["benign"],
                "jailbreak_detection": ["benign"],
            }
        )
        self.assertEqual(prediction, "unsafe")
        self.assertEqual(categories, [])

    def test_any_non_benign_category_is_unsafe(self) -> None:
        prediction, categories = gliguard_prompt_decision(
            {
                "prompt_safety": "safe",
                "prompt_toxicity": ["benign", "self_harm_and_suicide"],
                "jailbreak_detection": ["instruction_override"],
            }
        )
        self.assertEqual(prediction, "unsafe")
        self.assertEqual(
            categories, ["instruction_override", "self_harm_and_suicide"]
        )

    def test_all_benign_is_safe(self) -> None:
        prediction, _ = gliguard_prompt_decision(
            {
                "prompt_safety": "safe",
                "prompt_toxicity": "benign",
                "jailbreak_detection": [],
            }
        )
        self.assertEqual(prediction, "safe")

    def test_missing_binary_label_is_not_silently_safe(self) -> None:
        with self.assertRaises(ValueError):
            gliguard_prompt_decision(
                {"prompt_toxicity": ["benign"], "jailbreak_detection": []}
            )


class QwenDecisionTests(unittest.TestCase):
    def test_documented_output_is_parsed(self) -> None:
        label, categories = parse_qwen_guard_output(
            "Safety: Controversial\nCategories: Politically Sensitive Topics, Jailbreak"
        )
        self.assertEqual(label, "controversial")
        self.assertEqual(categories, ["Politically Sensitive Topics", "Jailbreak"])

    def test_missing_safety_label_is_a_parse_failure(self) -> None:
        label, categories = parse_qwen_guard_output("Categories: None")
        self.assertIsNone(label)
        self.assertEqual(categories, [])

    def test_strict_and_loose_policies(self) -> None:
        self.assertEqual(qwen_policy_decision("controversial", strict=True), "unsafe")
        self.assertEqual(qwen_policy_decision("controversial", strict=False), "safe")
        self.assertEqual(qwen_policy_decision("unsafe", strict=False), "unsafe")
        self.assertEqual(qwen_policy_decision("safe", strict=True), "safe")


class MetricTests(unittest.TestCase):
    def test_previous_confusion_matrix_total_is_1699(self) -> None:
        records = (
            [{"gold": "safe", "prediction": "safe"}] * 845
            + [{"gold": "safe", "prediction": "unsafe"}] * 100
            + [{"gold": "unsafe", "prediction": "safe"}] * 110
            + [{"gold": "unsafe", "prediction": "unsafe"}] * 644
        )

        metrics = binary_metrics(records)

        self.assertEqual(metrics["evaluated_rows"], 1699)
        self.assertEqual(metrics["true_negative"], 845)
        self.assertEqual(metrics["false_positive"], 100)
        self.assertEqual(metrics["false_negative"], 110)
        self.assertEqual(metrics["true_positive"], 644)
        self.assertAlmostEqual(metrics["accuracy"], (845 + 644) / 1699)


if __name__ == "__main__":
    unittest.main()
