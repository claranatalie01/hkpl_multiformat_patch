"""Test shared answer prompting and validated SQL table configuration.

These tests keep the live and evaluation answer contract stable while ensuring
operator-provided table names cannot bypass the single-identifier SQL policy.
They intentionally avoid database and model dependencies.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.answering import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    answer_completion_budget,
    build_grounded_answer_prompt,
    format_source_block,
)
from src.infrastructure.table_names import (
    configured_table_name,
    physical_vector_table_name,
    sql_identifier,
)


class AnsweringTests(unittest.TestCase):
    """Protect the canonical evidence-grounded answer instructions."""

    def test_prompt_contains_question_context_and_fallback(self) -> None:
        prompt = build_grounded_answer_prompt(
            question="When does the library open?",
            context="[Source 1]\nIt opens at 10:00 a.m.",
        )

        self.assertIn("When does the library open?", prompt)
        self.assertIn("It opens at 10:00 a.m.", prompt)
        self.assertIn(INSUFFICIENT_EVIDENCE_ANSWER, prompt)

    def test_source_title_and_reasoning_budget_are_deterministic(self) -> None:
        self.assertEqual(
            format_source_block("Evidence", 2, title="Opening hours"),
            "[Source 2: Opening hours]\nEvidence",
        )
        self.assertEqual(
            answer_completion_budget(
                512,
                enable_thinking=True,
                thinking_budget_tokens=500,
            ),
            1012,
        )


class TableNameTests(unittest.TestCase):
    """Verify shared identifier validation for dynamically selected tables."""

    def test_vector_table_name_uses_llamaindex_prefix(self) -> None:
        self.assertEqual(
            physical_vector_table_name("hkpl_knowledge_hybrid"),
            "data_hkpl_knowledge_hybrid",
        )

    def test_invalid_identifier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sql_identifier("knowledge; DROP TABLE knowledge_documents")

    def test_configured_table_reads_environment(self) -> None:
        with patch.dict(os.environ, {"TEST_TABLE": "evaluation_dataset_128"}):
            self.assertEqual(
                configured_table_name("TEST_TABLE", "evaluation_dataset"),
                "evaluation_dataset_128",
            )


if __name__ == "__main__":
    unittest.main()
