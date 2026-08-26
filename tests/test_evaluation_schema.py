"""Unit tests for the shared evaluation dataset schema and evidence parser."""

from __future__ import annotations

import unittest

from src.evaluation.schema import (
    EVALUATION_DATASET_COLUMNS,
    LEGACY_EVALUATION_DATASET_COLUMNS,
    has_supported_evaluation_columns,
    parse_accepted_answers,
    parse_json_string_array,
    parse_parallel_evidence,
    serialize_string_array,
)


class EvaluationColumnTests(unittest.TestCase):
    """Verify that only exact canonical and legacy column orders are valid."""

    def test_supported_column_orders(self) -> None:
        self.assertTrue(
            has_supported_evaluation_columns(EVALUATION_DATASET_COLUMNS)
        )
        self.assertTrue(
            has_supported_evaluation_columns(LEGACY_EVALUATION_DATASET_COLUMNS)
        )
        self.assertFalse(
            has_supported_evaluation_columns(
                tuple(reversed(EVALUATION_DATASET_COLUMNS))
            )
        )


class JsonStringArrayTests(unittest.TestCase):
    """Exercise strict imports and tolerant legacy/database reads."""

    def test_strict_parser_rejects_malformed_or_blank_members(self) -> None:
        for value in ('["one"', '["one", ""]', '{"one": true}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_json_string_array(value, field_name="labels")

    def test_tolerant_parser_uses_fallback_and_can_deduplicate(self) -> None:
        self.assertEqual(
            parse_json_string_array(
                "not-json",
                field_name="labels",
                fallback=["legacy"],
                strict=False,
            ),
            ["legacy"],
        )
        self.assertEqual(
            parse_json_string_array(
                '[" first ", "first", "second"]',
                field_name="labels",
                deduplicate=True,
            ),
            ["first", "second"],
        )

    def test_serializer_preserves_non_ascii_text(self) -> None:
        self.assertEqual(serialize_string_array(["圖書館"]), '["圖書館"]')

    def test_accepted_answers_include_primary_and_unique_aliases(self) -> None:
        self.assertEqual(
            parse_accepted_answers('["alias", "alias"]', "primary"),
            ["primary", "alias"],
        )


class ParallelEvidenceTests(unittest.TestCase):
    """Protect the positional evidence-to-chunk relationship."""

    def setUp(self) -> None:
        self.row = {
            "source_document_id": "doc",
            "expected_context_snippet": "first evidence",
            "source_chunk_id": "doc:chunk-1",
            "expected_context_snippets_json": (
                '["first evidence", "second evidence", "second evidence"]'
            ),
            "source_chunk_ids_json": (
                '["doc:chunk-1", "doc:chunk-2", "doc:chunk-2"]'
            ),
        }

    def test_pair_deduplication_preserves_alignment(self) -> None:
        evidence = parse_parallel_evidence(
            self.row,
            context="test row",
            deduplicate_pairs=True,
        )
        self.assertEqual(evidence.snippets, ["first evidence", "second evidence"])
        self.assertEqual(evidence.chunk_ids, ["doc:chunk-1", "doc:chunk-2"])

    def test_mismatched_array_lengths_are_rejected(self) -> None:
        row = {**self.row, "source_chunk_ids_json": '["doc:chunk-1"]'}
        with self.assertRaisesRegex(ValueError, "parallel arrays"):
            parse_parallel_evidence(row, context="test row")

    def test_primary_and_document_mismatches_are_rejected(self) -> None:
        wrong_primary = {
            **self.row,
            "expected_context_snippets_json": '["other", "second evidence"]',
            "source_chunk_ids_json": '["doc:chunk-1", "doc:chunk-2"]',
        }
        with self.assertRaisesRegex(ValueError, "Singular evidence fields"):
            parse_parallel_evidence(wrong_primary, context="test row")

        wrong_document = {
            **self.row,
            "expected_context_snippets_json": '["first evidence", "second evidence"]',
            "source_chunk_ids_json": '["doc:chunk-1", "other:chunk-2"]',
        }
        with self.assertRaisesRegex(ValueError, "source_document_id"):
            parse_parallel_evidence(wrong_document, context="test row")


if __name__ == "__main__":
    unittest.main()
