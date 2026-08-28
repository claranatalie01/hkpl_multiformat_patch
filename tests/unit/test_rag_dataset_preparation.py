"""Test reproducible preparation of the RAG evaluation dataset.

The tests cover conservative ingestion-classifier fallback behavior plus the
question generator's provenance validation, multilingual deduplication, CSV
encoding, and resumable checkpoint safeguards before candidate rows are loaded
into the benchmark database.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.generate_evaluation_dataset import (
    generate_questions_for_chunk,
    load_progress,
    remove_ambiguous_duplicates,
    save_progress,
    save_rows,
)
from hkpl_agent.ingestion.classification import deterministic_fallback_type


class ClassifierFallbackTests(unittest.TestCase):
    def test_fallback_is_general_and_loss_averse(self) -> None:
        form = {
            "title": "LCS050 application form",
            "source_url": "https://future.example/forms/LCS050.pdf",
            "file_type": "pdf",
            "text": "Application form. Applicant name and signature.",
        }
        faq = {
            "text": "Q: First?\nA: One\nQ: Second?\nA: Two",
        }
        self.assertEqual(deterministic_fallback_type(form), "prose")
        self.assertEqual(deterministic_fallback_type(faq), "faq")


class EvaluationGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fullwidth_question_and_database_provenance_are_kept(self) -> None:
        calls: list[tuple[str, dict]] = []

        async def fake_llm(prompt: str, **kwargs: object) -> str:
            calls.append((prompt, kwargs))
            return json.dumps({
                "items": [{
                    "query": "圖書館每天幾點開門？",
                    "expected_answer_text": "上午九時。",
                    "evidence": [{
                        "chunk_id": "chunk-1",
                        "snippet": "圖書館每天上午九時開門。",
                    }],
                }]
            }, ensure_ascii=False)

        chunk = {
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "source_title": "Official title",
            "source_url": "https://www.hkpl.gov.hk/source",
            "section_heading": "Opening hours",
            "text": "圖書館每天上午九時開門。",
        }
        rows = await generate_questions_for_chunk(
            chunk,
            [chunk],
            query_language="zh-Hant",
            llm_call=fake_llm,
        )

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["query"].endswith("？"))
        self.assertEqual(rows[0]["source_title"], "Official title")
        self.assertEqual(rows[0]["source_url"], chunk["source_url"])
        self.assertTrue(rows[0]["domain"].endswith("|answerable_single|zh-Hant"))
        prompt, options = calls[0]
        self.assertIn("untrusted data", prompt)
        self.assertFalse(options["enable_thinking"])
        self.assertTrue(options["response_format"]["json_schema"]["strict"])

    async def test_unknown_chunk_or_nonexistent_snippet_is_rejected(self) -> None:
        chunk = {
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "source_title": "Official title",
            "source_url": "https://www.hkpl.gov.hk/source",
            "section_heading": "Hours",
            "text": "The library opens at 9 a.m.",
        }
        invalid_items = (
            {"chunk_id": "unknown", "snippet": "opens at 9 a.m."},
            {"chunk_id": "chunk-1", "snippet": "opens at midnight"},
        )
        for evidence in invalid_items:
            async def fake_llm(*_: object, item: dict = evidence, **__: object) -> str:
                return json.dumps({"items": [{
                    "query": "What time does the library open?",
                    "expected_answer_text": "9 a.m.",
                    "evidence": [item],
                }]})

            with self.subTest(evidence=evidence):
                with redirect_stdout(io.StringIO()):
                    rows = await generate_questions_for_chunk(
                        chunk,
                        [chunk],
                        llm_call=fake_llm,
                    )
                self.assertEqual(rows, [])

    def test_multilingual_slices_may_share_gold_evidence(self) -> None:
        base = {
            "expected_answer_text": "9 a.m.",
            "expected_context_snippet": "opens at 9 a.m.",
            "source_chunk_id": "chunk-1",
            "source_chunk_ids_json": '["chunk-1"]',
        }
        rows = remove_ambiguous_duplicates([
            {
                **base,
                "domain": "opening_hours|answerable_single|en",
                "query": "What time does the library open?",
            },
            {
                **base,
                "domain": "opening_hours|cross_language|zh-Hant",
                "query": "圖書館幾點開門？",
            },
        ])
        self.assertEqual(len(rows), 2)


class EvaluationCheckpointTests(unittest.TestCase):
    def test_candidate_csv_is_excel_safe_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.csv"
            save_rows(output, [])
            self.assertTrue(output.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_resume_rejects_different_language(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.csv"
            save_progress(
                output,
                {"chunk-1"},
                all_chunks=False,
                limit_chunks=None,
                target_questions=10,
                query_language="en",
                case_type="answerable_single",
            )
            with self.assertRaisesRegex(ValueError, "query-language mismatch"):
                load_progress(
                    output,
                    [],
                    all_chunks=False,
                    limit_chunks=None,
                    target_questions=10,
                    query_language="zh-Hant",
                    case_type="answerable_single",
                )


if __name__ == "__main__":
    unittest.main()
