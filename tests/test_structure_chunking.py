from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.preview_ingestion import (
    CHUNK_TABLE,
    DOCUMENT_TABLE,
    postgres_json,
    postgres_text,
    run_preview,
)

import openpyxl
from llama_index.core import Document

from src.ingestion.chunking import build_search_text, chunk_documents
from src.ingestion.classification import MAX_BATCH_ITEMS, classify_batch_items
from src.ingestion.document_types import (
    chunk_policy_for,
    resolve_record_kind,
    validate_document_type,
)
from src.ingestion.readers import (
    extract_faq_pairs,
    extract_html_record_metadata,
    load_file,
)


FIXTURES = Path(__file__).parent / "fixtures"


class CharacterTokenizer:
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        return len(text)

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self):
        return self

    def encode(self, text: str, **_: object) -> list[int]:
        return list(range(len(text)))

    def __call__(self, text: str, **_: object) -> dict:
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def source_document(
    evidence_text: str,
    *,
    locator: dict | None = None,
    structural_kind: str = "hierarchical_leaf",
    record_kind: str = "prose",
    **metadata: object,
) -> Document:
    base = {
        "document_id": "doc-1",
        "kb_document_id": "doc-1",
        "source_version_id": "doc-1:v1",
        "document_version": 1,
        "source_title": "HKPL Source",
        "document_type": record_kind,
        "record_kind": record_kind,
        "structural_kind": structural_kind,
        "chunk_policy": chunk_policy_for(record_kind, structural_kind),
        "structure_path": [],
        "locator": locator or {"type": "section", "section": 1},
        "parser_version": "test-parser",
        "evidence_text": evidence_text,
        **metadata,
    }
    return Document(text=evidence_text, metadata=base)


class DocumentTypeTests(unittest.TestCase):
    def test_preview_tables_are_isolated_from_live_corpus(self):
        self.assertEqual(DOCUMENT_TABLE, "ingestion_preview_documents")
        self.assertEqual(CHUNK_TABLE, "ingestion_preview_chunks")
        self.assertNotIn(DOCUMENT_TABLE, {"knowledge_documents", "data_hkpl_knowledge"})
        self.assertNotIn(CHUNK_TABLE, {"knowledge_documents", "data_hkpl_knowledge"})

    def test_preview_removes_postgres_nul_characters(self):
        self.assertEqual(postgres_text("before\x00after"), "beforeafter")
        self.assertEqual(postgres_json({"text": "before\x00after"}), '{"text": "beforeafter"}')

    def test_physical_table_precedes_admin_faq_hint(self) -> None:
        metadata = {"document_type": "faq", "structural_kind": "table_row"}
        self.assertEqual(resolve_record_kind(metadata), "table")
        self.assertEqual(chunk_policy_for("faq", "table_row"), "table_rows")

    def test_unlabelled_input_does_not_guess_from_content(self) -> None:
        self.assertEqual(resolve_record_kind({"document_type": "auto"}), "prose")

    def test_legacy_admin_values_remain_valid_hints(self) -> None:
        self.assertEqual(validate_document_type("announcement"), "announcement")
        self.assertEqual(resolve_record_kind({"document_type": "announcement"}), "record")
        self.assertEqual(resolve_record_kind({"document_type": "directory"}), "record")


class PreviewRunTests(unittest.TestCase):
    def test_bad_classifier_batch_does_not_abort_later_batches(self) -> None:
        records = [{
            "document_id": str(index),
            "original_file_name": f"{index}.txt",
            "stored_file_name": f"{index}.txt",
            "source_type": "crawler",
        } for index in range(MAX_BATCH_ITEMS + 1)]
        decisions = {str(MAX_BATCH_ITEMS): {"document_type": "prose"}}

        with tempfile.TemporaryDirectory() as directory:
            for record in records:
                (Path(directory) / record["stored_file_name"]).write_text("text")
            with (
                patch("scripts.preview_ingestion.UPLOAD_DIR", Path(directory)),
                patch("scripts.preview_ingestion.ensure_preview_schema"),
                patch("scripts.preview_ingestion.list_documents", return_value=records),
                patch("scripts.preview_ingestion.extract_for_classification", return_value=("text", [])),
                patch("scripts.preview_ingestion.save_document_preview") as save,
                patch("scripts.preview_ingestion.preview_record") as preview,
                patch(
                    "scripts.preview_ingestion.classify_batch_items_sync",
                    side_effect=[ValueError("bad JSON"), decisions],
                ),
            ):
                run_preview(
                    run_id="run", limit=None, crawler_only=True, classify_only=False
                )

        self.assertEqual(preview.call_count, 1)
        self.assertTrue(any(
            call.kwargs.get("status") == "failed"
            and "Classification failed" in call.kwargs.get("error_message", "")
            for call in save.call_args_list
        ))


class BatchClassificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_one_compact_generation_call(self) -> None:
        calls: list[tuple[str, dict]] = []

        async def fake_llm(prompt: str, **kwargs: object) -> str:
            calls.append((prompt, kwargs))
            return json.dumps({"items": [
                {"id": "a", "type": "faq"},
                {"id": "b", "type": "skip"},
            ]})

        result = await classify_batch_items([
            {"id": "a", "title": "FAQ", "text": "Q: Renew? A: Online."},
            {"id": "b", "title": "Index", "text": "Links to notices"},
        ], llm_call=fake_llm)

        self.assertEqual(result, {
            "a": {"document_type": "faq"},
            "b": {"document_type": "skip"},
        })
        self.assertEqual(len(calls), 1)
        prompt, options = calls[0]
        self.assertIn("faq|record|prose|skip", prompt)
        self.assertIn("Tables and spreadsheets still need one of these labels", prompt)
        self.assertEqual(options["temperature"], 0.0)
        self.assertFalse(options["enable_thinking"])
        self.assertEqual(options["response_format"]["type"], "json_schema")
        self.assertTrue(options["response_format"]["json_schema"]["strict"])

    async def test_rejects_partial_or_invalid_output(self) -> None:
        items = [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}]
        invalid_outputs = (
            '{"items":[{"id":"a","type":"prose"}]}',
            '{"items":[{"id":"a","type":"prose"},{"id":"x","type":"faq"}]}',
            '{"items":[{"id":"a","type":"prose"},{"id":"a","type":"faq"}]}',
            '{"items":[{"id":"a","type":"event"},{"id":"b","type":"prose"}]}',
        )
        for raw in invalid_outputs:
            async def fake_llm(*_: object, output: str = raw, **__: object) -> str:
                return output

            with self.subTest(raw=raw), self.assertRaises(ValueError):
                await classify_batch_items(items, llm_call=fake_llm)

    async def test_rejects_oversized_batch_without_a_model_call(self) -> None:
        called = False

        async def fake_llm(*_: object, **__: object) -> str:
            nonlocal called
            called = True
            return "{}"

        with self.assertRaises(ValueError):
            await classify_batch_items(
                [{"id": str(index)} for index in range(MAX_BATCH_ITEMS + 1)],
                llm_call=fake_llm,
            )
        self.assertFalse(called)


class FaqAdapterTests(unittest.TestCase):
    def test_schema_org_has_highest_priority(self) -> None:
        html = """
        <script type="application/ld+json">{
          "@type":"FAQPage","mainEntity":[{
            "@type":"Question","name":"Schema question?",
            "acceptedAnswer":{"@type":"Answer","text":"Schema answer"}
          }]
        }</script>
        <details><summary>Details question?</summary><p>Details answer</p></details>
        """
        pairs = extract_faq_pairs(html)
        self.assertEqual([(pair.question, pair.answer) for pair in pairs], [
            ("Schema question?", "Schema answer")
        ])

    def test_details_multilingual_fixture(self) -> None:
        pairs = extract_faq_pairs(
            (FIXTURES / "hkpl_multilingual_faq.html").read_text(encoding="utf-8")
        )
        self.assertEqual(len(pairs), 2)
        self.assertIn("點樣", pairs[0].question)
        self.assertIn("電子書", pairs[1].question)

    def test_definition_list(self) -> None:
        pairs = extract_faq_pairs("<dl><dt>Borrowing?</dt><dd>Use a library card.</dd></dl>")
        self.assertEqual(pairs[0].answer, "Use a library card.")

    def test_aria_accordion(self) -> None:
        html = '<button aria-controls="answer-1">Opening hours?</button><div id="answer-1">See the branch page.</div>'
        pairs = extract_faq_pairs(html)
        self.assertEqual((pairs[0].question, pairs[0].answer), (
            "Opening hours?", "See the branch page."
        ))

    def test_heading_like_questions_require_repeated_evidence(self) -> None:
        html = """
        <h3>Can I renew?</h3><p>Yes, subject to the lending rules.</p>
        <h3>可以續借嗎？</h3><p>可以，但須符合借閱規則。</p>
        """
        self.assertEqual(len(extract_faq_pairs(html)), 2)
        self.assertEqual(
            extract_faq_pairs("<h3>Rhetorical question?</h3><p>Article prose.</p>"),
            [],
        )

    def test_all_marker_variants(self) -> None:
        html = """<pre>
Q: First?\nA: One
Q. Second?\nAnswer: Two
Q1: Third?\nA1: Three
Q.1: Fourth?\nA.1: Four
問題：第五？\n答案：五
问题：第六？\n答案：六
問：第七？\n答：七
</pre>"""
        pairs = extract_faq_pairs(html, hinted=True)
        self.assertEqual(len(pairs), 7)
        self.assertEqual(pairs[-1].answer, "七")


class ChunkConstructionTests(unittest.TestCase):
    def test_exact_evidence_and_normalized_search_text_are_separate(self) -> None:
        evidence = "ＡＢＣ\u00a0資料"
        document = source_document(
            evidence,
            operational_hash="must-not-be-embedded",
            access_level="public",
        )
        node = chunk_documents(
            [document], tokenizer=CharacterTokenizer(), max_tokens=512
        )[0]
        self.assertEqual(node.metadata["evidence_text"], evidence)
        self.assertIn("ABC 資料", node.text)
        self.assertNotIn("must-not-be-embedded", node.text)
        self.assertNotIn("public", node.text)

    def test_short_meaningful_record_is_retained(self) -> None:
        nodes = chunk_documents(
            [source_document("閉館")], tokenizer=CharacterTokenizer(), max_tokens=512
        )
        self.assertEqual(len(nodes), 1)

    def test_equal_text_at_distinct_locators_is_not_deduplicated(self) -> None:
        documents = [
            source_document("Same official wording", locator={"type": "page", "page": 1}),
            source_document("Same official wording", locator={"type": "page", "page": 2}),
        ]
        nodes = chunk_documents(documents, tokenizer=CharacterTokenizer(), max_tokens=512)
        self.assertEqual(len(nodes), 2)
        self.assertNotEqual(nodes[0].node_id, nodes[1].node_id)

    def test_oversized_faq_repeats_question_and_respects_cap(self) -> None:
        question = "How do I renew a book?"
        document = source_document(
            f"{question}\n" + ("answer " * 80),
            record_kind="faq",
            structural_kind="faq_pair",
            question=question,
            answer_text="answer " * 80,
            record_header=question,
            repeat_context=question,
        )
        nodes = chunk_documents(
            [document], tokenizer=CharacterTokenizer(160), max_tokens=160
        )
        self.assertGreater(len(nodes), 1)
        for node in nodes:
            self.assertLessEqual(node.metadata["token_count"], 160)
            self.assertTrue(node.metadata["evidence_text"].startswith(question))
            self.assertEqual(node.metadata["chunk_policy"], "oversized_leaf")
            self.assertEqual(node.metadata["chunk_overlap"], 64)

    def test_oversized_table_repeats_header(self) -> None:
        header = "Branch | Address"
        document = source_document(
            f"{header}\n" + ("Central Library | Causeway Bay " * 20),
            record_kind="table",
            structural_kind="table_row",
            table_header=header,
            repeat_context=header,
        )
        nodes = chunk_documents(
            [document], tokenizer=CharacterTokenizer(160), max_tokens=160
        )
        self.assertGreater(len(nodes), 1)
        self.assertTrue(all(node.metadata["evidence_text"].startswith(header) for node in nodes))

    def test_search_text_whitelist(self) -> None:
        search = build_search_text({
            "source_title": "Title",
            "structure_path": ["Policy", "Borrowing"],
            "search_aliases": ["借閱"],
            "record_header": "Renewal",
            "chunk_id": "uuid-not-searchable",
            "content_hash": "hash-not-searchable",
        }, "Evidence")
        self.assertEqual(search, "Title\n\nPolicy\n\nBorrowing\n\n借閱\n\nRenewal\n\nEvidence")

    def test_unlabelled_single_uses_512_64_fallback(self) -> None:
        document = source_document("word " * 120, record_kind="auto")
        nodes = chunk_documents(
            [document], tokenizer=CharacterTokenizer(180), max_tokens=180
        )
        self.assertGreater(len(nodes), 1)
        self.assertTrue(all(node.metadata["chunk_overlap"] == 64 for node in nodes))
        self.assertTrue(all(node.metadata["chunk_policy"] == "oversized_leaf" for node in nodes))
        self.assertTrue(all(node.metadata["classification_source"] == "fallback" for node in nodes))
        for previous, following in zip(nodes, nodes[1:]):
            self.assertEqual(
                previous.metadata["evidence_text"][-64:],
                following.metadata["evidence_text"][:64],
            )


class DeterministicReaderTests(unittest.TestCase):
    def _load(self, path: Path, **kwargs: object) -> list[Document]:
        return load_file(
            path,
            document_id="fixture-doc",
            original_file_name=path.name,
            document_type=str(kwargs.pop("document_type", "auto")),
            branch_ids=kwargs.pop("branch_ids", None),
            **kwargs,
        )

    def test_csv_rows_override_faq_hint(self) -> None:
        documents = self._load(FIXTURES / "records.csv", document_type="faq")
        self.assertEqual(len(documents), 2)
        self.assertTrue(all(doc.metadata["record_kind"] == "table" for doc in documents))
        self.assertEqual(documents[0].metadata["locator"]["row"], 2)

    def test_json_records_preserve_supplied_branch_ids(self) -> None:
        documents = self._load(
            FIXTURES / "branch_records.json",
            document_type="record",
            branch_ids=["HKCL"],
        )
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["branch_ids"], ["HKCL"])

    def test_xml_records_have_stable_locators(self) -> None:
        documents = self._load(FIXTURES / "records.xml")
        self.assertEqual(len(documents), 2)
        self.assertNotEqual(documents[0].metadata["locator"], documents[1].metadata["locator"])

    def test_multi_sheet_xlsx_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branches.xlsx"
            workbook = openpyxl.Workbook()
            first = workbook.active
            first.title = "Hong Kong"
            first.append(["branch", "hours"])
            first.append(["Central", "09:00"])
            second = workbook.create_sheet("Kowloon")
            second.append(["branch", "hours"])
            second.append(["Kowloon", "10:00"])
            workbook.save(path)
            documents = self._load(path)
        self.assertEqual(len(documents), 2)
        self.assertEqual(
            {document.metadata["sheet_name"] for document in documents},
            {"Hong Kong", "Kowloon"},
        )

    def test_docling_failure_is_not_silently_reparsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text("structured source text", encoding="utf-8")
            with patch("src.ingestion.readers._load_docling", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    self._load(path)


class RecordMetadataTests(unittest.TestCase):
    def test_notice_and_event_dates_are_typed_metadata(self) -> None:
        notice = extract_html_record_metadata(
            (FIXTURES / "notice_detail.html").read_text(encoding="utf-8"),
            "record",
        )
        event = extract_html_record_metadata(
            (FIXTURES / "event_detail.html").read_text(encoding="utf-8"),
            "record",
        )
        self.assertEqual(notice["datePublished"], "2026-08-01")
        self.assertEqual(event["startDate"], "2026-09-10T15:00:00+08:00")
        self.assertEqual(event["endDate"], "2026-09-10T16:00:00+08:00")


if __name__ == "__main__":
    unittest.main()
