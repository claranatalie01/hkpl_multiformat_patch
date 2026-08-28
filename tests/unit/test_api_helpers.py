"""Test dependency-light HTTP upload and server-sent event helpers."""

import unittest

from fastapi import HTTPException

from hkpl_agent.api.streaming import format_sse
from hkpl_agent.api.uploads import safe_filename, validate_file_signature


class StreamingTests(unittest.TestCase):
    """Protect the wire format emitted by streaming chat responses."""

    def test_multiline_payload_uses_one_data_field_per_line(self) -> None:
        self.assertEqual(
            format_sse("answer", "first\nsecond"),
            "event: answer\ndata: first\ndata: second\n\n",
        )


class UploadValidationTests(unittest.TestCase):
    """Protect filename normalization and basic binary signature checks."""

    def test_filename_cannot_escape_upload_directory(self) -> None:
        self.assertEqual(safe_filename("../../report 2026.pdf"), "report_2026.pdf")

    def test_pdf_signature_is_required(self) -> None:
        validate_file_signature(".pdf", b"%PDF-1.7")
        with self.assertRaises(HTTPException):
            validate_file_signature(".pdf", b"not a PDF")


if __name__ == "__main__":
    unittest.main()
