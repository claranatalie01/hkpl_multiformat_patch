"""Pydantic request contracts accepted by the FastAPI transport."""

from typing import Optional

from pydantic import BaseModel


class ProhibitedKeywordRequest(BaseModel):
    """Request to add one administrator-managed prohibited keyword."""

    keyword: str
    category: str = "general"
    language: str = "en"
    fallback_response: str
    created_by: str = "admin"


class KeywordStatusRequest(BaseModel):
    """Request to activate or deactivate a prohibited keyword."""

    is_active: bool
    staff_id: str = "admin"


class UrlIndexRequest(BaseModel):
    """Request to acquire and ingest one approved webpage."""

    url: str
    source_title: str = ""
    category: str | None = None
    language: str | None = None
    effective_date: str | None = None
    access_level: str = "public"
    document_type: str = "auto"


class TestQueryRequest(BaseModel):
    """Request to exercise the online graph through an admin endpoint."""

    question: str
    session_id: str = "admin-test-query"


class UserRequest(BaseModel):
    """Public streaming-chat request used by the current PoC API."""

    input_string: str
    session_id: str
    library_code: Optional[str] = None

