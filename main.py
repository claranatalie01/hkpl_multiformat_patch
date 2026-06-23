import hmac
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from src.graph import compiled_workflow
from src.ingestion.readers import SUPPORTED_EXTENSIONS
from src.ingestion.registry import (
    ensure_registry_schema,
    get_document,
    list_documents,
)
from src.ingestion.service import (
    UPLOAD_DIR,
    delete_registered_document,
    process_registered_document,
    register_upload,
)
from src.memory import load_conversation_history
from src.nodes import get_current_datetime


logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(
    os.getenv(
        "MAX_UPLOAD_BYTES",
        str(25 * 1024 * 1024),
    )
)
ADMIN_API_KEY = os.getenv(
    "ADMIN_API_KEY",
    "",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    ensure_registry_schema()

    if not ADMIN_API_KEY:
        logger.warning(
            "ADMIN_API_KEY is not set. "
            "Admin endpoints are open in development mode."
        )

    yield


app = FastAPI(
    title="HKPL Agentic RAG Service",
    lifespan=lifespan,
)


class UserRequest(BaseModel):
    input_string: str
    session_id: str
    is_voice: bool = False
    stt_confidence: float = 1.0
    library_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    user_memory: Optional[dict] = None


def require_admin(
    x_admin_key: Optional[str] = Header(
        default=None,
        alias="X-Admin-Key",
    ),
) -> None:
    if not ADMIN_API_KEY:
        return

    if (
        x_admin_key is None
        or not hmac.compare_digest(
            x_admin_key,
            ADMIN_API_KEY,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin API key.",
        )


def safe_filename(filename: str) -> str:
    basename = Path(filename).name
    cleaned = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        basename,
    )
    return cleaned or "uploaded_file"


def validate_file_signature(
    extension: str,
    content: bytes,
) -> None:
    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    signatures = {
        ".pdf": [b"%PDF"],
        ".png": [b"\x89PNG\r\n\x1a\n"],
        ".jpg": [b"\xff\xd8\xff"],
        ".jpeg": [b"\xff\xd8\xff"],
        ".tif": [b"II*\x00", b"MM\x00*"],
        ".tiff": [b"II*\x00", b"MM\x00*"],
        ".docx": [b"PK\x03\x04"],
        ".xlsx": [b"PK\x03\x04"],
        ".xlsm": [b"PK\x03\x04"],
        ".pptx": [b"PK\x03\x04"],
    }

    expected = signatures.get(extension)
    if expected and not any(
        content.startswith(signature)
        for signature in expected
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "The file content does not match "
                f"the {extension} extension."
            ),
        )


async def save_upload(
    file: UploadFile,
) -> tuple[Path, str, str]:
    original_name = safe_filename(
        file.filename or ""
    )
    extension = Path(
        original_name
    ).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension}. "
                f"Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    content = await file.read(
        MAX_UPLOAD_BYTES + 1
    )

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file is too large.",
        )

    validate_file_signature(
        extension,
        content,
    )

    stored_name = (
        f"{uuid4().hex}_{original_name}"
    )
    stored_path = (
        UPLOAD_DIR
        / stored_name
    )
    stored_path.write_bytes(content)

    return (
        stored_path,
        original_name,
        file.content_type or "",
    )


async def resolve_current_library(
    latitude: float,
    longitude: float,
) -> Optional[dict]:
    logger.warning(
        "Coordinate-based library resolution "
        "has not been implemented yet: %s, %s",
        latitude,
        longitude,
    )
    return None


def format_sse(
    event: str,
    data: str,
) -> str:
    lines = str(data).splitlines() or [""]
    payload = "".join(
        f"data: {line}\n"
        for line in lines
    )
    return f"event: {event}\n{payload}\n"


@app.post("/chat/stream")
async def chat_stream(
    payload: UserRequest,
):
    current_library = None

    if payload.library_code:
        name_map = {
            "HKCL": "Hong Kong Central Library",
            "STPL": "Shatin Public Library",
        }
        current_library = {
            "code": payload.library_code,
            "name": name_map.get(
                payload.library_code,
                payload.library_code,
            ),
        }
    elif (
        payload.latitude is not None
        and payload.longitude is not None
    ):
        current_library = (
            await resolve_current_library(
                payload.latitude,
                payload.longitude,
            )
        )

    history = load_conversation_history(
        payload.session_id
    )

    initial_state = {
        "messages": [
            HumanMessage(
                content=payload.input_string
            )
        ],
        "session_id": payload.session_id,
        "conversation_history": history,
        "original_query": payload.input_string,
        "rewritten_query": payload.input_string,
        "input_type": (
            "voice"
            if payload.is_voice
            else "text"
        ),
        "stt_confidence": payload.stt_confidence,
        "intent": "",
        "request_type": "normal_info",
        "retrieved_context": "",
        "retrieved_chunks": [],
        "retrieved_scores": [],
        "retrieved_sources": [],
        "generated_answer": "",
        "faithfulness_passed": True,
        "faithfulness_reason": "",
        "is_relevant": False,
        "rewrite_count": 0,
        "is_output_safe": True,
        "end_conversation": False,
        "tool_name": "",
        "tool_args": {},
        "current_library_code": (
            current_library["code"]
            if current_library
            else None
        ),
        "current_library_name": (
            current_library["name"]
            if current_library
            else None
        ),
        "current_datetime": (
            get_current_datetime()
        ),
        "user_memory": (
            payload.user_memory or {}
        ),
    }

    async def event_generator():
        async for chunk in compiled_workflow.astream(
            initial_state,
            stream_mode="updates",
        ):
            for node_name, updated in chunk.items():
                yield format_sse(
                    "node",
                    node_name,
                )

                if node_name in {
                    "safety",
                    "output_safety_filter",
                }:
                    if (
                        isinstance(updated, dict)
                        and "messages" in updated
                    ):
                        for message in updated["messages"]:
                            if isinstance(
                                message,
                                AIMessage,
                            ):
                                yield format_sse(
                                    "answer",
                                    message.content,
                                )

        yield format_sse(
            "end",
            "",
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/admin/documents/upload",
    status_code=202,
    dependencies=[Depends(require_admin)],
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_title: str = Form(""),
    source_url: str = Form(""),
    access_level: str = Form("public"),
):
    stored_path, original_name, mime_type = (
        await save_upload(file)
    )

    try:
        registration = register_upload(
            stored_path=stored_path,
            original_file_name=original_name,
            mime_type=mime_type,
            source_title=source_title,
            source_url=source_url,
            source_type="admin_upload",
            access_level=access_level,
        )
    except Exception:
        stored_path.unlink(
            missing_ok=True
        )
        raise

    if registration["duplicate"]:
        stored_path.unlink(
            missing_ok=True
        )
        document = registration["document"]
        return {
            "status": "duplicate",
            "document_id": str(
                document["document_id"]
            ),
            "file_name": document[
                "original_file_name"
            ],
        }

    document_id = str(
        registration["document"][
            "document_id"
        ]
    )

    background_tasks.add_task(
        process_registered_document,
        document_id,
    )

    return {
        "status": "uploaded",
        "document_id": document_id,
        "file_name": original_name,
        "message": (
            "Extraction, chunking, and embedding "
            "have been queued."
        ),
    }


@app.post(
    "/admin/documents/{document_id}/replace",
    status_code=202,
    dependencies=[Depends(require_admin)],
)
async def replace_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_title: str = Form(""),
    source_url: str = Form(""),
    access_level: str = Form("public"),
):
    stored_path, original_name, mime_type = (
        await save_upload(file)
    )

    try:
        registration = register_upload(
            stored_path=stored_path,
            original_file_name=original_name,
            mime_type=mime_type,
            source_title=source_title,
            source_url=source_url,
            source_type="admin_upload",
            access_level=access_level,
            replace_document_id=document_id,
        )
    except Exception:
        stored_path.unlink(
            missing_ok=True
        )
        raise

    if registration["duplicate"]:
        stored_path.unlink(
            missing_ok=True
        )
        return {
            "status": "unchanged",
            "document_id": document_id,
        }

    background_tasks.add_task(
        process_registered_document,
        document_id,
    )

    return {
        "status": "uploaded",
        "document_id": document_id,
        "file_name": original_name,
        "message": (
            "The replacement has been queued. "
            "The previous chunks remain available "
            "until the new version is indexed."
        ),
    }


@app.get(
    "/admin/documents",
    dependencies=[Depends(require_admin)],
)
async def get_documents():
    return {
        "documents": list_documents()
    }


@app.get(
    "/admin/documents/{document_id}",
    dependencies=[Depends(require_admin)],
)
async def get_document_status(
    document_id: str,
):
    document = get_document(
        document_id
    )
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )
    return document


@app.delete(
    "/admin/documents/{document_id}",
    dependencies=[Depends(require_admin)],
)
async def delete_document(
    document_id: str,
):
    try:
        return delete_registered_document(
            document_id
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
