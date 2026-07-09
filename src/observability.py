import logging
import os

logger = logging.getLogger(__name__)
_tracing_initialized = False


def setup_phoenix_tracing() -> None:
    global _tracing_initialized

    if _tracing_initialized:
        return

    if os.getenv("PHOENIX_ENABLED", "false").lower() != "true":
        logger.info("Phoenix tracing disabled.")
        return

    try:
        from phoenix.otel import register
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        endpoint = os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://phoenix:6006/v1/traces",
        )

        tracer_provider = register(
            project_name="hkpl-rag",
            endpoint=endpoint,
        )

        LlamaIndexInstrumentor().instrument(
            tracer_provider=tracer_provider,
        )

        _tracing_initialized = True
        logger.info("Phoenix tracing enabled: %s", endpoint)

    except Exception:
        logger.exception("Failed to enable Phoenix tracing.")