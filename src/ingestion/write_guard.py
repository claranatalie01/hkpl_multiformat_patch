import os


READ_ONLY_ENV = "KNOWLEDGE_CORPUS_READ_ONLY"
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class CorpusReadOnlyError(RuntimeError):
    """Raised when a knowledge-corpus mutation is attempted while frozen."""


def corpus_is_read_only() -> bool:
    return os.getenv(READ_ONLY_ENV, "false").strip().lower() in TRUE_VALUES


def ensure_corpus_writable(operation: str = "modify the knowledge corpus") -> None:
    if corpus_is_read_only():
        raise CorpusReadOnlyError(
            f"Cannot {operation}: {READ_ONLY_ENV}=true. "
            "The benchmark corpus is frozen to keep evaluation evidence stable."
        )
