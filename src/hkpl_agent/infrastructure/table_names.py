"""Validate configured PostgreSQL table and trigger identifiers.

Several maintenance scripts interpolate operator-selected table names into SQL
because PostgreSQL bind parameters cannot represent identifiers.  Keeping the
validation in this module prevents each caller from maintaining a subtly
different regular expression and gives vector, evaluation, and lock tables one
consistent naming policy.
"""

from __future__ import annotations

import os
import re


_SQL_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def sql_identifier(value: str, *, setting: str = "SQL identifier") -> str:
    """Return ``value`` when it is safe to interpolate as one SQL identifier."""

    normalized = value.strip()
    if not _SQL_IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"Unsafe {setting}: {value!r}")
    return normalized


def configured_table_name(environment_name: str, default: str) -> str:
    """Read and validate a table name from the process environment."""

    return sql_identifier(
        os.getenv(environment_name, default),
        setting=environment_name,
    )


def physical_vector_table_name(collection_name: str) -> str:
    """Return LlamaIndex's physical table for a logical vector collection."""

    logical_name = sql_identifier(collection_name, setting="VECTOR_TABLE")
    return sql_identifier(
        f"data_{logical_name}",
        setting="physical vector table name",
    )
