#!/usr/bin/env python3
"""Coordinate the reproducible RAG benchmark preparation workflow.

Subcommands audit the frozen corpus, prepare and validate candidate evaluation
data, promote an approved candidate, run evaluation, and report corpus state.
It delegates each stage to the specialized scripts in this directory.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hkpl_agent.infrastructure.db import engine
from hkpl_agent.infrastructure.table_names import configured_table_name
from hkpl_agent.infrastructure.vector_store import VECTOR_TABLE_NAME


SCRIPTS = PROJECT_ROOT / "scripts"
DEFAULT_CANDIDATE = PROJECT_ROOT / "data" / "evaluation_dataset.candidate.csv"
DEFAULT_ACTIVE = PROJECT_ROOT / "data" / "evaluation_dataset.csv"
CANDIDATE_TABLE = configured_table_name(
    "EVALUATION_CANDIDATE_TABLE",
    "evaluation_dataset_100",
)
ACTIVE_TABLE = configured_table_name(
    "EVALUATION_ACTIVE_TABLE",
    "evaluation_dataset",
)


def run_script(name: str, *arguments: str, env: dict[str, str] | None = None) -> None:
    command = [sys.executable, str(SCRIPTS / name), *arguments]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env or os.environ.copy(),
            check=False,
        )
    except KeyboardInterrupt:
        print("Workflow interrupted. Candidate checkpoints were preserved.")
        raise SystemExit(130) from None
    if completed.returncode:
        raise SystemExit(completed.returncode)


def corpus_counts() -> dict[str, int]:
    with engine.connect() as connection:
        rows = connection.execute(text(f"""
            SELECT
                COALESCE(NULLIF(metadata_->>'dataset', ''), 'hkpl') AS dataset,
                COUNT(*) AS vectors
            FROM {VECTOR_TABLE_NAME}
            GROUP BY dataset
            ORDER BY dataset
        """)).mappings().all()
    return {str(row["dataset"]): int(row["vectors"]) for row in rows}


def print_status(*, require_distractors: bool = False) -> None:
    counts = corpus_counts()
    total = sum(counts.values())
    print("Vector corpus status")
    for dataset in ("hkpl", "hotpotqa", "webz_news"):
        print(f"- {dataset}: {counts.get(dataset, 0)}")
    print(f"- total: {total}")

    if counts.get("hkpl", 0) <= 0:
        raise SystemExit("HKPL corpus is empty. Ingest authoritative documents first.")
    if require_distractors:
        missing = [
            dataset
            for dataset in ("hotpotqa", "webz_news")
            if counts.get(dataset, 0) <= 0
        ]
        if missing:
            raise SystemExit(
                "Missing distractor corpora: " + ", ".join(missing)
            )


def candidate_environment(candidate: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["EVALUATION_DATASET_PATH"] = str(candidate)
    environment["EVALUATION_DATASET_TABLE"] = CANDIDATE_TABLE
    return environment


def audit_corpus() -> None:
    print_status()
    run_script("ingest_pgvector_llamaindex.py", "--audit-chunks")


def validate_candidate(candidate: Path) -> None:
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    run_script(
        "normalize_evaluation_schema.py",
        "--path",
        str(candidate),
        "--check",
    )
    environment = candidate_environment(candidate)
    run_script(
        "ingest_pgvector_llamaindex.py",
        "--evaluation-only",
        env=environment,
    )
    run_script("validate_evaluation_dataset.py", env=environment)


def prepare_candidate(args: argparse.Namespace) -> None:
    audit_corpus()
    candidate = args.output.resolve()
    generation_arguments = ["--output", str(candidate)]
    if args.resume:
        generation_arguments.append("--resume")
    if args.all_chunks:
        generation_arguments.append("--all-chunks")
    if args.limit_chunks is not None:
        generation_arguments.extend(["--limit-chunks", str(args.limit_chunks)])
    if args.target_questions is not None:
        generation_arguments.extend([
            "--target-questions",
            str(args.target_questions),
        ])
    run_script("generate_evaluation_dataset.py", *generation_arguments)
    validate_candidate(candidate)
    print()
    print(f"Candidate ready for semantic label review: {candidate}")
    print("Do not promote it until every ambiguous/time-sensitive label is reviewed.")


def promote_candidate(args: argparse.Namespace) -> None:
    if not args.yes:
        raise SystemExit("Promotion requires --yes after semantic label review.")
    candidate = args.candidate.resolve()
    active = args.active.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    validate_candidate(candidate)

    active.parent.mkdir(parents=True, exist_ok=True)
    if active.is_file():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = active.with_name(f"{active.stem}.before-promotion-{timestamp}.csv")
        shutil.copy2(active, backup)
        print(f"Active benchmark backup: {backup}")
    shutil.copy2(candidate, active)

    active_environment = os.environ.copy()
    active_environment["EVALUATION_DATASET_PATH"] = str(active)
    active_environment["EVALUATION_DATASET_TABLE"] = ACTIVE_TABLE
    run_script(
        "ingest_pgvector_llamaindex.py",
        "--evaluation-only",
        env=active_environment,
    )
    run_script("validate_evaluation_dataset.py", env=active_environment)
    print(f"Promoted active benchmark: {active}")


def evaluate(args: argparse.Namespace) -> None:
    print_status(require_distractors=True)
    run_script("validate_evaluation_dataset.py")
    arguments: list[str] = []
    if args.limit is not None:
        arguments.extend(["--limit", str(args.limit)])
    if args.phoenix_project:
        arguments.extend(["--phoenix-project", args.phoenix_project])
    if args.rerun_answer_failures_from is not None:
        arguments.extend(
            [
                "--rerun-answer-failures-from",
                str(args.rerun_answer_failures_from),
            ]
        )
    if args.answer_reasoning:
        arguments.append("--answer-reasoning")
    if args.evaluator_reasoning:
        arguments.append("--evaluator-reasoning")
    arguments.extend(["--reasoning-budget", str(args.reasoning_budget)])
    run_script("evaluate_rag.py", *arguments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enforce corpus-first HKPL benchmark generation, review, promotion, "
            "and evaluation."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show HKPL and distractor vector counts.")
    commands.add_parser("audit-corpus", help="Audit the finalized vector corpus.")

    validate = commands.add_parser(
        "validate-candidate",
        help="Validate an already generated candidate without regenerating it.",
    )
    validate.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)

    prepare = commands.add_parser(
        "prepare-candidate",
        help="Audit HKPL vectors, then generate and evidence-validate a candidate.",
    )
    prepare.add_argument("--output", type=Path, default=DEFAULT_CANDIDATE)
    prepare.add_argument("--resume", action="store_true")
    prepare.add_argument("--all-chunks", action="store_true")
    prepare.add_argument("--limit-chunks", type=int, default=None)
    prepare.add_argument("--target-questions", type=int, default=None)

    promote = commands.add_parser(
        "promote",
        help="Promote a reviewed, evidence-valid candidate to the active benchmark.",
    )
    promote.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    promote.add_argument("--active", type=Path, default=DEFAULT_ACTIVE)
    promote.add_argument("--yes", action="store_true")

    evaluation = commands.add_parser(
        "evaluate",
        help="Require a valid benchmark and both distractors, then run evaluation.",
    )
    evaluation.add_argument("--limit", type=int, default=None)
    evaluation.add_argument("--phoenix-project", default="")
    evaluation.add_argument("--rerun-answer-failures-from", type=Path)
    evaluation.add_argument("--answer-reasoning", action="store_true")
    evaluation.add_argument("--evaluator-reasoning", action="store_true")
    evaluation.add_argument("--reasoning-budget", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "status":
        print_status()
    elif args.command == "audit-corpus":
        audit_corpus()
    elif args.command == "validate-candidate":
        validate_candidate(args.candidate)
    elif args.command == "prepare-candidate":
        prepare_candidate(args)
    elif args.command == "promote":
        promote_candidate(args)
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
