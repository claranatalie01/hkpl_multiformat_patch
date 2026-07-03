#!/usr/bin/env python3

import asyncio
import csv
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import retrieve_nodes


DATASET = PROJECT_ROOT / "data" / "evaluation_dataset.csv"
OUTPUT = PROJECT_ROOT / "data" / "retrieval_results.csv"


def reciprocal_rank(expected, retrieved):
    """
    Compute Reciprocal Rank.

    If expected document appears

    position 1 -> 1.0
    position 2 -> 0.5
    position 3 -> 0.333
    ...
    """

    for rank, doc in enumerate(retrieved, start=1):
        if doc == expected:
            return 1.0 / rank

    return 0.0


async def evaluate():

    df = pd.read_csv(DATASET)

    results = []

    hit1 = 0
    hit3 = 0
    hit5 = 0

    rr_total = 0.0

    total = len(df)

    print("=" * 80)
    print("Evaluating retrieval...")
    print("=" * 80)

    for index, row in df.iterrows():

        query = row["query"]

        expected_document = row["source_document_id"]

        try:
            nodes = await retrieve_nodes(query)

        except Exception as e:
            print("=" * 80)
            print("FAILED QUERY")
            print(query)
            print(e)
            print("=" * 80)

        retrieved_documents = []

        retrieved_titles = []

        retrieved_scores = []

        for node in nodes:

            metadata = node.metadata

            document = (
                metadata.get("kb_document_id")
                or metadata.get("document_id")
                or ""
            )

            retrieved_documents.append(document)

            retrieved_titles.append(
                metadata.get("source_title", "")
            )

            retrieved_scores.append(node.score)

        h1 = (
            len(retrieved_documents) >= 1
            and retrieved_documents[0] == expected_document
        )

        h3 = expected_document in retrieved_documents[:3]

        h5 = expected_document in retrieved_documents[:5]

        rr = reciprocal_rank(
            expected_document,
            retrieved_documents,
        )

        hit1 += h1
        hit3 += h3
        hit5 += h5

        rr_total += rr

        results.append(
            {
                "query": query,

                "expected_document": expected_document,

                "retrieved_document_1":
                    retrieved_documents[0]
                    if len(retrieved_documents) > 0 else "",

                "retrieved_document_2":
                    retrieved_documents[1]
                    if len(retrieved_documents) > 1 else "",

                "retrieved_document_3":
                    retrieved_documents[2]
                    if len(retrieved_documents) > 2 else "",

                "title_1":
                    retrieved_titles[0]
                    if len(retrieved_titles) > 0 else "",

                "title_2":
                    retrieved_titles[1]
                    if len(retrieved_titles) > 1 else "",

                "title_3":
                    retrieved_titles[2]
                    if len(retrieved_titles) > 2 else "",

                "score_1":
                    retrieved_scores[0]
                    if len(retrieved_scores) > 0 else "",

                "score_2":
                    retrieved_scores[1]
                    if len(retrieved_scores) > 1 else "",

                "score_3":
                    retrieved_scores[2]
                    if len(retrieved_scores) > 2 else "",

                "hit@1": h1,
                "hit@3": h3,
                "hit@5": h5,
                "reciprocal_rank": rr,
            }
        )

        print(f"[{index+1}/{total}] {query}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()

        writer.writerows(results)

    print()

    print("=" * 80)

    print("Retrieval Evaluation Summary")

    print("=" * 80)

    print(f"Questions          : {total}")

    print(f"Hit@1              : {hit1/total:.2%}")

    print(f"Recall@3 (Hit@3)   : {hit3/total:.2%}")

    print(f"Recall@5 (Hit@5)   : {hit5/total:.2%}")

    print(f"MRR                : {rr_total/total:.4f}")

    print()

    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(evaluate())