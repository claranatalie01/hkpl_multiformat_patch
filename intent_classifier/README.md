# Safety intent classifier evaluation

This folder keeps input-safety classifier experiments separate from the RAG
retrieval and answer-quality evaluations under `scripts/`.

## Contents

| File | Purpose |
| --- | --- |
| `evaluate_safety_guards.py` | Compares GLiGuard 300M and Qwen3Guard-Gen-0.6B on WildGuardTest prompt harmfulness. |
| `tests/test_evaluate_safety_guards.py` | Tests dataset accounting, decision rules, parsers, and metrics without downloading models. |

## Dataset

The benchmark downloads the official `allenai/wildguardmix` dataset with the
`wildguardtest` configuration. It evaluates only `prompt_harm_label` because
the HKPL input guard runs before answer generation. The 1,725 test records
contain 1,699 usable prompt-harmfulness labels and 26 records without an agreed
prompt label.

Accept the dataset conditions at
<https://huggingface.co/datasets/allenai/wildguardmix>, then authenticate:

```bash
uv run hf auth login
```

Download and audit the dataset without loading models:

```bash
uv run python intent_classifier/evaluate_safety_guards.py \
  --audit-dataset-only \
  --output-dir data/safety_evaluation/wildguardtest_dataset_audit
```

Run a 20-row smoke test:

```bash
uv run python intent_classifier/evaluate_safety_guards.py \
  --limit 20 \
  --device cuda \
  --output-dir data/safety_evaluation/wildguardtest_smoke
```

Run the complete comparison:

```bash
uv run python intent_classifier/evaluate_safety_guards.py \
  --device cuda \
  --output-dir data/safety_evaluation/wildguardtest
```

Run its model-independent tests:

```bash
uv run python -m unittest intent_classifier.tests.test_evaluate_safety_guards
```
