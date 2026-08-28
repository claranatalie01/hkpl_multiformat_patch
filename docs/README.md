# Documentation

The root `README.md` is intentionally a short landing page. Detailed guidance
is split by responsibility so that operational commands do not obscure the
system design.

| Guide | Use it when you need to |
|---|---|
| [Architecture and code map](architecture.md) | Understand how packages and files connect |
| [Ingestion](ingestion.md) | Crawl, extract, chunk, embed, or rebuild vectors |
| [Evaluation](evaluation.md) | Generate datasets, validate rows, evaluate RAG, or inspect Phoenix |
| [Operations](operations.md) | Manage locks, tables, services, cron, or common failures |
| [Development](development.md) | Run checks, tests, CI, and container release workflows |

The target production requirements remain in `AGENTS.md`. These guides
describe the code that is currently present in this repository.
