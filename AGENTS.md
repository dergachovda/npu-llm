# Project Guidelines & Workflow

## 🛠 Tech Stack & Standards
- **Runtime:** Python 3.14 (experimental/latest)
- **Package Manager:** `uv` (for dependency management and script execution)
- **Linting & Formatting:** `ruff` (check and format)
- **Static Analysis:** `pyright` (strict mode)
- **Configuration:** Centralized via `.env` and `python-dotenv` (stored in `src/npu_llm/config.py`)
- **Structure:** `src/` layout for package source, `tests/` for testing.

## 📦 Project Commands
- `uv run npu-llm`: Starts the LLM chat application.
- `uv run validation`: Runs the full quality suite (`ruff check` $\rightarrow$ `ruff format` $\rightarrow$ `pyright`).

## 📝 Python Best Practices
- **Type Hinting:** Strict type hints for all function signatures.
- **Docstrings:** Google-style docstrings for all public modules and functions.
- **Encoding:** Force UTF-8 for Windows consoles to support Cyrillic text.
- **Configuration:** No hardcoded secrets or environment-specific paths in business logic; use `src/npu_llm/config.py`.
- **Tooling:** Prefer `uv` over `pip` for faster, deterministic builds.

## 🔄 IPD Workflow Quick Reference
This project uses the **Ideas $\rightarrow$ Plans $\rightarrow$ Decisions** (IPD) workflow to ensure traceability. All workflow artifacts are stored in the hidden `.ipd/` directory.

### Core Loop
1. **Idea**: Capture a spark in `.ipd/ideas/`.
2. **Plan**: Formalize the approach in `.ipd/plans/`.
3. **Decision**: Document the final choice in `.ipd/decisions/`.

See `.ipd/AGENTS.md` for the full specification and command routing.
