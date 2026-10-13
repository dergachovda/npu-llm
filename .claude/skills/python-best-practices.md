# Python Project Best Practices (UV)

This skill ensures the project adheres to modern Python standards, prioritizing speed, type safety, and maintainability using `uv`.

## Tooling Stack
- **Package Manager**: `uv` (Replacement for pip, pip-tools, and virtualenv)
- **Build System**: `pyproject.toml` (PEP 621)
- **Linting & Formatting**: `Ruff` (Replacement for Flake8, Black, isort)
- **Type Checking**: `Mypy` or `Pyright`
- **Data Validation**: `Pydantic`
- **Testing**: `Pytest` (with `pytest-asyncio` for async code)

## Implementation Guidelines

### 1. Dependency Management
- Always use `uv` for adding/removing dependencies:
  - Add: `uv add <package>`
  - Remove: `uv remove <package>`
  - Sync: `uv sync`
- Use `pyproject.toml` for all metadata and tool configurations.

### 2. Code Quality & Style
- **Formatting**: Run `ruff format .`
- **Linting**: Run `ruff check . --fix`
- **Type Hints**: 
  - Every function signature must have type hints.
  - Use `typing` module and Python 3.10+ union types (`X | Y`).
  - Run `mypy .` to verify.

### 3. Architecture
- **Layout**: Maintain the `src/` layout to prevent import ambiguity.
- **Validation**: Use Pydantic models for all API request/response types and configuration.
- **Async**: Use `asyncio` for I/O bound tasks (LLM calls, NPU drivers).
- **Environment**: Use `.env` files for secrets and config; load via `python-dotenv` or Pydantic `BaseSettings`.
- **Logging**: Use structured logging (via `logging` module or `loguru`) with appropriate levels (DEBUG, INFO, ERROR).

### 4. Verification
- Run tests via `uv run pytest`.
- Ensure critical paths in `model_service.py` and `chat_service.py` have unit tests.

## Quality Checklist
- [ ] Is `pyproject.toml` used for configuration?
- [ ] Are dependencies managed via `uv`?
- [ ] Does `ruff` pass without errors?
- [ ] Does `mypy` pass without errors?
- [ ] Are Pydantic models used for data structures?
- [ ] Is the `src/` layout maintained?
