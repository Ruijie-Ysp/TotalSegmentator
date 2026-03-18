# Repository Guidelines

## Project Structure & Module Organization
- `totalsegmentator/`: core Python package (inference, preprocessing, postprocessing, Python API).
- `totalsegmentator/bin/`: CLI entry points such as `TotalSegmentator` and utility commands.
- `api/`: FastAPI backend services (task orchestration, storage, Label Studio integration).
- `web/`: Next.js + TypeScript viewer UI (`web/src/components`, `web/src/lib`, `web/src/app`).
- `tests/`: regression/integration tests; test assets live in `tests/reference_files/`.
- Root utilities include `app.py` (Gradio demo), `download_all_weights.py`, and `docker-compose.yml`.

## Build, Test, and Development Commands
- `python -m pip install -e .`: install the Python package in editable mode.
- `pip install -r api/requirements.txt`: install backend API dependencies.
- `npm run install:web`: install frontend dependencies in `web/`.
- `npm run start:debug`: run API (`:28000`) and web app (`:23000`) with reload.
- `npm run build`: build the frontend bundle.
- `pre-commit run -a`: run lint/quality hooks (Ruff, pyupgrade, codespell, checks).
- `pytest -v tests/test_device_type.py`: quick Python smoke test.
- `./tests/tests.sh`: full CLI regression flow (long-running; downloads/uses model weights).

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes.
- Keep public CLI behavior stable unless the change is explicitly breaking.
- Linting is configured via Ruff in `pyproject.toml`; prefer `pre-commit` as the source of truth.
- Frontend: React components use `PascalCase` filenames (for example `NiiVueViewer.tsx`); shared helpers in `web/src/lib/` use concise lower-case names.

## Testing Guidelines
- Put new tests under `tests/` and follow `test_*.py` naming.
- Start with targeted tests (`pytest -v path/to/test_file.py::test_name`) before running broader suites.
- For API-related changes, extend focused tests such as `tests/test_labelstudio_service.py` where possible.
- For CLI/inference changes, run at least one script-based regression (`tests/tests.sh`, `tests/tests_os.py`, or `tests/tests_nnunet.py`).

## Commit & Pull Request Guidelines
- Match existing history style: short, imperative commit subjects (for example `fix linting errors`, `update readme`).
- Keep commits focused; separate refactors from behavior changes.
- PRs should include: change summary, motivation, validation commands run, and any required env/config updates.
- Include screenshots for UI changes and sample request/response payloads for API changes.

## Security & Configuration Tips
- Copy `.env.example` to `.env`; do not commit secrets, license keys, or machine-specific paths.
- Keep generated outputs and downloaded weights out of Git (`outputs/`, `api_data/`, `~/.totalsegmentator/`).
- Enable optional annotation services only when needed (for example `docker compose --profile labelstudio up -d`).
