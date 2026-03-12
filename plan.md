# Shiny to FastAPI Migration Plan (TM_PYTHON)

## Goal
Migrate the app from Python Shiny to FastAPI while preserving the core data processing, BERTopic modeling, visualization generation, and session-output behavior.

## Inputs Used
- `codebase-analysis-docs/README.md`
- `codebase-analysis-docs/CODEBASE_KNOWLEDGE.md`
- `codebase-analysis-docs/assets/ARCHITECTURE_DIAGRAMS.md`
- Current source under `app_files/`

---

## 1) FastAPI + Python Best-Practice Approach

### Architectural target
Use a layered FastAPI design:
- **API layer**: routers only (request parsing + response shaping)
- **Service layer**: business logic and orchestration
- **Core layer**: session lifecycle, config, exceptions, dependencies
- **Schema layer**: Pydantic request/response/domain models

### Recommended package structure
```text
fastapi_app/
  main.py
  api/routers/
    sessions.py
    upload.py
    processing.py
    modeling.py
    visualizations.py
    downloads.py
    seed_topics.py
  services/
    data_processing.py
    topic_modeling.py
    visualization.py
    orchestration.py
  core/
    session_manager.py
    dependencies.py
    exceptions.py
    logging.py
  schemas/
    requests.py
    responses.py
    domain.py
  tasks/
    worker.py
```

### Practices to follow
1. Keep routers thin; move workflow logic into `services/orchestration.py`.
2. Validate all payloads with Pydantic (`requests.py`, `responses.py`).
3. Map typed domain exceptions to explicit HTTP responses (no silent failures).
4. Use structured logging and include session/job IDs in logs.
5. Keep CPU-heavy BERTopic work off the event loop (threadpool/background worker).
6. Start with simple in-process background execution; move to Celery/Redis only if needed.
7. Keep output directory conventions stable (`outputs/{session_id}/...`) for compatibility.

---

## 2) What to Replace, Modify, and Keep

## Replace (Shiny framework-specific)
- `app_files/app.py` -> replace with `fastapi_app/main.py`
- `app_files/modules/ui.py` -> replace with frontend UI (SPA/static HTML)
- `app_files/modules/server.py` -> replace with FastAPI routers + orchestration services
- `launcher.py` -> replace with uvicorn startup (`uvicorn fastapi_app.main:app ...`)

## Keep but Modify (preserve logic, adapt interfaces)
- `app_files/modules/app_core.py`
  - Keep path/session storage concepts
  - Refactor `SessionManager` to explicit API session lifecycle
- `app_files/modules/config.py`
  - Keep defaults/constants
  - Add environment settings layer (`pydantic-settings`)
- `app_files/modules/core_types.py`
  - Keep domain types/protocol intent
  - Remove/adjust framework-coupled aliases/protocols
- `app_files/modules/data_processing.py`
  - Keep CSV, validation, cleaning, split logic
  - Replace Shiny status cadence assumptions with API/job-status updates
- `app_files/modules/topic_modeling.py`
  - Keep BERTopic + seed-topic pipeline
  - Remove Shiny/decorator coupling; expose service-friendly methods
- `app_files/modules/visualization.py`
  - Keep visualization generation and output writes
  - Return API-friendly metadata where appropriate
- `app_files/modules/decorators.py`
  - Keep only generic cross-cutting decorators if still useful
  - Remove Shiny-specific behavior
- `app_files/modules/utils.py`
  - Keep utility functions; split by concern if needed
- `app_files/modules/__init__.py`
  - Rewrite exports for new package boundaries

## Keep As-Is or Near As-Is
- `app_files/www/style.css` (reuse in new UI)
- Most algorithm/config defaults in `config.py`
- Core data/model processing logic when detached from Shiny callbacks

## Add New Files
- `fastapi_app/main.py`
- `fastapi_app/api/routers/*.py`
- `fastapi_app/services/orchestration.py`
- `fastapi_app/schemas/{requests.py,responses.py,domain.py}`
- `fastapi_app/core/{dependencies.py,exceptions.py,logging.py,session_manager.py}`
- `fastapi_app/tasks/worker.py`

## Scripts/Docs to Update
- `run.bat` -> run uvicorn
- `setup.bat` -> install FastAPI stack and remove Shiny checks
- `app_files/requirements.txt` -> remove Shiny-only deps, add FastAPI deps
- `readme.md` -> new startup/API usage/migration notes

---

## 3) Reactive Shiny Flow -> FastAPI Endpoint Mapping

- `input.load_data` -> `POST /api/upload`
- `input.process_attachments` -> `POST /api/process/attachments`
- `input.clean_data` -> `POST /api/process/clean`
- `input.run_modeling` -> `POST /api/modeling/start`
- status reactive updates -> `GET /api/modeling/{job_id}/status`
- seed topic add/remove -> `POST/DELETE /api/seed-topics`
- rendered visual outputs -> `GET /api/visualizations/{type}`
- downloads -> `GET /api/downloads/{session_id}/{filename}`
- implicit Shiny session lifecycle -> `POST/DELETE /api/sessions`

---

## 4) Phased Migration Plan

## Phase 0 - Baseline parity capture
- Freeze current behavior with fixture datasets and output expectations.
- Define parity checks: required columns, outputs generated, error behavior.

## Phase 1 - FastAPI foundation
- Create `fastapi_app` app factory and health endpoint.
- Add config/settings, logging, exception handlers.

## Phase 2 - Session + upload API
- Refactor session/path management for explicit API sessions.
- Implement session create/get/delete and upload endpoints.

## Phase 3 - Port service layer
- Port `data_processing`, `topic_modeling`, `visualization` logic.
- Remove Shiny imports/callback assumptions.

## Phase 4 - Orchestration + background jobs
- Add `modeling/start` + `modeling/status` endpoints.
- Execute BERTopic in background worker/threadpool.

## Phase 5 - UI migration
- Replace Shiny UI with either:
  - Minimal static client for parity, then
  - SPA (React/Vue) for long-term maintainability.

## Phase 6 - Hardening + cutover
- Run integration/regression tests.
- Validate performance and cleanup behavior.
- Switch runtime scripts/docs and deprecate Shiny entrypoints.

---

## 5) Testing/Validation Strategy

### Unit tests
- Data cleaning/splitting
- Seed topic parsing and model setup
- Visualization generation and output file creation

### API tests
- Session lifecycle
- Upload and validation behavior
- Modeling start/status lifecycle
- Downloads and error responses

### Integration test
- Full workflow: create session -> upload -> process -> model -> visualize -> download

### Non-functional checks
- CPU/memory behavior during BERTopic tasks
- Session cleanup correctness (temp/output lifecycle)
- Output compatibility with existing expected artifacts

---

## 6) Risks and Mitigations

1. **Reactive -> request/response mismatch**
   - Mitigate with explicit orchestration state + status endpoint.
2. **CPU-bound modeling blocking async server**
   - Mitigate with background execution model.
3. **Session/file leaks**
   - Mitigate with TTL cleanup + explicit delete endpoint + startup sweep.
4. **Behavior drift during refactor**
   - Mitigate by preserving defaults and validating parity with fixtures.
5. **UI rewrite scope creep**
   - Mitigate via staged UI migration (parity first, enhancements later).

---

## 7) Direct Answer to Your Core Requirement

To keep most core logic intact and only replace framework parts:
- **Keep and port** business modules (`data_processing.py`, `topic_modeling.py`, `visualization.py`) with minimal internal changes.
- **Refactor boundaries** (`app_core.py`, `core_types.py`, `decorators.py`, imports/config glue).
- **Replace only Shiny-specific pieces** (`app_files/app.py`, `modules/server.py`, `modules/ui.py`, `launcher.py`, and Shiny dependencies/runtime scripts).

This gives the lowest-risk migration path with highest logic reuse.
