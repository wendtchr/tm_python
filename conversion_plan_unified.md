# Shiny-to-FastAPI Migration Plan

### Goal

Migrate the existing Python Shiny app to a FastAPI-based architecture while preserving behavior parity for data validation, cleaning, topic modeling, visualizations, session outputs, and downloads.

### Non-Goals (v1)

- No algorithmic redesign of NLP, BERTopic, or visualization logic
- No distributed task queue (Celery/Redis)
- No SSE/WebSocket push updates (polling only)
- No auth hardening beyond local/service-level safeguards unless required by deployment

## 2. Unified Best Practices (from v1 + v2)

1. Keep routers thin and business logic in services.
2. Use Pydantic models for every request and response contract.
3. Maintain deterministic session artifact paths (`outputs/{session_id}/...`).
4. Offload blocking/CPU-heavy work from request handlers.
5. Use explicit orchestration states instead of implicit reactive chains.
6. Keep migration parity-first before feature expansion.
7. Use structured error payloads and stable status semantics.
8. Add regression tests before cutover.

## 3. Source of Truth and Scope Baseline

### Code Source of Truth

- Existing runtime behavior in `app_files/modules/server.py`
- UI controls/defaults in `app_files/modules/ui.py`
- Config defaults in `app_files/modules/config.py`
- Data pipeline and validation in `app_files/modules/data_processing.py`
- Modeling behavior in `app_files/modules/topic_modeling.py`
- Visualization output behavior in `app_files/modules/visualization.py`
- Session/status lifecycle in `app_files/modules/app_core.py`

### Documentation Inputs

- Architecture and pipeline references in `codebase-analysis-docs/CODEBASE_KNOWLEDGE.md`
- Flow diagrams and artifact expectations in `codebase-analysis-docs/assets/ARCHITECTURE_DIAGRAMS.md`

## 4. Target Architecture

## 4.1 Layers

- API Layer: FastAPI routers only (validation wiring, HTTP response mapping)
- Service Layer: data processing, modeling orchestration, visualization generation, downloads
- Core Layer: settings, session context, task registry, exceptions, logging
- Schema Layer: request/response/domain models
- Frontend Layer: workflow UI consuming REST endpoints

## 4.2 Suggested Package Structure

```text
fastapi_app/
  main.py
  api/
    routers/
      health.py
      sessions.py
      data.py
      model.py
      visualizations.py
      downloads.py
  services/
    session_service.py
    data_service.py
    attachments_service.py
    cleaning_service.py
    modeling_service.py
    visualization_service.py
    artifact_service.py
    orchestration_service.py
  core/
    settings.py
    dependencies.py
    task_registry.py
    exceptions.py
    logging.py
    types.py
  schemas/
    requests.py
    responses.py
    domain.py
```

## 5. Parity Workflow and Stage State Machine

State transitions are explicit and enforced:

- `INIT` -> `LOADED` via upload
- `LOADED` -> `ATTACHMENTS_PROCESSED` via attachment step (optional)
- `LOADED` or `ATTACHMENTS_PROCESSED` -> `CLEANED`
- `CLEANED` -> `MODELING_RUNNING` -> `MODELED`
- Any state -> `ERROR` on failure
- Any active state -> `DELETED` via session delete

Rules:

1. Modeling endpoint rejects requests unless state is `CLEANED`.
2. Duplicate model-run submission while already running returns conflict.
3. GET endpoints are read-only and never mutate stage.
4. Session delete is idempotent and performs cleanup.

## 6. Unified API Surface

### Session and Health

- `POST /api/sessions`
- `DELETE /api/sessions/{session_id}`
- `GET /api/health`

### Data Workflow

- `POST /api/sessions/{session_id}/upload`
- `POST /api/sessions/{session_id}/attachments/process`
- `POST /api/sessions/{session_id}/clean`
- `GET /api/sessions/{session_id}/files`

### Modeling Workflow

- `POST /api/sessions/{session_id}/model/run`
- `GET /api/sessions/{session_id}/tasks/{task_id}`
- `GET /api/sessions/{session_id}/model/summary`
- `GET /api/sessions/{session_id}/model/results`

### Visualizations

- `GET /api/sessions/{session_id}/visualizations/topic`
- `GET /api/sessions/{session_id}/visualizations/hierarchy`
- `GET /api/sessions/{session_id}/visualizations/wordcloud`
- `GET /api/sessions/{session_id}/visualizations/word-scores`
- `GET /api/sessions/{session_id}/visualizations/alignment`

### Downloads

- `GET /api/sessions/{session_id}/downloads/{artifact_name}`

## 7. Request/Response Contracts

Minimum required schemas:

- `SessionCreateRequest`
- `SessionResponse`
- `UploadRequest` (multipart metadata)
- `AttachmentProcessRequest`
- `CleanDataRequest`
- `ModelRunRequest`
- `TaskAcceptedResponse`
- `TaskStatusResponse`
- `DataTableResponse`
- `TopicSummaryResponse`
- `VisualizationResponse`
- `FileListResponse`
- `ErrorResponse`

`TaskStatusResponse` should include:

- `task_id`
- `session_id`
- `status` (`pending|running|completed|failed|cancelled`)
- `progress` (0-100)
- `message`
- `created_at`
- `updated_at`
- `result` (optional)
- `error` (optional)

Validation defaults and bounds come from:

- UI defaults in `app_files/modules/ui.py`
- Config defaults in `app_files/modules/config.py`

## 8. Session, Artifact, and Download Strategy

## 8.1 Session Context

Each request resolves a session-scoped filesystem context:

- `outputs/{session_id}/temp`
- `outputs/{session_id}/visualizations`
- `outputs/{session_id}/reports`

Use collision-resistant session IDs (timestamp + random suffix or UUID).

## 8.2 Artifact Registry

Downloads are key-based, not free-path access. Define a whitelist map:

- Data outputs (`df_initial`, `df_clean`, `df_processed`)
- Model outputs (`model`, `topic_info`)
- Visualization outputs (`topic`, `hierarchy`, `wordcloud`, `word_scores`, `alignment`)
- Reports (`topic_summary`, optional comparison outputs)

Unknown artifact key returns validation error; missing file returns not-found.

## 9. Async, Concurrency, and Task Execution

1. Route handlers are `async def`.
2. Blocking pandas/modeling/plot generation is offloaded to executor workers.
3. Model run endpoint submits background task and returns `task_id` immediately.
4. Progress is exposed via polling endpoint.
5. v1 restart behavior: in-flight tasks are marked failed and must be rerun.
6. Add dedupe guard per session for model run to prevent concurrent duplicate jobs.

## 10. Frontend Delivery Plan (Unified)

### Stage D1 (Parity UI)

Build a minimal workflow client for:

- Upload
- Process attachments
- Clean
- Run model
- Poll status
- View tables/visualizations/downloads

### Stage D2 (React SPA)

Implement full React UI with dynamic seed-topic editing and richer dashboard behavior after API contracts stabilize.

This preserves parity speed while still delivering a React target architecture.

## 11. Sequenced Implementation Plan

### Phase A: Foundation

- FastAPI app factory, routers registration, lifespan hooks
- settings via pydantic-settings
- shared dependencies
- exception and logging framework

### Phase B: Service Extraction

- move app logic from `app_files/modules/*` into service classes
- preserve algorithmic behavior
- remove Shiny coupling

### Phase C: Router Implementation

- implement endpoints with response models only
- enforce stage gating
- integrate task registry and session context

### Phase D: Frontend

- D1 parity UI
- D2 React SPA

### Phase E: Integration and Deployment

- Docker Compose runtime
- endpoint, workflow, and parity tests
- performance and cleanup checks

Dependency order:

- B depends on A
- C depends on B
- D1 starts after C contracts are stable
- D2 can parallelize once contracts freeze
- E after C + D integration

## 12. Testing and Acceptance

## 12.1 Contract Tests

- Every endpoint validates request schema and returns declared response shape.
- Every error path returns structured `ErrorResponse`.

## 12.2 Workflow Tests

- Required stage ordering is enforced.
- Duplicate model submission is rejected while running.
- Optional attachment path works when applicable.

## 12.3 Parity Tests

- Required-column and file validation behavior matches current app.
- Equivalent topic summary/result semantics.
- Equivalent artifact availability and download behavior for completed sessions.
- Equivalent visualization availability by completion stage.

## 12.4 Concurrency and Cleanup Tests

- Parallel sessions remain isolated.
- Background jobs do not block API responsiveness.
- Session cleanup works for both successful and failed runs.

## 12.5 Docker Readiness

- health endpoint reports ready
- writable output volume exists
- non-blocking behavior under modeling load

## 13. Risk Register and Mitigations

1. Reactive-to-request mismatch

- Mitigation: explicit orchestration service + state machine tests

1. CPU-heavy modeling blocking API

- Mitigation: executor offload + task polling + dedupe guard

1. Session/file leakage

- Mitigation: scoped session dirs + cleanup policy + delete endpoint

1. Behavior drift during refactor

- Mitigation: parity fixtures and regression tests before cutover

1. UI rewrite scope creep

- Mitigation: D1 parity UI, then D2 React enhancements

## 14. Deliverables

1. FastAPI backend with validated REST contracts
2. Service-layer migration of existing core logic
3. Session-scoped deterministic artifact pipeline
4. Polling-based task status endpoint
5. Staged frontend (D1 parity UI, D2 React SPA)
6. Docker Compose deployment assets
7. Test suite for contracts, workflow, and parity

## 15. Execution Rule

Implement sequentially by phases and preserve parity semantics first. Defer non-parity enhancements until post-cutover stabilization.
