# Shiny to FastAPI Migration Progress

Last updated: 2026-03-12

## Overall Status

- Current phase: Phase 6 (verification complete, rechecked after Phase 5)
- Parity mode: enabled
- Hardening changes: deferred

## Phase Checklist

- Phase 0: Parity Lock and Baseline Matrix
Status: complete
Evidence: fastapi_app/PARITY_BASELINE.md

- Phase 1: Minimal FastAPI Foundation
Status: complete

Completed:

1. FastAPI app entrypoint and router wiring
2. Health endpoint (`GET /api/health`)
3. Session create/delete endpoints (`POST /api/sessions`, `DELETE /api/sessions/{session_id}`)
4. Deterministic session directory creation under outputs/{session_id}/
5. session.json persistence with explicit stage metadata
6. Targeted tests passed for health/sessions contracts and idempotent delete

Pending:

1. Expand shared error response usage across future routers

- Phase 2: Data Workflow Endpoints
Status: complete

Completed in this ticket:

1. Added `POST /api/sessions/{session_id}/upload` using parity validation for required `Comment` column
2. Persisted upload artifact as `df_initial.csv` in session output root (matching current Shiny behavior)
3. Added stage transition `INIT -> LOADED` persisted in `session.json`
4. Added `GET /api/sessions/{session_id}/files` for session-scoped artifact listing
5. Added `POST /api/sessions/{session_id}/attachments/process` with artifact `df_initial_attach.csv`
6. Added `POST /api/sessions/{session_id}/clean` with artifact `df_initial_attach_clean.csv`
7. Added parity tests for attachments/clean stage transitions and artifact persistence

- Phase 3: Modeling and Polling
Status: complete

Completed in this ticket:

1. Added task registry service with session-scoped active-task guard
2. Added `POST /api/sessions/{session_id}/model/run` enqueue endpoint with immediate `task_id` response
3. Added background modeling orchestration off request thread
4. Added `GET /api/sessions/{session_id}/tasks/{task_id}` polling endpoint
5. Added stage transitions for `MODELING_RUNNING -> MODELED` and error transition to `ERROR`
6. Added duplicate model-run conflict guard (`409 MODEL_ALREADY_RUNNING`)
7. Added phase-specific tests for enqueue, polling success, duplicate conflict, and task-not-found behavior

- Phase 4: Visualization and Downloads
Status: complete

Completed in this ticket:

1. Added read-only visualization endpoints:
   - `GET /api/sessions/{session_id}/visualizations/topic`
   - `GET /api/sessions/{session_id}/visualizations/hierarchy`
   - `GET /api/sessions/{session_id}/visualizations/wordcloud`
   - `GET /api/sessions/{session_id}/visualizations/word-scores`
   - `GET /api/sessions/{session_id}/visualizations/alignment`
2. Added key-based download endpoint:
   - `GET /api/sessions/{session_id}/downloads/{artifact_name}`
3. Added artifact registry with session-scoped safe-path resolution and known-key mapping
4. Added topic-comparison artifact exposure when available (`topic_comparison/*`)
5. Added parity tests covering known key success, unknown key validation errors, and missing artifact not-found behavior

- Phase 5: Thin Parity UI
Status: complete

Completed in this ticket:

1. Added thin parity UI at `GET /` backed by `fastapi_app/frontend/index.html`
2. Implemented minimal end-to-end workflow controls for:
   - session create/delete
   - upload, attachments processing, data cleaning
   - model run + task polling
   - output file refresh and model summary refresh
3. Preserved seed-topic editing and defaults in the UI using locked defaults from Shiny config
4. Added phase-specific UI tests (`tests/fastapi/test_phase5_ui.py`)

- Phase 6: Verification and Cutover
Status: complete (verified after Phase 5)

Completed in this ticket:

1. Added missing model contracts:
   - `GET /api/sessions/{session_id}/model/summary`
   - `GET /api/sessions/{session_id}/model/results`
2. Added structured model-results not-found errors (`MODEL_RESULTS_NOT_FOUND`)
3. Added Phase 6 verification tests for:
   - model summary/results contract behavior
   - model failure stage transition to `ERROR`
   - validation error contract shape
   - parallel session isolation and session cleanup
   - API responsiveness while model background task is running
4. Fixed validation error serialization to guarantee JSON-safe structured payloads
5. Executed full FastAPI regression suite with all tests passing
6. Re-ran full FastAPI regression suite after Phase 5 UI implementation

## API Coverage Snapshot

Implemented:

1. GET /api/health
2. POST /api/sessions
3. DELETE /api/sessions/{session_id}
4. POST /api/sessions/{session_id}/upload
5. GET /api/sessions/{session_id}/files
6. POST /api/sessions/{session_id}/attachments/process
7. POST /api/sessions/{session_id}/clean
8. POST /api/sessions/{session_id}/model/run
9. GET /api/sessions/{session_id}/tasks/{task_id}
10. GET /api/sessions/{session_id}/visualizations/topic
11. GET /api/sessions/{session_id}/visualizations/hierarchy
12. GET /api/sessions/{session_id}/visualizations/wordcloud
13. GET /api/sessions/{session_id}/visualizations/word-scores
14. GET /api/sessions/{session_id}/visualizations/alignment
15. GET /api/sessions/{session_id}/downloads/{artifact_name}
16. GET /api/sessions/{session_id}/model/summary
17. GET /api/sessions/{session_id}/model/results

## Verification Log

2026-03-12:

1. Installed FastAPI-side dependencies from fastapi_app/requirements.txt
2. Ran: C:/python/python.exe -m pytest tests/fastapi/test_health_and_sessions.py -q
3. Result: 3 passed
4. Added upload/files endpoints and reran: C:/python/python.exe -m pytest tests/fastapi/test_health_and_sessions.py -q
5. Result: 6 passed
6. Added attachments/clean endpoints and reran: C:/python/python.exe -m pytest tests/fastapi/test_health_and_sessions.py -q
7. Result: 8 passed
8. Added phase 3 endpoints/services and ran: C:/python/python.exe -m pytest tests/fastapi -q
9. Result: 11 passed
10. Added phase 4 visualization/download endpoints and ran: C:/python/python.exe -m pytest tests/fastapi -q
11. Result: 16 passed
12. Added phase 6 verification tests and model summary/results endpoints; ran: C:/python/python.exe -m pytest tests/fastapi -q
13. Result: 22 passed
14. Added phase 5 thin parity UI with seed topic defaults/editing and ran: C:/python/python.exe -m pytest tests/fastapi -q
15. Result: 24 passed
