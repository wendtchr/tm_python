Here's a robust prompt designed for this conversion task:

---

## Prompt: Convert Python Shiny App to FastAPI App

You are an expert Python developer specializing in FastAPI, async web applications, and UI/API separation. I need to convert a Python Shiny app to a FastAPI application.

### Context & Goals

- Convert a reactive Shiny app to a REST API + frontend architecture
- Preserve all existing business logic and data processing
- Follow FastAPI best practices: async patterns, Pydantic models, dependency injection
- Produce clean, production-ready code

---

### Phase 1 — Codebase Audit

Before writing any code, analyze the provided Shiny app and produce a structured inventory:

**Inputs & Reactive Sources**

- List every `ui.input_*` widget (type, id, default value, purpose)
- Identify reactive dependencies between inputs

**Outputs & Renders**

- List every `@render.*` function (type: plot/table/text/ui)
- Map each render to its upstream inputs and data sources

**Reactive Logic**

- List all `@reactive.calc`, `@reactive.effect`, and `reactive.Value` usages
- Identify side effects (file writes, DB calls, session state mutations)

**Data Layer**

- List all data loading functions, file reads, database connections
- Identify what is static (load-once) vs. dynamic (per-request)

**External Dependencies**

- List all imports, third-party packages, and version constraints

---

### Phase 2 — Architecture Design

Design the FastAPI architecture **before writing code**, addressing:

**API Structure**

- Propose URL routes for each Shiny output/action, using RESTful conventions
- Identify which endpoints are GET vs POST
- Flag any endpoints requiring request bodies (define Pydantic input models)
- Flag any endpoints returning files/streams (plots, CSVs)

**State Management Strategy**

- Shiny uses server-side reactive session state. Specify how each piece of state will be handled in FastAPI:
  - URL query parameters
  - Request body
  - Server-side cache (e.g., `functools.lru_cache`, Redis)
  - No state needed (pure functions)

**Response Models**

- Define Pydantic `BaseModel` schemas for every API response
- Define Pydantic models for every request body

**Async Strategy**

- Identify which functions should be `async def` vs `def`
- Flag any blocking I/O (pandas reads, model inference, DB queries) that needs `run_in_executor` or a task queue

**Frontend Strategy** — choose one and justify:

- Jinja2 server-side templates
- Static HTML/JS (fetch API calls to FastAPI)
- React/Vue SPA
- Keep a minimal Shiny-like UI using a framework like Reflex or Gradio

**Dependency Injection Plan**

- Identify shared resources (DB connections, config, ML models) to expose via `Depends()`

---

### Phase 3 — Implementation Plan

Produce a sequenced, file-by-file implementation plan:

```
project/
├── main.py               # FastAPI app, router registration, lifespan events
├── routers/
│   └── [domain].py       # One router per logical feature group
├── models/
│   ├── requests.py       # Pydantic input models
│   └── responses.py      # Pydantic response models
├── services/
│   └── [domain].py       # Business logic (ported from Shiny server functions)
├── data/
│   └── loader.py         # Data loading, caching
├── dependencies.py        # FastAPI Depends() providers
├── config.py             # Settings via pydantic-settings
└── frontend/             # Templates or static files
```

For each file, specify:

- What Shiny code maps to it
- Key functions/classes to implement
- Any async considerations

---

### Phase 4 — Shiny-to-FastAPI Mapping Rules

Apply these translation rules explicitly:

| Shiny Pattern                       | FastAPI Equivalent                              |
| ----------------------------------- | ----------------------------------------------- |
| `ui.input_slider(id, ...)`          | Query param or request body field               |
| `ui.input_select(id, ...)`          | Query param with `Enum` validation              |
| `@render.plot`                      | `Response(content=..., media_type="image/png")` |
| `@render.table` / `render.DataGrid` | JSON endpoint returning list of dicts           |
| `@render.text`                      | JSON endpoint returning `{"value": ...}`        |
| `@reactive.calc`                    | Service function (cached if pure)               |
| `@reactive.effect`                  | Background task or event hook                   |
| `reactive.Value`                    | Request body state or cache entry               |
| `@reactive.event`                   | POST endpoint triggered by user action          |
| `session.send_custom_message`       | WebSocket or SSE if real-time push is needed    |

---

### Phase 5 — Constraints & Quality Gates

The output must satisfy:

- [ ] All Pydantic models have field-level validation and docstrings
- [ ] No business logic in route handlers — delegated to `services/`
- [ ] All blocking I/O is either `async` or wrapped in `run_in_executor`
- [ ] Every route has a response model declared
- [ ] Config is loaded via `pydantic-settings`, never hardcoded
- [ ] Error handling uses `HTTPException` with meaningful status codes
- [ ] Startup/shutdown resource management uses `@asynccontextmanager` lifespan
- [ ] All functions have type hints and docstrings

---

### Inputs Required From Me

Before proceeding, confirm you have:

1. The full Shiny app source code
2. Any `.requirements.txt` / `pyproject.toml`
3. Any data files or schema definitions
4. The intended frontend strategy (or ask me to decide after Phase 2)
5. Deployment target (bare server, Docker, cloud function) — affects async/worker choices

---

### Output Format

Deliver phases sequentially. Do not write implementation code until Phase 1 audit and Phase 2 architecture design have been reviewed and approved.

---

This prompt enforces a plan-before-code workflow, prevents scope creep, and gives the AI enough structure to handle the stateful→stateless paradigm shift that's the core challenge of this conversion. Want me to add a WebSocket handling section or a testing strategy phase?
