# Codebase Analysis Documentation Index

**Project:** TM_PYTHON (Topic Modeling Web Application)  
**Analysis Date:** 2026-03-11  
**Status:** Complete - All 6 Analysis Phases Completed

---

## 📚 Documentation Files

### Master Knowledge Document

**File:** `CODEBASE_KNOWLEDGE.md` (12 sections, ~8000 lines)

The complete brain dump for all developers. Contains:

| Section | Content | Use For |
|---------|---------|---------|
| 1. High-Level Overview | What the app does, tech stack, features | Quick orientation |
| 2. System Architecture | Layered design, data flows, component interactions | Understanding design |
| 3. Feature-by-Feature Analysis | 8 features with implementation details | How things work |
| 4. Nuances & Gotchas | Gotchas, error handling, performance | Things to know before editing |
| 5. Technical Reference | APIs, classes, schemas, glossary | Developer reference |
| 6. Dependency Graph | Module imports, import order | Understanding structure |
| 7. Quick Reference | File locations by task | Finding code fast |
| 8. State Tracking | What was analyzed, open questions | Future work |

**Jump to section:**

- Feature 1: File Upload & Processing → Section 3, subsection "FEATURE 1"
- Topic modeling details → Section 3, subsection "FEATURE 4"
- Session management → Section 3, subsection "FEATURE 8"

---

### Architecture Diagrams

**File:** `assets/ARCHITECTURE_DIAGRAMS.md` (text-based diagrams)

Visual representations of codebase structure:

1. **Component Dependency Graph**
   - Shows how modules depend on each other
   - Identifies layers (UI → business logic → infra)
   - Use when: understanding module interaction

2. **Data Processing Pipeline**
   - Step-by-step data transformation flow
   - CSV → embeddings → clustering → visualizations
   - Use when: implementing data processing

3. **Reactive Flow in Shiny**
   - How user interactions trigger updates
   - Input → effect → render → display
   - Use when: troubleshooting UI reactivity

4. **Session Lifecycle**
   - App startup → user session → cleanup
   - When resources created/destroyed
   - Use when: managing file/memory resources

5. **Module Import Order**
   - Why modules import in specific order
   - Circular dependency prevention
   - Use when: adding new modules

6. **Configuration Hierarchy**
   - How config overrides work
   - Where each setting lives
   - Use when: changing application parameters

---

## 🎯 Quick Start Navigation

### "I need to understand..."

| Topic | Start Here |
|-------|-----------|
| **What the app does** | CODEBASE_KNOWLEDGE.md § 1.1 |
| **How the data flows** | CODEBASE_KNOWLEDGE.md § 2.2, then ARCHITECTURE_DIAGRAMS.md |
| **How file upload works** | CODEBASE_KNOWLEDGE.md § 3, FEATURE 1 |
| **Topic modeling algorithm** | CODEBASE_KNOWLEDGE.md § 3, FEATURE 4 |
| **How the UI works** | CODEBASE_KNOWLEDGE.md § 2.4 (Reactive), then ARCHITECTURE_DIAGRAMS.md "Reactive Flow" |
| **Module dependencies** | CODEBASE_KNOWLEDGE.md § 6, then ARCHITECTURE_DIAGRAMS.md "Module Import Order" |
| **API documentation** | CODEBASE_KNOWLEDGE.md § 5.1 & 5.4 |
| **Common mistakes** | CODEBASE_KNOWLEDGE.md § 7 |

---

### "I need to modify..."

| Task | Start Here | Then Read |
|------|-----------|-----------|
| **Add visualization** | § 7.4 Extension Points | visualization.py docs in § 5.1 |
| **Add model parameter** | § 7.4 Extension Points | config.py, ui.py, topic_modeling.py in § 5.1 |
| **Fix data validation** | § 3 FEATURE 2 | data_processing.py in § 5.1 |
| **Modify UI layout** | § 3 FEATURE 6 | ui.py in § 5.1 |
| **Change topic modeling** | § 3 FEATURE 4 | TopicModeler in § 5.1 |
| **Debug session issues** | § 3 FEATURE 8 | app_core.py SessionManager in § 5.1 |

---

### "I'm getting an error..."

| Error | Solution |
|-------|----------|
| **File encoding error** | § 4.2, search "encoding mismatch" |
| **Module not found** | § 6 Dependency Graph, check import order in § 5.1 |
| **Visualization not showing** | § 7.3 "Common Pitfalls", point 4 |
| **Reactive value not updating** | § 7.3 "Common Pitfalls", point 1 |
| **Session cleanup issue** | § 7.2 "Don't Break These", point 3 |
| **OOM (out of memory)** | § 4.2 "Error Scenarios", then § 7.2 "Performance Seams" |

---

## 📋 File-to-Docs Reference

### Core Application Files

| File | Type | Analysis In | Key Topics |
|------|------|------------|-----------|
| `app.py` (entry) | Python | § 5.1 | Initialization, env vars, static mounting |
| `ui.py` | Python | § 5.1, § 3 FEATURE 6 | Layout, components, seed topics UI |
| `server.py` | Python | § 5.1, § 3 FEATURE 8 | Handlers, reactivity, events |
| `topic_modeling.py` | Python | § 5.1, § 3 FEATURE 4 | BERTopic wrapper, seed processing, model config |
| `data_processing.py` | Python | § 5.1, § 3 FEATURES 1-3 | File upload, cleaning, paragraph splitting |
| `visualization.py` | Python | § 5.1, § 3 FEATURE 5 | Plotly/Matplotlib, output generation |
| `app_core.py` | Python | § 5.1, § 3 FEATURE 8 | SessionManager, PathManager, StatusManager |
| `config.py` | Python | § 5.1, § 5.2, § 5.3 | All configuration constants |
| `core_types.py` | Python | § 5.1, § 5.2 | Type definitions, protocols |
| `utils.py` | Python | § 5.1 | File I/O, encoding, utilities |
| `decorators.py` | Python | § 5.1 | Error handling, status context |

---

## 🏗️ Architecture Overview (from § 2.1)

**Layered Architecture:**

```
Layer 1: UI (Shiny reactive components)
    ↓ through server handlers
Layer 2: Business Logic (data processing, modeling, visualization)
    ↓ orchestrated by
Layer 3: Infrastructure (session, path, status management)
    ↓ configured via
Layer 4: Configuration & Types (config.py, core_types.py)
```

**Key Principles:**

- Separation of concerns: UI, logic, infrastructure
- Dependency injection: modules receive config/managers
- Reactive pattern: Shiny handles state updates
- Circular import prevention: stdlib modules at leaf

---

## 📊 Analyzed Components

### Modules (8 total)

- ✅ app.py: Initialization
- ✅ ui.py: UI layout (15 sections, seed topics, visualizations)
- ✅ server.py: Event handlers (10+ reactive effects)
- ✅ topic_modeling.py: BERTopic wrapper + configurations
- ✅ data_processing.py: CSV reading, cleaning, splitting
- ✅ visualization.py: Interactive + static visualizations
- ✅ app_core.py: Session, path, status management
- ✅ config.py: 15+ configuration sections
- ✅ core_types.py: 10+ type definitions
- ✅ utils.py: Generic utilities
- ✅ decorators.py: Error handling, async patterns

### Features Documented (8 total)

1. ✅ File Upload & Data Processing
2. ✅ Data Cleaning & Validation
3. ✅ Paragraph Splitting (Semantic Chunking)
4. ✅ Topic Modeling with BERTopic
5. ✅ Visualization & Report Generation
6. ✅ Seed Topic Management
7. ✅ Status Tracking & Progress
8. ✅ Session Management & File Serving

### Dependencies Mapped

- ✅ Module import order analyzed
- ✅ Circular dependencies eliminated
- ✅ Type protocols documented
- ✅ API signatures documented
- ✅ 20+ external packages catalogued

---

## 🔍 Key Insights Documented

### Architecture Decisions (Why)

- Why Shiny over Streamlit/React → § 4.4
- Why BERTopic over LDA → § 4.4
- Why HDBSCAN over KMeans → § 4.4
- Why guided modeling with seeds → § 4.1

### Performance Characteristics

- Embedding generation: 30-120s → § 4.3
- Paragraph splitting: 10-30s → § 4.3
- BERTopic clustering: 20-60s → § 4.3
- Total workflow: 2-5 min for 1k-5k docs → § 4.3

### Critical Code Paths

- File upload → validation → cleaning → splitting → modeling → visualization
- Reactive state propagation in Shiny
- Session creation → processing → cleanup lifecycle

---

## ⚡ Extension Points

All documented in § 7.4:

1. **Add Visualization**: Create method in VisualizationService
2. **Add Model Parameter**: config.py → ui.py → topic_modeling.py
3. **Custom Stopwords**: Edit STOPWORDS set in config.py
4. **Parallel Processing**: Wrap in asyncio.gather()
5. **Custom Embeddings**: Change EMBEDDING.model in config.py

With code examples for each.

---

## 🚀 Getting Started with the Docs

### For New Developers

1. Read **§ 1: High-Level Overview** (5 min)
2. Scan **§ 2: System Architecture** (10 min)
3. Focus on **§ 3: FEATURE 1-3** (your first task area) (20 min)
4. Reference **§ 5** as needed for APIs (ongoing)

### For Bug Fixes

1. Check **§ 4: Gotchas** (5 min)
2. Find error in **§ 4.2: Error Scenarios** (2 min)
3. Look up file location in **§ 8: Quick Reference** (1 min)
4. Read module docs in **§ 5.1** (varies)

### For New Features

1. Check **§ 7.4: Extension Points** (5 min)
2. Read **§ 3** relevant FEATURE section (15 min)
3. Study dependency graph in **§ 6** (5 min)
4. Review code and implement

---

## 📝 Document Metadata

**Analysis Phases Completed:**

- ✅ Phase 1: Initial Context Scan
- ✅ Phase 2: System Architecture Deep Dive
- ✅ Phase 3: Feature-by-Feature Analysis
- ✅ Phase 4: Nuances, Subtleties & Gotchas
- ✅ Phase 5: Technical Reference & Glossary
- ✅ Phase 6: Final Knowledge Document Assembly

**Files Referenced:** 11 core files analyzed  
**Code Lines Examined:** ~3000+ lines  
**Configuration Items:** 50+  
**Diagrams Created:** 6 ASCII diagrams  

**For Questions:**

- Check if your question is in § 7 first
- Search the document (all sections are titled)
- Cross-references to related sections provided throughout

---

**Version:** 1.0  
**Last Updated:** 2026-03-11  
**Access:** Read `CODEBASE_KNOWLEDGE.md` → Reference `ARCHITECTURE_DIAGRAMS.md` as needed
