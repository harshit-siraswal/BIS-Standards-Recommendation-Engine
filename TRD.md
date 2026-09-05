# Technical Requirements Document (TRD)

## Project
BIS Saarthi: AI-powered Intelligent Assistant for Indian Standards and BIS Services

## SIH26107 Technical Alignment
The implementation is aligned to SIH26107 by combining:
- deterministic standards retrieval,
- BIS service route guidance,
- step-by-step compliance roadmap output,
- multilingual, user-facing web experience.

## System Architecture
1. Data Layer
- Parsed BIS catalog artifacts and indexed retrieval data.
- Key files: standards JSON, FAISS index, BM25 pickle, code maps, synonyms.

2. Retrieval Layer
- Query normalization and domain synonym expansion.
- Hybrid retrieval and deterministic ranking/boosting.
- Top-k standards output with latency.

3. Guidance Layer
- Deterministic business guidance generator.
- BIS service guidance generator:
  - audience classification,
  - service buckets,
  - official links.
- Compliance roadmap generator:
  - schemes,
  - lab/test path,
  - process summary,
  - sequential action steps with links.

4. API Layer (FastAPI)
- /recommend
- /chat
- /health
- /api/status

5. Presentation Layer
- Single-page interface embedded in app.py.
- Assistant panel + roadmap toggle.
- Multilingual text and accessibility controls.

## API Contract Additions
Recommendation response includes:
- business_guidance
- bis_services
- compliance_roadmap

compliance_roadmap structure:
- audience
- applicable_schemes
- suggested_labs
- process_summary
- steps: [{ step_no, title, action, official_link }]
- verification_note

## Major Components
- src.parser: catalog parsing utilities.
- src.indexer: index build pipelines.
- src.query_processor: query normalization and warning support.
- src.retriever: retrieval and ranking.
- app.py:
  - endpoint handlers,
  - deterministic guidance builders,
  - roadmap generator,
  - web UI and roadmap rendering logic.

## Compliance Roadmap Feature Design
Backend:
- Build roadmap deterministically from query context and retrieval outcome.
- Include out-of-scope safe pathway when no in-catalog standards are found.
- Include official BIS/Manak/lab links only.

Frontend:
- Roadmap button in assistant panel.
- Expand/collapse roadmap panel.
- Render schemes, labs, process checkpoints, and step-by-step links.

## Guardrails
- Guidance code references are restricted to retrieved standards.
- Verification note is always included.
- Optional LLM guidance is validated and rejected if unsafe.
- Out-of-scope products produce advisory paths rather than fabricated standards.

## Performance and Reliability
- Deterministic retrieval path supports judge use and predictable latency.
- /health and /api/status expose operational readiness and artifact checks.
- Caching used for pipeline and standard lookup initialization.

## Testing Strategy
- Unit/integration tests for API behavior and UI text contract in tests/test_app.py.
- Roadmap feature tests:
  - roadmap button rendered in HTML,
  - /recommend includes compliance_roadmap fields.

## Deployment Notes
- Vercel-compatible FastAPI entrypoint in app.py.
- Keep inference.py unchanged for judge-evaluable deterministic path.
- Environment variable GROQ_API_KEY remains optional for guidance polish.

## Known Constraints
- Catalog breadth depends on bundled data artifacts.
- Official BIS processes may change over time; links and advisory text should be reviewed periodically.
- Local test execution may require limiting BLAS threads on memory-constrained environments.

## Traceability Matrix (SIH26107)
- AI assistant behavior: /recommend + /chat + UI assistant panel.
- BIS services guidance: bis_services payload.
- Step-wise compliance process: compliance_roadmap payload + roadmap panel.
- Industry + consumer orientation: audience detection + consumer routes.
- Multilingual UX: language selector and translated interface strings.
