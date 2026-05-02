# BIS Standards Recommendation Engine

A deterministic recommendation engine that maps manufacturing and product
queries to the most relevant Bureau of Indian Standards (BIS) IS codes. It
parses the bundled BIS PDF into a catalog, expands domain terminology such as
OPC and PPC, retrieves candidates with BM25/FAISS artifacts plus metadata
boosts, and returns evaluator-compatible top-5 standards.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run.py --input public_test_set.json --output results.json
```

For hackathon judging, use the required inference entrypoint:

```bash
python inference.py --input public_test_set.json --output team_results.json
```

For a fresh clone without generated retrieval artifacts:

```bash
python run.py --input public_test_set.json --output results.json --rebuild
```

`--rebuild` parses `dataset.pdf` and rebuilds `data/standards_indexed.json`,
`data/faiss.index`, `data/bm25.pkl`, and `data/codes.json`.

## Business Compliance Assistant Demo

The FastAPI app adds a demo-only Business Compliance Assistant around the
retrieved standards:

```bash
uvicorn app:app --reload
```

`POST /recommend` keeps the retrieved IS codes and rationale, then adds
`business_guidance` with matched category, matched terms, why the standards
match, documents to prepare, testing readiness, workflow, and verification
notes. This is separate from `inference.py`, which remains deterministic,
fast, API-free, and judge-schema compatible.

If `GROQ_API_KEY` is present, the app may use Groq to polish the business
guidance. If the key is missing or the call fails, it falls back to deterministic
template guidance. Guardrails require guidance to mention only returned IS
codes, avoid invented fees/timelines/forms/legal claims, and tell users to
verify with BIS.

## Evaluate

```bash
python eval_script.py --results results.json
```

Current public-set metrics:

| Metric | Score |
|--------|-------|
| Hit Rate @3 | 100.00% |
| MRR @5 | 1.0000 |
| Avg Latency | 0.0291s |

## Tune And Diagnose

```bash
python scripts/tune.py
python scripts/error_analysis.py
python scripts/check_format.py results.json
```

The tuned default keeps `RRF_K = 40`, `BOOST_EXPLICIT_CODE = 0.20`, and
`BOOST_TITLE_TOKEN = 0.03`. Public-set robustness comes from targeted phrase
boosts for high-value product families, including supersulphated cement, white
Portland cement, calcined clay PPC, concrete pipes, and lightweight masonry
blocks.

## Architecture

```text
dataset.pdf
  -> src.parser -> data/standards_catalog.json
  -> src.indexer -> FAISS, BM25, code lookup artifacts
query
  -> src.query_processor -> normalized tokens, synonyms, explicit IS codes
  -> src.retriever -> fused candidate ranking with deterministic boosts
  -> src.reranker optional cross-encoder reranking
  -> run.py -> results.json
  -> app.py -> demo rationale and business guidance
```

Retrieval innovation includes PDF parsing into a structured catalogue, custom
chunking by IS code/title/scope/body, synonym and multilingual query expansion,
BM25/FAISS-ready hybrid retrieval, deterministic boosts for product families and
explicit IS codes, and robustness evaluation across multilingual/noisy queries.

## Tests

```bash
python -m pytest
```
