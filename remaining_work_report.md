# Remaining Work Report

## BIS Standards Recommendation Engine

Generated from:
- `C:\Users\ASUS\Downloads\content (11).md` - Product Requirements Document
- `C:\Users\ASUS\Downloads\content (12).md` - Technical Requirements Document

Checked against the current repository state on 2026-05-02.

## Executive Summary

There is no evaluator-blocking product work left for the current public test set. The repository already meets the PRD acceptance metrics and the main CLI path is working.

Current verification:

| Check | Result |
|---|---|
| `python -m pytest` | 39 passed |
| `python eval_script.py --results results.json` | Hit Rate @3: 100.00%, MRR @5: 1.0000, Avg Latency: 0.0288s |
| `python scripts/check_format.py results.json` | Passed |

The remaining work is mainly spec alignment, offline-readiness hardening, performance/stress reporting, and documentation cleanup.

## Completed Against PRD/TRD

| Requirement Area | Current Status |
|---|---|
| CLI entry point | Implemented via `run.py --input ... --output ...` |
| Public-set metrics | Passing above target: 100% Hit@3 and 1.0000 MRR@5 |
| Output schema | Format checker passes |
| Parser/catalog/index artifacts | Implemented and generated under `data/` |
| Query normalization, synonyms, explicit IS-code detection | Implemented |
| BM25 retrieval, RRF-style fusion, metadata boosts | Implemented |
| Optional reranker | Implemented behind `--use-reranker` |
| Tests | Unit, integration, reranker, retriever, parser, robustness tests pass |
| Diagnostic scripts | `scripts/check_format.py`, `scripts/error_analysis.py`, `scripts/evaluate.py`, `scripts/tune.py` exist |

## Remaining Work

### 1. Resolve retrieval-path spec drift

Priority: Medium

The PRD/TRD describe dense FAISS retrieval plus sparse BM25 retrieval as the normal hybrid path, followed by cross-encoder reranking. The current implementation keeps FAISS artifacts available but defaults to BM25 plus deterministic boosts, with dense retrieval disabled by default and reranking opt-in through `--use-reranker`.

Decision needed:
- If the docs are authoritative, enable dense retrieval by default and decide whether reranking should be default.
- If the implementation is authoritative, update the PRD/TRD to say the default evaluator path is BM25 plus deterministic boosts, while dense retrieval and reranking are optional.

Recommended action: Keep the current default because it is fast, deterministic, offline-friendly, and passes the public metrics. Update the docs to reflect it.

### 2. Harden offline rebuild/model-cache readiness

Priority: Medium

The PRD/TRD require offline capability after model download. The checked-in artifacts let the normal run work without rebuilding, but `--rebuild`, dense retrieval, and reranker usage still depend on local Hugging Face model availability.

Remaining tasks:
- Document exact model cache requirements for `BAAI/bge-small-en-v1.5` and `BAAI/bge-reranker-base`.
- Add a preflight command or script that checks whether required model files are available locally.
- Decide whether the evaluator is expected to run `--rebuild` offline. If yes, provide explicit cache setup instructions before submission.

### 3. Add formal stress/performance harness

Priority: Medium

The TRD asks for 1000 paraphrased queries and p95 latency checks. The repo has robustness tests and public-set latency reporting, but not a formal stress benchmark.

Remaining tasks:
- Add a script such as `scripts/stress_test.py`.
- Generate or load paraphrased queries.
- Report p50/p95/p99 latency and crash count.
- Optionally fail if p95 exceeds the PRD/TRD budget.

### 4. Add artifact integrity checks

Priority: Low

The TRD lists recovery for corrupted FAISS/index artifacts. The current code rebuilds when artifacts are missing, but it does not verify hashes or detect corrupted-but-present artifacts before loading.

Remaining tasks:
- Store artifact metadata or checksums after index build.
- Validate `faiss.index`, `bm25.pkl`, `codes.json`, and `standards_indexed.json` before use.
- On validation failure, show a clear rebuild instruction or auto-rebuild when `--rebuild` is allowed.

### 5. Clean up and commit PRD/TRD documentation

Priority: Low

The supplied PRD/TRD files are not currently tracked in the repo as `PRD.md` and `TRD.md`, even though the TRD directory structure lists them. The downloaded Markdown also appears to contain encoding artifacts in diagrams and arrows.

Remaining tasks:
- Add cleaned `PRD.md` and `TRD.md` to the repository if they are required deliverables.
- Fix mojibake/encoding issues in diagrams and arrows.
- Align dependency versions and architecture notes with the current implementation.
- Remove stale references such as `nltk` if not used.

### 6. Decide on optional LLM query expansion

Priority: Low

The PRD mentions optional LLM-based query expansion behind a `--use-llm` flag. This is not implemented. It is also marked optional and conflicts with the stated offline/no-external-API hot path.

Recommended action: Explicitly mark this as out of scope for the hackathon submission unless there is a strong reason to add it.

### 7. Re-run CodeRabbit after rate limit and commit changes

Priority: Low

CodeRabbit CLI was available through WSL, but the follow-up uncommitted review hit a rate limit. The current fixes and report are still uncommitted in the local worktree.

Remaining tasks:
- Re-run `coderabbit review --agent -t uncommitted` after the rate limit resets.
- Review any new findings.
- Commit and push the final changes.

## Not Required For Current Scope

The PRD explicitly marks the following as out of scope for the hackathon:

- Web UI or chatbot interface
- Multi-language queries
- Standards outside `dataset.pdf`
- Real-time index updates

These should not be treated as remaining work unless the product scope changes.

## Final Assessment

The implementation is submission-ready for the current evaluator path. The highest-value remaining work is to align the PRD/TRD with the chosen fast deterministic retrieval path, document offline model-cache expectations, and add a formal stress benchmark if time permits.
