# BIS Standards Recommendation Engine

The current modules include the PDF parser/catalog extractor, query processor,
index builder, and Plan 04 hybrid retriever. The parser reads `dataset.pdf`,
detects BIS IS codes, extracts titles, scopes, body snippets, and keywords, then
writes the structured catalog used by later retrieval modules.

## Build the catalog

```bash
pip install -r requirements.txt
python -m src.parser
```

The parser writes `data/standards_catalog.json`.

## Build indexes

```bash
python -m src.indexer --catalog data/standards_catalog.json --output-dir data
```

The indexer writes dense, sparse, and exact-code lookup artifacts under `data/`.

## Process a query

```bash
python -m src.query_processor "OPC requirements under IS 269: 1989"
```

Plan 03 normalizes raw user queries, detects explicit year-qualified IS codes,
and expands catalog vocabulary from `data/synonyms.json` before BM25
tokenization. The curated synonym set covers OPC, PPC, Portland slag cement,
concrete pipes, aggregates, masonry blocks, asbestos cement sheets, and common
abbreviations such as SSC, HDPE, UPVC, GRP, and CMU.

## Retrieve candidates

```bash
python -m src.indexer --catalog data/standards_catalog.json --output-dir data
python -c "from src.query_processor import QueryProcessor; from src.retriever import HybridRetriever; qp=QueryProcessor(); r=HybridRetriever(); pq=qp.process('33 Grade Ordinary Portland Cement chemical requirements'); [print(x.standard['is_code'], x.score, x.sources) for x in r.retrieve(pq, top_k=5)]"
```

Plan 04 combines dense FAISS search, sparse BM25 search, exact IS-code matches,
reciprocal rank fusion, and deterministic metadata boosts. It returns the top
20 `RetrievalResult` candidates for downstream reranking.

## Validate

```bash
python -m pytest
```
