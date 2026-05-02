# BIS Standards Recommendation Engine

Plan 01 implements the PDF parser and standards extractor. It reads
`dataset.pdf`, detects BIS IS codes, extracts titles, scopes, body snippets, and
keywords, then writes the structured catalog used by later retrieval modules.

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

The indexer builds a FAISS HNSW dense index with `BAAI/bge-small-en-v1.5`,
a BM25 sparse index over the same chunk text, a normalized IS-code lookup, and
an indexed catalog with `chunk_text` attached:

- `data/faiss.index`
- `data/bm25.pkl`
- `data/codes.json`
- `data/standards_indexed.json`

## Process a query

```bash
python -m src.query_processor "OPC requirements under IS 269: 1989"
```

Plan 03 normalizes raw user queries, detects explicit year-qualified IS codes,
and expands catalog vocabulary from `data/synonyms.json` before BM25
tokenization. The curated synonym set covers OPC, PPC, Portland slag cement,
concrete pipes, aggregates, masonry blocks, asbestos cement sheets, and common
abbreviations such as SSC, HDPE, UPVC, GRP, and CMU.

## Validate

```bash
python -m pytest
```
