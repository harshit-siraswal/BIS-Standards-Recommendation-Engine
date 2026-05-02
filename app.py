"""Vercel-compatible FastAPI entrypoint for the BIS recommendation engine."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import BISPipeline, MAX_QUERY_CHARS


ARTIFACT_PATHS = (
    Path("data/standards_indexed.json"),
    Path("data/faiss.index"),
    Path("data/bm25.pkl"),
    Path("data/codes.json"),
    Path("data/synonyms.json"),
)

app = FastAPI(
    title="BIS Standards Recommendation Engine",
    version="1.0.0",
    description="Recommend Bureau of Indian Standards IS codes from product and manufacturing queries.",
)


class RecommendationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int = Field(default=5, ge=1, le=5)


class RecommendationResponse(BaseModel):
    query: str
    retrieved_standards: list[str]
    latency_seconds: float
    compliance_warnings: list[str]


def _missing_artifacts() -> list[str]:
    return [str(path) for path in ARTIFACT_PATHS if not path.exists()]


@lru_cache(maxsize=1)
def _pipeline() -> BISPipeline:
    missing = _missing_artifacts()
    if missing:
        raise FileNotFoundError(
            "Required retrieval artifacts are missing from the deployment: "
            + ", ".join(missing)
        )
    return BISPipeline(use_reranker=False)


@app.get("/")
def root() -> dict[str, Any]:
    missing = _missing_artifacts()
    return {
        "service": "BIS Standards Recommendation Engine",
        "status": "ready" if not missing else "missing_artifacts",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend",
        },
        "missing_artifacts": missing,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    missing = _missing_artifacts()
    return {
        "ok": not missing,
        "missing_artifacts": missing,
    }


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest) -> dict[str, Any]:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    try:
        pipeline = _pipeline()
        processed = pipeline.query_processor.process(query)
        result = pipeline.run_query(query, top_k=payload.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc

    return {
        **result,
        "compliance_warnings": processed.compliance_warnings,
    }


@app.post("/api/recommend", response_model=RecommendationResponse, include_in_schema=False)
def recommend_api(payload: RecommendationRequest) -> dict[str, Any]:
    return recommend(payload)
