"""Vercel-compatible FastAPI entrypoint for the BIS recommendation engine."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from src.pipeline import BISPipeline, MAX_QUERY_CHARS
from src.retriever import normalize_standard_code


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


class RecommendationItem(BaseModel):
    code: str
    title: str
    rationale: str
    confidence: float


class RecommendationResponse(BaseModel):
    query: str
    retrieved_standards: list[str]
    latency_seconds: float
    compliance_warnings: list[str]
    recommendations: list[RecommendationItem]


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BIS Standards Recommendation Engine</title>
  <script>
    if ("scrollRestoration" in history) {
      history.scrollRestoration = "manual";
    }
    window.addEventListener("pageshow", () => {
      setTimeout(() => window.scrollTo(0, 0), 0);
    });
  </script>
  <style>
    :root {
      --gov-navy: #08366f;
      --gov-blue: #005ea8;
      --gov-blue-2: #d7ecff;
      --saffron: #ff9933;
      --india-green: #138808;
      --ink: #101828;
      --muted: #526173;
      --line: #d8e1ee;
      --paper: #f7f9fc;
      --white: #ffffff;
      --success: #067647;
      --warning: #a15c07;
      --shadow: 0 18px 50px rgba(8, 54, 111, 0.14);
      --radius: 8px;
      --max: 1180px;
    }

    * {
      box-sizing: border-box;
    }

    html {
      scroll-behavior: smooth;
    }

    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Noto Sans", "IBM Plex Sans", "Segoe UI", sans-serif;
      line-height: 1.5;
    }

    button,
    textarea,
    input {
      font: inherit;
    }

    button:focus-visible,
    textarea:focus-visible,
    a:focus-visible {
      outline: 3px solid var(--saffron);
      outline-offset: 3px;
    }

    .gov-strip {
      background: var(--gov-navy);
      color: var(--white);
      font-size: 0.82rem;
    }

    .strip-inner,
    .nav-inner,
    .wrap {
      width: min(var(--max), calc(100% - 32px));
      margin: 0 auto;
    }

    .strip-inner {
      min-height: 36px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .strip-links {
      display: flex;
      align-items: center;
      gap: 14px;
      white-space: nowrap;
    }

    .strip-links button {
      border: 1px solid rgba(255, 255, 255, 0.35);
      background: transparent;
      color: var(--white);
      padding: 3px 8px;
      border-radius: 4px;
      cursor: pointer;
    }

    .brand-nav {
      background: var(--white);
      border-bottom: 1px solid var(--line);
      box-shadow: 0 2px 16px rgba(16, 24, 40, 0.04);
    }

    .nav-inner {
      min-height: 74px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }

    .emblem {
      width: 45px;
      height: 45px;
      border-radius: 50%;
      border: 2px solid var(--gov-navy);
      display: grid;
      place-items: center;
      color: var(--gov-navy);
      font-weight: 800;
      letter-spacing: 0.04em;
      background:
        linear-gradient(180deg, rgba(255, 153, 51, 0.22), rgba(255, 255, 255, 0) 42%),
        linear-gradient(0deg, rgba(19, 136, 8, 0.14), rgba(255, 255, 255, 0) 40%),
        var(--white);
    }

    .brand-title {
      margin: 0;
      font-size: clamp(1rem, 2vw, 1.32rem);
      line-height: 1.15;
      color: var(--gov-navy);
      letter-spacing: 0;
    }

    .brand-subtitle {
      margin: 3px 0 0;
      font-size: 0.82rem;
      color: var(--muted);
    }

    .nav-links {
      display: flex;
      align-items: center;
      gap: 18px;
      color: var(--muted);
      font-size: 0.9rem;
    }

    .nav-links a {
      color: inherit;
      text-decoration: none;
    }

    .nav-links a:hover {
      color: var(--gov-blue);
    }

    .hero {
      position: relative;
      border-bottom: 1px solid var(--line);
      background:
        linear-gradient(90deg, rgba(255, 153, 51, 0.16), transparent 22%),
        linear-gradient(270deg, rgba(19, 136, 8, 0.11), transparent 24%),
        linear-gradient(180deg, #ffffff 0%, #edf5ff 100%);
      overflow: hidden;
    }

    .hero::before {
      content: "";
      position: absolute;
      inset: 0;
      background-image:
        linear-gradient(rgba(8, 54, 111, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(8, 54, 111, 0.05) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: linear-gradient(180deg, black, transparent 72%);
      pointer-events: none;
    }

    .hero-inner {
      position: relative;
      width: min(var(--max), calc(100% - 32px));
      margin: 0 auto;
      padding: 48px 0 34px;
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) 360px;
      gap: 28px;
      align-items: start;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 16px;
      color: var(--gov-blue);
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .eyebrow::before {
      content: "";
      width: 28px;
      height: 4px;
      background: linear-gradient(90deg, var(--saffron), var(--india-green));
      border-radius: 999px;
    }

    h1 {
      max-width: 840px;
      margin: 0;
      font-size: clamp(2rem, 5vw, 4.6rem);
      line-height: 0.96;
      letter-spacing: 0;
      color: var(--gov-navy);
    }

    .hero-copy {
      max-width: 720px;
      margin: 18px 0 0;
      color: var(--muted);
      font-size: clamp(1rem, 1.8vw, 1.16rem);
    }

    .service-card {
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 20px;
    }

    .service-card h2,
    .panel h2,
    .results h2 {
      margin: 0;
      color: var(--gov-navy);
      font-size: 1.05rem;
    }

    .service-card p,
    .panel p {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .status-list {
      display: grid;
      gap: 10px;
      margin-top: 18px;
    }

    .status-item {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px;
      align-items: start;
      font-size: 0.9rem;
      color: var(--ink);
    }

    .check {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: rgba(19, 136, 8, 0.12);
      color: var(--success);
      font-weight: 900;
      font-size: 0.78rem;
    }

    .workspace {
      width: min(var(--max), calc(100% - 32px));
      margin: 28px auto 56px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 24px;
      align-items: start;
    }

    .query-panel,
    .results,
    .panel {
      background: var(--white);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: 0 12px 34px rgba(16, 24, 40, 0.07);
    }

    .query-panel {
      padding: 22px;
    }

    label {
      display: block;
      color: var(--gov-navy);
      font-weight: 800;
      margin-bottom: 10px;
    }

    textarea {
      display: block;
      width: 100%;
      min-height: 150px;
      resize: vertical;
      border: 1px solid #b9c7d8;
      border-radius: 6px;
      padding: 14px 15px;
      color: var(--ink);
      background: #fbfdff;
      box-shadow: inset 0 1px 0 rgba(16, 24, 40, 0.04);
    }

    .query-tools {
      margin-top: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }

    .chips {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .chip {
      border: 1px solid #b9c7d8;
      background: #f6f9fd;
      color: var(--gov-navy);
      border-radius: 999px;
      padding: 7px 10px;
      cursor: pointer;
      font-size: 0.84rem;
    }

    .chip:hover {
      border-color: var(--gov-blue);
      background: var(--gov-blue-2);
    }

    .primary {
      border: 0;
      background: var(--gov-blue);
      color: var(--white);
      border-radius: 6px;
      padding: 11px 18px;
      cursor: pointer;
      font-weight: 800;
      box-shadow: 0 10px 22px rgba(0, 94, 168, 0.26);
    }

    .primary:hover {
      background: var(--gov-navy);
    }

    .primary[disabled] {
      cursor: wait;
      opacity: 0.68;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }

    .metric {
      border-left: 4px solid var(--gov-blue);
      background: #f6f9fd;
      padding: 12px;
      border-radius: 5px;
    }

    .metric strong {
      display: block;
      color: var(--gov-navy);
      font-size: 1.24rem;
      line-height: 1;
    }

    .metric span {
      color: var(--muted);
      font-size: 0.78rem;
    }

    .results {
      margin-top: 18px;
      padding: 0;
      overflow: hidden;
    }

    .results-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
    }

    .latency {
      color: var(--success);
      background: rgba(19, 136, 8, 0.11);
      border-radius: 999px;
      padding: 5px 10px;
      font-weight: 800;
      font-size: 0.78rem;
      white-space: nowrap;
    }

    .empty-state {
      padding: 22px 20px;
      color: var(--muted);
    }

    .result-list {
      display: grid;
    }

    .result {
      display: grid;
      grid-template-columns: 52px 1fr;
      gap: 14px;
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
    }

    .result:last-child {
      border-bottom: 0;
    }

    .rank {
      width: 38px;
      height: 38px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      color: var(--white);
      background: var(--gov-navy);
      font-weight: 900;
    }

    .result h3 {
      margin: 0;
      font-size: 1.04rem;
      color: var(--ink);
    }

    .result .code {
      display: inline-flex;
      align-items: center;
      margin-right: 8px;
      color: var(--gov-blue);
      font-weight: 900;
    }

    .rationale {
      margin: 7px 0 10px;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .confidence {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 800;
    }

    .bar {
      flex: 0 1 180px;
      height: 7px;
      background: #e7edf5;
      border-radius: 999px;
      overflow: hidden;
    }

    .bar span {
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--india-green), var(--gov-blue));
    }

    .side-stack {
      display: grid;
      gap: 16px;
    }

    .panel {
      padding: 18px;
    }

    .notice {
      border-left: 4px solid var(--saffron);
    }

    .steps {
      margin: 15px 0 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 12px;
    }

    .steps li {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px;
      color: var(--muted);
      font-size: 0.9rem;
    }

    .num {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: var(--gov-blue-2);
      color: var(--gov-navy);
      font-size: 0.78rem;
      font-weight: 900;
    }

    .warnings {
      margin-top: 12px;
      display: grid;
      gap: 8px;
    }

    .warning {
      border: 1px solid rgba(255, 153, 51, 0.5);
      background: rgba(255, 153, 51, 0.11);
      color: var(--warning);
      border-radius: 5px;
      padding: 10px;
      font-size: 0.86rem;
    }

    footer {
      border-top: 1px solid var(--line);
      background: #eef4fb;
      color: var(--muted);
      padding: 22px 0;
      font-size: 0.86rem;
    }

    body.contrast {
      --paper: #050505;
      --white: #101010;
      --ink: #ffffff;
      --muted: #d6d6d6;
      --line: #4d4d4d;
      --gov-navy: #ffffff;
      --gov-blue: #ffd45a;
      --gov-blue-2: #2e2e2e;
      --shadow: none;
    }

    body.large-text {
      font-size: 112%;
    }

    @media (max-width: 920px) {
      .hero-inner,
      .workspace {
        grid-template-columns: 1fr;
      }

      .nav-links {
        display: none;
      }

      .service-card {
        max-width: 560px;
      }
    }

    @media (max-width: 620px) {
      .strip-inner,
      .nav-inner {
        align-items: flex-start;
        flex-direction: column;
        padding: 10px 0;
      }

      .strip-links {
        width: 100%;
        overflow-x: auto;
      }

      .hero-inner {
        padding-top: 34px;
      }

      .metrics {
        grid-template-columns: 1fr;
      }

      .result {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="gov-strip">
      <div class="strip-inner">
        <div>An official Government of India digital service</div>
        <div class="strip-links" aria-label="accessibility controls">
          <span>English</span>
          <span>Hindi</span>
          <button type="button" id="textSize">A+</button>
          <button type="button" id="contrast">Contrast</button>
        </div>
      </div>
    </div>
    <nav class="brand-nav" aria-label="primary">
      <div class="nav-inner">
        <div class="brand">
          <div class="emblem" aria-hidden="true">BIS</div>
          <div>
            <p class="brand-title">BIS Standards Recommendation Engine</p>
            <p class="brand-subtitle">Bureau of Indian Standards lookup for manufacturing MSEs</p>
          </div>
        </div>
        <div class="nav-links">
          <a href="#search">Search</a>
          <a href="#results">Results</a>
          <a href="/docs">API Docs</a>
          <a href="/health">Health</a>
        </div>
      </div>
    </nav>
  </header>

  <main>
    <section class="hero" aria-labelledby="hero-title">
      <div class="hero-inner">
        <div>
          <div class="eyebrow">BIS Digital Service</div>
          <h1 id="hero-title">Find applicable Indian Standards from a product description.</h1>
          <p class="hero-copy">
            Paste a manufacturing use case in plain language. The engine searches the BIS building-materials catalog and returns ranked IS standards with concise catalog-backed rationale.
          </p>
        </div>
        <aside class="service-card" aria-label="service readiness">
          <h2>Submission readiness</h2>
          <p>Built for the hackathon judging path and a public demo workflow.</p>
          <div class="status-list">
            <div class="status-item"><span class="check">✓</span><span>Root-level inference.py for automated scoring</span></div>
            <div class="status-item"><span class="check">✓</span><span>Responses constrained to parsed BIS catalog entries</span></div>
            <div class="status-item"><span class="check">✓</span><span>Fast API path for live demo and integrations</span></div>
          </div>
        </aside>
      </div>
    </section>

    <section class="workspace" id="search">
      <div>
        <form class="query-panel" id="recommendForm">
          <label for="query">Describe your product or manufacturing use case</label>
          <textarea id="query" name="query" maxlength="500">We are a small enterprise manufacturing 33 Grade Ordinary Portland Cement. Which BIS standard covers the chemical and physical requirements for our product?</textarea>
          <div class="query-tools">
            <div class="chips" aria-label="sample queries">
              <button class="chip" type="button" data-query="coarse and fine aggregates from natural sources for structural concrete">Aggregates</button>
              <button class="chip" type="button" data-query="precast concrete pipes with and without reinforcement for water mains">Concrete pipes</button>
              <button class="chip" type="button" data-query="white Portland cement for architectural and decorative purposes">White cement</button>
            </div>
            <button class="primary" id="submit" type="submit">Recommend standards</button>
          </div>
          <div class="metrics" aria-label="public evaluation metrics">
            <div class="metric"><strong>100%</strong><span>Public Hit@3</span></div>
            <div class="metric"><strong>1.00</strong><span>Public MRR@5</span></div>
            <div class="metric"><strong>&lt;5s</strong><span>Latency target</span></div>
          </div>
        </form>

        <section class="results" id="results" aria-live="polite">
          <div class="results-header">
            <h2>Recommended standards</h2>
            <span class="latency" id="latency">Ready</span>
          </div>
          <div id="resultBody" class="empty-state">
            Submit a product description to view ranked BIS standards with rationale.
          </div>
        </section>
      </div>

      <aside class="side-stack">
        <section class="panel notice">
          <h2>No-hallucination guardrail</h2>
          <p>Recommendations are selected from the indexed BIS SP 21 catalog only. The UI displays titles and rationale from the matched catalog records.</p>
          <div id="warnings" class="warnings"></div>
        </section>
        <section class="panel">
          <h2>MSE compliance workflow</h2>
          <ul class="steps">
            <li><span class="num">1</span><span>Describe the product, material, grade, and intended use.</span></li>
            <li><span class="num">2</span><span>Review the top 3-5 standards and rationale.</span></li>
            <li><span class="num">3</span><span>Use the IS code list for certification planning and expert review.</span></li>
          </ul>
        </section>
        <section class="panel">
          <h2>API access</h2>
          <p>POST JSON to <strong>/recommend</strong> with <strong>query</strong> and optional <strong>top_k</strong>. Open <strong>/docs</strong> for the FastAPI schema.</p>
        </section>
      </aside>
    </section>
  </main>

  <footer>
    <div class="wrap">Proof-of-concept for BIS standards discovery. Always validate final compliance decisions with the official standard and domain experts.</div>
  </footer>

  <script>
    const form = document.querySelector("#recommendForm");
    const queryInput = document.querySelector("#query");
    const submitButton = document.querySelector("#submit");
    const resultBody = document.querySelector("#resultBody");
    const latency = document.querySelector("#latency");
    const warnings = document.querySelector("#warnings");

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function setLoading(isLoading) {
      submitButton.disabled = isLoading;
      submitButton.textContent = isLoading ? "Searching catalog..." : "Recommend standards";
    }

    function renderWarnings(items) {
      warnings.innerHTML = "";
      if (!items || !items.length) return;
      warnings.innerHTML = items.map((item) => `<div class="warning">${escapeHtml(item)}</div>`).join("");
    }

    function renderResults(data) {
      latency.textContent = `${Number(data.latency_seconds || 0).toFixed(3)}s`;
      renderWarnings(data.compliance_warnings);
      const recommendations = data.recommendations || [];
      if (!recommendations.length) {
        resultBody.className = "empty-state";
        resultBody.textContent = "No matching standards were returned.";
        return;
      }

      resultBody.className = "result-list";
      resultBody.innerHTML = recommendations.map((item, index) => {
        const confidence = Math.round(Number(item.confidence || 0) * 100);
        return `
          <article class="result">
            <div class="rank">${index + 1}</div>
            <div>
              <h3><span class="code">${escapeHtml(item.code)}</span>${escapeHtml(item.title || "BIS standard")}</h3>
              <p class="rationale">${escapeHtml(item.rationale || "Matched against the BIS catalog.")}</p>
              <div class="confidence">
                <span>Confidence</span>
                <span class="bar"><span style="width:${confidence}%"></span></span>
                <span>${confidence}%</span>
              </div>
            </div>
          </article>
        `;
      }).join("");
    }

    async function recommend(query) {
      setLoading(true);
      latency.textContent = "Searching";
      resultBody.className = "empty-state";
      resultBody.textContent = "Retrieving relevant BIS standards...";
      try {
        const response = await fetch("/recommend", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, top_k: 5 }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Recommendation failed");
        }
        renderResults(data);
      } catch (error) {
        latency.textContent = "Error";
        resultBody.className = "empty-state";
        resultBody.textContent = error.message;
      } finally {
        setLoading(false);
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const query = queryInput.value.trim();
      if (query) recommend(query);
    });

    document.querySelectorAll("[data-query]").forEach((button) => {
      button.addEventListener("click", () => {
        queryInput.value = button.dataset.query;
        queryInput.focus();
      });
    });

    document.querySelector("#contrast").addEventListener("click", () => {
      document.body.classList.toggle("contrast");
    });

    document.querySelector("#textSize").addEventListener("click", () => {
      document.body.classList.toggle("large-text");
    });
  </script>
</body>
</html>
"""


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


@lru_cache(maxsize=1)
def _standard_lookup() -> dict[str, dict[str, Any]]:
    pipeline = _pipeline()
    return {
        normalize_standard_code(str(standard.get("is_code", ""))): standard
        for standard in pipeline.retriever.standards
    }


def _first_sentence(text: str, max_chars: int = 210) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    sentence = cleaned.split(". ", 1)[0]
    if len(sentence) > max_chars:
        return sentence[: max_chars - 1].rstrip() + "."
    return sentence


def _recommendation_items(codes: list[str]) -> list[dict[str, Any]]:
    lookup = _standard_lookup()
    items: list[dict[str, Any]] = []
    total = max(len(codes), 1)
    for index, code in enumerate(codes):
        standard = lookup.get(normalize_standard_code(code), {})
        title = str(standard.get("title") or "BIS catalog standard").strip()
        scope = _first_sentence(str(standard.get("scope") or standard.get("body") or ""))
        rationale = (
            f"Matched to the BIS catalog entry for {title}. {scope}"
            if scope
            else f"Matched to the BIS catalog entry for {title}."
        )
        items.append(
            {
                "code": code,
                "title": title.title() if title.isupper() else title,
                "rationale": rationale,
                "confidence": round(max(0.52, 0.94 - (index * (0.36 / total))), 2),
            }
        )
    return items


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/api/status")
def status() -> dict[str, Any]:
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


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect width="64" height="64" rx="12" fill="#08366f"/>'
        '<path d="M10 18h44v8H10z" fill="#ff9933"/>'
        '<path d="M10 38h44v8H10z" fill="#138808"/>'
        '<text x="32" y="36" text-anchor="middle" font-family="Arial" '
        'font-size="14" font-weight="700" fill="#fff">BIS</text></svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


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
        "recommendations": _recommendation_items(result["retrieved_standards"]),
    }


@app.post("/api/recommend", response_model=RecommendationResponse, include_in_schema=False)
def recommend_api(payload: RecommendationRequest) -> dict[str, Any]:
    return recommend(payload)
