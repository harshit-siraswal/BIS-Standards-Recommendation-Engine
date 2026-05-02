"""Vercel-compatible FastAPI entrypoint for the BIS recommendation engine."""

from __future__ import annotations

from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from src.pipeline import BISPipeline, MAX_QUERY_CHARS
from src.query_processor import is_out_of_scope_product
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
    language: str = Field(default="en", max_length=20)


class RecommendationItem(BaseModel):
    code: str
    title: str
    rationale: str
    confidence: float


class ExternalStandard(BaseModel):
    code: str
    title: str
    rationale: str
    source_url: str


class RecommendationResponse(BaseModel):
    query: str
    retrieved_standards: list[str]
    latency_seconds: float
    compliance_warnings: list[str]
    recommendations: list[RecommendationItem]
    out_of_scope: bool = False
    external_standards: list[ExternalStandard] = Field(default_factory=list)


FAVICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="BIS">
  <rect width="64" height="64" rx="10" fill="#08366f"/>
  <path d="M8 12h48v10H8z" fill="#ff9933"/>
  <path d="M8 42h48v10H8z" fill="#138808"/>
  <circle cx="32" cy="32" r="12" fill="#ffffff"/>
  <circle cx="32" cy="32" r="8" fill="none" stroke="#08366f" stroke-width="2"/>
  <text x="32" y="36.5" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="800" fill="#08366f">BIS</text>
</svg>
""".strip()


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BIS Standards Recommendation Engine</title>
  <meta name="theme-color" content="#08366f" />
  <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
  <link rel="alternate icon" href="/favicon.ico" />
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

    .language-select {
      border: 1px solid rgba(255, 255, 255, 0.45);
      background: var(--gov-navy);
      color: var(--white);
      border-radius: 4px;
      padding: 4px 8px;
      max-width: 190px;
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
      width: 24px;
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
      background: #082f63;
      color: rgba(255, 255, 255, 0.84);
      font-size: 0.9rem;
    }

    .footer-inner {
      display: grid;
      grid-template-columns: 1.25fr repeat(3, 1fr);
      gap: 28px;
      padding: 34px 0 26px;
    }

    .footer-brand {
      display: grid;
      gap: 10px;
      align-content: start;
    }

    .footer-mark {
      width: 52px;
      height: 52px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background:
        linear-gradient(180deg, var(--saffron) 0 32%, #ffffff 32% 66%, var(--india-green) 66% 100%);
      color: var(--gov-navy);
      font-weight: 900;
      border: 2px solid rgba(255, 255, 255, 0.75);
    }

    .footer-brand strong,
    .footer-links h2 {
      color: var(--white);
      font-size: 0.98rem;
      margin: 0 0 10px;
      letter-spacing: 0;
    }

    .footer-brand p,
    .footer-note p {
      margin: 0;
    }

    .footer-links ul {
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 8px;
    }

    .footer-links a,
    .footer-bottom a {
      color: rgba(255, 255, 255, 0.9);
      text-decoration: none;
    }

    .footer-links a:hover,
    .footer-bottom a:hover {
      text-decoration: underline;
    }

    .footer-bottom {
      border-top: 1px solid rgba(255, 255, 255, 0.18);
      padding: 14px 0 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }

    .footer-note {
      background: #06264e;
      color: rgba(255, 255, 255, 0.78);
      font-size: 0.82rem;
      padding: 10px 0;
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

      .footer-inner {
        grid-template-columns: 1fr 1fr;
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

      .footer-inner {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="gov-strip">
      <div class="strip-inner">
        <div data-i18n="govStrip">BIS standards discovery digital service</div>
        <div class="strip-links" aria-label="accessibility controls">
          <label for="languageSelect" data-i18n="languageLabel">Language</label>
          <select id="languageSelect" class="language-select" aria-label="Language">
            <option value="en">English</option>
            <option value="hi">हिन्दी</option>
            <option value="hinglish">Hinglish</option>
            <option value="bn">বাংলা</option>
            <option value="ta">தமிழ்</option>
            <option value="te">తెలుగు</option>
            <option value="mr">मराठी</option>
            <option value="gu">ગુજરાતી</option>
            <option value="kn">ಕನ್ನಡ</option>
            <option value="ml">മലയാളം</option>
            <option value="pa">ਪੰਜਾਬੀ</option>
            <option value="ur">اردو</option>
          </select>
          <button type="button" id="textSize">A+</button>
          <button type="button" id="contrast" data-i18n="contrast">Contrast</button>
        </div>
      </div>
    </div>
    <nav class="brand-nav" aria-label="primary">
      <div class="nav-inner">
        <div class="brand">
          <div class="emblem" aria-hidden="true">BIS</div>
          <div>
            <p class="brand-title">BIS Standards Recommendation Engine</p>
            <p class="brand-subtitle" data-i18n="brandSubtitle">Bureau of Indian Standards lookup for manufacturing enterprises</p>
          </div>
        </div>
        <div class="nav-links">
          <a href="#search" data-i18n="navSearch">Search</a>
          <a href="#results" data-i18n="navResults">Results</a>
          <a href="/api/status" data-i18n="navStatus">Service Status</a>
          <a href="/docs" data-i18n="navDocs">Developer API</a>
        </div>
      </div>
    </nav>
  </header>

  <main>
    <section class="hero" aria-labelledby="hero-title">
      <div class="hero-inner">
        <div>
          <div class="eyebrow" data-i18n="eyebrow">BIS Digital Service</div>
          <h1 id="hero-title" data-i18n="heroTitle">Search applicable Indian Standards for your product.</h1>
          <p class="hero-copy" data-i18n="heroCopy">Enter the product name, material, grade, and intended use. This service searches available BIS catalogue records and provides relevant Indian Standards for guidance.</p>
        </div>
        <aside class="service-card" aria-label="service status">
          <h2 data-i18n="serviceStatusTitle">Service status</h2>
          <p data-i18n="serviceStatusCopy">Digital assistance for standards discovery. Final compliance decisions should be verified with official BIS documents.</p>
          <div class="status-list">
            <div class="status-item"><span class="check">OK</span><span data-i18n="statusOne">BIS catalogue records loaded</span></div>
            <div class="status-item"><span class="check">OK</span><span data-i18n="statusTwo">Results limited to known standards or verified external guidance</span></div>
            <div class="status-item"><span class="check">OK</span><span data-i18n="statusThree">Accessible interface with Indian language support</span></div>
          </div>
        </aside>
      </div>
    </section>

    <section class="workspace" id="search">
      <div>
        <form class="query-panel" id="recommendForm">
          <label for="query" data-i18n="queryLabel">Describe the product, material, grade, and intended use</label>
          <textarea id="query" name="query" maxlength="500"></textarea>
          <div class="query-tools">
            <div class="chips" aria-label="sample queries">
              <button class="chip" type="button" data-sample="aggregates">Aggregates</button>
              <button class="chip" type="button" data-sample="pipes">Concrete pipes</button>
              <button class="chip" type="button" data-sample="whiteCement">White cement</button>
            </div>
            <button class="primary" id="submit" type="submit" data-i18n="submit">Search standards</button>
          </div>
          <div class="metrics" aria-label="service information">
            <div class="metric"><strong>SP 21</strong><span data-i18n="metricOne">Building materials catalogue</span></div>
            <div class="metric"><strong>Top 5</strong><span data-i18n="metricTwo">Relevant standard guidance</span></div>
            <div class="metric"><strong>&lt;5s</strong><span data-i18n="metricThree">Typical search target</span></div>
          </div>
        </form>

        <section class="results" id="results" aria-live="polite">
          <div class="results-header">
            <h2 data-i18n="resultsTitle">Relevant Indian Standards</h2>
            <span class="latency" id="latency" data-i18n="ready">Ready</span>
          </div>
          <div id="resultBody" class="empty-state" data-i18n="emptyState">
            Enter product details to view standards guidance.
          </div>
        </section>
      </div>

      <aside class="side-stack">
        <section class="panel notice">
          <h2 data-i18n="scopeTitle">Catalogue scope</h2>
          <p data-i18n="scopeCopy">The service uses available BIS catalogue data. If a product is outside the current catalogue, it will show a verification advisory instead of unrelated standards.</p>
          <div id="warnings" class="warnings"></div>
        </section>
        <section class="panel">
          <h2 data-i18n="workflowTitle">How to use this service</h2>
          <ul class="steps">
            <li><span class="num">1</span><span data-i18n="stepOne">Enter the product, material, grade, and intended use.</span></li>
            <li><span class="num">2</span><span data-i18n="stepTwo">Review the relevant IS codes and catalogue rationale.</span></li>
            <li><span class="num">3</span><span data-i18n="stepThree">Verify requirements on the official BIS portal before certification action.</span></li>
          </ul>
        </section>
        <section class="panel">
          <h2 data-i18n="advisoryTitle">Important advisory</h2>
          <p data-i18n="advisoryCopy">This digital service supports standards discovery. It does not replace official BIS standards, certification rules, testing requirements, or expert assessment.</p>
        </section>
      </aside>
    </section>
  </main>

  <footer>
    <div class="wrap footer-inner">
      <div class="footer-brand">
        <div class="footer-mark" aria-hidden="true">BIS</div>
        <strong data-i18n="footerTitle">BIS Standards Recommendation Engine</strong>
        <p data-i18n="footer">Digital aid for BIS standards discovery. Validate final compliance decisions with official BIS documents and competent authorities.</p>
      </div>
      <div class="footer-links">
        <h2 data-i18n="footerBis">BIS services</h2>
        <ul>
          <li><a href="https://www.bis.gov.in/" target="_blank" rel="noopener">BIS official website</a></li>
          <li><a href="https://standardsbis.bsbedge.com/" target="_blank" rel="noopener">Download Indian Standards</a></li>
          <li><a href="https://www.manakonline.in/" target="_blank" rel="noopener">Manak Online</a></li>
          <li><a href="https://www.bis.gov.in/product-certification/product-certification-overview/?lang=en" target="_blank" rel="noopener">Product certification</a></li>
        </ul>
      </div>
      <div class="footer-links">
        <h2 data-i18n="footerGov">Government links</h2>
        <ul>
          <li><a href="https://www.india.gov.in/" target="_blank" rel="noopener">National Portal of India</a></li>
          <li><a href="https://consumeraffairs.gov.in/" target="_blank" rel="noopener">Department of Consumer Affairs</a></li>
          <li><a href="https://consumerhelpline.gov.in/" target="_blank" rel="noopener">National Consumer Helpline</a></li>
          <li><a href="https://www.ux4g.gov.in/design-system.php" target="_blank" rel="noopener">UX4G Design System</a></li>
        </ul>
      </div>
      <div class="footer-links">
        <h2 data-i18n="footerService">Service access</h2>
        <ul>
          <li><a href="/api/status">Service status</a></li>
          <li><a href="/health">Health endpoint</a></li>
          <li><a href="/docs">Developer API</a></li>
          <li><a href="#search">Search standards</a></li>
        </ul>
      </div>
    </div>
    <div class="footer-note">
      <div class="wrap">
        <p data-i18n="footerNote">This application is a standards discovery aid. It is not an official BIS service.</p>
      </div>
    </div>
    <div class="wrap footer-bottom">
      <span data-i18n="footerUpdated">Last reviewed: 02 May 2026</span>
      <span><a href="#search" data-i18n="footerTop">Back to search</a></span>
    </div>
  </footer>

  <script>
    const baseText = {
      govStrip: "BIS standards discovery digital service",
      languageLabel: "Language",
      contrast: "Contrast",
      brandSubtitle: "Bureau of Indian Standards lookup for manufacturing enterprises",
      navSearch: "Search",
      navResults: "Results",
      navStatus: "Service Status",
      navDocs: "Developer API",
      eyebrow: "BIS Digital Service",
      heroTitle: "Search applicable Indian Standards for your product.",
      heroCopy: "Enter the product name, material, grade, and intended use. This service searches available BIS catalogue records and provides relevant Indian Standards for guidance.",
      serviceStatusTitle: "Service status",
      serviceStatusCopy: "Digital assistance for standards discovery. Final compliance decisions should be verified with official BIS documents.",
      statusOne: "BIS catalogue records loaded",
      statusTwo: "Results limited to known standards or verified external guidance",
      statusThree: "Accessible interface with Indian language support",
      queryLabel: "Describe the product, material, grade, and intended use",
      submit: "Search standards",
      loading: "Searching catalogue...",
      searching: "Searching",
      retrieving: "Retrieving relevant BIS standards...",
      metricOne: "Building materials catalogue",
      metricTwo: "Relevant standard guidance",
      metricThree: "Typical search target",
      resultsTitle: "Relevant Indian Standards",
      ready: "Ready",
      emptyState: "Enter product details to view standards guidance.",
      noResults: "No matching standards were returned.",
      confidence: "Confidence",
      outsideCatalog: "Outside current SP 21 catalogue",
      verifyBis: "Verify on BIS preview",
      scopeTitle: "Catalogue scope",
      scopeCopy: "The service uses available BIS catalogue data. If a product is outside the current catalogue, it will show a verification advisory instead of unrelated standards.",
      workflowTitle: "How to use this service",
      stepOne: "Enter the product, material, grade, and intended use.",
      stepTwo: "Review the relevant IS codes and catalogue rationale.",
      stepThree: "Verify requirements on the official BIS portal before certification action.",
      advisoryTitle: "Important advisory",
      advisoryCopy: "This digital service supports standards discovery. It does not replace official BIS standards, certification rules, testing requirements, or expert assessment.",
      footerTitle: "BIS Standards Recommendation Engine",
      footerBis: "BIS services",
      footerGov: "Government links",
      footerService: "Service access",
      footer: "Digital aid for BIS standards discovery. Validate final compliance decisions with official BIS documents and competent authorities.",
      footerNote: "This application is a standards discovery aid. It is not an official BIS service.",
      footerUpdated: "Last reviewed: 02 May 2026",
      footerTop: "Back to search",
      sampleDefault: "We manufacture 33 Grade Ordinary Portland Cement for general building construction. Which Indian Standard is applicable?",
      sampleAggregatesLabel: "Aggregates",
      sampleAggregates: "coarse and fine aggregates from natural sources for structural concrete",
      samplePipesLabel: "Concrete pipes",
      samplePipes: "precast concrete pipes with and without reinforcement for water mains",
      sampleWhiteCementLabel: "White cement",
      sampleWhiteCement: "white Portland cement for architectural and decorative purposes"
    };

    const translations = {
      en: baseText,
      hi: {
        ...baseText,
        govStrip: "BIS मानक खोज डिजिटल सेवा",
        languageLabel: "भाषा",
        contrast: "कॉन्ट्रास्ट",
        brandSubtitle: "विनिर्माण उद्यमों के लिए भारतीय मानक ब्यूरो मानक खोज",
        navSearch: "खोजें",
        navResults: "परिणाम",
        navStatus: "सेवा स्थिति",
        navDocs: "डेवलपर API",
        eyebrow: "BIS डिजिटल सेवा",
        heroTitle: "अपने उत्पाद के लिए लागू भारतीय मानक खोजें।",
        heroCopy: "उत्पाद का नाम, सामग्री, ग्रेड और उपयोग दर्ज करें। यह सेवा उपलब्ध BIS कैटलॉग रिकॉर्ड खोजकर संबंधित भारतीय मानक बताती है।",
        serviceStatusTitle: "सेवा स्थिति",
        serviceStatusCopy: "मानक खोज के लिए डिजिटल सहायता। अंतिम अनुपालन निर्णय आधिकारिक BIS दस्तावेजों से सत्यापित करें।",
        statusOne: "BIS कैटलॉग रिकॉर्ड लोड हैं",
        statusTwo: "परिणाम ज्ञात मानकों या सत्यापित बाहरी मार्गदर्शन तक सीमित हैं",
        statusThree: "भारतीय भाषा समर्थन के साथ सुलभ इंटरफेस",
        queryLabel: "उत्पाद, सामग्री, ग्रेड और उपयोग का वर्णन करें",
        submit: "मानक खोजें",
        loading: "कैटलॉग खोजा जा रहा है...",
        searching: "खोज जारी है",
        retrieving: "संबंधित BIS मानक खोजे जा रहे हैं...",
        metricOne: "निर्माण सामग्री कैटलॉग",
        metricTwo: "संबंधित मानक मार्गदर्शन",
        metricThree: "सामान्य खोज लक्ष्य",
        resultsTitle: "संबंधित भारतीय मानक",
        ready: "तैयार",
        emptyState: "मानक मार्गदर्शन देखने के लिए उत्पाद विवरण दर्ज करें।",
        noResults: "कोई मिलान करने वाला मानक नहीं मिला।",
        confidence: "विश्वास स्तर",
        outsideCatalog: "वर्तमान SP 21 कैटलॉग से बाहर",
        verifyBis: "BIS पूर्वावलोकन पर सत्यापित करें",
        scopeTitle: "कैटलॉग सीमा",
        scopeCopy: "यह सेवा उपलब्ध BIS कैटलॉग डेटा का उपयोग करती है। उत्पाद वर्तमान कैटलॉग से बाहर होने पर असंबंधित मानक दिखाने के बजाय सत्यापन सलाह दी जाएगी।",
        workflowTitle: "इस सेवा का उपयोग कैसे करें",
        stepOne: "उत्पाद, सामग्री, ग्रेड और उपयोग दर्ज करें।",
        stepTwo: "संबंधित IS कोड और कैटलॉग कारण देखें।",
        stepThree: "प्रमाणीकरण से पहले आधिकारिक BIS पोर्टल पर आवश्यकताएं सत्यापित करें।",
        advisoryTitle: "महत्वपूर्ण सलाह",
        advisoryCopy: "यह डिजिटल सेवा मानक खोज में सहायता करती है। यह आधिकारिक BIS मानकों, प्रमाणन नियमों, परीक्षण आवश्यकताओं या विशेषज्ञ आकलन का विकल्प नहीं है।",
        footer: "BIS मानक खोज के लिए डिजिटल सहायता। अंतिम अनुपालन निर्णय आधिकारिक BIS दस्तावेजों और सक्षम प्राधिकारियों से सत्यापित करें।",
        sampleDefault: "हम सामान्य भवन निर्माण के लिए 33 ग्रेड ऑर्डिनरी पोर्टलैंड सीमेंट बनाते हैं। कौन सा भारतीय मानक लागू है?",
        sampleAggregatesLabel: "एग्रीगेट",
        samplePipesLabel: "कंक्रीट पाइप",
        sampleWhiteCementLabel: "व्हाइट सीमेंट"
      },
      hinglish: {
        ...baseText,
        govStrip: "BIS standards discovery digital seva",
        languageLabel: "Bhasha",
        contrast: "Contrast",
        brandSubtitle: "Manufacturing enterprises ke liye BIS standards lookup",
        navSearch: "Search",
        navResults: "Results",
        navStatus: "Service Status",
        navDocs: "Developer API",
        eyebrow: "BIS Digital Seva",
        heroTitle: "Apne product ke liye applicable Indian Standards search karein.",
        heroCopy: "Product ka naam, material, grade aur use likhein. Service available BIS catalogue records se relevant Indian Standards guidance deti hai.",
        serviceStatusTitle: "Service status",
        serviceStatusCopy: "Standards discovery ke liye digital sahayata. Final compliance decision official BIS documents se verify karein.",
        statusOne: "BIS catalogue records loaded",
        statusTwo: "Results known standards ya verified external guidance tak limited hain",
        statusThree: "Indian language support ke saath accessible interface",
        queryLabel: "Product, material, grade aur intended use describe karein",
        submit: "Standards search karein",
        loading: "Catalogue search ho raha hai...",
        searching: "Searching",
        retrieving: "Relevant BIS standards retrieve ho rahe hain...",
        resultsTitle: "Relevant Indian Standards",
        emptyState: "Standards guidance dekhne ke liye product details enter karein.",
        noResults: "Matching standards nahi mile.",
        confidence: "Confidence",
        outsideCatalog: "Current SP 21 catalogue ke bahar",
        verifyBis: "BIS preview par verify karein",
        scopeTitle: "Catalogue scope",
        scopeCopy: "Service available BIS catalogue data use karti hai. Product current catalogue ke bahar ho to unrelated standards ki jagah verification advisory dikhegi.",
        workflowTitle: "Is service ka istemal kaise karein",
        stepOne: "Product, material, grade aur intended use enter karein.",
        stepTwo: "Relevant IS codes aur catalogue rationale review karein.",
        stepThree: "Certification action se pehle official BIS portal par requirements verify karein.",
        advisoryTitle: "Important advisory",
        advisoryCopy: "Ye digital service standards discovery mein help karti hai. Ye official BIS standards, certification rules, testing requirements ya expert assessment ka replacement nahi hai.",
        footer: "BIS standards discovery ke liye digital aid. Final compliance decisions official BIS documents aur competent authorities se verify karein.",
        sampleDefault: "Hum general building construction ke liye 33 Grade Ordinary Portland Cement manufacture karte hain. Kaunsa Indian Standard applicable hai?",
        sampleAggregatesLabel: "Aggregates",
        samplePipesLabel: "Concrete pipes",
        sampleWhiteCementLabel: "White cement"
      },
      bn: {
        ...baseText,
        govStrip: "BIS মান অনুসন্ধান ডিজিটাল পরিষেবা",
        languageLabel: "ভাষা",
        heroTitle: "আপনার পণ্যের জন্য প্রযোজ্য ভারতীয় মান অনুসন্ধান করুন।",
        queryLabel: "পণ্য, উপাদান, গ্রেড এবং ব্যবহারের বিবরণ দিন",
        submit: "মান অনুসন্ধান করুন",
        resultsTitle: "সম্পর্কিত ভারতীয় মান",
        emptyState: "মান নির্দেশিকা দেখতে পণ্যের বিবরণ লিখুন।"
      },
      ta: {
        ...baseText,
        govStrip: "BIS தர தேடல் டிஜிட்டல் சேவை",
        languageLabel: "மொழி",
        heroTitle: "உங்கள் தயாரிப்பிற்கு பொருந்தும் இந்திய தரங்களைத் தேடுங்கள்.",
        queryLabel: "தயாரிப்பு, பொருள், தரம் மற்றும் பயன்பாட்டை விவரிக்கவும்",
        submit: "தரங்களைத் தேடுங்கள்",
        resultsTitle: "பொருத்தமான இந்திய தரங்கள்",
        emptyState: "தர வழிகாட்டுதலை காண தயாரிப்பு விவரங்களை உள்ளிடவும்."
      },
      te: {
        ...baseText,
        govStrip: "BIS ప్రమాణాల శోధన డిజిటల్ సేవ",
        languageLabel: "భాష",
        heroTitle: "మీ ఉత్పత్తికి వర్తించే భారతీయ ప్రమాణాలను శోధించండి.",
        queryLabel: "ఉత్పత్తి, పదార్థం, గ్రేడ్ మరియు వినియోగాన్ని వివరించండి",
        submit: "ప్రమాణాలను శోధించండి",
        resultsTitle: "సంబంధిత భారతీయ ప్రమాణాలు",
        emptyState: "ప్రమాణ మార్గదర్శకాన్ని చూడటానికి ఉత్పత్తి వివరాలు నమోదు చేయండి."
      },
      mr: {
        ...baseText,
        govStrip: "BIS मानक शोध डिजिटल सेवा",
        languageLabel: "भाषा",
        heroTitle: "आपल्या उत्पादनासाठी लागू भारतीय मानके शोधा.",
        queryLabel: "उत्पादन, साहित्य, ग्रेड आणि वापराचे वर्णन करा",
        submit: "मानके शोधा",
        resultsTitle: "संबंधित भारतीय मानके",
        emptyState: "मानक मार्गदर्शन पाहण्यासाठी उत्पादन तपशील द्या."
      },
      gu: {
        ...baseText,
        govStrip: "BIS ધોરણ શોધ ડિજિટલ સેવા",
        languageLabel: "ભાષા",
        heroTitle: "તમારા ઉત્પાદન માટે લાગુ ભારતીય ધોરણો શોધો.",
        queryLabel: "ઉત્પાદન, સામગ્રી, ગ્રેડ અને ઉપયોગનું વર્ણન કરો",
        submit: "ધોરણો શોધો",
        resultsTitle: "સંબંધિત ભારતીય ધોરણો",
        emptyState: "ધોરણ માર્ગદર્શન જોવા ઉત્પાદન વિગતો દાખલ કરો."
      },
      kn: {
        ...baseText,
        govStrip: "BIS ಮಾನದಂಡ ಹುಡುಕಾಟ ಡಿಜಿಟಲ್ ಸೇವೆ",
        languageLabel: "ಭಾಷೆ",
        heroTitle: "ನಿಮ್ಮ ಉತ್ಪನ್ನಕ್ಕೆ ಅನ್ವಯಿಸುವ ಭಾರತೀಯ ಮಾನದಂಡಗಳನ್ನು ಹುಡುಕಿ.",
        queryLabel: "ಉತ್ಪನ್ನ, ವಸ್ತು, ಗ್ರೇಡ್ ಮತ್ತು ಬಳಕೆಯನ್ನು ವಿವರಿಸಿ",
        submit: "ಮಾನದಂಡಗಳನ್ನು ಹುಡುಕಿ",
        resultsTitle: "ಸಂಬಂಧಿತ ಭಾರತೀಯ ಮಾನದಂಡಗಳು",
        emptyState: "ಮಾನದಂಡ ಮಾರ್ಗದರ್ಶನಕ್ಕಾಗಿ ಉತ್ಪನ್ನ ವಿವರಗಳನ್ನು ನಮೂದಿಸಿ."
      },
      ml: {
        ...baseText,
        govStrip: "BIS സ്റ്റാൻഡേർഡ് തിരച്ചിൽ ഡിജിറ്റൽ സേവനം",
        languageLabel: "ഭാഷ",
        heroTitle: "നിങ്ങളുടെ ഉൽപ്പന്നത്തിന് ബാധകമായ ഇന്ത്യൻ സ്റ്റാൻഡേർഡുകൾ തിരയുക.",
        queryLabel: "ഉൽപ്പന്നം, വസ്തു, ഗ്രേഡ്, ഉപയോഗം എന്നിവ വിവരിക്കുക",
        submit: "സ്റ്റാൻഡേർഡുകൾ തിരയുക",
        resultsTitle: "ബന്ധപ്പെട്ട ഇന്ത്യൻ സ്റ്റാൻഡേർഡുകൾ",
        emptyState: "സ്റ്റാൻഡേർഡ് മാർഗ്ഗനിർദ്ദേശം കാണാൻ ഉൽപ്പന്ന വിശദാംശങ്ങൾ നൽകുക."
      },
      pa: {
        ...baseText,
        govStrip: "BIS ਮਿਆਰ ਖੋਜ ਡਿਜ਼ਿਟਲ ਸੇਵਾ",
        languageLabel: "ਭਾਸ਼ਾ",
        heroTitle: "ਆਪਣੇ ਉਤਪਾਦ ਲਈ ਲਾਗੂ ਭਾਰਤੀ ਮਿਆਰ ਖੋਜੋ।",
        queryLabel: "ਉਤਪਾਦ, ਸਮੱਗਰੀ, ਗ੍ਰੇਡ ਅਤੇ ਵਰਤੋਂ ਦਾ ਵੇਰਵਾ ਦਿਓ",
        submit: "ਮਿਆਰ ਖੋਜੋ",
        resultsTitle: "ਸੰਬੰਧਿਤ ਭਾਰਤੀ ਮਿਆਰ",
        emptyState: "ਮਿਆਰ ਮਾਰਗਦਰਸ਼ਨ ਵੇਖਣ ਲਈ ਉਤਪਾਦ ਵੇਰਵੇ ਦਿਓ।"
      },
      ur: {
        ...baseText,
        govStrip: "BIS معیارات کی تلاش ڈیجیٹل خدمت",
        languageLabel: "زبان",
        heroTitle: "اپنی مصنوعات کے لیے قابل اطلاق بھارتی معیارات تلاش کریں۔",
        queryLabel: "مصنوعات، مواد، گریڈ اور استعمال کی تفصیل درج کریں",
        submit: "معیارات تلاش کریں",
        resultsTitle: "متعلقہ بھارتی معیارات",
        emptyState: "معیاری رہنمائی دیکھنے کے لیے مصنوعات کی تفصیل درج کریں۔"
      }
    };

    const footerTranslations = {
      hi: {
        footerBis: "BIS \u0938\u0947\u0935\u093e\u090f\u0902",
        footerGov: "\u0938\u0930\u0915\u093e\u0930\u0940 \u0932\u093f\u0902\u0915",
        footerService: "\u0938\u0947\u0935\u093e \u092a\u0939\u0941\u0902\u091a",
        footerNote: "\u092f\u0939 \u090f\u092a\u094d\u0932\u093f\u0915\u0947\u0936\u0928 \u092e\u093e\u0928\u0915 \u0916\u094b\u091c \u0915\u0947 \u0932\u093f\u090f \u0938\u0939\u093e\u092f\u0924\u093e \u0939\u0948\u0964 \u092f\u0939 \u0906\u0927\u093f\u0915\u093e\u0930\u093f\u0915 BIS \u0938\u0947\u0935\u093e \u0928\u0939\u0940\u0902 \u0939\u0948\u0964",
        footerUpdated: "\u0905\u0902\u0924\u093f\u092e \u0938\u092e\u0940\u0915\u094d\u0937\u093e: 02 \u092e\u0908 2026",
        footerTop: "\u0916\u094b\u091c \u092a\u0930 \u0935\u093e\u092a\u0938 \u091c\u093e\u090f\u0902"
      },
      hinglish: {
        footerBis: "BIS services",
        footerGov: "Government links",
        footerService: "Service access",
        footerNote: "Ye application standards discovery aid hai. Ye official BIS service nahi hai.",
        footerUpdated: "Last reviewed: 02 May 2026",
        footerTop: "Search par wapas"
      },
      bn: {
        footerBis: "BIS \u09aa\u09b0\u09bf\u09b7\u09c7\u09ac\u09be",
        footerGov: "\u09b8\u09b0\u0995\u09be\u09b0\u09bf \u09b2\u09bf\u0999\u09cd\u0995",
        footerService: "\u09aa\u09b0\u09bf\u09b7\u09c7\u09ac\u09be \u09aa\u09cd\u09b0\u09ac\u09c7\u09b6",
        footerNote: "\u098f\u0987 \u0985\u09cd\u09af\u09be\u09aa\u09cd\u09b2\u09bf\u0995\u09c7\u09b6\u09a8\u099f\u09bf \u09ae\u09be\u09a8 \u0985\u09a8\u09c1\u09b8\u09a8\u09cd\u09a7\u09be\u09a8\u09c7\u09b0 \u09b8\u09b9\u09be\u09df\u0995\u0964 \u098f\u099f\u09bf \u0986\u09a7\u09bf\u0995\u09be\u09b0\u09bf\u0995 BIS \u09aa\u09b0\u09bf\u09b7\u09c7\u09ac\u09be \u09a8\u09df\u0964",
        footerUpdated: "\u09b6\u09c7\u09b7 \u09aa\u09b0\u09cd\u09af\u09be\u09b2\u09cb\u099a\u09a8\u09be: 02 \u09ae\u09c7 2026",
        footerTop: "\u0985\u09a8\u09c1\u09b8\u09a8\u09cd\u09a7\u09be\u09a8\u09c7 \u09ab\u09bf\u09b0\u09c1\u09a8"
      },
      ta: {
        footerBis: "BIS \u0b9a\u0bc7\u0bb5\u0bc8\u0b95\u0bb3\u0bcd",
        footerGov: "\u0b85\u0bb0\u0b9a\u0bc1 \u0b87\u0ba3\u0bc8\u0baa\u0bcd\u0baa\u0bc1\u0b95\u0bb3\u0bcd",
        footerService: "\u0b9a\u0bc7\u0bb5\u0bc8 \u0b85\u0ba3\u0bc1\u0b95\u0bb2\u0bcd",
        footerNote: "\u0b87\u0ba8\u0bcd\u0ba4 \u0baa\u0baf\u0ba9\u0bcd\u0baa\u0bbe\u0b9f\u0bc1 \u0ba4\u0bb0 \u0ba4\u0bc7\u0b9f\u0bb2\u0bc1\u0b95\u0bcd\u0b95\u0bbe\u0ba9 \u0b89\u0ba4\u0bb5\u0bbf\u0baf\u0bbe\u0b95\u0bc1\u0bae\u0bcd. \u0b87\u0ba4\u0bc1 \u0b85\u0ba4\u0bbf\u0b95\u0bbe\u0bb0\u0baa\u0bcd\u0baa\u0bc2\u0bb0\u0bcd\u0bb5 BIS \u0b9a\u0bc7\u0bb5\u0bc8 \u0b85\u0bb2\u0bcd\u0bb2.",
        footerUpdated: "\u0b95\u0b9f\u0bc8\u0b9a\u0bbf \u0bae\u0ba4\u0bbf\u0baa\u0bcd\u0baa\u0bbe\u0baf\u0bcd\u0bb5\u0bc1: 02 \u0bae\u0bc7 2026",
        footerTop: "\u0ba4\u0bc7\u0b9f\u0bb2\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0ba4\u0bbf\u0bb0\u0bc1\u0bae\u0bcd\u0baa\u0bc1"
      },
      te: {
        footerBis: "BIS \u0c38\u0c47\u0c35\u0c32\u0c41",
        footerGov: "\u0c2a\u0c4d\u0c30\u0c2d\u0c41\u0c24\u0c4d\u0c35 \u0c32\u0c3f\u0c02\u0c15\u0c4d\u0c32\u0c41",
        footerService: "\u0c38\u0c47\u0c35 \u0c2a\u0c4d\u0c30\u0c35\u0c47\u0c36\u0c02",
        footerNote: "\u0c08 \u0c05\u0c2a\u0c4d\u0c32\u0c3f\u0c15\u0c47\u0c37\u0c28\u0c4d \u0c2a\u0c4d\u0c30\u0c2e\u0c3e\u0c23\u0c3e\u0c32 \u0c36\u0c4b\u0c27\u0c28\u0c15\u0c41 \u0c38\u0c39\u0c3e\u0c2f\u0c02. \u0c07\u0c26\u0c3f \u0c05\u0c27\u0c3f\u0c15\u0c3e\u0c30\u0c3f\u0c15 BIS \u0c38\u0c47\u0c35 \u0c15\u0c3e\u0c26\u0c41.",
        footerUpdated: "\u0c1a\u0c3f\u0c35\u0c30\u0c3f \u0c38\u0c2e\u0c40\u0c15\u0c4d\u0c37: 02 \u0c2e\u0c47 2026",
        footerTop: "\u0c36\u0c4b\u0c27\u0c28\u0c15\u0c41 \u0c24\u0c3f\u0c30\u0c3f\u0c17\u0c3f \u0c35\u0c46\u0c33\u0c4d\u0c32\u0c02\u0c21\u0c3f"
      },
      mr: {
        footerBis: "BIS \u0938\u0947\u0935\u093e",
        footerGov: "\u0936\u093e\u0938\u0915\u0940\u092f \u0926\u0941\u0935\u0947",
        footerService: "\u0938\u0947\u0935\u093e \u092a\u094d\u0930\u0935\u0947\u0936",
        footerNote: "\u0939\u0947 \u0905\u0945\u092a\u094d\u0932\u093f\u0915\u0947\u0936\u0928 \u092e\u093e\u0928\u0915 \u0936\u094b\u0927\u0923\u094d\u092f\u093e\u0938\u093e\u0920\u0940 \u0938\u0939\u093e\u092f\u094d\u092f\u0915 \u0906\u0939\u0947. \u0939\u0940 \u0905\u0927\u093f\u0915\u0943\u0924 BIS \u0938\u0947\u0935\u093e \u0928\u093e\u0939\u0940.",
        footerUpdated: "\u0936\u0947\u0935\u091f\u091a\u0940 \u0938\u092e\u0940\u0915\u094d\u0937\u093e: 02 \u092e\u0947 2026",
        footerTop: "\u0936\u094b\u0927\u093e\u0915\u0921\u0947 \u092a\u0930\u0924 \u091c\u093e"
      },
      gu: {
        footerBis: "BIS \u0ab8\u0ac7\u0ab5\u0abe\u0a93",
        footerGov: "\u0ab8\u0ab0\u0a95\u0abe\u0ab0\u0ac0 \u0ab2\u0abf\u0a82\u0a95\u0acd\u0ab8",
        footerService: "\u0ab8\u0ac7\u0ab5\u0abe \u0a8d\u0a95\u0acd\u0ab8\u0ac7\u0ab8",
        footerNote: "\u0a86 \u0a8d\u0aaa\u0acd\u0ab2\u0abf\u0a95\u0ac7\u0ab6\u0aa8 \u0aa7\u0acb\u0ab0\u0aa3 \u0ab6\u0acb\u0aa7\u0ab5\u0abe \u0aae\u0abe\u0a9f\u0ac7 \u0ab8\u0ab9\u0abe\u0aaf\u0a95 \u0a9b\u0ac7. \u0a86 \u0ab8\u0aa4\u0acd\u0aa4\u0abe\u0ab5\u0abe\u0ab0 BIS \u0ab8\u0ac7\u0ab5\u0abe \u0aa8\u0aa5\u0ac0.",
        footerUpdated: "\u0a9b\u0ac7\u0ab2\u0acd\u0ab2\u0ac0 \u0ab8\u0aae\u0ac0\u0a95\u0acd\u0ab7\u0abe: 02 \u0aae\u0ac7 2026",
        footerTop: "\u0ab6\u0acb\u0aa7 \u0aaa\u0ab0 \u0aaa\u0abe\u0a9b\u0abe \u0a9c\u0abe\u0a93"
      },
      kn: {
        footerBis: "BIS \u0cb8\u0cc7\u0cb5\u0cc6\u0c97\u0cb3\u0cc1",
        footerGov: "\u0cb8\u0cb0\u0ccd\u0c95\u0cbe\u0cb0\u0cbf \u0cb2\u0cbf\u0c82\u0c95\u0ccd\u0c97\u0cb3\u0cc1",
        footerService: "\u0cb8\u0cc7\u0cb5\u0cc6 \u0caa\u0ccd\u0cb0\u0cb5\u0cc7\u0cb6",
        footerNote: "\u0c88 \u0c85\u0caa\u0ccd\u0cb2\u0cbf\u0c95\u0cc7\u0cb6\u0ca8\u0ccd \u0cae\u0cbe\u0ca8\u0ca6\u0c82\u0ca1 \u0cb9\u0cc1\u0ca1\u0cc1\u0c95\u0cbe\u0c9f\u0c95\u0ccd\u0c95\u0cc6 \u0cb8\u0cb9\u0cbe\u0caf\u0cb5\u0cbe\u0c97\u0cbf\u0ca6\u0cc6. \u0c87\u0ca6\u0cc1 \u0c85\u0ca7\u0cbf\u0c95\u0cc3\u0ca4 BIS \u0cb8\u0cc7\u0cb5\u0cc6 \u0c85\u0cb2\u0ccd\u0cb2.",
        footerUpdated: "\u0c95\u0cca\u0ca8\u0cc6\u0caf \u0caa\u0cb0\u0cbf\u0cb6\u0cc0\u0cb2\u0ca8\u0cc6: 02 \u0cae\u0cc7 2026",
        footerTop: "\u0cb9\u0cc1\u0ca1\u0cc1\u0c95\u0cbe\u0c9f\u0c95\u0ccd\u0c95\u0cc6 \u0cb9\u0cbf\u0c82\u0ca4\u0cbf\u0cb0\u0cc1\u0c97\u0cbf"
      },
      ml: {
        footerBis: "BIS \u0d38\u0d47\u0d35\u0d28\u0d19\u0d4d\u0d19\u0d7e",
        footerGov: "\u0d38\u0d7c\u0d15\u0d4d\u0d15\u0d3e\u0d7c \u0d32\u0d3f\u0d19\u0d4d\u0d15\u0d41\u0d15\u0d7e",
        footerService: "\u0d38\u0d47\u0d35\u0d28 \u0d05\u0d23\u0d41\u0d15\u0d7d",
        footerNote: "\u0d08 \u0d06\u0d2a\u0d4d\u0d32\u0d3f\u0d15\u0d4d\u0d15\u0d47\u0d37\u0d7b \u0d38\u0d4d\u0d31\u0d4d\u0d31\u0d3e\u0d7b\u0d21\u0d47\u0d7c\u0d21\u0d4d \u0d24\u0d3f\u0d30\u0d1a\u0d4d\u0d1a\u0d3f\u0d32\u0d3f\u0d28\u0d41\u0d33\u0d4d\u0d33 \u0d38\u0d39\u0d3e\u0d2f\u0d2e\u0d3e\u0d23\u0d4d. \u0d07\u0d24\u0d4d \u0d14\u0d26\u0d4d\u0d2f\u0d4b\u0d17\u0d3f\u0d15 BIS \u0d38\u0d47\u0d35\u0d28\u0d02 \u0d05\u0d32\u0d4d\u0d32.",
        footerUpdated: "\u0d05\u0d35\u0d38\u0d3e\u0d28 \u0d05\u0d35\u0d32\u0d4b\u0d15\u0d28\u0d02: 02 \u0d2e\u0d46\u0d2f\u0d4d 2026",
        footerTop: "\u0d24\u0d3f\u0d30\u0d1a\u0d4d\u0d1a\u0d3f\u0d32\u0d3f\u0d32\u0d47\u0d15\u0d4d\u0d15\u0d4d \u0d2e\u0d1f\u0d19\u0d4d\u0d19\u0d41\u0d15"
      },
      pa: {
        footerBis: "BIS \u0a38\u0a47\u0a35\u0a3e\u0a35\u0a3e\u0a02",
        footerGov: "\u0a38\u0a30\u0a15\u0a3e\u0a30\u0a40 \u0a32\u0a3f\u0a70\u0a15",
        footerService: "\u0a38\u0a47\u0a35\u0a3e \u0a2a\u0a39\u0a41\u0a70\u0a1a",
        footerNote: "\u0a07\u0a39 \u0a10\u0a2a\u0a32\u0a40\u0a15\u0a47\u0a38\u0a3c\u0a28 \u0a2e\u0a3f\u0a06\u0a30 \u0a16\u0a4b\u0a1c \u0a32\u0a08 \u0a38\u0a39\u0a3e\u0a07\u0a24\u0a3e \u0a39\u0a48\u0964 \u0a07\u0a39 \u0a05\u0a27\u0a3f\u0a15\u0a3e\u0a30\u0a3f\u0a15 BIS \u0a38\u0a47\u0a35\u0a3e \u0a28\u0a39\u0a40\u0a02 \u0a39\u0a48\u0964",
        footerUpdated: "\u0a06\u0a16\u0a30\u0a40 \u0a38\u0a2e\u0a40\u0a16\u0a3f\u0a06: 02 \u0a2e\u0a08 2026",
        footerTop: "\u0a16\u0a4b\u0a1c \u0a35\u0a3f\u0a71\u0a1a \u0a35\u0a3e\u0a2a\u0a38"
      },
      ur: {
        footerBis: "BIS \u062e\u062f\u0645\u0627\u062a",
        footerGov: "\u0633\u0631\u06a9\u0627\u0631\u06cc \u0631\u0627\u0628\u0637\u06d2",
        footerService: "\u0633\u0631\u0648\u0633 \u062a\u06a9 \u0631\u0633\u0627\u0626\u06cc",
        footerNote: "\u06cc\u06c1 \u0627\u06cc\u067e\u0644\u06cc \u06a9\u06cc\u0634\u0646 \u0645\u0639\u06cc\u0627\u0631\u0627\u062a \u06a9\u06cc \u062f\u0631\u06cc\u0627\u0641\u062a \u06a9\u06d2 \u0644\u06cc\u06d2 \u0645\u0639\u0627\u0648\u0646 \u06c1\u06d2\u06d4 \u06cc\u06c1 \u0633\u0631\u06a9\u0627\u0631\u06cc BIS \u0633\u0631\u0648\u0633 \u0646\u06c1\u06cc\u06ba \u06c1\u06d2\u06d4",
        footerUpdated: "\u0622\u062e\u0631\u06cc \u062c\u0627\u0626\u0632\u06c1: 02 \u0645\u0626\u06cc 2026",
        footerTop: "\u062a\u0644\u0627\u0634 \u067e\u0631 \u0648\u0627\u067e\u0633"
      }
    };

    Object.entries(footerTranslations).forEach(([lang, values]) => {
      translations[lang] = { ...translations[lang], ...values };
    });

    const form = document.querySelector("#recommendForm");
    const queryInput = document.querySelector("#query");
    const submitButton = document.querySelector("#submit");
    const resultBody = document.querySelector("#resultBody");
    const latency = document.querySelector("#latency");
    const warnings = document.querySelector("#warnings");
    const languageSelect = document.querySelector("#languageSelect");
    let currentLang = "en";

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function t(key) {
      return (translations[currentLang] && translations[currentLang][key]) || baseText[key] || key;
    }

    function applyLanguage(lang) {
      currentLang = translations[lang] ? lang : "en";
      document.documentElement.lang = currentLang === "hinglish" ? "en-IN" : currentLang;
      document.documentElement.dir = currentLang === "ur" ? "rtl" : "ltr";

      document.querySelectorAll("[data-i18n]").forEach((node) => {
        node.textContent = t(node.dataset.i18n);
      });

      document.querySelectorAll("[data-sample]").forEach((button) => {
        const key = button.dataset.sample;
        button.textContent = t(`sample${key[0].toUpperCase()}${key.slice(1)}Label`);
        button.dataset.query = t(`sample${key[0].toUpperCase()}${key.slice(1)}`);
      });

      const knownSamples = Object.values(translations).map((locale) => locale.sampleDefault);
      if (!queryInput.value || knownSamples.includes(queryInput.value)) {
        queryInput.value = t("sampleDefault");
      }
      queryInput.placeholder = t("queryLabel");
      if (latency.textContent === baseText.ready || latency.textContent === translations.hi.ready) {
        latency.textContent = t("ready");
      }
    }

    function setLoading(isLoading) {
      submitButton.disabled = isLoading;
      submitButton.textContent = isLoading ? t("loading") : t("submit");
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
      const external = data.external_standards || [];
      if (!recommendations.length && external.length) {
        resultBody.className = "result-list";
        resultBody.innerHTML = external.map((item, index) => `
          <article class="result">
            <div class="rank">${index + 1}</div>
            <div>
              <h3><span class="code">${escapeHtml(item.code)}</span>${escapeHtml(item.title)}</h3>
              <p class="rationale">${escapeHtml(item.rationale)}</p>
              <div class="confidence">
                <span>${escapeHtml(t("outsideCatalog"))}</span>
                <a href="${escapeHtml(item.source_url)}" target="_blank" rel="noopener">${escapeHtml(t("verifyBis"))}</a>
              </div>
            </div>
          </article>
        `).join("");
        return;
      }
      if (!recommendations.length) {
        resultBody.className = "empty-state";
        resultBody.textContent = t("noResults");
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
              <p class="rationale">${escapeHtml(item.rationale || "Matched against the BIS catalogue.")}</p>
              <div class="confidence">
                <span>${escapeHtml(t("confidence"))}</span>
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
      latency.textContent = t("searching");
      resultBody.className = "empty-state";
      resultBody.textContent = t("retrieving");
      try {
        const response = await fetch("/recommend", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, top_k: 5, language: currentLang }),
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

    document.querySelectorAll("[data-sample]").forEach((button) => {
      button.addEventListener("click", () => {
        queryInput.value = button.dataset.query;
        queryInput.focus();
      });
    });

    languageSelect.addEventListener("change", () => {
      applyLanguage(languageSelect.value);
    });

    document.querySelector("#contrast").addEventListener("click", () => {
      document.body.classList.toggle("contrast");
    });

    document.querySelector("#textSize").addEventListener("click", () => {
      document.body.classList.toggle("large-text");
    });

    applyLanguage("en");
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


RATIONALE_PREFIX = {
    "hi": "BIS कैटलॉग प्रविष्टि के आधार पर मिलान किया गया",
    "hinglish": "BIS catalogue entry ke basis par match kiya gaya",
}


def _recommendation_items(codes: list[str], language: str = "en") -> list[dict[str, Any]]:
    lookup = _standard_lookup()
    items: list[dict[str, Any]] = []
    total = max(len(codes), 1)
    prefix = RATIONALE_PREFIX.get(language, "Matched to the BIS catalogue entry for")
    for index, code in enumerate(codes):
        standard = lookup.get(normalize_standard_code(code), {})
        title = str(standard.get("title") or "BIS catalog standard").strip()
        scope = _first_sentence(str(standard.get("scope") or standard.get("body") or ""))
        rationale = (
            f"{prefix} {title}. {scope}"
            if scope
            else f"{prefix} {title}."
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


def _external_standards(query: str, language: str = "en") -> list[dict[str, str]]:
    if not is_out_of_scope_product(query):
        return []
    if language == "hi":
        first_rationale = (
            "यह BIS मानक ड्राइंग पेंसिल, कारपेंटर पेंसिल, स्टेनोग्राफर/रिपोर्टर पेंसिल "
            "और सामान्य लेखन पेंसिल की आवश्यकताओं को कवर करता है।"
        )
        second_rationale = (
            "यह BIS मानक पेंसिल स्लिप यानी pencil lead बनाने में उपयोग होने वाले graphite "
            "grades की आवश्यकताओं को कवर करता है।"
        )
    elif language == "hinglish":
        first_rationale = (
            "Ye BIS standard drawing pencils, carpenter's pencils, stenographer/reporter pencils "
            "aur general writing pencils ki requirements cover karta hai."
        )
        second_rationale = (
            "Ye BIS standard pencil slips, yaani pencil lead, banane ke liye graphite grades ki "
            "requirements cover karta hai."
        )
    else:
        first_rationale = (
            "This BIS standard covers requirements for drawing pencils, carpenter's pencils, "
            "stenographer's and reporter's pencils, and pencils for general writing."
        )
        second_rationale = (
            "This BIS standard covers graphite grades intended for manufacturing slips for pencils, "
            "also commonly referred to as pencil lead."
        )
    return [
        {
            "code": "IS 1375:2021",
            "title": "Black Lead Pencils - Specification",
            "rationale": first_rationale,
            "source_url": "https://standardsbis.bsbedge.com/BIS_Preview.aspx?id=1375_2021",
        },
        {
            "code": "IS 2079:2022",
            "title": "Graphite for Pencil Slips - Specification",
            "rationale": second_rationale,
            "source_url": "https://standardsbis.bsbedge.com/BIS_Preview.aspx?id=2079_2022",
        },
    ]


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


def _status_payload() -> dict[str, Any]:
    missing = _missing_artifacts()
    return {
        "service": "BIS Standards Recommendation Engine",
        "status": "ready" if not missing else "missing_artifacts",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend",
            "developer_api": "/docs",
        },
        "missing_artifacts": missing,
    }


def _status_html(payload: dict[str, Any]) -> str:
    status_label = str(payload["status"])
    missing = payload["missing_artifacts"]
    missing_html = (
        "<li>No missing service artifacts</li>"
        if not missing
        else "".join(f"<li>{escape(str(item))}</li>" for item in missing)
    )
    badge_class = "ready" if status_label == "ready" else "attention"
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="theme-color" content="#08366f" />
  <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
  <title>Service Status | BIS Standards Recommendation Engine</title>
  <style>
    :root {{
      --navy: #08366f;
      --blue: #005ea8;
      --saffron: #ff9933;
      --green: #138808;
      --paper: #f7f9fc;
      --line: #d8e1ee;
      --ink: #101828;
      --muted: #526173;
      --white: #ffffff;
      --success: #067647;
      --warning: #a15c07;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Noto Sans", "Segoe UI", Arial, sans-serif;
      line-height: 1.5;
    }}
    .top {{ background: var(--navy); color: var(--white); font-size: 0.86rem; }}
    .wrap {{ width: min(1120px, calc(100% - 32px)); margin: 0 auto; }}
    .top .wrap, nav .wrap {{
      min-height: 38px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    nav {{
      background: var(--white);
      border-bottom: 1px solid var(--line);
      box-shadow: 0 2px 16px rgba(16, 24, 40, 0.05);
    }}
    .brand {{ display: flex; align-items: center; gap: 12px; min-height: 74px; }}
    .mark {{
      width: 46px;
      height: 46px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: linear-gradient(180deg, var(--saffron) 0 32%, #fff 32% 66%, var(--green) 66% 100%);
      color: var(--navy);
      border: 2px solid var(--navy);
      font-weight: 900;
    }}
    .brand strong {{ color: var(--navy); font-size: 1.05rem; }}
    nav a {{ color: var(--navy); font-weight: 700; text-decoration: none; }}
    main {{ padding: 42px 0 54px; }}
    .panel {{
      background: var(--white);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 50px rgba(8, 54, 111, 0.12);
      overflow: hidden;
    }}
    .panel-head {{
      border-top: 5px solid var(--saffron);
      padding: 28px;
      display: grid;
      gap: 10px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0; color: var(--navy); font-size: clamp(1.6rem, 4vw, 2.5rem); letter-spacing: 0; }}
    .lead {{ margin: 0; max-width: 760px; color: var(--muted); }}
    .status-grid {{
      display: grid;
      grid-template-columns: minmax(220px, 0.8fr) 1.2fr;
      gap: 24px;
      padding: 28px;
    }}
    .badge {{
      display: inline-flex;
      width: fit-content;
      border-radius: 999px;
      padding: 7px 12px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 0.76rem;
    }}
    .badge.ready {{ background: #e8f7ef; color: var(--success); }}
    .badge.attention {{ background: #fff4e5; color: var(--warning); }}
    .detail {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #fbfdff;
    }}
    .detail h2 {{ margin: 0 0 10px; color: var(--navy); font-size: 1rem; }}
    .detail ul {{ margin: 0; padding-left: 18px; color: var(--muted); }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
    .actions a {{
      border: 1px solid var(--blue);
      border-radius: 6px;
      padding: 9px 12px;
      color: var(--blue);
      font-weight: 800;
      text-decoration: none;
      background: var(--white);
    }}
    .actions a.primary {{ background: var(--blue); color: var(--white); }}
    footer {{ background: #082f63; color: rgba(255, 255, 255, 0.84); padding: 18px 0; font-size: 0.86rem; }}
    footer a {{ color: var(--white); }}
    @media (max-width: 720px) {{
      .top .wrap, nav .wrap {{ align-items: flex-start; flex-direction: column; padding: 10px 0; }}
      .status-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="top"><div class="wrap"><span>Government-style digital service</span><span>Service status</span></div></div>
    <nav><div class="wrap"><div class="brand"><div class="mark">BIS</div><strong>BIS Standards Recommendation Engine</strong></div><a href="/">Back to search</a></div></nav>
  </header>
  <main class="wrap">
    <section class="panel">
      <div class="panel-head">
        <span class="badge {badge_class}">{escape(status_label.replace("_", " "))}</span>
        <h1>Service Status</h1>
        <p class="lead">Operational summary for the recommendation service. The JSON health endpoint remains available for automated monitoring.</p>
      </div>
      <div class="status-grid">
        <div class="detail">
          <h2>Current status</h2>
          <p><strong>{escape(status_label.replace("_", " ").title())}</strong></p>
          <div class="actions">
            <a class="primary" href="/">Search standards</a>
            <a href="/docs">Developer API</a>
          </div>
        </div>
        <div class="detail">
          <h2>Artifact checks</h2>
          <ul>{missing_html}</ul>
          <div class="actions">
            <a href="/health">View health JSON</a>
            <a href="/api/status?format=json">View status JSON</a>
          </div>
        </div>
      </div>
    </section>
  </main>
  <footer><div class="wrap">Digital aid for BIS standards discovery. Validate final compliance decisions with official BIS documents and competent authorities.</div></footer>
</body>
</html>
""".strip()


@app.get("/api/status", response_model=None)
def status(request: Request) -> Any:
    payload = _status_payload()
    wants_json = request.query_params.get("format") == "json"
    accepts_html = "text/html" in request.headers.get("accept", "")
    if accepts_html and not wants_json:
        return HTMLResponse(_status_html(payload))
    return payload


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg() -> Response:
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return favicon_svg()




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
        "recommendations": _recommendation_items(result["retrieved_standards"], payload.language),
        "external_standards": _external_standards(query, payload.language),
    }


@app.post("/api/recommend", response_model=RecommendationResponse, include_in_schema=False)
def recommend_api(payload: RecommendationRequest) -> dict[str, Any]:
    return recommend(payload)
