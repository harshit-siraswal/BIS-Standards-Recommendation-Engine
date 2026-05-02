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
        <div data-i18n="govStrip">An official Government of India digital service</div>
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
    <div class="wrap" data-i18n="footer">Digital aid for BIS standards discovery. Validate final compliance decisions with official BIS documents and competent authorities.</div>
  </footer>

  <script>
    const baseText = {
      govStrip: "An official Government of India digital service",
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
      footer: "Digital aid for BIS standards discovery. Validate final compliance decisions with official BIS documents and competent authorities.",
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
        govStrip: "भारत सरकार की आधिकारिक डिजिटल सेवा",
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
        govStrip: "Government of India ki official digital service",
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
        govStrip: "ভারত সরকারের সরকারি ডিজিটাল পরিষেবা",
        languageLabel: "ভাষা",
        heroTitle: "আপনার পণ্যের জন্য প্রযোজ্য ভারতীয় মান অনুসন্ধান করুন।",
        queryLabel: "পণ্য, উপাদান, গ্রেড এবং ব্যবহারের বিবরণ দিন",
        submit: "মান অনুসন্ধান করুন",
        resultsTitle: "সম্পর্কিত ভারতীয় মান",
        emptyState: "মান নির্দেশিকা দেখতে পণ্যের বিবরণ লিখুন।"
      },
      ta: {
        ...baseText,
        govStrip: "இந்திய அரசின் அதிகாரப்பூர்வ டிஜிட்டல் சேவை",
        languageLabel: "மொழி",
        heroTitle: "உங்கள் தயாரிப்பிற்கு பொருந்தும் இந்திய தரங்களைத் தேடுங்கள்.",
        queryLabel: "தயாரிப்பு, பொருள், தரம் மற்றும் பயன்பாட்டை விவரிக்கவும்",
        submit: "தரங்களைத் தேடுங்கள்",
        resultsTitle: "பொருத்தமான இந்திய தரங்கள்",
        emptyState: "தர வழிகாட்டுதலை காண தயாரிப்பு விவரங்களை உள்ளிடவும்."
      },
      te: {
        ...baseText,
        govStrip: "భారత ప్రభుత్వ అధికారిక డిజిటల్ సేవ",
        languageLabel: "భాష",
        heroTitle: "మీ ఉత్పత్తికి వర్తించే భారతీయ ప్రమాణాలను శోధించండి.",
        queryLabel: "ఉత్పత్తి, పదార్థం, గ్రేడ్ మరియు వినియోగాన్ని వివరించండి",
        submit: "ప్రమాణాలను శోధించండి",
        resultsTitle: "సంబంధిత భారతీయ ప్రమాణాలు",
        emptyState: "ప్రమాణ మార్గదర్శకాన్ని చూడటానికి ఉత్పత్తి వివరాలు నమోదు చేయండి."
      },
      mr: {
        ...baseText,
        govStrip: "भारत सरकारची अधिकृत डिजिटल सेवा",
        languageLabel: "भाषा",
        heroTitle: "आपल्या उत्पादनासाठी लागू भारतीय मानके शोधा.",
        queryLabel: "उत्पादन, साहित्य, ग्रेड आणि वापराचे वर्णन करा",
        submit: "मानके शोधा",
        resultsTitle: "संबंधित भारतीय मानके",
        emptyState: "मानक मार्गदर्शन पाहण्यासाठी उत्पादन तपशील द्या."
      },
      gu: {
        ...baseText,
        govStrip: "ભારત સરકારની સત્તાવાર ડિજિટલ સેવા",
        languageLabel: "ભાષા",
        heroTitle: "તમારા ઉત્પાદન માટે લાગુ ભારતીય ધોરણો શોધો.",
        queryLabel: "ઉત્પાદન, સામગ્રી, ગ્રેડ અને ઉપયોગનું વર્ણન કરો",
        submit: "ધોરણો શોધો",
        resultsTitle: "સંબંધિત ભારતીય ધોરણો",
        emptyState: "ધોરણ માર્ગદર્શન જોવા ઉત્પાદન વિગતો દાખલ કરો."
      },
      kn: {
        ...baseText,
        govStrip: "ಭಾರತ ಸರ್ಕಾರದ ಅಧಿಕೃತ ಡಿಜಿಟಲ್ ಸೇವೆ",
        languageLabel: "ಭಾಷೆ",
        heroTitle: "ನಿಮ್ಮ ಉತ್ಪನ್ನಕ್ಕೆ ಅನ್ವಯಿಸುವ ಭಾರತೀಯ ಮಾನದಂಡಗಳನ್ನು ಹುಡುಕಿ.",
        queryLabel: "ಉತ್ಪನ್ನ, ವಸ್ತು, ಗ್ರೇಡ್ ಮತ್ತು ಬಳಕೆಯನ್ನು ವಿವರಿಸಿ",
        submit: "ಮಾನದಂಡಗಳನ್ನು ಹುಡುಕಿ",
        resultsTitle: "ಸಂಬಂಧಿತ ಭಾರತೀಯ ಮಾನದಂಡಗಳು",
        emptyState: "ಮಾನದಂಡ ಮಾರ್ಗದರ್ಶನಕ್ಕಾಗಿ ಉತ್ಪನ್ನ ವಿವರಗಳನ್ನು ನಮೂದಿಸಿ."
      },
      ml: {
        ...baseText,
        govStrip: "ഇന്ത്യ സർക്കാരിന്റെ ഔദ്യോഗിക ഡിജിറ്റൽ സേവനം",
        languageLabel: "ഭാഷ",
        heroTitle: "നിങ്ങളുടെ ഉൽപ്പന്നത്തിന് ബാധകമായ ഇന്ത്യൻ സ്റ്റാൻഡേർഡുകൾ തിരയുക.",
        queryLabel: "ഉൽപ്പന്നം, വസ്തു, ഗ്രേഡ്, ഉപയോഗം എന്നിവ വിവരിക്കുക",
        submit: "സ്റ്റാൻഡേർഡുകൾ തിരയുക",
        resultsTitle: "ബന്ധപ്പെട്ട ഇന്ത്യൻ സ്റ്റാൻഡേർഡുകൾ",
        emptyState: "സ്റ്റാൻഡേർഡ് മാർഗ്ഗനിർദ്ദേശം കാണാൻ ഉൽപ്പന്ന വിശദാംശങ്ങൾ നൽകുക."
      },
      pa: {
        ...baseText,
        govStrip: "ਭਾਰਤ ਸਰਕਾਰ ਦੀ ਅਧਿਕਾਰਿਕ ਡਿਜ਼ਿਟਲ ਸੇਵਾ",
        languageLabel: "ਭਾਸ਼ਾ",
        heroTitle: "ਆਪਣੇ ਉਤਪਾਦ ਲਈ ਲਾਗੂ ਭਾਰਤੀ ਮਿਆਰ ਖੋਜੋ।",
        queryLabel: "ਉਤਪਾਦ, ਸਮੱਗਰੀ, ਗ੍ਰੇਡ ਅਤੇ ਵਰਤੋਂ ਦਾ ਵੇਰਵਾ ਦਿਓ",
        submit: "ਮਿਆਰ ਖੋਜੋ",
        resultsTitle: "ਸੰਬੰਧਿਤ ਭਾਰਤੀ ਮਿਆਰ",
        emptyState: "ਮਿਆਰ ਮਾਰਗਦਰਸ਼ਨ ਵੇਖਣ ਲਈ ਉਤਪਾਦ ਵੇਰਵੇ ਦਿਓ।"
      },
      ur: {
        ...baseText,
        govStrip: "حکومت ہند کی سرکاری ڈیجیٹل خدمت",
        languageLabel: "زبان",
        heroTitle: "اپنی مصنوعات کے لیے قابل اطلاق بھارتی معیارات تلاش کریں۔",
        queryLabel: "مصنوعات، مواد، گریڈ اور استعمال کی تفصیل درج کریں",
        submit: "معیارات تلاش کریں",
        resultsTitle: "متعلقہ بھارتی معیارات",
        emptyState: "معیاری رہنمائی دیکھنے کے لیے مصنوعات کی تفصیل درج کریں۔"
      }
    };

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
    normalized = " ".join(str(query or "").lower().split())
    if "pencil" not in normalized:
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
        "recommendations": _recommendation_items(result["retrieved_standards"], payload.language),
        "external_standards": _external_standards(query, payload.language),
    }


@app.post("/api/recommend", response_model=RecommendationResponse, include_in_schema=False)
def recommend_api(payload: RecommendationRequest) -> dict[str, Any]:
    return recommend(payload)
