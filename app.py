"""Vercel-compatible FastAPI entrypoint for the BIS recommendation engine."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from src.pipeline import BISPipeline, MAX_QUERY_CHARS
from src.query_processor import is_edible_oil_query, is_pencil_query
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


class ChatMessage(BaseModel):
    role: str = Field(..., max_length=20)
    content: str = Field(..., min_length=1, max_length=1200)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    message: str = Field(..., min_length=1, max_length=1200)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)
    top_k: int = Field(default=5, ge=1, le=5)
    language: str = Field(default="en", max_length=20)


class RecommendationItem(BaseModel):
    code: str
    title: str
    rationale: str
    confidence: float
    source_url: str


class ExternalStandard(BaseModel):
    code: str
    title: str
    rationale: str
    source_url: str


class BusinessGuidance(BaseModel):
    matched_category: str
    matched_terms: list[str]
    why_these_standards: list[str]
    documents_to_prepare: list[str]
    testing_lab_readiness: list[str]
    bis_workflow: list[str]
    verification_notes: list[str]
    ai_generated: bool = False


class RecommendationResponse(BaseModel):
    query: str
    retrieved_standards: list[str]
    latency_seconds: float
    compliance_warnings: list[str]
    recommendations: list[RecommendationItem]
    business_guidance: BusinessGuidance
    out_of_scope: bool = False
    external_standards: list[ExternalStandard] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    retrieved_standards: list[str]
    compliance_warnings: list[str]
    ai_generated: bool = False


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
  <title>Business Compliance Assistant | BIS Standards</title>
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
      --info: #175cd3;
      --shadow: 0 18px 50px rgba(8, 54, 111, 0.14);
      --radius: 8px;
      --max: 1380px;
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
      grid-template-columns: minmax(0, 1fr) minmax(420px, 0.78fr);
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

    .result-meta {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 800;
    }

    .result-meta a {
      color: var(--gov-blue);
      overflow-wrap: anywhere;
      text-decoration: none;
    }

    .result-meta a:hover {
      text-decoration: underline;
    }

    .side-stack {
      display: grid;
      gap: 16px;
    }

    .panel {
      padding: 18px;
    }

    .panel.assistant-panel {
      padding: 0;
      overflow: hidden;
    }

    .assistant-panel {
      position: sticky;
      top: 16px;
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

    .guidance {
      display: grid;
      gap: 14px;
      padding: 18px 20px;
    }

    .assistant-panel {
      border-top: 4px solid var(--india-green);
    }

    .assistant-empty {
      color: var(--muted);
      font-size: 0.92rem;
    }

    .guidance-card {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfdff;
      padding: 12px;
    }

    .guidance-card h3 {
      margin: 0 0 8px;
      color: var(--gov-navy);
      font-size: 0.9rem;
    }

    .guidance-card p,
    .guidance-card ul {
      margin: 0;
      color: var(--muted);
      font-size: 0.88rem;
    }

    .guidance-card ul {
      padding-left: 18px;
    }

    .term-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .term-pill {
      border: 1px solid rgba(0, 94, 168, 0.22);
      border-radius: 999px;
      background: var(--gov-blue-2);
      color: var(--gov-navy);
      padding: 5px 9px;
      font-size: 0.78rem;
      font-weight: 800;
    }

    .guidance-source {
      display: inline-flex;
      width: fit-content;
      border-radius: 999px;
      background: #e8f1ff;
      color: var(--info);
      padding: 5px 9px;
      font-size: 0.76rem;
      font-weight: 900;
      text-transform: uppercase;
    }

    .chat {
      border-top: 1px solid var(--line);
      padding: 18px 20px;
      display: grid;
      gap: 12px;
    }

    .chat h3 {
      margin: 0;
      color: var(--gov-navy);
      font-size: 0.96rem;
    }

    .chat-log {
      min-height: 112px;
      max-height: 260px;
      overflow: auto;
      display: grid;
      align-content: start;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfdff;
      padding: 10px;
    }

    .chat-message {
      border-radius: 6px;
      padding: 9px 10px;
      font-size: 0.88rem;
      color: var(--ink);
      background: var(--white);
      border: 1px solid var(--line);
    }

    .chat-message.user {
      background: var(--gov-blue);
      border-color: var(--gov-blue);
      color: var(--white);
      justify-self: end;
      max-width: 92%;
    }

    .chat-message.assistant {
      justify-self: start;
      max-width: 96%;
    }

    .chat-form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
    }

    .chat-form input {
      min-width: 0;
      border: 1px solid #b7c7da;
      border-radius: 6px;
      padding: 10px 11px;
      color: var(--ink);
      background: var(--white);
    }

    .chat-form button {
      border: 0;
      border-radius: 6px;
      background: var(--gov-blue);
      color: var(--white);
      padding: 10px 13px;
      font-weight: 800;
      cursor: pointer;
    }

    .chat-form button:disabled {
      opacity: 0.65;
      cursor: wait;
    }

    .chat-note {
      margin: 0;
      color: var(--muted);
      font-size: 0.78rem;
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

      .assistant-panel {
        position: static;
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
            <p class="brand-title" data-i18n="footerTitle">Business Compliance Assistant</p>
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
          <h1 id="hero-title" data-i18n="heroTitle">Business Compliance Assistant for BIS standards.</h1>
          <p class="hero-copy" data-i18n="heroCopy">Enter a messy product description. The assistant retrieves relevant Indian Standards, explains why they match, and lists practical next steps to verify with BIS.</p>
        </div>
        <aside class="service-card" aria-label="service status">
          <h2 data-i18n="serviceStatusTitle">Service status</h2>
          <p data-i18n="serviceStatusCopy">Digital assistance for standards discovery. Final compliance decisions should be verified with official BIS documents.</p>
          <div class="status-list">
            <div class="status-item"><span class="check" data-i18n="statusOk">OK</span><span data-i18n="statusOne">BIS catalogue records loaded</span></div>
            <div class="status-item"><span class="check" data-i18n="statusOk">OK</span><span data-i18n="statusTwo">Results limited to known standards or verified external guidance</span></div>
            <div class="status-item"><span class="check" data-i18n="statusOk">OK</span><span data-i18n="statusThree">Accessible interface with Indian language support</span></div>
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
        <section class="panel assistant-panel" aria-live="polite">
          <div class="results-header">
            <h2 data-assistant-i18n="title">Business Compliance Assistant</h2>
            <span class="latency" data-assistant-i18n="ready">Next steps</span>
          </div>
          <div id="guidanceBody" class="guidance assistant-empty">
            <p data-assistant-i18n="empty">Run a standards search to view matched category, key terms, document readiness, testing readiness, and verification notes.</p>
          </div>
          <div class="chat">
            <h3 data-assistant-i18n="chatTitle">Ask follow-up questions</h3>
            <div id="chatLog" class="chat-log">
              <div class="chat-message assistant" data-assistant-i18n="chatEmpty">Search a product, then ask about applicability, documents, testing, or verification.</div>
            </div>
            <form id="chatForm" class="chat-form">
              <input id="chatInput" type="text" maxlength="1200" data-assistant-placeholder="chatPlaceholder" placeholder="Ask what to do next..." />
              <button id="chatSubmit" type="submit" data-assistant-i18n="chatSend">Ask</button>
            </form>
            <p class="chat-note" data-assistant-i18n="chatNote">The chatbot only uses standards returned by this retriever and asks you to verify with BIS.</p>
          </div>
        </section>
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
        <strong data-i18n="footerTitle">Business Compliance Assistant</strong>
        <p data-i18n="footer">Digital aid for BIS standards discovery. Validate final compliance decisions with official BIS documents and competent authorities.</p>
      </div>
      <div class="footer-links">
        <h2 data-i18n="footerBis">BIS services</h2>
        <ul>
          <li><a href="https://www.bis.gov.in/" target="_blank" rel="noopener" data-i18n="footerBisWebsite">BIS official website</a></li>
          <li><a href="https://standardsbis.bsbedge.com/" target="_blank" rel="noopener" data-i18n="footerDownload">Download Indian Standards</a></li>
          <li><a href="https://www.manakonline.in/" target="_blank" rel="noopener" data-i18n="footerManak">Manak Online</a></li>
          <li><a href="https://www.bis.gov.in/product-certification/product-certification-overview/?lang=en" target="_blank" rel="noopener" data-i18n="footerCertification">Product certification</a></li>
        </ul>
      </div>
      <div class="footer-links">
        <h2 data-i18n="footerGov">Government links</h2>
        <ul>
          <li><a href="https://www.india.gov.in/" target="_blank" rel="noopener" data-i18n="footerIndiaPortal">National Portal of India</a></li>
          <li><a href="https://consumeraffairs.gov.in/" target="_blank" rel="noopener" data-i18n="footerConsumerAffairs">Department of Consumer Affairs</a></li>
          <li><a href="https://consumerhelpline.gov.in/" target="_blank" rel="noopener" data-i18n="footerConsumerHelpline">National Consumer Helpline</a></li>
          <li><a href="https://www.ux4g.gov.in/design-system.php" target="_blank" rel="noopener" data-i18n="footerUx4g">UX4G Design System</a></li>
        </ul>
      </div>
      <div class="footer-links">
        <h2 data-i18n="footerService">Service access</h2>
        <ul>
          <li><a href="/api/status" data-i18n="footerStatus">Service status</a></li>
          <li><a href="/health" data-i18n="footerHealth">Health endpoint</a></li>
          <li><a href="/docs" data-i18n="footerApi">Developer API</a></li>
          <li><a href="#search" data-i18n="footerSearch">Search standards</a></li>
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
      heroTitle: "Business Compliance Assistant for BIS standards.",
      heroCopy: "Enter a messy product description. The assistant retrieves relevant Indian Standards, explains why they match, and lists practical next steps to verify with BIS.",
      serviceStatusTitle: "Service status",
      serviceStatusCopy: "Digital assistance for standards discovery. Final compliance decisions should be verified with official BIS documents.",
      statusOk: "OK",
      statusOne: "BIS catalogue records loaded",
      statusTwo: "Results limited to known standards or verified external guidance",
      statusThree: "Accessible interface with Indian language support",
      queryLabel: "Describe the product, material, grade, and intended use",
      submit: "Search standards",
      loading: "Searching catalogue...",
      searching: "Searching",
      retrieving: "Retrieving relevant BIS standards...",
      error: "Error",
      recommendationFailed: "Recommendation failed",
      metricOne: "Building materials catalogue",
      metricTwo: "Relevant standard guidance",
      metricThree: "Typical search target",
      resultsTitle: "Relevant Indian Standards",
      ready: "Ready",
      emptyState: "Enter product details to view standards guidance.",
      noResults: "No matching standards were returned.",
      outsideCatalog: "Outside current SP 21 catalogue",
      verifyBis: "Verify on BIS preview",
      pencilWarning: "Graphite or black lead pencils are outside the bundled BIS SP 21 building-materials catalogue. Verify pencil-specific BIS standards such as IS 1375:2021 and IS 2079:2022.",
      edibleOilWarning: "Edible oil is outside the bundled BIS SP 21 building-materials catalogue. Use oil and fats standards such as IS 548 and the relevant oil-type specification.",
      scopeTitle: "Catalogue scope",
      scopeCopy: "The service uses available BIS catalogue data. If a product is outside the current catalogue, it will show a verification advisory instead of unrelated standards.",
      workflowTitle: "How to use this service",
      stepOne: "Enter the product, material, grade, and intended use.",
      stepTwo: "Review the relevant IS codes and catalogue rationale.",
      stepThree: "Verify requirements on the official BIS portal before certification action.",
      advisoryTitle: "Important advisory",
      advisoryCopy: "This digital service supports standards discovery. It does not replace official BIS standards, certification rules, testing requirements, or expert assessment.",
      footerTitle: "Business Compliance Assistant",
      footerBis: "BIS services",
      footerGov: "Government links",
      footerService: "Service access",
      footerBisWebsite: "BIS official website",
      footerDownload: "Download Indian Standards",
      footerManak: "Manak Online",
      footerCertification: "Product certification",
      footerIndiaPortal: "National Portal of India",
      footerConsumerAffairs: "Department of Consumer Affairs",
      footerConsumerHelpline: "National Consumer Helpline",
      footerUx4g: "UX4G Design System",
      footerStatus: "Service status",
      footerHealth: "Health endpoint",
      footerApi: "Developer API",
      footerSearch: "Search standards",
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
        heroTitle: "BIS मानकों के लिए व्यावसायिक अनुपालन सहायक।",
        heroCopy: "अव्यवस्थित उत्पाद विवरण दर्ज करें। सहायक संबंधित भारतीय मानक खोजता है, मिलान का कारण बताता है, और BIS से सत्यापित करने के लिए व्यावहारिक अगले कदम देता है।",
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
        heroTitle: "BIS standards ke liye Business Compliance Assistant.",
        heroCopy: "Messy product description enter karein. Assistant relevant Indian Standards retrieve karta hai, match ka reason batata hai, aur BIS se verify karne ke practical next steps deta hai.",
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

    const localeCompletion = {
      hi: {
        statusOk: "ठीक",
        error: "त्रुटि",
        recommendationFailed: "अनुशंसा प्राप्त नहीं हो सकी",
        pencilWarning: "ग्रेफाइट या ब्लैक लेड पेंसिल वर्तमान BIS SP 21 निर्माण-सामग्री कैटलॉग से बाहर हैं। पेंसिल से संबंधित IS 1375:2021 और IS 2079:2022 जैसे BIS मानक सत्यापित करें।",
        edibleOilWarning: "खाद्य तेल वर्तमान BIS SP 21 निर्माण-सामग्री कैटलॉग से बाहर है। तेल और वसा से संबंधित IS 548 और संबंधित तेल-प्रकार विनिर्देश देखें।",
        footerTitle: "व्यावसायिक अनुपालन सहायक",
        footerBis: "BIS सेवाएं",
        footerGov: "सरकारी लिंक",
        footerService: "सेवा पहुंच",
        footerBisWebsite: "BIS आधिकारिक वेबसाइट",
        footerDownload: "भारतीय मानक डाउनलोड करें",
        footerManak: "मानक ऑनलाइन",
        footerCertification: "उत्पाद प्रमाणन",
        footerIndiaPortal: "भारत का राष्ट्रीय पोर्टल",
        footerConsumerAffairs: "उपभोक्ता मामले विभाग",
        footerConsumerHelpline: "राष्ट्रीय उपभोक्ता हेल्पलाइन",
        footerUx4g: "UX4G डिजाइन प्रणाली",
        footerStatus: "सेवा स्थिति",
        footerHealth: "स्वास्थ्य एंडपॉइंट",
        footerApi: "डेवलपर API",
        footerSearch: "मानक खोजें",
        footerNote: "यह एप्लिकेशन मानक खोज के लिए सहायता है। यह आधिकारिक BIS सेवा नहीं है।",
        footerUpdated: "अंतिम समीक्षा: 02 मई 2026",
        footerTop: "खोज पर वापस जाएं",
        sampleAggregates: "संरचनात्मक कंक्रीट के लिए प्राकृतिक स्रोतों से मोटे और महीन एग्रीगेट",
        samplePipes: "जल मुख्य लाइनों के लिए सुदृढ़ीकरण सहित और बिना सुदृढ़ीकरण के प्रीकास्ट कंक्रीट पाइप",
        sampleWhiteCement: "वास्तु और सजावटी उपयोग के लिए सफेद पोर्टलैंड सीमेंट"
      },
      hinglish: {
        contrast: "Contrast mode",
        navSearch: "Khoj",
        navResults: "Parinam",
        navStatus: "Seva status",
        navDocs: "Developer API dastavez",
        serviceStatusTitle: "Seva status",
        statusOk: "Theek",
        statusOne: "BIS catalogue records loaded hain",
        statusTwo: "Results known standards ya verified external guidance tak limited hain",
        statusThree: "Indian language support ke saath accessible interface",
        loading: "Catalogue search ho raha hai...",
        searching: "Search ho raha hai",
        retrieving: "Relevant BIS standards laaye ja rahe hain...",
        error: "Truti",
        recommendationFailed: "Recommendation nahi mil payi",
        metricOne: "Building materials catalogue",
        metricTwo: "Relevant standard guidance",
        metricThree: "Typical search target",
        resultsTitle: "Relevant Bharatiya Standards",
        ready: "Taiyar",
        verifyBis: "BIS preview par verify karein",
        pencilWarning: "Graphite ya black lead pencils bundled BIS SP 21 building-materials catalogue ke bahar hain. Pencil-specific BIS standards jaise IS 1375:2021 aur IS 2079:2022 verify karein.",
        edibleOilWarning: "Edible oil bundled BIS SP 21 building-materials catalogue ke bahar hai. Oils and fats ke liye IS 548 aur relevant oil-type specification dekhein.",
        footerTitle: "Business Compliance Assistant",
        footerBis: "BIS sevayen",
        footerGov: "Sarkari links",
        footerService: "Seva access",
        footerBisWebsite: "BIS official website",
        footerDownload: "Indian Standards download karein",
        footerManak: "Manak Online",
        footerCertification: "Product certification",
        footerIndiaPortal: "National Portal of India",
        footerConsumerAffairs: "Department of Consumer Affairs",
        footerConsumerHelpline: "National Consumer Helpline",
        footerUx4g: "UX4G Design System",
        footerStatus: "Seva status",
        footerHealth: "Health endpoint",
        footerApi: "Developer API",
        footerSearch: "Standards search",
        footerNote: "Ye application standards discovery aid hai. Ye official BIS service nahi hai.",
        footerUpdated: "Last reviewed: 02 May 2026",
        footerTop: "Search par wapas",
        sampleAggregates: "structural concrete ke liye natural sources se coarse aur fine aggregates",
        samplePipes: "water mains ke liye reinforcement ke saath aur bina reinforcement ke precast concrete pipes",
        sampleWhiteCement: "architectural aur decorative use ke liye white Portland cement"
      },
      bn: {
        contrast: "কনট্রাস্ট",
        brandSubtitle: "উৎপাদন প্রতিষ্ঠানের জন্য ভারতীয় মান ব্যুরো মান অনুসন্ধান",
        navSearch: "অনুসন্ধান",
        navResults: "ফলাফল",
        navStatus: "পরিষেবার অবস্থা",
        navDocs: "ডেভেলপার API",
        eyebrow: "BIS ডিজিটাল পরিষেবা",
        heroCopy: "পণ্যের নাম, উপাদান, গ্রেড এবং ব্যবহার লিখুন। এই পরিষেবা উপলব্ধ BIS ক্যাটালগ রেকর্ড অনুসন্ধান করে প্রাসঙ্গিক ভারতীয় মানের নির্দেশনা দেয়।",
        serviceStatusTitle: "পরিষেবার অবস্থা",
        serviceStatusCopy: "মান অনুসন্ধানের জন্য ডিজিটাল সহায়তা। চূড়ান্ত কমপ্লায়েন্স সিদ্ধান্ত সরকারি BIS নথি দিয়ে যাচাই করুন।",
        statusOk: "ঠিক",
        statusOne: "BIS ক্যাটালগ রেকর্ড লোড হয়েছে",
        statusTwo: "ফলাফল পরিচিত মান বা যাচাইকৃত বাহ্যিক নির্দেশনায় সীমিত",
        statusThree: "ভারতীয় ভাষা সহায়তাসহ প্রবেশযোগ্য ইন্টারফেস",
        loading: "ক্যাটালগ অনুসন্ধান চলছে...",
        searching: "অনুসন্ধান চলছে",
        retrieving: "প্রাসঙ্গিক BIS মান আনা হচ্ছে...",
        error: "ত্রুটি",
        recommendationFailed: "সুপারিশ পাওয়া যায়নি",
        metricOne: "নির্মাণ সামগ্রী ক্যাটালগ",
        metricTwo: "প্রাসঙ্গিক মান নির্দেশনা",
        metricThree: "সাধারণ অনুসন্ধান লক্ষ্য",
        ready: "প্রস্তুত",
        noResults: "কোনো মিল থাকা মান পাওয়া যায়নি।",
        outsideCatalog: "বর্তমান SP 21 ক্যাটালগের বাইরে",
        verifyBis: "BIS প্রিভিউতে যাচাই করুন",
        pencilWarning: "গ্রাফাইট বা ব্ল্যাক লেড পেন্সিল bundled BIS SP 21 নির্মাণ-সামগ্রী ক্যাটালগের বাইরে। IS 1375:2021 এবং IS 2079:2022-এর মতো পেন্সিল-নির্দিষ্ট BIS মান যাচাই করুন।",
        edibleOilWarning: "ভোজ্য তেল bundled BIS SP 21 নির্মাণ-সামগ্রী ক্যাটালগের বাইরে। তেল ও চর্বির জন্য IS 548 এবং সংশ্লিষ্ট তেল-প্রকার স্পেসিফিকেশন দেখুন।",
        scopeTitle: "ক্যাটালগের সীমা",
        scopeCopy: "পরিষেবাটি উপলব্ধ BIS ক্যাটালগ ডেটা ব্যবহার করে। পণ্য বর্তমান ক্যাটালগের বাইরে হলে এটি অপ্রাসঙ্গিক মান দেখানোর বদলে যাচাই নির্দেশনা দেখায়।",
        workflowTitle: "এই পরিষেবা কীভাবে ব্যবহার করবেন",
        stepOne: "পণ্য, উপাদান, গ্রেড এবং ব্যবহার লিখুন।",
        stepTwo: "প্রাসঙ্গিক IS কোড এবং ক্যাটালগ কারণ পর্যালোচনা করুন।",
        stepThree: "সার্টিফিকেশন পদক্ষেপের আগে সরকারি BIS পোর্টালে প্রয়োজনীয়তা যাচাই করুন।",
        advisoryTitle: "গুরুত্বপূর্ণ পরামর্শ",
        advisoryCopy: "এই ডিজিটাল পরিষেবা মান অনুসন্ধানে সহায়তা করে। এটি সরকারি BIS মান, সার্টিফিকেশন নিয়ম, পরীক্ষা প্রয়োজনীয়তা বা বিশেষজ্ঞ মূল্যায়নের বিকল্প নয়।",
        footerTitle: "Business Compliance Assistant",
        footerBis: "BIS পরিষেবা",
        footerGov: "সরকারি লিঙ্ক",
        footerService: "পরিষেবা প্রবেশ",
        footerBisWebsite: "BIS সরকারি ওয়েবসাইট",
        footerDownload: "ভারতীয় মান ডাউনলোড করুন",
        footerManak: "মানক অনলাইন",
        footerCertification: "পণ্য সার্টিফিকেশন",
        footerIndiaPortal: "ভারতের জাতীয় পোর্টাল",
        footerConsumerAffairs: "ভোক্তা বিষয়ক বিভাগ",
        footerConsumerHelpline: "জাতীয় ভোক্তা হেল্পলাইন",
        footerUx4g: "UX4G ডিজাইন সিস্টেম",
        footerStatus: "পরিষেবার অবস্থা",
        footerHealth: "হেলথ এন্ডপয়েন্ট",
        footerApi: "ডেভেলপার API",
        footerSearch: "মান অনুসন্ধান",
        footer: "BIS মান অনুসন্ধানের জন্য ডিজিটাল সহায়তা। চূড়ান্ত কমপ্লায়েন্স সিদ্ধান্ত সরকারি BIS নথি এবং সক্ষম কর্তৃপক্ষের সঙ্গে যাচাই করুন।",
        footerNote: "এই অ্যাপ্লিকেশনটি মান অনুসন্ধানের সহায়ক। এটি সরকারি BIS পরিষেবা নয়।",
        footerUpdated: "শেষ পর্যালোচনা: 02 মে 2026",
        footerTop: "অনুসন্ধানে ফিরুন",
        sampleDefault: "আমরা সাধারণ ভবন নির্মাণের জন্য 33 গ্রেড অর্ডিনারি পোর্টল্যান্ড সিমেন্ট উৎপাদন করি। কোন ভারতীয় মান প্রযোজ্য?",
        sampleAggregatesLabel: "এগ্রিগেট",
        sampleAggregates: "স্ট্রাকচারাল কংক্রিটের জন্য প্রাকৃতিক উৎসের মোটা ও সূক্ষ্ম এগ্রিগেট",
        samplePipesLabel: "কংক্রিট পাইপ",
        samplePipes: "জল মেইনের জন্য রিইনফোর্সমেন্টসহ এবং ছাড়া প্রিকাস্ট কংক্রিট পাইপ",
        sampleWhiteCementLabel: "সাদা সিমেন্ট",
        sampleWhiteCement: "স্থাপত্য ও সজ্জার কাজে সাদা পোর্টল্যান্ড সিমেন্ট"
      },
      ta: {
        contrast: "மாறுபாடு",
        brandSubtitle: "உற்பத்தி நிறுவனங்களுக்கான இந்திய தரநிலைகள் பணியக தர தேடல்",
        navSearch: "தேடல்",
        navResults: "முடிவுகள்",
        navStatus: "சேவை நிலை",
        navDocs: "டெவலப்பர் API",
        eyebrow: "BIS டிஜிட்டல் சேவை",
        heroCopy: "தயாரிப்பு பெயர், பொருள், தரம் மற்றும் பயன்பாட்டை உள்ளிடவும். இந்த சேவை கிடைக்கும் BIS பட்டியல் பதிவுகளைத் தேடி தொடர்புடைய இந்திய தரங்களை வழிகாட்டலாக வழங்குகிறது.",
        serviceStatusTitle: "சேவை நிலை",
        serviceStatusCopy: "தர தேடலுக்கான டிஜிட்டல் உதவி. இறுதி இணக்கத் தீர்மானங்களை அதிகாரப்பூர்வ BIS ஆவணங்களுடன் சரிபார்க்கவும்.",
        statusOk: "சரி",
        statusOne: "BIS பட்டியல் பதிவுகள் ஏற்றப்பட்டன",
        statusTwo: "முடிவுகள் அறியப்பட்ட தரங்கள் அல்லது சரிபார்க்கப்பட்ட வெளிப்புற வழிகாட்டல்களுக்கு மட்டுப்படுத்தப்பட்டவை",
        statusThree: "இந்திய மொழி ஆதரவுடன் அணுகக்கூடிய இடைமுகம்",
        loading: "பட்டியல் தேடப்படுகிறது...",
        searching: "தேடப்படுகிறது",
        retrieving: "தொடர்புடைய BIS தரங்கள் பெறப்படுகின்றன...",
        error: "பிழை",
        recommendationFailed: "பரிந்துரையை பெற முடியவில்லை",
        metricOne: "கட்டுமானப் பொருள் பட்டியல்",
        metricTwo: "தொடர்புடைய தர வழிகாட்டல்",
        metricThree: "வழக்கமான தேடல் இலக்கு",
        ready: "தயார்",
        noResults: "பொருந்தும் தரங்கள் எதுவும் கிடைக்கவில்லை.",
        outsideCatalog: "தற்போதைய SP 21 பட்டியலுக்கு வெளியே",
        verifyBis: "BIS முன்னோட்டத்தில் சரிபார்க்கவும்",
        pencilWarning: "கிராஃபைட் அல்லது கருப்பு லீட் பென்சில்கள் bundled BIS SP 21 கட்டுமானப் பொருள் பட்டியலில் இல்லை. IS 1375:2021 மற்றும் IS 2079:2022 போன்ற பென்சில் சார்ந்த BIS தரங்களை சரிபார்க்கவும்.",
        edibleOilWarning: "உணவு எண்ணெய் bundled BIS SP 21 கட்டுமானப் பொருள் பட்டியலில் இல்லை. எண்ணெய்கள் மற்றும் கொழுப்புகளுக்கு IS 548 மற்றும் தொடர்புடைய எண்ணெய் வகை விவரக்குறிப்பைப் பார்க்கவும்.",
        scopeTitle: "பட்டியல் வரம்பு",
        scopeCopy: "இந்த சேவை கிடைக்கும் BIS பட்டியல் தரவைப் பயன்படுத்துகிறது. தயாரிப்பு தற்போதைய பட்டியலுக்கு வெளியே இருந்தால் தொடர்பற்ற தரங்களை காட்டாமல் சரிபார்ப்பு அறிவுரையை காட்டும்.",
        workflowTitle: "இந்த சேவையைப் பயன்படுத்துவது எப்படி",
        stepOne: "தயாரிப்பு, பொருள், தரம் மற்றும் பயன்பாட்டை உள்ளிடவும்.",
        stepTwo: "தொடர்புடைய IS குறியீடுகள் மற்றும் பட்டியல் காரணத்தைப் பரிசீலிக்கவும்.",
        stepThree: "சான்றிதழ் நடவடிக்கைக்கு முன் அதிகாரப்பூர்வ BIS தளத்தில் தேவைகளை சரிபார்க்கவும்.",
        advisoryTitle: "முக்கிய அறிவுரை",
        advisoryCopy: "இந்த டிஜிட்டல் சேவை தர தேடலுக்கு உதவுகிறது. இது அதிகாரப்பூர்வ BIS தரங்கள், சான்றிதழ் விதிகள், சோதனை தேவைகள் அல்லது நிபுணர் மதிப்பீட்டிற்கு மாற்றாகாது.",
        footerTitle: "Business Compliance Assistant",
        footerBis: "BIS சேவைகள்",
        footerGov: "அரசு இணைப்புகள்",
        footerService: "சேவை அணுகல்",
        footerBisWebsite: "BIS அதிகாரப்பூர்வ இணையதளம்",
        footerDownload: "இந்திய தரங்களைப் பதிவிறக்கவும்",
        footerManak: "மானக் ஆன்லைன்",
        footerCertification: "தயாரிப்பு சான்றிதழ்",
        footerIndiaPortal: "இந்திய தேசிய தளம்",
        footerConsumerAffairs: "நுகர்வோர் விவகாரத் துறை",
        footerConsumerHelpline: "தேசிய நுகர்வோர் உதவி எண்",
        footerUx4g: "UX4G வடிவமைப்பு அமைப்பு",
        footerStatus: "சேவை நிலை",
        footerHealth: "சுகாதார எண்ட்பாயிண்ட்",
        footerApi: "டெவலப்பர் API",
        footerSearch: "தரங்களைத் தேடல்",
        footer: "BIS தர தேடலுக்கான டிஜிட்டல் உதவி. இறுதி இணக்கத் தீர்மானங்களை அதிகாரப்பூர்வ BIS ஆவணங்கள் மற்றும் தகுதியான அதிகாரிகளுடன் சரிபார்க்கவும்.",
        footerNote: "இந்த பயன்பாடு தர தேடலுக்கான உதவி. இது அதிகாரப்பூர்வ BIS சேவை அல்ல.",
        footerUpdated: "கடைசி மதிப்பாய்வு: 02 மே 2026",
        footerTop: "தேடலுக்கு திரும்பு",
        sampleDefault: "பொது கட்டிட கட்டுமானத்திற்கு 33 தர Ordinary Portland Cement தயாரிக்கிறோம். எந்த இந்திய தரம் பொருந்தும்?",
        sampleAggregatesLabel: "அக்ரிகேட்கள்",
        sampleAggregates: "கட்டமைப்பு கான்கிரீட்டிற்கு இயற்கை மூலங்களில் இருந்து பெறப்படும் பெரிய மற்றும் சிறிய அக்ரிகேட்கள்",
        samplePipesLabel: "கான்கிரீட் குழாய்கள்",
        samplePipes: "நீர் மெயின்களுக்கு உறுதிப்படுத்தலுடன் மற்றும் இன்றி முன் தயாரிக்கப்பட்ட கான்கிரீட் குழாய்கள்",
        sampleWhiteCementLabel: "வெள்ளை சிமெண்டு",
        sampleWhiteCement: "கட்டிடக்கலை மற்றும் அலங்கார பயன்பாட்டிற்கான வெள்ளை போர்ட்லண்ட் சிமெண்டு"
      },
      te: {
        contrast: "కాంట్రాస్ట్",
        brandSubtitle: "తయారీ సంస్థల కోసం భారతీయ ప్రమాణాల బ్యూరో ప్రమాణాల శోధన",
        navSearch: "శోధన",
        navResults: "ఫలితాలు",
        navStatus: "సేవ స్థితి",
        navDocs: "డెవలపర్ API",
        eyebrow: "BIS డిజిటల్ సేవ",
        heroCopy: "ఉత్పత్తి పేరు, పదార్థం, గ్రేడ్ మరియు వినియోగాన్ని నమోదు చేయండి. ఈ సేవ అందుబాటులో ఉన్న BIS కాటలాగ్ రికార్డులను శోధించి సంబంధిత భారతీయ ప్రమాణాల మార్గదర్శకాన్ని అందిస్తుంది.",
        serviceStatusTitle: "సేవ స్థితి",
        serviceStatusCopy: "ప్రమాణాల శోధనకు డిజిటల్ సహాయం. తుది అనుసరణ నిర్ణయాలను అధికారిక BIS పత్రాలతో ధృవీకరించండి.",
        statusOk: "సరే",
        statusOne: "BIS కాటలాగ్ రికార్డులు లోడ్ అయ్యాయి",
        statusTwo: "ఫలితాలు తెలిసిన ప్రమాణాలు లేదా ధృవీకరించిన బాహ్య మార్గదర్శకానికే పరిమితం",
        statusThree: "భారతీయ భాషా మద్దతుతో సులభంగా ఉపయోగించగల ఇంటర్‌ఫేస్",
        loading: "కాటలాగ్ శోధిస్తోంది...",
        searching: "శోధిస్తోంది",
        retrieving: "సంబంధిత BIS ప్రమాణాలు పొందుతోంది...",
        error: "లోపం",
        recommendationFailed: "సిఫార్సు పొందలేకపోయింది",
        metricOne: "నిర్మాణ సామగ్రి కాటలాగ్",
        metricTwo: "సంబంధిత ప్రమాణ మార్గదర్శకం",
        metricThree: "సాధారణ శోధన లక్ష్యం",
        ready: "సిద్ధం",
        noResults: "సరిపోలే ప్రమాణాలు లభించలేదు.",
        outsideCatalog: "ప్రస్తుత SP 21 కాటలాగ్ వెలుపల",
        verifyBis: "BIS ప్రీవ్యూలో ధృవీకరించండి",
        pencilWarning: "గ్రాఫైట్ లేదా బ్లాక్ లీడ్ పెన్సిల్స్ bundled BIS SP 21 నిర్మాణ సామగ్రి కాటలాగ్‌లో లేవు. IS 1375:2021 మరియు IS 2079:2022 వంటి పెన్సిల్‌కు సంబంధించిన BIS ప్రమాణాలను ధృవీకరించండి.",
        edibleOilWarning: "తినే నూనె bundled BIS SP 21 నిర్మాణ సామగ్రి కాటలాగ్‌లో లేదు. నూనెలు మరియు కొవ్వులకు IS 548 మరియు సంబంధిత నూనె రకం స్పెసిఫికేషన్ చూడండి.",
        scopeTitle: "కాటలాగ్ పరిధి",
        scopeCopy: "ఈ సేవ అందుబాటులో ఉన్న BIS కాటలాగ్ డేటాను ఉపయోగిస్తుంది. ఉత్పత్తి ప్రస్తుత కాటలాగ్ వెలుపల ఉంటే సంబంధం లేని ప్రమాణాల బదులు ధృవీకరణ సూచన చూపుతుంది.",
        workflowTitle: "ఈ సేవను ఎలా ఉపయోగించాలి",
        stepOne: "ఉత్పత్తి, పదార్థం, గ్రేడ్ మరియు వినియోగాన్ని నమోదు చేయండి.",
        stepTwo: "సంబంధిత IS కోడ్లు మరియు కాటలాగ్ కారణాన్ని సమీక్షించండి.",
        stepThree: "సర్టిఫికేషన్ చర్యకు ముందు అధికారిక BIS పోర్టల్‌లో అవసరాలను ధృవీకరించండి.",
        advisoryTitle: "ముఖ్య సూచన",
        advisoryCopy: "ఈ డిజిటల్ సేవ ప్రమాణాల శోధనకు సహాయం చేస్తుంది. ఇది అధికారిక BIS ప్రమాణాలు, సర్టిఫికేషన్ నియమాలు, పరీక్ష అవసరాలు లేదా నిపుణుల అంచనాకు ప్రత్యామ్నాయం కాదు.",
        footerTitle: "Business Compliance Assistant",
        footerBis: "BIS సేవలు",
        footerGov: "ప్రభుత్వ లింకులు",
        footerService: "సేవ ప్రవేశం",
        footerBisWebsite: "BIS అధికారిక వెబ్‌సైట్",
        footerDownload: "భారతీయ ప్రమాణాలను డౌన్‌లోడ్ చేయండి",
        footerManak: "మానక్ ఆన్‌లైన్",
        footerCertification: "ఉత్పత్తి ధృవీకరణ",
        footerIndiaPortal: "భారత జాతీయ పోర్టల్",
        footerConsumerAffairs: "వినియోగదారుల వ్యవహారాల శాఖ",
        footerConsumerHelpline: "జాతీయ వినియోగదారుల హెల్ప్‌లైన్",
        footerUx4g: "UX4G డిజైన్ సిస్టమ్",
        footerStatus: "సేవ స్థితి",
        footerHealth: "హెల్త్ ఎండ్పాయింట్",
        footerApi: "డెవలపర్ API",
        footerSearch: "ప్రమాణాల శోధన",
        footer: "BIS ప్రమాణాల శోధనకు డిజిటల్ సహాయం. తుది అనుసరణ నిర్ణయాలను అధికారిక BIS పత్రాలు మరియు సమర్థ అధికారులతో ధృవీకరించండి.",
        footerNote: "ఈ అప్లికేషన్ ప్రమాణాల శోధనకు సహాయకం. ఇది అధికారిక BIS సేవ కాదు.",
        footerUpdated: "చివరి సమీక్ష: 02 మే 2026",
        footerTop: "శోధనకు తిరిగి వెళ్లండి",
        sampleDefault: "మేము సాధారణ భవన నిర్మాణానికి 33 గ్రేడ్ Ordinary Portland Cement తయారు చేస్తున్నాము. ఏ భారతీయ ప్రమాణం వర్తిస్తుంది?",
        sampleAggregatesLabel: "అగ్రిగేట్లు",
        sampleAggregates: "నిర్మాణ కాంక్రీటుకు సహజ మూలాల నుంచి మోటు మరియు సన్నని అగ్రిగేట్లు",
        samplePipesLabel: "కాంక్రీట్ పైపులు",
        samplePipes: "నీటి మెయిన్ల కోసం బలపరచిన మరియు బలపరచని ప్రీకాస్ట్ కాంక్రీట్ పైపులు",
        sampleWhiteCementLabel: "వైట్ సిమెంట్",
        sampleWhiteCement: "ఆర్కిటెక్చరల్ మరియు అలంకార వినియోగానికి వైట్ పోర్ట్‌ల్యాండ్ సిమెంట్"
      },
      mr: {
        contrast: "कॉन्ट्रास्ट",
        brandSubtitle: "उत्पादन उद्योगांसाठी भारतीय मानक ब्युरो मानक शोध",
        navSearch: "शोध",
        navResults: "निकाल",
        navStatus: "सेवा स्थिती",
        navDocs: "डेव्हलपर API",
        eyebrow: "BIS डिजिटल सेवा",
        heroCopy: "उत्पादनाचे नाव, साहित्य, ग्रेड आणि वापर नोंदवा. ही सेवा उपलब्ध BIS कॅटलॉग नोंदी शोधून संबंधित भारतीय मानकांचे मार्गदर्शन देते.",
        serviceStatusTitle: "सेवा स्थिती",
        serviceStatusCopy: "मानक शोधासाठी डिजिटल सहाय्य. अंतिम अनुपालन निर्णय अधिकृत BIS दस्तऐवजांद्वारे पडताळा.",
        statusOk: "ठीक",
        statusOne: "BIS कॅटलॉग नोंदी लोड झाल्या",
        statusTwo: "निकाल ज्ञात मानके किंवा पडताळलेल्या बाह्य मार्गदर्शनापुरते मर्यादित",
        statusThree: "भारतीय भाषा समर्थनासह सुलभ इंटरफेस",
        loading: "कॅटलॉग शोधला जात आहे...",
        searching: "शोध सुरू आहे",
        retrieving: "संबंधित BIS मानके मिळवत आहे...",
        error: "त्रुटी",
        recommendationFailed: "शिफारस मिळू शकली नाही",
        metricOne: "बांधकाम साहित्य कॅटलॉग",
        metricTwo: "संबंधित मानक मार्गदर्शन",
        metricThree: "सामान्य शोध लक्ष्य",
        ready: "तयार",
        noResults: "जुळणारी मानके मिळाली नाहीत.",
        outsideCatalog: "सध्याच्या SP 21 कॅटलॉगच्या बाहेर",
        verifyBis: "BIS पूर्वावलोकनात पडताळा",
        pencilWarning: "ग्रेफाइट किंवा ब्लॅक लीड पेन्सिली bundled BIS SP 21 बांधकाम-साहित्य कॅटलॉगच्या बाहेर आहेत. IS 1375:2021 आणि IS 2079:2022 सारखी पेन्सिल-विशिष्ट BIS मानके पडताळा.",
        edibleOilWarning: "खाद्य तेल bundled BIS SP 21 बांधकाम-साहित्य कॅटलॉगच्या बाहेर आहे. तेल आणि चरबींसाठी IS 548 आणि संबंधित तेल-प्रकार तपशील पहा.",
        scopeTitle: "कॅटलॉगची व्याप्ती",
        scopeCopy: "ही सेवा उपलब्ध BIS कॅटलॉग डेटा वापरते. उत्पादन सध्याच्या कॅटलॉगच्या बाहेर असल्यास असंबंधित मानके न दाखवता पडताळणी सूचना दाखवते.",
        workflowTitle: "ही सेवा कशी वापरावी",
        stepOne: "उत्पादन, साहित्य, ग्रेड आणि वापर नोंदवा.",
        stepTwo: "संबंधित IS कोड आणि कॅटलॉग कारण तपासा.",
        stepThree: "प्रमाणन कृतीपूर्वी अधिकृत BIS पोर्टलवर आवश्यकता पडताळा.",
        advisoryTitle: "महत्त्वाची सूचना",
        advisoryCopy: "ही डिजिटल सेवा मानक शोधासाठी मदत करते. ती अधिकृत BIS मानके, प्रमाणन नियम, चाचणी आवश्यकता किंवा तज्ज्ञ मूल्यांकनाची जागा घेत नाही.",
        footerTitle: "Business Compliance Assistant",
        footerBis: "BIS सेवा",
        footerGov: "शासकीय दुवे",
        footerService: "सेवा प्रवेश",
        footerBisWebsite: "BIS अधिकृत संकेतस्थळ",
        footerDownload: "भारतीय मानके डाउनलोड करा",
        footerManak: "मानक ऑनलाइन",
        footerCertification: "उत्पादन प्रमाणन",
        footerIndiaPortal: "भारताचे राष्ट्रीय पोर्टल",
        footerConsumerAffairs: "ग्राहक व्यवहार विभाग",
        footerConsumerHelpline: "राष्ट्रीय ग्राहक हेल्पलाइन",
        footerUx4g: "UX4G डिझाइन प्रणाली",
        footerStatus: "सेवा स्थिती",
        footerHealth: "हेल्थ एंडपॉइंट",
        footerApi: "डेव्हलपर API",
        footerSearch: "मानके शोधा",
        footer: "BIS मानक शोधासाठी डिजिटल सहाय्य. अंतिम अनुपालन निर्णय अधिकृत BIS दस्तऐवज आणि सक्षम प्राधिकरणांशी पडताळा.",
        footerNote: "हे अॅप्लिकेशन मानक शोधण्यासाठी सहाय्यक आहे. ही अधिकृत BIS सेवा नाही.",
        footerUpdated: "शेवटची समीक्षा: 02 मे 2026",
        footerTop: "शोधाकडे परत जा",
        sampleDefault: "आम्ही सामान्य इमारत बांधकामासाठी 33 ग्रेड Ordinary Portland Cement तयार करतो. कोणते भारतीय मानक लागू आहे?",
        sampleAggregatesLabel: "एग्रीगेट",
        sampleAggregates: "स्ट्रक्चरल कंक्रीटसाठी नैसर्गिक स्रोतांमधील जाड आणि बारीक एग्रीगेट",
        samplePipesLabel: "कंक्रीट पाइप",
        samplePipes: "पाणी मुख्य वाहिन्यांसाठी मजबुतीकरणासह आणि त्याशिवाय प्रीकास्ट कंक्रीट पाइप",
        sampleWhiteCementLabel: "पांढरे सिमेंट",
        sampleWhiteCement: "वास्तुशिल्प आणि सजावटीच्या वापरासाठी पांढरे पोर्टलँड सिमेंट"
      },
      gu: {
        contrast: "કોન્ટ્રાસ્ટ",
        brandSubtitle: "ઉત્પાદન એકમો માટે ભારતીય ધોરણ બ્યુરો ધોરણ શોધ",
        navSearch: "શોધ",
        navResults: "પરિણામો",
        navStatus: "સેવા સ્થિતિ",
        navDocs: "ડેવલપર API",
        eyebrow: "BIS ડિજિટલ સેવા",
        heroCopy: "ઉત્પાદનનું નામ, સામગ્રી, ગ્રેડ અને ઉપયોગ દાખલ કરો. આ સેવા ઉપલબ્ધ BIS કેટલોગ રેકોર્ડ્સ શોધીને સંબંધિત ભારતીય ધોરણ માર્ગદર્શન આપે છે.",
        serviceStatusTitle: "સેવા સ્થિતિ",
        serviceStatusCopy: "ધોરણ શોધ માટે ડિજિટલ સહાય. અંતિમ અનુપાલન નિર્ણયો સત્તાવાર BIS દસ્તાવેજોથી ચકાસો.",
        statusOk: "બરાબર",
        statusOne: "BIS કેટલોગ રેકોર્ડ્સ લોડ થયા",
        statusTwo: "પરિણામો જાણીતા ધોરણો અથવા ચકાસેલા બાહ્ય માર્ગદર્શન સુધી મર્યાદિત",
        statusThree: "ભારતીય ભાષા સપોર્ટ સાથે સુલભ ઇન્ટરફેસ",
        loading: "કેટલોગ શોધાઈ રહ્યો છે...",
        searching: "શોધાઈ રહ્યું છે",
        retrieving: "સંબંધિત BIS ધોરણો મેળવાઈ રહ્યા છે...",
        error: "ભૂલ",
        recommendationFailed: "ભલામણ મળી શકી નથી",
        metricOne: "બાંધકામ સામગ્રી કેટલોગ",
        metricTwo: "સંબંધિત ધોરણ માર્ગદર્શન",
        metricThree: "સામાન્ય શોધ લક્ષ્ય",
        ready: "તૈયાર",
        noResults: "મેળ ખાતા ધોરણો મળ્યા નથી.",
        outsideCatalog: "વર્તમાન SP 21 કેટલોગની બહાર",
        verifyBis: "BIS પ્રિવ્યૂમાં ચકાસો",
        pencilWarning: "ગ્રેફાઇટ અથવા બ્લેક લીડ પેન્સિલ bundled BIS SP 21 બાંધકામ-સામગ્રી કેટલોગની બહાર છે. IS 1375:2021 અને IS 2079:2022 જેવા પેન્સિલ-વિશિષ્ટ BIS ધોરણો ચકાસો.",
        edibleOilWarning: "ખાદ્ય તેલ bundled BIS SP 21 બાંધકામ-સામગ્રી કેટલોગની બહાર છે. તેલ અને ચરબી માટે IS 548 અને સંબંધિત તેલ-પ્રકાર સ્પેસિફિકેશન જુઓ.",
        scopeTitle: "કેટલોગ વ્યાપ",
        scopeCopy: "આ સેવા ઉપલબ્ધ BIS કેટલોગ ડેટાનો ઉપયોગ કરે છે. ઉત્પાદન વર્તમાન કેટલોગની બહાર હોય તો તે અસંબંધિત ધોરણો બદલે ચકાસણી સલાહ બતાવે છે.",
        workflowTitle: "આ સેવાનો ઉપયોગ કેવી રીતે કરવો",
        stepOne: "ઉત્પાદન, સામગ્રી, ગ્રેડ અને ઉપયોગ દાખલ કરો.",
        stepTwo: "સંબંધિત IS કોડ અને કેટલોગ કારણની સમીક્ષા કરો.",
        stepThree: "પ્રમાણન કાર્યવાહી પહેલાં સત્તાવાર BIS પોર્ટલ પર આવશ્યકતાઓ ચકાસો.",
        advisoryTitle: "મહત્વપૂર્ણ સલાહ",
        advisoryCopy: "આ ડિજિટલ સેવા ધોરણ શોધમાં મદદ કરે છે. તે સત્તાવાર BIS ધોરણો, પ્રમાણન નિયમો, પરીક્ષણ આવશ્યકતાઓ અથવા નિષ્ણાત મૂલ્યાંકનનો વિકલ્પ નથી.",
        footerTitle: "Business Compliance Assistant",
        footerBis: "BIS સેવાઓ",
        footerGov: "સરકારી લિંક્સ",
        footerService: "સેવા ઍક્સેસ",
        footerBisWebsite: "BIS સત્તાવાર વેબસાઇટ",
        footerDownload: "ભારતીય ધોરણો ડાઉનલોડ કરો",
        footerManak: "માનક ઑનલાઇન",
        footerCertification: "ઉત્પાદન પ્રમાણન",
        footerIndiaPortal: "ભારતનું રાષ્ટ્રીય પોર્ટલ",
        footerConsumerAffairs: "ગ્રાહક બાબતો વિભાગ",
        footerConsumerHelpline: "રાષ્ટ્રીય ગ્રાહક હેલ્પલાઇન",
        footerUx4g: "UX4G ડિઝાઇન સિસ્ટમ",
        footerStatus: "સેવા સ્થિતિ",
        footerHealth: "હેલ્થ એન્ડપૉઇન્ટ",
        footerApi: "ડેવલપર API",
        footerSearch: "ધોરણો શોધો",
        footer: "BIS ધોરણ શોધ માટે ડિજિટલ સહાય. અંતિમ અનુપાલન નિર્ણયો સત્તાવાર BIS દસ્તાવેજો અને સક્ષમ સત્તાધિકારીઓ સાથે ચકાસો.",
        footerNote: "આ એપ્લિકેશન ધોરણ શોધ માટે સહાયક છે. આ સત્તાવાર BIS સેવા નથી.",
        footerUpdated: "છેલ્લી સમીક્ષા: 02 મે 2026",
        footerTop: "શોધ પર પાછા જાઓ",
        sampleDefault: "અમે સામાન્ય બિલ્ડિંગ બાંધકામ માટે 33 Grade Ordinary Portland Cement બનાવીએ છીએ. કયું ભારતીય ધોરણ લાગુ પડે છે?",
        sampleAggregatesLabel: "એગ્રીગેટ",
        sampleAggregates: "રચનાત્મક કૉંક્રિટ માટે કુદરતી સ્ત્રોતોમાંથી મોટા અને નાના એગ્રીગેટ",
        samplePipesLabel: "કૉંક્રિટ પાઇપ",
        samplePipes: "વૉટર મેઇન્સ માટે મજબૂતીકરણ સાથે અને વગર પ્રીકાસ્ટ કૉંક્રિટ પાઇપ",
        sampleWhiteCementLabel: "વ્હાઇટ સિમેન્ટ",
        sampleWhiteCement: "આર્કિટેક્ચરલ અને ડેકોરેટિવ ઉપયોગ માટે વ્હાઇટ પોર્ટલેન્ડ સિમેન્ટ"
      },
      kn: {
        contrast: "ಕಾಂಟ್ರಾಸ್ಟ್",
        brandSubtitle: "ತಯಾರಿಕಾ ಉದ್ಯಮಗಳಿಗೆ ಭಾರತೀಯ ಮಾನದಂಡಗಳ ಬ್ಯೂರೋ ಮಾನದಂಡ ಹುಡುಕಾಟ",
        navSearch: "ಹುಡುಕು",
        navResults: "ಫಲಿತಾಂಶಗಳು",
        navStatus: "ಸೇವೆಯ ಸ್ಥಿತಿ",
        navDocs: "ಡೆವಲಪರ್ API",
        eyebrow: "BIS ಡಿಜಿಟಲ್ ಸೇವೆ",
        heroCopy: "ಉತ್ಪನ್ನದ ಹೆಸರು, ವಸ್ತು, ಗ್ರೇಡ್ ಮತ್ತು ಬಳಕೆಯನ್ನು ನಮೂದಿಸಿ. ಈ ಸೇವೆ ಲಭ್ಯವಿರುವ BIS ಕ್ಯಾಟಲಾಗ್ ದಾಖಲೆಗಳನ್ನು ಹುಡುಕಿ ಸಂಬಂಧಿತ ಭಾರತೀಯ ಮಾನದಂಡಗಳ ಮಾರ್ಗದರ್ಶನ ನೀಡುತ್ತದೆ.",
        serviceStatusTitle: "ಸೇವೆಯ ಸ್ಥಿತಿ",
        serviceStatusCopy: "ಮಾನದಂಡ ಹುಡುಕಾಟಕ್ಕೆ ಡಿಜಿಟಲ್ ಸಹಾಯ. ಅಂತಿಮ ಅನುಸರಣಾ ನಿರ್ಧಾರಗಳನ್ನು ಅಧಿಕೃತ BIS ದಾಖಲೆಗಳೊಂದಿಗೆ ಪರಿಶೀಲಿಸಿ.",
        statusOk: "ಸರಿ",
        statusOne: "BIS ಕ್ಯಾಟಲಾಗ್ ದಾಖಲೆಗಳು ಲೋಡ್ ಆಗಿವೆ",
        statusTwo: "ಫಲಿತಾಂಶಗಳು ತಿಳಿದಿರುವ ಮಾನದಂಡಗಳು ಅಥವಾ ಪರಿಶೀಲಿತ ಬಾಹ್ಯ ಮಾರ್ಗದರ್ಶನಕ್ಕೆ ಮಾತ್ರ ಸೀಮಿತ",
        statusThree: "ಭಾರತೀಯ ಭಾಷಾ ಬೆಂಬಲದೊಂದಿಗೆ ಸುಲಭ ಪ್ರವೇಶದ ಇಂಟರ್‌ಫೇಸ್",
        loading: "ಕ್ಯಾಟಲಾಗ್ ಹುಡುಕಲಾಗುತ್ತಿದೆ...",
        searching: "ಹುಡುಕಲಾಗುತ್ತಿದೆ",
        retrieving: "ಸಂಬಂಧಿತ BIS ಮಾನದಂಡಗಳನ್ನು ಪಡೆಯಲಾಗುತ್ತಿದೆ...",
        error: "ದೋಷ",
        recommendationFailed: "ಶಿಫಾರಸು ಪಡೆಯಲಾಗಲಿಲ್ಲ",
        metricOne: "ನಿರ್ಮಾಣ ಸಾಮಗ್ರಿ ಕ್ಯಾಟಲಾಗ್",
        metricTwo: "ಸಂಬಂಧಿತ ಮಾನದಂಡ ಮಾರ್ಗದರ್ಶನ",
        metricThree: "ಸಾಮಾನ್ಯ ಹುಡುಕಾಟ ಗುರಿ",
        ready: "ಸಿದ್ಧ",
        noResults: "ಹೊಂದುವ ಮಾನದಂಡಗಳು ಲಭ್ಯವಿಲ್ಲ.",
        outsideCatalog: "ಪ್ರಸ್ತುತ SP 21 ಕ್ಯಾಟಲಾಗ್ ಹೊರಗೆ",
        verifyBis: "BIS preview ನಲ್ಲಿ ಪರಿಶೀಲಿಸಿ",
        pencilWarning: "ಗ್ರಾಫೈಟ್ ಅಥವಾ ಬ್ಲಾಕ್ ಲೀಡ್ ಪೆನ್ಸಿಲ್‌ಗಳು bundled BIS SP 21 ನಿರ್ಮಾಣ-ಸಾಮಗ್ರಿ ಕ್ಯಾಟಲಾಗ್‌ನ ಹೊರಗಿವೆ. IS 1375:2021 ಮತ್ತು IS 2079:2022 ಮುಂತಾದ ಪೆನ್ಸಿಲ್-ನಿರ್ದಿಷ್ಟ BIS ಮಾನದಂಡಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
        edibleOilWarning: "ಆಹಾರ ಎಣ್ಣೆ bundled BIS SP 21 ನಿರ್ಮಾಣ-ಸಾಮಗ್ರಿ ಕ್ಯಾಟಲಾಗ್‌ನ ಹೊರಗಿದೆ. ಎಣ್ಣೆ ಮತ್ತು ಕೊಬ್ಬುಗಳಿಗೆ IS 548 ಮತ್ತು ಸಂಬಂಧಿತ ಎಣ್ಣೆ-ಪ್ರಕಾರದ ವಿವರಣೆಯನ್ನು ನೋಡಿ.",
        scopeTitle: "ಕ್ಯಾಟಲಾಗ್ ವ್ಯಾಪ್ತಿ",
        scopeCopy: "ಈ ಸೇವೆ ಲಭ್ಯವಿರುವ BIS ಕ್ಯಾಟಲಾಗ್ ಡೇಟಾವನ್ನು ಬಳಸುತ್ತದೆ. ಉತ್ಪನ್ನ ಪ್ರಸ್ತುತ ಕ್ಯಾಟಲಾಗ್‌ನ ಹೊರಗಿದ್ದರೆ ಸಂಬಂಧವಿಲ್ಲದ ಮಾನದಂಡಗಳ ಬದಲು ಪರಿಶೀಲನಾ ಸಲಹೆಯನ್ನು ತೋರಿಸುತ್ತದೆ.",
        workflowTitle: "ಈ ಸೇವೆಯನ್ನು ಹೇಗೆ ಬಳಸುವುದು",
        stepOne: "ಉತ್ಪನ್ನ, ವಸ್ತು, ಗ್ರೇಡ್ ಮತ್ತು ಬಳಕೆಯನ್ನು ನಮೂದಿಸಿ.",
        stepTwo: "ಸಂಬಂಧಿತ IS ಕೋಡ್‌ಗಳು ಮತ್ತು ಕ್ಯಾಟಲಾಗ್ ಕಾರಣವನ್ನು ಪರಿಶೀಲಿಸಿ.",
        stepThree: "ಪ್ರಮಾಣೀಕರಣ ಕ್ರಮಕ್ಕೂ ಮೊದಲು ಅಧಿಕೃತ BIS ಪೋರ್ಟಲ್‌ನಲ್ಲಿ ಅವಶ್ಯಕತೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
        advisoryTitle: "ಮುಖ್ಯ ಸಲಹೆ",
        advisoryCopy: "ಈ ಡಿಜಿಟಲ್ ಸೇವೆ ಮಾನದಂಡ ಹುಡುಕಾಟಕ್ಕೆ ಸಹಾಯ ಮಾಡುತ್ತದೆ. ಇದು ಅಧಿಕೃತ BIS ಮಾನದಂಡಗಳು, ಪ್ರಮಾಣೀಕರಣ ನಿಯಮಗಳು, ಪರೀಕ್ಷಾ ಅವಶ್ಯಕತೆಗಳು ಅಥವಾ ಪರಿಣಿತರ ಮೌಲ್ಯಮಾಪನಕ್ಕೆ ಪರ್ಯಾಯವಲ್ಲ.",
        footerTitle: "Business Compliance Assistant",
        footerBis: "BIS ಸೇವೆಗಳು",
        footerGov: "ಸರ್ಕಾರಿ ಲಿಂಕ್‌ಗಳು",
        footerService: "ಸೇವೆಯ ಪ್ರವೇಶ",
        footerBisWebsite: "BIS ಅಧಿಕೃತ ವೆಬ್‌ಸೈಟ್",
        footerDownload: "ಭಾರತೀಯ ಮಾನದಂಡಗಳನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        footerManak: "ಮಾನಕ್ ಆನ್‌ಲೈನ್",
        footerCertification: "ಉತ್ಪನ್ನ ಪ್ರಮಾಣೀಕರಣ",
        footerIndiaPortal: "ಭಾರತದ ರಾಷ್ಟ್ರೀಯ ಪೋರ್ಟಲ್",
        footerConsumerAffairs: "ಗ್ರಾಹಕ ವ್ಯವಹಾರಗಳ ಇಲಾಖೆ",
        footerConsumerHelpline: "ರಾಷ್ಟ್ರೀಯ ಗ್ರಾಹಕ ಸಹಾಯವಾಣಿ",
        footerUx4g: "UX4G ವಿನ್ಯಾಸ ವ್ಯವಸ್ಥೆ",
        footerStatus: "ಸೇವೆಯ ಸ್ಥಿತಿ",
        footerHealth: "ಹೆಲ್ತ್ ಎಂಡ್ಪಾಯಿಂಟ್",
        footerApi: "ಡೆವಲಪರ್ API",
        footerSearch: "ಮಾನದಂಡಗಳನ್ನು ಹುಡುಕಿ",
        footer: "BIS ಮಾನದಂಡ ಹುಡುಕಾಟಕ್ಕೆ ಡಿಜಿಟಲ್ ಸಹಾಯ. ಅಂತಿಮ ಅನುಸರಣಾ ನಿರ್ಧಾರಗಳನ್ನು ಅಧಿಕೃತ BIS ದಾಖಲೆಗಳು ಮತ್ತು ಸಮರ್ಥ ಅಧಿಕಾರಿಗಳೊಂದಿಗೆ ಪರಿಶೀಲಿಸಿ.",
        footerNote: "ಈ ಅಪ್ಲಿಕೇಶನ್ ಮಾನದಂಡ ಹುಡುಕಾಟಕ್ಕೆ ಸಹಾಯಕ. ಇದು ಅಧಿಕೃತ BIS ಸೇವೆಯಲ್ಲ.",
        footerUpdated: "ಕೊನೆಯ ಪರಿಶೀಲನೆ: 02 ಮೇ 2026",
        footerTop: "ಹುಡುಕಾಟಕ್ಕೆ ಹಿಂತಿರುಗಿ",
        sampleDefault: "ನಾವು ಸಾಮಾನ್ಯ ಕಟ್ಟಡ ನಿರ್ಮಾಣಕ್ಕೆ 33 Grade Ordinary Portland Cement ತಯಾರಿಸುತ್ತೇವೆ. ಯಾವ ಭಾರತೀಯ ಮಾನದಂಡ ಅನ್ವಯಿಸುತ್ತದೆ?",
        sampleAggregatesLabel: "ಅಗ್ರಿಗೇಟ್‌ಗಳು",
        sampleAggregates: "ಸಂರಚನಾ ಕಾಂಕ್ರೀಟಿಗೆ ನೈಸರ್ಗಿಕ ಮೂಲಗಳಿಂದ ದೊಡ್ಡ ಮತ್ತು ಸಣ್ಣ ಅಗ್ರಿಗೇಟ್‌ಗಳು",
        samplePipesLabel: "ಕಾಂಕ್ರೀಟ್ ಪೈಪ್‌ಗಳು",
        samplePipes: "ನೀರು ಮುಖ್ಯ ಮಾರ್ಗಗಳಿಗೆ ಬಲವರ್ಧನೆಯೊಂದಿಗೆ ಮತ್ತು ಇಲ್ಲದೆ ಪೂರ್ವ ತಯಾರಿತ ಕಾಂಕ್ರೀಟ್ ಪೈಪ್‌ಗಳು",
        sampleWhiteCementLabel: "ಬಿಳಿ ಸಿಮೆಂಟ್",
        sampleWhiteCement: "ವಾಸ್ತುಶಿಲ್ಪ ಮತ್ತು ಅಲಂಕಾರಿಕ ಬಳಕೆಗೆ ಬಿಳಿ ಪೋರ್ಟ್‌ಲ್ಯಾಂಡ್ ಸಿಮೆಂಟ್"
      },
      ml: {
        contrast: "കോൺട്രാസ്റ്റ്",
        brandSubtitle: "നിർമാണ സ്ഥാപനങ്ങൾക്കായുള്ള ഇന്ത്യൻ സ്റ്റാൻഡേർഡ്സ് ബ്യൂറോ മാനദണ്ഡ തിരച്ചിൽ",
        navSearch: "തിരച്ചിൽ",
        navResults: "ഫലങ്ങൾ",
        navStatus: "സേവന നില",
        navDocs: "ഡെവലപ്പർ API",
        eyebrow: "BIS ഡിജിറ്റൽ സേവനം",
        heroCopy: "ഉൽപ്പന്നത്തിന്റെ പേര്, വസ്തു, ഗ്രേഡ്, ഉപയോഗം എന്നിവ നൽകുക. ലഭ്യമായ BIS കാറ്റലോഗ് രേഖകൾ തിരഞ്ഞ് ബന്ധപ്പെട്ട ഇന്ത്യൻ സ്റ്റാൻഡേർഡുകളുടെ മാർഗ്ഗനിർദ്ദേശം ഈ സേവനം നൽകുന്നു.",
        serviceStatusTitle: "സേവന നില",
        serviceStatusCopy: "സ്റ്റാൻഡേർഡ് കണ്ടെത്തലിനുള്ള ഡിജിറ്റൽ സഹായം. അന്തിമ അനുസരണ തീരുമാനങ്ങൾ ഔദ്യോഗിക BIS രേഖകളിലൂടെ പരിശോധിക്കുക.",
        statusOk: "ശരി",
        statusOne: "BIS കാറ്റലോഗ് രേഖകൾ ലോഡ് ചെയ്തു",
        statusTwo: "ഫലങ്ങൾ അറിയപ്പെടുന്ന സ്റ്റാൻഡേർഡുകളോ പരിശോധിച്ച പുറം മാർഗ്ഗനിർദ്ദേശമോ വരെ പരിമിതം",
        statusThree: "ഇന്ത്യൻ ഭാഷാ പിന്തുണയുള്ള ആക്സസിബിൾ ഇന്റർഫേസ്",
        loading: "കാറ്റലോഗ് തിരയുന്നു...",
        searching: "തിരയുന്നു",
        retrieving: "ബന്ധപ്പെട്ട BIS സ്റ്റാൻഡേർഡുകൾ നേടുന്നു...",
        error: "പിശക്",
        recommendationFailed: "ശുപാർശ ലഭിച്ചില്ല",
        metricOne: "നിർമാണ വസ്തു കാറ്റലോഗ്",
        metricTwo: "ബന്ധപ്പെട്ട സ്റ്റാൻഡേർഡ് മാർഗ്ഗനിർദ്ദേശം",
        metricThree: "സാധാരണ തിരച്ചിൽ ലക്ഷ്യം",
        ready: "തയ്യാർ",
        noResults: "പൊരുത്തമുള്ള സ്റ്റാൻഡേർഡുകൾ ലഭിച്ചില്ല.",
        outsideCatalog: "നിലവിലെ SP 21 കാറ്റലോഗിന് പുറത്താണ്",
        verifyBis: "BIS preview ൽ പരിശോധിക്കുക",
        pencilWarning: "ഗ്രാഫൈറ്റ് അല്ലെങ്കിൽ ബ്ലാക്ക് ലീഡ് പെൻസിലുകൾ bundled BIS SP 21 നിർമാണ-വസ്തു കാറ്റലോഗിന് പുറത്താണ്. IS 1375:2021, IS 2079:2022 പോലുള്ള പെൻസിൽ-നിർദ്ദിഷ്ട BIS സ്റ്റാൻഡേർഡുകൾ പരിശോധിക്കുക.",
        edibleOilWarning: "ഭക്ഷ്യ എണ്ണ bundled BIS SP 21 നിർമാണ-വസ്തു കാറ്റലോഗിന് പുറത്താണ്. എണ്ണകൾക്കും കൊഴുപ്പുകൾക്കും IS 548യും ബന്ധപ്പെട്ട എണ്ണ-തരം സ്പെസിഫിക്കേഷനും പരിശോധിക്കുക.",
        scopeTitle: "കാറ്റലോഗ് പരിധി",
        scopeCopy: "ഈ സേവനം ലഭ്യമായ BIS കാറ്റലോഗ് ഡാറ്റ ഉപയോഗിക്കുന്നു. ഉൽപ്പന്നം നിലവിലെ കാറ്റലോഗിന് പുറത്താണെങ്കിൽ ബന്ധമില്ലാത്ത സ്റ്റാൻഡേർഡുകൾ കാണിക്കാതെ സ്ഥിരീകരണ ഉപദേശം കാണിക്കും.",
        workflowTitle: "ഈ സേവനം എങ്ങനെ ഉപയോഗിക്കാം",
        stepOne: "ഉൽപ്പന്നം, വസ്തു, ഗ്രേഡ്, ഉപയോഗം എന്നിവ നൽകുക.",
        stepTwo: "ബന്ധപ്പെട്ട IS കോഡുകളും കാറ്റലോഗ് കാരണവും പരിശോധിക്കുക.",
        stepThree: "സർട്ടിഫിക്കേഷൻ നടപടിക്ക് മുമ്പ് ഔദ്യോഗിക BIS പോർട്ടലിൽ ആവശ്യകതകൾ പരിശോധിക്കുക.",
        advisoryTitle: "പ്രധാന ഉപദേശം",
        advisoryCopy: "ഈ ഡിജിറ്റൽ സേവനം സ്റ്റാൻഡേർഡ് കണ്ടെത്തലിന് സഹായിക്കുന്നു. ഇത് ഔദ്യോഗിക BIS സ്റ്റാൻഡേർഡുകൾ, സർട്ടിഫിക്കേഷൻ നിയമങ്ങൾ, പരിശോധന ആവശ്യകതകൾ അല്ലെങ്കിൽ വിദഗ്ധ വിലയിരുത്തൽ എന്നിവയ്ക്ക് പകരമല്ല.",
        footerTitle: "Business Compliance Assistant",
        footerBis: "BIS സേവനങ്ങൾ",
        footerGov: "സർക്കാർ ലിങ്കുകൾ",
        footerService: "സേവന പ്രവേശനം",
        footerBisWebsite: "BIS ഔദ്യോഗിക വെബ്‌സൈറ്റ്",
        footerDownload: "ഇന്ത്യൻ സ്റ്റാൻഡേർഡുകൾ ഡൗൺലോഡ് ചെയ്യുക",
        footerManak: "മാനക് ഓൺലൈൻ",
        footerCertification: "ഉൽപ്പന്ന സർട്ടിഫിക്കേഷൻ",
        footerIndiaPortal: "ഇന്ത്യയുടെ ദേശീയ പോർട്ടൽ",
        footerConsumerAffairs: "ഉപഭോക്തൃ കാര്യ വകുപ്പ്",
        footerConsumerHelpline: "ദേശീയ ഉപഭോക്തൃ ഹെൽപ്‌ലൈൻ",
        footerUx4g: "UX4G ഡിസൈൻ സിസ്റ്റം",
        footerStatus: "സേവന നില",
        footerHealth: "ഹെൽത്ത് എൻഡ്പോയിന്റ്",
        footerApi: "ഡെവലപ്പർ API",
        footerSearch: "സ്റ്റാൻഡേർഡുകൾ തിരയുക",
        footer: "BIS സ്റ്റാൻഡേർഡ് കണ്ടെത്തലിനുള്ള ഡിജിറ്റൽ സഹായം. അന്തിമ അനുസരണ തീരുമാനങ്ങൾ ഔദ്യോഗിക BIS രേഖകളും യോഗ്യ അധികാരികളും വഴി പരിശോധിക്കുക.",
        footerNote: "ഈ ആപ്ലിക്കേഷൻ സ്റ്റാൻഡേർഡ് കണ്ടെത്തലിനുള്ള സഹായമാണ്. ഇത് ഔദ്യോഗിക BIS സേവനമല്ല.",
        footerUpdated: "അവസാന അവലോകനം: 02 മേയ് 2026",
        footerTop: "തിരച്ചിലിലേക്ക് മടങ്ങുക",
        sampleDefault: "സാധാരണ കെട്ടിട നിർമാണത്തിനായി ഞങ്ങൾ 33 Grade Ordinary Portland Cement നിർമ്മിക്കുന്നു. ഏത് ഇന്ത്യൻ സ്റ്റാൻഡേർഡ് ബാധകമാണ്?",
        sampleAggregatesLabel: "അഗ്രിഗേറ്റുകൾ",
        sampleAggregates: "സ്ട്രക്ചറൽ കോൺക്രീറ്റിനായി പ്രകൃതിദത്ത ഉറവിടങ്ങളിൽ നിന്നുള്ള വലിയതും ചെറുതുമായ അഗ്രിഗേറ്റുകൾ",
        samplePipesLabel: "കോൺക്രീറ്റ് പൈപ്പുകൾ",
        samplePipes: "വാട്ടർ മെയിനുകൾക്കായി ബലവൽക്കരണത്തോടെയും ഇല്ലാതെയും മുൻകൂട്ടി നിർമ്മിച്ച കോൺക്രീറ്റ് പൈപ്പുകൾ",
        sampleWhiteCementLabel: "വെളുത്ത സിമന്റ്",
        sampleWhiteCement: "വാസ്തുവിദ്യാ, അലങ്കാര ഉപയോഗത്തിനായുള്ള വെളുത്ത പോർട്ട്‌ലാൻഡ് സിമന്റ്"
      },
      pa: {
        contrast: "ਕਾਂਟ੍ਰਾਸਟ",
        brandSubtitle: "ਨਿਰਮਾਣ ਇਕਾਈਆਂ ਲਈ ਭਾਰਤੀ ਮਿਆਰ ਬਿਊਰੋ ਮਿਆਰ ਖੋਜ",
        navSearch: "ਖੋਜ",
        navResults: "ਨਤੀਜੇ",
        navStatus: "ਸੇਵਾ ਸਥਿਤੀ",
        navDocs: "ਡਿਵੈਲਪਰ API",
        eyebrow: "BIS ਡਿਜ਼ਿਟਲ ਸੇਵਾ",
        heroCopy: "ਉਤਪਾਦ ਦਾ ਨਾਮ, ਸਮੱਗਰੀ, ਗ੍ਰੇਡ ਅਤੇ ਵਰਤੋਂ ਦਰਜ ਕਰੋ। ਇਹ ਸੇਵਾ ਉਪਲਬਧ BIS ਕੈਟਾਲਾਗ ਰਿਕਾਰਡ ਖੋਜ ਕੇ ਸੰਬੰਧਿਤ ਭਾਰਤੀ ਮਿਆਰਾਂ ਦੀ ਰਹਿਨੁਮਾਈ ਦਿੰਦੀ ਹੈ।",
        serviceStatusTitle: "ਸੇਵਾ ਸਥਿਤੀ",
        serviceStatusCopy: "ਮਿਆਰ ਖੋਜ ਲਈ ਡਿਜ਼ਿਟਲ ਸਹਾਇਤਾ। ਆਖਰੀ ਅਨੁਕੂਲਤਾ ਫੈਸਲੇ ਅਧਿਕਾਰਤ BIS ਦਸਤਾਵੇਜ਼ਾਂ ਨਾਲ ਪੱਕੇ ਕਰੋ।",
        statusOk: "ਠੀਕ",
        statusOne: "BIS ਕੈਟਾਲਾਗ ਰਿਕਾਰਡ ਲੋਡ ਹੋਏ",
        statusTwo: "ਨਤੀਜੇ ਜਾਣੇ ਮਿਆਰਾਂ ਜਾਂ ਪੱਕੀ ਕੀਤੀ ਬਾਹਰੀ ਰਹਿਨੁਮਾਈ ਤੱਕ ਸੀਮਿਤ",
        statusThree: "ਭਾਰਤੀ ਭਾਸ਼ਾ ਸਹਾਇਤਾ ਨਾਲ ਪਹੁੰਚਯੋਗ ਇੰਟਰਫੇਸ",
        loading: "ਕੈਟਾਲਾਗ ਖੋਜਿਆ ਜਾ ਰਿਹਾ ਹੈ...",
        searching: "ਖੋਜ ਜਾਰੀ ਹੈ",
        retrieving: "ਸੰਬੰਧਿਤ BIS ਮਿਆਰ ਪ੍ਰਾਪਤ ਕੀਤੇ ਜਾ ਰਹੇ ਹਨ...",
        error: "ਗਲਤੀ",
        recommendationFailed: "ਸਿਫ਼ਾਰਸ਼ ਨਹੀਂ ਮਿਲੀ",
        metricOne: "ਨਿਰਮਾਣ ਸਮੱਗਰੀ ਕੈਟਾਲਾਗ",
        metricTwo: "ਸੰਬੰਧਿਤ ਮਿਆਰ ਰਹਿਨੁਮਾਈ",
        metricThree: "ਆਮ ਖੋਜ ਟੀਚਾ",
        ready: "ਤਿਆਰ",
        noResults: "ਕੋਈ ਮਿਲਦਾ ਮਿਆਰ ਨਹੀਂ ਮਿਲਿਆ।",
        outsideCatalog: "ਮੌਜੂਦਾ SP 21 ਕੈਟਾਲਾਗ ਤੋਂ ਬਾਹਰ",
        verifyBis: "BIS ਪ੍ਰੀਵਿਊ ਵਿੱਚ ਪੱਕਾ ਕਰੋ",
        pencilWarning: "ਗ੍ਰਾਫਾਈਟ ਜਾਂ ਬਲੈਕ ਲੀਡ ਪੈਂਸਿਲਾਂ bundled BIS SP 21 ਨਿਰਮਾਣ-ਸਮੱਗਰੀ ਕੈਟਾਲਾਗ ਤੋਂ ਬਾਹਰ ਹਨ। IS 1375:2021 ਅਤੇ IS 2079:2022 ਵਰਗੇ ਪੈਂਸਿਲ-ਖਾਸ BIS ਮਿਆਰ ਪੱਕੇ ਕਰੋ।",
        edibleOilWarning: "ਖਾਦ ਤੇਲ bundled BIS SP 21 ਨਿਰਮਾਣ-ਸਮੱਗਰੀ ਕੈਟਾਲਾਗ ਤੋਂ ਬਾਹਰ ਹੈ। ਤੇਲ ਅਤੇ ਚਰਬੀ ਲਈ IS 548 ਅਤੇ ਸੰਬੰਧਿਤ ਤੇਲ-ਕਿਸਮ ਵਿਵਰਣ ਵੇਖੋ।",
        scopeTitle: "ਕੈਟਾਲਾਗ ਦਾਇਰਾ",
        scopeCopy: "ਇਹ ਸੇਵਾ ਉਪਲਬਧ BIS ਕੈਟਾਲਾਗ ਡਾਟਾ ਵਰਤਦੀ ਹੈ। ਜੇ ਉਤਪਾਦ ਮੌਜੂਦਾ ਕੈਟਾਲਾਗ ਤੋਂ ਬਾਹਰ ਹੈ ਤਾਂ ਇਹ ਅਸੰਬੰਧਤ ਮਿਆਰਾਂ ਦੀ ਥਾਂ ਜਾਂਚ ਸਲਾਹ ਦਿਖਾਉਂਦੀ ਹੈ।",
        workflowTitle: "ਇਹ ਸੇਵਾ ਕਿਵੇਂ ਵਰਤਣੀ ਹੈ",
        stepOne: "ਉਤਪਾਦ, ਸਮੱਗਰੀ, ਗ੍ਰੇਡ ਅਤੇ ਵਰਤੋਂ ਦਰਜ ਕਰੋ।",
        stepTwo: "ਸੰਬੰਧਿਤ IS ਕੋਡ ਅਤੇ ਕੈਟਾਲਾਗ ਕਾਰਣ ਵੇਖੋ।",
        stepThree: "ਸਰਟੀਫਿਕੇਸ਼ਨ ਕਾਰਵਾਈ ਤੋਂ ਪਹਿਲਾਂ ਅਧਿਕਾਰਤ BIS ਪੋਰਟਲ ਤੇ ਲੋੜਾਂ ਪੱਕੀਆਂ ਕਰੋ।",
        advisoryTitle: "ਮਹੱਤਵਪੂਰਨ ਸਲਾਹ",
        advisoryCopy: "ਇਹ ਡਿਜ਼ਿਟਲ ਸੇਵਾ ਮਿਆਰ ਖੋਜ ਵਿੱਚ ਸਹਾਇਤਾ ਕਰਦੀ ਹੈ। ਇਹ ਅਧਿਕਾਰਤ BIS ਮਿਆਰਾਂ, ਸਰਟੀਫਿਕੇਸ਼ਨ ਨਿਯਮਾਂ, ਟੈਸਟ ਲੋੜਾਂ ਜਾਂ ਮਾਹਰ ਅੰਦਾਜ਼ੇ ਦਾ ਬਦਲ ਨਹੀਂ ਹੈ।",
        footerTitle: "Business Compliance Assistant",
        footerBis: "BIS ਸੇਵਾਵਾਂ",
        footerGov: "ਸਰਕਾਰੀ ਲਿੰਕ",
        footerService: "ਸੇਵਾ ਪਹੁੰਚ",
        footerBisWebsite: "BIS ਅਧਿਕਾਰਤ ਵੈੱਬਸਾਈਟ",
        footerDownload: "ਭਾਰਤੀ ਮਿਆਰ ਡਾਊਨਲੋਡ ਕਰੋ",
        footerManak: "ਮਾਨਕ ਆਨਲਾਈਨ",
        footerCertification: "ਉਤਪਾਦ ਸਰਟੀਫਿਕੇਸ਼ਨ",
        footerIndiaPortal: "ਭਾਰਤ ਦਾ ਰਾਸ਼ਟਰੀ ਪੋਰਟਲ",
        footerConsumerAffairs: "ਉਪਭੋਗਤਾ ਮਾਮਲੇ ਵਿਭਾਗ",
        footerConsumerHelpline: "ਰਾਸ਼ਟਰੀ ਉਪਭੋਗਤਾ ਹੈਲਪਲਾਈਨ",
        footerUx4g: "UX4G ਡਿਜ਼ਾਈਨ ਸਿਸਟਮ",
        footerStatus: "ਸੇਵਾ ਸਥਿਤੀ",
        footerHealth: "ਹੈਲਥ ਐਂਡਪੌਇੰਟ",
        footerApi: "ਡਿਵੈਲਪਰ API",
        footerSearch: "ਮਿਆਰ ਖੋਜੋ",
        footer: "BIS ਮਿਆਰ ਖੋਜ ਲਈ ਡਿਜ਼ਿਟਲ ਸਹਾਇਤਾ। ਆਖਰੀ ਅਨੁਕੂਲਤਾ ਫੈਸਲੇ ਅਧਿਕਾਰਤ BIS ਦਸਤਾਵੇਜ਼ਾਂ ਅਤੇ ਯੋਗ ਅਧਿਕਾਰੀਆਂ ਨਾਲ ਪੱਕੇ ਕਰੋ।",
        footerNote: "ਇਹ ਐਪਲੀਕੇਸ਼ਨ ਮਿਆਰ ਖੋਜ ਲਈ ਸਹਾਇਕ ਹੈ। ਇਹ ਅਧਿਕਾਰਤ BIS ਸੇਵਾ ਨਹੀਂ ਹੈ।",
        footerUpdated: "ਆਖਰੀ ਸਮੀਖਿਆ: 02 ਮਈ 2026",
        footerTop: "ਖੋਜ ਵੱਲ ਵਾਪਸ",
        sampleDefault: "ਅਸੀਂ ਆਮ ਇਮਾਰਤ ਨਿਰਮਾਣ ਲਈ 33 Grade Ordinary Portland Cement ਬਣਾਉਂਦੇ ਹਾਂ। ਕਿਹੜਾ ਭਾਰਤੀ ਮਿਆਰ ਲਾਗੂ ਹੈ?",
        sampleAggregatesLabel: "ਐਗਰੀਗੇਟ",
        sampleAggregates: "ਸਟਰੱਕਚਰਲ ਕੌਂਕਰੀਟ ਲਈ ਕੁਦਰਤੀ ਸਰੋਤਾਂ ਤੋਂ ਮੋਟੇ ਅਤੇ ਬਰੀਕ ਐਗਰੀਗੇਟ",
        samplePipesLabel: "ਕੌਂਕਰੀਟ ਪਾਈਪ",
        samplePipes: "ਵਾਟਰ ਮੇਨ ਲਈ ਰੀਇਨਫੋਰਸਮੈਂਟ ਨਾਲ ਅਤੇ ਬਿਨਾਂ ਪ੍ਰੀਕਾਸਟ ਕੌਂਕਰੀਟ ਪਾਈਪ",
        sampleWhiteCementLabel: "ਵਾਈਟ ਸਿਮੈਂਟ",
        sampleWhiteCement: "ਆਰਕੀਟੈਕਚਰਲ ਅਤੇ ਸਜਾਵਟੀ ਵਰਤੋਂ ਲਈ ਵਾਈਟ ਪੋਰਟਲੈਂਡ ਸਿਮੈਂਟ"
      },
      ur: {
        contrast: "کنٹراسٹ",
        brandSubtitle: "مینوفیکچرنگ اداروں کے لیے بھارتی معیارات بیورو کی تلاش",
        navSearch: "تلاش",
        navResults: "نتائج",
        navStatus: "سروس کی حالت",
        navDocs: "ڈویلپر API",
        eyebrow: "BIS ڈیجیٹل سروس",
        heroCopy: "مصنوعات کا نام، مواد، گریڈ اور استعمال درج کریں۔ یہ سروس دستیاب BIS کیٹلاگ ریکارڈز تلاش کر کے متعلقہ بھارتی معیارات کی رہنمائی دیتی ہے۔",
        serviceStatusTitle: "سروس کی حالت",
        serviceStatusCopy: "معیارات کی تلاش کے لیے ڈیجیٹل معاونت۔ حتمی تعمیل کے فیصلے سرکاری BIS دستاویزات سے تصدیق کریں۔",
        statusOk: "ٹھیک",
        statusOne: "BIS کیٹلاگ ریکارڈز لوڈ ہو گئے",
        statusTwo: "نتائج معلوم معیارات یا تصدیق شدہ بیرونی رہنمائی تک محدود ہیں",
        statusThree: "بھارتی زبانوں کی مدد کے ساتھ قابل رسائی انٹرفیس",
        loading: "کیٹلاگ تلاش ہو رہا ہے...",
        searching: "تلاش جاری ہے",
        retrieving: "متعلقہ BIS معیارات حاصل ہو رہے ہیں...",
        error: "خرابی",
        recommendationFailed: "سفارش حاصل نہیں ہو سکی",
        metricOne: "تعمیراتی مواد کیٹلاگ",
        metricTwo: "متعلقہ معیار رہنمائی",
        metricThree: "عام تلاش کا ہدف",
        ready: "تیار",
        noResults: "کوئی مماثل معیار نہیں ملا۔",
        outsideCatalog: "موجودہ SP 21 کیٹلاگ سے باہر",
        verifyBis: "BIS پری ویو میں تصدیق کریں",
        pencilWarning: "گریفائٹ یا بلیک لیڈ پنسلیں bundled BIS SP 21 تعمیراتی مواد کیٹلاگ سے باہر ہیں۔ IS 1375:2021 اور IS 2079:2022 جیسے پنسل مخصوص BIS معیارات کی تصدیق کریں۔",
        edibleOilWarning: "خوردنی تیل bundled BIS SP 21 تعمیراتی مواد کیٹلاگ سے باہر ہے۔ تیل اور چکنائی کے لیے IS 548 اور متعلقہ تیل قسم کی تفصیل دیکھیں۔",
        scopeTitle: "کیٹلاگ کا دائرہ",
        scopeCopy: "یہ سروس دستیاب BIS کیٹلاگ ڈیٹا استعمال کرتی ہے۔ اگر مصنوعات موجودہ کیٹلاگ سے باہر ہو تو غیر متعلقہ معیارات کے بجائے تصدیقی مشورہ دکھایا جائے گا۔",
        workflowTitle: "اس سروس کا استعمال کیسے کریں",
        stepOne: "مصنوعات، مواد، گریڈ اور استعمال درج کریں۔",
        stepTwo: "متعلقہ IS کوڈز اور کیٹلاگ وجہ کا جائزہ لیں۔",
        stepThree: "سرٹیفیکیشن کارروائی سے پہلے سرکاری BIS پورٹل پر ضروریات کی تصدیق کریں۔",
        advisoryTitle: "اہم مشورہ",
        advisoryCopy: "یہ ڈیجیٹل سروس معیارات کی تلاش میں مدد کرتی ہے۔ یہ سرکاری BIS معیارات، سرٹیفیکیشن قواعد، ٹیسٹنگ ضروریات یا ماہر تشخیص کا بدل نہیں ہے۔",
        footerTitle: "Business Compliance Assistant",
        footerBis: "BIS خدمات",
        footerGov: "سرکاری رابطے",
        footerService: "سروس تک رسائی",
        footerBisWebsite: "BIS سرکاری ویب سائٹ",
        footerDownload: "بھارتی معیارات ڈاؤن لوڈ کریں",
        footerManak: "مانک آن لائن",
        footerCertification: "مصنوعات سرٹیفیکیشن",
        footerIndiaPortal: "بھارت کا قومی پورٹل",
        footerConsumerAffairs: "محکمہ صارف امور",
        footerConsumerHelpline: "قومی صارف ہیلپ لائن",
        footerUx4g: "UX4G ڈیزائن سسٹم",
        footerStatus: "سروس کی حالت",
        footerHealth: "ہیلتھ اینڈپوائنٹ",
        footerApi: "ڈویلپر API",
        footerSearch: "معیارات تلاش کریں",
        footer: "BIS معیارات کی تلاش کے لیے ڈیجیٹل معاونت۔ حتمی تعمیل کے فیصلے سرکاری BIS دستاویزات اور مجاز حکام سے تصدیق کریں۔",
        footerNote: "یہ ایپلی کیشن معیارات کی تلاش کے لیے معاون ہے۔ یہ سرکاری BIS سروس نہیں ہے۔",
        footerUpdated: "آخری جائزہ: 02 مئی 2026",
        footerTop: "تلاش پر واپس",
        sampleDefault: "ہم عام عمارت تعمیر کے لیے 33 Grade Ordinary Portland Cement بناتے ہیں۔ کون سا بھارتی معیار لاگو ہے؟",
        sampleAggregatesLabel: "ایگریگیٹس",
        sampleAggregates: "ساختی کنکریٹ کے لیے قدرتی ذرائع سے موٹے اور باریک ایگریگیٹس",
        samplePipesLabel: "کنکریٹ پائپ",
        samplePipes: "واٹر مینز کے لیے مضبوطی کے ساتھ اور بغیر پری کاسٹ کنکریٹ پائپ",
        sampleWhiteCementLabel: "سفید سیمنٹ",
        sampleWhiteCement: "تعمیراتی اور آرائشی استعمال کے لیے سفید پورٹ لینڈ سیمنٹ"
      }
    };

    const assistantText = {
      en: {
        title: "Business Compliance Assistant",
        ready: "Next steps",
        empty: "Run a standards search to view matched category, key terms, document readiness, testing readiness, and verification notes.",
        fallback: "Verify with BIS before taking action.",
        sourceAi: "AI-assisted",
        sourceDeterministic: "Deterministic",
        categoryTitle: "Matched product category",
        categoryFallback: "Catalogue match requires BIS verification.",
        termsTitle: "Key matched terms",
        termsFallback: "Review the returned catalogue titles and verify with BIS.",
        whyTitle: "Why these standards match",
        documentsTitle: "Documents to prepare",
        testingTitle: "Testing and lab readiness",
        workflowTitle: "BIS certification workflow",
        notesTitle: "Warnings and verification notes",
        chatTitle: "Ask follow-up questions",
        chatEmpty: "Search a product, then ask about applicability, documents, testing, or verification.",
        chatPlaceholder: "Ask what to do next...",
        chatSend: "Ask",
        chatSending: "Thinking...",
        chatNote: "The chatbot only uses standards returned by this retriever and asks you to verify with BIS.",
        chatNeedQuery: "Enter a product description first.",
        chatError: "Chatbot response failed."
      },
      hi: {
        title: "व्यावसायिक अनुपालन सहायक",
        ready: "अगले कदम",
        empty: "मिलान श्रेणी, मुख्य शब्द, दस्तावेज़ तैयारी, परीक्षण तैयारी और सत्यापन नोट देखने के लिए मानक खोज चलाएँ।",
        fallback: "कार्रवाई से पहले BIS से सत्यापित करें।",
        sourceAi: "AI-सहायता प्राप्त",
        sourceDeterministic: "नियम-आधारित",
        categoryTitle: "मिलान उत्पाद श्रेणी",
        categoryFallback: "कैटलॉग मिलान के लिए BIS सत्यापन आवश्यक है।",
        termsTitle: "मुख्य मिलान शब्द",
        termsFallback: "लौटाए गए कैटलॉग शीर्षक देखें और BIS से सत्यापित करें।",
        whyTitle: "ये मानक क्यों मेल खाते हैं",
        documentsTitle: "तैयार करने वाले दस्तावेज़",
        testingTitle: "परीक्षण और लैब तैयारी",
        workflowTitle: "BIS प्रमाणन कार्यप्रवाह",
        notesTitle: "चेतावनी और सत्यापन नोट",
        chatTitle: "फॉलो-अप प्रश्न पूछें",
        chatEmpty: "उत्पाद खोजें, फिर लागू मानक, दस्तावेज़, परीक्षण या सत्यापन के बारे में पूछें।",
        chatPlaceholder: "अगला कदम पूछें...",
        chatSend: "पूछें",
        chatSending: "सोच रहा है...",
        chatNote: "चैटबॉट केवल इसी retriever से लौटे मानकों का उपयोग करता है और BIS से सत्यापन कहता है।",
        chatNeedQuery: "पहले उत्पाद विवरण दर्ज करें।",
        chatError: "चैटबॉट उत्तर नहीं दे सका।"
      },
      hinglish: {
        title: "Business Compliance Assistant",
        ready: "Next steps",
        empty: "Matched category, key terms, documents, testing readiness aur verification notes dekhne ke liye standards search chalayein.",
        fallback: "Action lene se pehle BIS se verify karein.",
        sourceAi: "AI-assisted",
        sourceDeterministic: "Rule-based",
        categoryTitle: "Matched product category",
        categoryFallback: "Catalogue match ko BIS se verify karna zaroori hai.",
        termsTitle: "Key matched terms",
        termsFallback: "Returned catalogue titles review karein aur BIS se verify karein.",
        whyTitle: "Ye standards kyun match hue",
        documentsTitle: "Documents prepare karein",
        testingTitle: "Testing aur lab readiness",
        workflowTitle: "BIS certification workflow",
        notesTitle: "Warnings aur verification notes",
        chatTitle: "Follow-up questions poochein",
        chatEmpty: "Product search karein, phir applicability, documents, testing, ya verification ke baare mein poochein.",
        chatPlaceholder: "Next step poochein...",
        chatSend: "Ask",
        chatSending: "Soch raha hai...",
        chatNote: "Chatbot sirf retriever ke returned standards use karta hai aur BIS verification bolta hai.",
        chatNeedQuery: "Pehle product description enter karein.",
        chatError: "Chatbot response fail ho gaya."
      }
    };

    Object.entries(localeCompletion).forEach(([lang, values]) => {
      translations[lang] = { ...translations[lang], ...values };
    });

    const form = document.querySelector("#recommendForm");
    const queryInput = document.querySelector("#query");
    const submitButton = document.querySelector("#submit");
    const resultBody = document.querySelector("#resultBody");
    const latency = document.querySelector("#latency");
    const warnings = document.querySelector("#warnings");
    const guidanceBody = document.querySelector("#guidanceBody");
    const chatLog = document.querySelector("#chatLog");
    const chatForm = document.querySelector("#chatForm");
    const chatInput = document.querySelector("#chatInput");
    const chatSubmit = document.querySelector("#chatSubmit");
    const languageSelect = document.querySelector("#languageSelect");
    let currentLang = "en";
    let latestResultData = null;
    let chatHistory = [];

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

    function a(key) {
      return (assistantText[currentLang] && assistantText[currentLang][key]) || assistantText.en[key] || key;
    }

    function bisPortalSearchUrl(code) {
      const match = String(code || "").match(/IS\\s*(\\d{2,5})(?:\\s*\\(\\s*Part\\s*(\\d+)(?:\\s*\\/\\s*Sec\\s*(\\d+))?\\s*\\))?\\s*[:\\-]\\s*(\\d{4})/i);
      if (!match) return "https://standardsbis.bsbedge.com/";
      const terms = ["IS", match[1]];
      if (match[2]) terms.push("Part", match[2]);
      if (match[3]) terms.push("Sec", match[3]);
      return `https://standardsbis.bsbedge.com/BIS_SearchStandard.aspx?Standard_Number=${encodeURIComponent(terms.join(" "))}&id=0`;
    }

    function applyLanguage(lang) {
      currentLang = translations[lang] ? lang : "en";
      document.documentElement.lang = currentLang === "hinglish" ? "en-IN" : currentLang;
      document.documentElement.dir = currentLang === "ur" ? "rtl" : "ltr";

      document.querySelectorAll("[data-i18n]").forEach((node) => {
        node.textContent = t(node.dataset.i18n);
      });

      document.querySelectorAll("[data-assistant-i18n]").forEach((node) => {
        node.textContent = a(node.dataset.assistantI18n);
      });

      document.querySelectorAll("[data-assistant-placeholder]").forEach((node) => {
        node.placeholder = a(node.dataset.assistantPlaceholder);
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
      const readyLabels = Object.values(translations).map((locale) => locale.ready);
      if (readyLabels.includes(latency.textContent)) {
        latency.textContent = t("ready");
      }
      if (latestResultData) {
        renderResults(latestResultData);
      } else {
        renderGuidance(null);
        renderChatLog();
      }
    }

    function setLoading(isLoading) {
      submitButton.disabled = isLoading;
      submitButton.textContent = isLoading ? t("loading") : t("submit");
    }

    function renderWarnings(items) {
      warnings.innerHTML = "";
      if (!items || !items.length) return;
      warnings.innerHTML = items.map((item) => `<div class="warning">${escapeHtml(localizedWarning(item))}</div>`).join("");
    }

    function localizedWarning(item) {
      const warning = String(item || "");
      const lower = warning.toLowerCase();
      if (lower.includes("pencil")) return t("pencilWarning");
      if (lower.includes("edible oil")) return t("edibleOilWarning");
      return warning;
    }

    function listItems(items) {
      const safeItems = (items || []).filter(Boolean);
      if (!safeItems.length) return `<p>${escapeHtml(a("fallback"))}</p>`;
      return `<ul>${safeItems.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
    }

    function renderGuidance(guidance) {
      if (!guidance) {
        guidanceBody.className = "guidance assistant-empty";
        guidanceBody.innerHTML = `<p>${escapeHtml(a("empty"))}</p>`;
        return;
      }
      const terms = guidance.matched_terms || [];
      const source = guidance.ai_generated ? a("sourceAi") : a("sourceDeterministic");
      guidanceBody.className = "guidance";
      guidanceBody.innerHTML = `
        <span class="guidance-source">${escapeHtml(source)}</span>
        <div class="guidance-card">
          <h3>${escapeHtml(a("categoryTitle"))}</h3>
          <p>${escapeHtml(guidance.matched_category || a("categoryFallback"))}</p>
        </div>
        <div class="guidance-card">
          <h3>${escapeHtml(a("termsTitle"))}</h3>
          ${
            terms.length
              ? `<div class="term-list">${terms.map((term) => `<span class="term-pill">${escapeHtml(term)}</span>`).join("")}</div>`
              : `<p>${escapeHtml(a("termsFallback"))}</p>`
          }
        </div>
        <div class="guidance-card">
          <h3>${escapeHtml(a("whyTitle"))}</h3>
          ${listItems(guidance.why_these_standards)}
        </div>
        <div class="guidance-card">
          <h3>${escapeHtml(a("documentsTitle"))}</h3>
          ${listItems(guidance.documents_to_prepare)}
        </div>
        <div class="guidance-card">
          <h3>${escapeHtml(a("testingTitle"))}</h3>
          ${listItems(guidance.testing_lab_readiness)}
        </div>
        <div class="guidance-card">
          <h3>${escapeHtml(a("workflowTitle"))}</h3>
          ${listItems(guidance.bis_workflow)}
        </div>
        <div class="guidance-card">
          <h3>${escapeHtml(a("notesTitle"))}</h3>
          ${listItems(guidance.verification_notes)}
        </div>
      `;
    }

    function renderChatLog() {
      if (!chatHistory.length) {
        chatLog.innerHTML = `<div class="chat-message assistant">${escapeHtml(a("chatEmpty"))}</div>`;
        return;
      }
      chatLog.innerHTML = chatHistory.map((item) => {
        const role = item.role === "user" ? "user" : "assistant";
        return `<div class="chat-message ${role}">${escapeHtml(item.content)}</div>`;
      }).join("");
      chatLog.scrollTop = chatLog.scrollHeight;
    }

    function appendChat(role, content) {
      chatHistory.push({ role, content });
      chatHistory = chatHistory.slice(-8);
      renderChatLog();
    }

    function renderResults(data) {
      latestResultData = data;
      latency.textContent = `${Number(data.latency_seconds || 0).toFixed(3)}s`;
      renderWarnings(data.compliance_warnings);
      renderGuidance(data.business_guidance);
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
              <div class="result-meta">
                <span>${escapeHtml(t("outsideCatalog"))}</span>
                <a href="${escapeHtml(item.source_url)}" target="_blank" rel="noopener">${escapeHtml(item.source_url)}</a>
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
        return `
          <article class="result">
            <div class="rank">${index + 1}</div>
            <div>
              <h3><span class="code">${escapeHtml(item.code)}</span>${escapeHtml(item.title || "BIS standard")}</h3>
              <p class="rationale">${escapeHtml(item.rationale || "Matched against the BIS catalogue.")}</p>
              <div class="result-meta">
                <a href="${escapeHtml(item.source_url || bisPortalSearchUrl(item.code))}" target="_blank" rel="noopener">${escapeHtml(item.source_url || bisPortalSearchUrl(item.code))}</a>
              </div>
            </div>
          </article>
        `;
      }).join("");
    }

    async function recommend(query) {
      latestResultData = null;
      chatHistory = [];
      renderChatLog();
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
          throw new Error(data.detail || t("recommendationFailed"));
        }
        renderResults(data);
      } catch (error) {
        latency.textContent = t("error");
        resultBody.className = "empty-state";
        resultBody.textContent = error.message || t("recommendationFailed");
      } finally {
        setLoading(false);
      }
    }

    async function askAssistant(message) {
      const query = queryInput.value.trim();
      if (!query) {
        appendChat("assistant", a("chatNeedQuery"));
        return;
      }
      appendChat("user", message);
      chatSubmit.disabled = true;
      chatSubmit.textContent = a("chatSending");
      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            message,
            history: chatHistory.slice(0, -1),
            top_k: 5,
            language: currentLang,
          }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || a("chatError"));
        }
        appendChat("assistant", data.answer || a("chatError"));
      } catch (error) {
        appendChat("assistant", error.message || a("chatError"));
      } finally {
        chatSubmit.disabled = false;
        chatSubmit.textContent = a("chatSend");
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const query = queryInput.value.trim();
      if (query) recommend(query);
    });

    chatForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const message = chatInput.value.trim();
      if (!message) return;
      chatInput.value = "";
      askAssistant(message);
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

IS_CODE_PREVIEW_PATTERN = re.compile(
    r"\bIS\s*(\d{2,5})"
    r"(?:\s*\(\s*Part\s*(\d+)(?:\s*/\s*Sec\s*(\d+))?\s*\))?"
    r"\s*[:\-]\s*(\d{4})",
    re.IGNORECASE,
)
BIS_STANDARD_SEARCH_BASE_URL = "https://standardsbis.bsbedge.com/BIS_SearchStandard.aspx"


def _bis_portal_search_url_for_code(code: str) -> str:
    match = IS_CODE_PREVIEW_PATTERN.search(str(code or ""))
    if not match:
        return "https://standardsbis.bsbedge.com/"
    number, part, section, _year = match.groups()
    search_terms = ["IS", number]
    if part:
        search_terms.extend(["Part", part])
    if section:
        search_terms.extend(["Sec", section])
    return f"{BIS_STANDARD_SEARCH_BASE_URL}?Standard_Number={quote_plus(' '.join(search_terms))}&id=0"


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
                "source_url": _bis_portal_search_url_for_code(code),
            }
        )
    return items


GUIDANCE_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
GUIDANCE_IS_CODE_PATTERN = re.compile(
    r"\bIS\s*\d{2,5}(?:\s*\(\s*Part\s*\d+(?:\s*/\s*Sec\s*\d+)?\s*\))?\s*[:\-]\s*\d{4}",
    re.IGNORECASE,
)
GUIDANCE_STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "indian",
    "into",
    "make",
    "material",
    "product",
    "standard",
    "standards",
    "the",
    "use",
    "we",
    "which",
    "with",
}
GENERIC_DOCUMENTS = [
    "Product description with material, grade, dimensions, and intended use.",
    "Manufacturing process note and quality-control checkpoints.",
    "Raw material specifications and supplier records.",
    "Batch or lot identification records for samples submitted for testing.",
]
GENERIC_TESTING_READINESS = [
    "Shortlist BIS-recognized or otherwise competent labs for the returned standards.",
    "Prepare representative samples and retain traceability to production lots.",
    "Compare test parameters against the official standard text before submission.",
]
GENERIC_BIS_WORKFLOW = [
    "Confirm the applicable standard on the official BIS portal.",
    "Map product variants and grades to the returned IS codes.",
    "Prepare documents and test evidence before starting certification activity.",
    "Use Manak Online or the relevant BIS channel for the official process.",
]
VERIFY_WITH_BIS_NOTE = (
    "Verify with BIS before relying on this guidance for certification, testing, fees, timelines, "
    "or legal compliance."
)
FALLBACK_CHAT_DISCLOSURE = "Verify with BIS before taking certification or legal action."


def _tokenize_guidance_text(text: str) -> list[str]:
    return [token.lower() for token in GUIDANCE_TOKEN_PATTERN.findall(str(text or ""))]


def _matched_terms(query: str, recommendations: list[dict[str, Any]], limit: int = 8) -> list[str]:
    query_tokens = set(_tokenize_guidance_text(query))
    if not query_tokens:
        return []
    matched: list[str] = []
    for item in recommendations:
        text = " ".join(str(item.get(field) or "") for field in ("title", "rationale", "code"))
        item_tokens = set(_tokenize_guidance_text(text))
        for token in sorted(query_tokens & item_tokens):
            if token not in GUIDANCE_STOP_WORDS and token not in matched:
                matched.append(token)
                if len(matched) >= limit:
                    return matched
    return matched


def _matched_category(query: str, recommendations: list[dict[str, Any]]) -> str:
    query_tokens = set(_tokenize_guidance_text(query))
    category_rules = (
        (("cement", "opc", "ppc", "portland"), "Cement and cementitious building material"),
        (("aggregate", "aggregates", "sand", "gravel"), "Concrete aggregates and granular material"),
        (("pipe", "pipes", "water", "mains"), "Concrete pipes and drainage/water infrastructure"),
        (("block", "blocks", "masonry"), "Masonry units and concrete blocks"),
        (("sheet", "roof", "roofing", "cladding"), "Roofing and cladding material"),
        (("steel", "tmt", "reinforcement", "reinforced", "bar", "bars"), "Steel and reinforcement product"),
    )
    for terms, category in category_rules:
        if any(term in query_tokens for term in terms):
            return category
    if recommendations:
        return str(recommendations[0].get("title") or "BIS catalogue product family")
    return "No current catalogue match"


def _fallback_business_guidance(
    query: str,
    retrieved_codes: list[str],
    recommendations: list[dict[str, Any]],
    out_of_scope: bool,
) -> dict[str, Any]:
    if out_of_scope or not retrieved_codes:
        return {
            "matched_category": "Outside current catalogue or no returned standard",
            "matched_terms": [
                token for token in _tokenize_guidance_text(query) if token not in GUIDANCE_STOP_WORDS
            ][:5],
            "why_these_standards": [],
            "documents_to_prepare": [],
            "testing_lab_readiness": [],
            "bis_workflow": ["Verify the product category and applicable standards directly with BIS."],
            "verification_notes": [VERIFY_WITH_BIS_NOTE],
            "ai_generated": False,
        }

    why = []
    for index, item in enumerate(recommendations, start=1):
        code = str(item.get("code") or retrieved_codes[index - 1])
        title = str(item.get("title") or "BIS catalogue standard")
        if index == 1:
            why.append(f"Top candidate: {code} aligns with the catalogue title/scope for {title}.")
        else:
            why.append(
                f"Additional candidate rank {index}: {code} is related in the catalogue, but verify "
                f"whether {title} applies to this exact product grade and use."
            )

    return {
        "matched_category": _matched_category(query, recommendations),
        "matched_terms": _matched_terms(query, recommendations),
        "why_these_standards": why,
        "documents_to_prepare": GENERIC_DOCUMENTS,
        "testing_lab_readiness": GENERIC_TESTING_READINESS,
        "bis_workflow": GENERIC_BIS_WORKFLOW,
        "verification_notes": [
            "Only the returned IS codes are used in this guidance.",
            VERIFY_WITH_BIS_NOTE,
        ],
        "ai_generated": False,
    }


def _strings_in_guidance(guidance: dict[str, Any]) -> list[str]:
    strings: list[str] = []
    for key in (
        "matched_category",
        "matched_terms",
        "why_these_standards",
        "documents_to_prepare",
        "testing_lab_readiness",
        "bis_workflow",
        "verification_notes",
    ):
        value = guidance.get(key)
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, list):
            strings.extend(str(item) for item in value)
    return strings


def _guidance_mentions_only_allowed_codes(guidance: dict[str, Any], allowed_codes: set[str]) -> bool:
    for text in _strings_in_guidance(guidance):
        for match in GUIDANCE_IS_CODE_PATTERN.findall(text):
            if normalize_standard_code(match) not in allowed_codes:
                return False
    return True


def _normalize_guidance_payload(payload: Any, allowed_codes: set[str]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    normalized: dict[str, Any] = {
        "matched_category": str(payload.get("matched_category") or "BIS catalogue product family")[:140],
        "matched_terms": [str(item)[:48] for item in payload.get("matched_terms") or []][:8],
        "why_these_standards": [str(item)[:220] for item in payload.get("why_these_standards") or []][:5],
        "documents_to_prepare": [str(item)[:180] for item in payload.get("documents_to_prepare") or []][:5],
        "testing_lab_readiness": [str(item)[:180] for item in payload.get("testing_lab_readiness") or []][:5],
        "bis_workflow": [str(item)[:180] for item in payload.get("bis_workflow") or []][:5],
        "verification_notes": [str(item)[:220] for item in payload.get("verification_notes") or []][:5],
        "ai_generated": True,
    }
    if not any("verify with bis" in note.lower() for note in normalized["verification_notes"]):
        normalized["verification_notes"].append(VERIFY_WITH_BIS_NOTE)
    if not _guidance_mentions_only_allowed_codes(normalized, allowed_codes):
        return None
    return normalized


def _groq_business_guidance(
    query: str,
    retrieved_codes: list[str],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not retrieved_codes:
        return None

    standards = [
        {
            "code": item.get("code"),
            "title": item.get("title"),
            "rationale": item.get("rationale"),
        }
        for item in recommendations
    ]
    prompt = {
        "query": query,
        "retrieved_standards": standards,
        "rules": [
            "Return JSON only with the requested keys.",
            "Only mention IS codes present in retrieved_standards.",
            "Do not invent fees, timelines, forms, legal claims, certification guarantees, or standards.",
            "Include Verify with BIS in verification_notes.",
        ],
    }
    request_body = {
        "model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write cautious BIS standards discovery guidance for MSE users. "
                    "Use only the supplied retrieved standards and return compact JSON."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
        ],
        "temperature": 0.1,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.5) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None

    content = (
        response_payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    try:
        guidance_payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    allowed_codes = {normalize_standard_code(code) for code in retrieved_codes}
    return _normalize_guidance_payload(guidance_payload, allowed_codes)


def _answer_mentions_only_allowed_codes(answer: str, allowed_codes: set[str]) -> bool:
    for match in GUIDANCE_IS_CODE_PATTERN.findall(answer):
        if normalize_standard_code(match) not in allowed_codes:
            return False
    return True


def _fallback_chat_answer(
    message: str,
    retrieved_codes: list[str],
    recommendations: list[dict[str, Any]],
    out_of_scope: bool,
) -> str:
    if out_of_scope or not retrieved_codes:
        return (
            "I do not have a returned BIS catalogue match for this product in the current retrieval result. "
            f"Use the official BIS portal or Manak Online to verify the applicable standard. {FALLBACK_CHAT_DISCLOSURE}"
        )

    lower = message.lower()
    top = recommendations[0] if recommendations else {}
    top_code = str(top.get("code") or retrieved_codes[0])
    top_title = str(top.get("title") or "the top returned standard")
    codes = ", ".join(retrieved_codes)
    other_codes = ", ".join(retrieved_codes[1:]) or "no additional returned candidates"

    if any(term in lower for term in ("why", "match", "selected", "applicable", "which")):
        return (
            f"The strongest candidate is {top_code} because its catalogue title/scope is closest to the "
            f"product description: {top_title}. Other returned candidates ({other_codes}) are related catalogue "
            f"matches and should be treated as candidates until the exact grade, material, and intended use "
            f"are checked against the official BIS text. {FALLBACK_CHAT_DISCLOSURE}"
        )
    if any(term in lower for term in ("document", "prepare", "paper", "record")):
        return (
            "Prepare a product description, grade/material details, manufacturing process note, quality-control "
            "records, raw material specifications, supplier records, and sample batch traceability. "
            f"Use these documents to verify the returned standards ({codes}) with BIS. {FALLBACK_CHAT_DISCLOSURE}"
        )
    if any(term in lower for term in ("test", "lab", "sample")):
        return (
            "Prepare representative samples with batch traceability, identify competent labs, and compare required "
            f"test parameters against the official text for the returned standards ({codes}). "
            f"{FALLBACK_CHAT_DISCLOSURE}"
        )
    if any(term in lower for term in ("next", "workflow", "certification", "apply", "manak")):
        return (
            f"Next, verify {top_code} and the other returned candidates on the official BIS portal, map each product "
            "variant to the correct code, prepare test evidence and documents, then use Manak Online or the relevant "
            f"BIS channel for the official process. {FALLBACK_CHAT_DISCLOSURE}"
        )
    return (
        f"I can answer using only the returned standards: {codes}. The top candidate is {top_code} ({top_title}). "
        "Ask about why it matched, documents, testing readiness, or next steps. "
        f"{FALLBACK_CHAT_DISCLOSURE}"
    )


def _groq_chat_answer(
    query: str,
    message: str,
    history: list[ChatMessage],
    retrieved_codes: list[str],
    recommendations: list[dict[str, Any]],
) -> str | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not retrieved_codes:
        return None

    standards = [
        {
            "code": item.get("code"),
            "title": item.get("title"),
            "rationale": item.get("rationale"),
        }
        for item in recommendations
    ]
    safe_history = [
        {"role": item.role if item.role in {"user", "assistant"} else "user", "content": item.content}
        for item in history[-6:]
    ]
    prompt = {
        "product_query": query,
        "user_message": message,
        "history": safe_history,
        "retrieved_standards": standards,
        "rules": [
            "Answer only using retrieved_standards.",
            "Only mention IS codes present in retrieved_standards.",
            "Do not invent fees, timelines, forms, legal claims, certification guarantees, or standards.",
            "If unsure, say to verify with BIS.",
        ],
    }
    request_body = {
        "model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cautious BIS standards assistant for MSE users. Keep answers short, "
                    "grounded only in supplied retrieved standards, and include BIS verification when needed."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
        ],
        "temperature": 0.1,
        "max_tokens": 450,
    }
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None

    answer = str(
        response_payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    ).strip()
    allowed_codes = {normalize_standard_code(code) for code in retrieved_codes}
    if not answer or not _answer_mentions_only_allowed_codes(answer, allowed_codes):
        return None
    if "verify with bis" not in answer.lower():
        answer = f"{answer} {FALLBACK_CHAT_DISCLOSURE}"
    return answer[:1800]


def _chat_answer(
    query: str,
    message: str,
    history: list[ChatMessage],
    retrieved_codes: list[str],
    recommendations: list[dict[str, Any]],
    out_of_scope: bool,
) -> tuple[str, bool]:
    generated = _groq_chat_answer(query, message, history, retrieved_codes, recommendations)
    if generated is not None:
        return generated, True
    return _fallback_chat_answer(message, retrieved_codes, recommendations, out_of_scope), False


def _business_guidance(
    query: str,
    retrieved_codes: list[str],
    recommendations: list[dict[str, Any]],
    out_of_scope: bool,
) -> dict[str, Any]:
    fallback = _fallback_business_guidance(query, retrieved_codes, recommendations, out_of_scope)
    generated = _groq_business_guidance(query, retrieved_codes, recommendations)
    if generated is None:
        return fallback
    if not generated.get("matched_terms"):
        generated["matched_terms"] = fallback["matched_terms"]
    return generated


def _external_standards(query: str, language: str = "en") -> list[dict[str, str]]:
    if is_edible_oil_query(query):
        return [
            {
                "code": "IS 548 (Part 1/Sec 1):2021",
                "title": "Method of Sampling and Test for Oils and Fats - Sampling",
                "rationale": (
                    "Use this Indian Standard for sampling crude or processed animal and vegetable "
                    "fats and oils, including individual and blended edible oils."
                ),
                "source_url": "https://standardsbis.bsbedge.com/BIS_Preview.aspx?id=548_1_1",
            },
            {
                "code": "IS 548 (Part 1/Sec 2):2021",
                "title": "Method of Sampling and Test for Oils and Fats - Physical and Chemical Tests",
                "rationale": (
                    "Use this Indian Standard for physical and chemical tests of individual oils, "
                    "blended oils, fortified oils, fats and related products."
                ),
                "source_url": "https://standardsbis.bsbedge.com/BIS_Preview.aspx?id=548_1_2",
            },
            {
                "code": "IS 548 (Part 2):1976",
                "title": "Methods of Sampling and Test for Oils and Fats - Purity Tests",
                "rationale": "Use this Indian Standard for purity testing requirements for oils and fats.",
                "source_url": "https://standardsbis.bsbedge.com/BIS_Preview.aspx?id=548_2_1976_Reff2020",
            },
            {
                "code": "IS 14349:2025",
                "title": "Code for Hygienic Conditions for Processing Units of Edible Oils and Fats",
                "rationale": (
                    "Use this Indian Standard for hygienic conditions in edible oils and fats "
                    "manufacturing, packing, storage and transport facilities."
                ),
                "source_url": "https://www.bis.gov.in/index.php/standard-of-the-month/",
            },
            {
                "code": "IS 14636:1998",
                "title": "Flexible Packaging Materials for Packaging of Edible Oils, Ghee and Vanaspati",
                "rationale": (
                    "Use this Indian Standard when edible oil is packed in flexible packaging "
                    "materials."
                ),
                "source_url": "https://standardsbis.bsbedge.com/BIS_Preview.aspx?id=14636",
            },
        ]
    if not is_pencil_query(query):
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

    recommendations = _recommendation_items(result["retrieved_standards"], payload.language)
    external_standards = _external_standards(query, payload.language)
    return {
        **result,
        "compliance_warnings": processed.compliance_warnings,
        "recommendations": recommendations,
        "business_guidance": _business_guidance(
            query=query,
            retrieved_codes=result["retrieved_standards"],
            recommendations=recommendations,
            out_of_scope=bool(result.get("out_of_scope")),
        ),
        "external_standards": external_standards,
    }


@app.post("/api/recommend", response_model=RecommendationResponse, include_in_schema=False)
def recommend_api(payload: RecommendationRequest) -> dict[str, Any]:
    return recommend(payload)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> dict[str, Any]:
    query = payload.query.strip()
    message = payload.message.strip()
    if not query or not message:
        raise HTTPException(status_code=422, detail="query and message must not be empty")

    try:
        pipeline = _pipeline()
        processed = pipeline.query_processor.process(query)
        result = pipeline.run_query(query, top_k=payload.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    recommendations = _recommendation_items(result["retrieved_standards"], payload.language)
    answer, ai_generated = _chat_answer(
        query=query,
        message=message,
        history=payload.history,
        retrieved_codes=result["retrieved_standards"],
        recommendations=recommendations,
        out_of_scope=bool(result.get("out_of_scope")),
    )
    return {
        "answer": answer,
        "retrieved_standards": result["retrieved_standards"],
        "compliance_warnings": processed.compliance_warnings,
        "ai_generated": ai_generated,
    }


@app.post("/api/chat", response_model=ChatResponse, include_in_schema=False)
def chat_api(payload: ChatRequest) -> dict[str, Any]:
    return chat(payload)
