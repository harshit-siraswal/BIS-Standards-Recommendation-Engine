# Product Requirements Document (PRD)

## Project
AI-powered Intelligent Assistant for Indian Standards and BIS Services for Industries and Consumers

## SIH Alignment
- Problem Code: SIH26107
- Theme: Smart Automation
- Ministry: Ministry of Consumer Affairs, Food and Public Distribution
- Goal: Provide an AI-powered assistant that helps industries, MSMEs, and consumers discover relevant Indian Standards and navigate BIS services.

## Problem Statement Summary
Users struggle to find applicable standards, certification routes, and official BIS processes quickly. The solution should provide standard discovery, service routing, and guided next steps with multilingual support.

## Product Vision
Deliver a trustworthy assistant that:
- retrieves relevant IS standards from known catalog data,
- gives deterministic compliance guidance,
- routes users to official BIS schemes, labs, and process links,
- supports multiple Indian languages,
- avoids fabricated standards or legal claims.

## Target Users
- Manufacturers and MSMEs seeking applicable IS standards.
- Compliance and quality teams preparing certification applications.
- Consumers looking for BIS service routes, hallmarking info, and complaint channels.
- Students and early-stage entrepreneurs learning BIS compliance workflows.

## Core Use Cases
1. Standards Discovery
- User provides product, material, grade, and intended use.
- System returns top relevant IS codes with rationale and official search links.

2. Compliance Roadmap
- User clicks roadmap button after search.
- System shows step-by-step compliance flow including scheme selection, Manak route, testing readiness, and verification points.

3. BIS Service Navigation
- System identifies audience context (industry, consumer, mixed).
- System suggests relevant BIS service buckets and official links.

4. Conversational Assistant
- User asks follow-up questions.
- Assistant provides safe, grounded replies tied to retrieved standards.

## Functional Requirements
- FR-1: Return top-k relevant standards for each query with confidence and source link.
- FR-2: Include business guidance with documents, testing readiness, and workflow.
- FR-3: Include BIS service guidance with audience, relevant services, next steps, and official links.
- FR-4: Include compliance roadmap with:
  - applicable schemes,
  - suggested labs/test facilities,
  - process checkpoints,
  - step-by-step action plan with official links.
- FR-5: Provide multilingual UI support for Indian languages.
- FR-6: Provide deterministic fallback when optional LLM path is unavailable.
- FR-7: Do not mention IS codes not present in retrieved or verified external data.

## Non-Functional Requirements
- NFR-1: API response latency should be suitable for interactive use (target under 5 seconds for common queries).
- NFR-2: Deterministic judge path must remain API-free and stable.
- NFR-3: High availability with clear health/status endpoints.
- NFR-4: Accessibility-friendly UI with contrast and text-size controls.

## Safety and Trust Requirements
- No hallucinated standards.
- Explicit verification note: users must verify with BIS for certification, legal, timelines, and fee decisions.
- Out-of-scope handling should redirect users to official BIS routes rather than return unrelated codes.

## Success Metrics
- Retrieval quality: high hit rate and MRR on public/robustness sets.
- Guidance quality: roadmap completeness (schemes + labs + process + links).
- Trust quality: zero fabricated IS code incidents in guarded responses.
- User utility: reduced time to identify next BIS action.

## Scope
In scope:
- BIS standards discovery assistant experience.
- Compliance roadmap and BIS service routing.
- Multilingual UI and conversational guidance.

Out of scope:
- Automatic certification filing submission.
- Legal certification decision automation.
- Replacing official BIS advisory or scheme documents.

## User Journey (Happy Path)
1. User opens web app and enters product details.
2. System returns relevant IS standards.
3. User reviews rationale and service guidance.
4. User clicks Show compliance roadmap.
5. System displays step-wise process with official BIS/Manak/lab links.
6. User asks follow-up in chat and receives grounded guidance.
7. User proceeds to official BIS channels for final action.

## Release Readiness Checklist
- API responses include business_guidance, bis_services, and compliance_roadmap.
- UI includes roadmap toggle and renders roadmap details.
- Tests cover roadmap presence and schema behavior.
- README documents roadmap usage.
