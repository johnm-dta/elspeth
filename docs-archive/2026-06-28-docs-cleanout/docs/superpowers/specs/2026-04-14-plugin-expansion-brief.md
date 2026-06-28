# ELSPETH Plugin Expansion — Case for Work

**Date**: 2026-04-14
**Scope**: Phased expansion of ELSPETH's plugin ecosystem across search, analytical, and reporting capabilities
**Present-state evidence**: For the evaluation evidence underpinning the audit-discipline claim this case-for-work rests on, see `docs/composer/evidence/composer-briefing-2026-05-03.md` (the SES briefing on the composer hard-mode evaluation, including past-extension evidence that the audit discipline holds when the platform is extended).

---

## What ELSPETH Does Today

ELSPETH is an auditable Sense/Decide/Act pipeline framework. Every decision — from data ingestion to LLM classification to final output — is traced back to source data, configuration, and code version. The audit trail is the legal record; it doesn't approximate what happened, it proves it.

Today, ELSPETH pipelines can ingest structured data (CSV, JSON, Dataverse, Azure Blob), process it through LLM transforms with full audit recording, apply content safety screening, perform retrieval-augmented generation against vector stores, and write results to files or databases. The system handles the hard problems well: deterministic hashing, checkpoint/resume, retry with backoff, fork/join DAG execution, and row-level quarantine for bad data.

What it cannot do is reach out into the broader information landscape. Pipelines operate on data that has already been collected and placed in a file or database. The web scraping capability exists but is limited to static HTML. There is no way to index processed data for later retrieval, no way to generate reports from pipeline output, and no way to monitor sources for changes over time.

## What This Work Delivers

This expansion turns ELSPETH from a data processing framework into a research and intelligence platform — while preserving the audit guarantees that make it valuable in the first place.

### Search and Retrieval Infrastructure

A provider-based search abstraction brings full-text, vector, and hybrid search into the pipeline model. OpenSearch serves as the primary backend, providing analytical aggregations (count by category, trend over time, statistical summaries) that are essential for research workflows. Qdrant adds optimised vector search for RAG retrieval. Meilisearch provides a lightweight option for development and smaller deployments. The existing ChromaDB integration is upgraded from a write-only sink to a full search provider.

Pipelines gain three new capabilities: query a search index as a data source (search_source), write processed data into an index (search_sink), and retrieve context mid-pipeline for augmented generation (search RAG provider). The query model supports structured boolean filters, aggregations, facets, highlighting, and cursor-based pagination for large result sets — each backend declares its capabilities, and the system validates at configuration time rather than failing at runtime.

### Intelligent Web Scraping

The existing web scrape transform gains a browser mode powered by Playwright. This handles JavaScript-rendered content, single-page applications, and pages that require interaction (clicking load-more buttons, scrolling through infinite feeds, filling search forms). A declarative interaction model keeps every browser action auditable — no arbitrary JavaScript execution, just structured steps that the audit trail can record. Visual capture (screenshots, page PDFs) lets pipelines preserve evidence of what a page looked like at scrape time. Security controls carry over from the existing HTTP mode: SSRF prevention, IP pinning, DNS rebinding defence, and resource blocking.

### Reporting and Notifications

A report sink generates structured output from pipeline data — Markdown for version-controlled documentation, HTML for self-contained web reports, PDF for formal distribution, and JSON for downstream consumption by dashboards or other pipelines. Jinja2 templates give full control over report structure, from simple data tables to narrative research documents where LLM-generated summaries are woven into a coherent report.

A notification sink delivers alerts through email, Slack, Teams, or generic webhooks. Combined with monitoring pipelines, this enables automated alerting when conditions are met — a competitor changes pricing, a regulatory document is updated, or a research query returns new results.

### Document Processing and Knowledge Management

Later phases add document ingestion (PDF, DOCX, PPTX), text chunking with configurable strategies, and an embedding transform with a provider registry for OpenAI, Azure, and local models. Together with the search infrastructure, this creates a complete knowledge management pipeline: ingest organisational documents, chunk and embed them, index into a search backend, and surface answers through RAG-powered queries. Every step — from the original document to the chunked embedding to the retrieved context — maintains full audit lineage.

### Monitoring and Analysis

Change detection transforms compare current scrapes against previous versions using content fingerprints, surfacing what has changed since the last run. Entity extraction pulls structured data (people, organisations, dates, monetary values) from unstructured text. Scheduled pipeline triggers enable periodic execution for continuous monitoring. Dashboard sinks write structured data to visualisation-friendly formats for ongoing operational intelligence.

## Use Cases

The expanded capabilities compose into four pipeline patterns that recur across corporate and government contexts. In each case, the audit trail is what transforms automation from a convenience into a defensible process.

### Monitor & Alert

*Scheduled scrape → extract changes → diff against previous → index → notify*

A compliance team tracks legislative changes across parliamentary websites, gazette notifications, and regulatory body publications. The pipeline scrapes these sources on a schedule, the LLM extracts material changes, the diff transform identifies what's new since the last run, and the notification sink alerts the compliance inbox. When a regulator asks "how did you become aware of this change?", the audit trail shows the exact scrape timestamp, what was extracted, and when the notification was sent.

The same pattern serves competitive price monitoring (scraping JS-rendered product pages, extracting pricing, alerting on threshold changes), supply chain risk monitoring (news sentiment on suppliers), patent and IP tracking (new filings in relevant technology areas), and public health surveillance (disease indicators from health authority portals). In each case, the organisational question isn't just "did you know?" — it's "when did you know, and can you prove it?"

### Research & Report

*Multi-source scrape/ingest → LLM extraction → index → aggregation queries → generate report*

A policy team needs to brief a minister on housing affordability. The pipeline scrapes statistical agencies, state government portals, academic repositories, and news sources. The LLM extracts relevant findings, the search sink indexes them, aggregation queries produce trend data, and a Jinja2 template generates the briefing document. Every claim in the briefing links back to the specific source that supported it.

Corporate equivalents include M&A due diligence (researching a target across corporate registries, financial filings, court records, and sanctions databases — the report is a legal document), ESG reporting (environmental and governance data collection with verifiable source provenance), and market entry research (regulatory landscape, competitor analysis, and customer demographics synthesised into an investment case). In all cases, research reports inform consequential decisions. When those decisions are questioned — by auditors, regulators, or opposing counsel — "we ran a search" doesn't hold up. A documented research process with timestamped evidence does.

### Screen & Assess

*Ingest entities → search against reference databases → LLM classify/score → generate assessment*

Every government payment, visa application, or contract award must be screened against sanctions lists. The pipeline ingests the entity, fuzzy-searches against OFAC, DFAT, EU, and UN lists, the LLM disambiguates near-matches, and the output is a risk score with a screening certificate. If a sanctioned entity slips through, the organisation must prove it screened. If a false positive blocks a legitimate entity, it must prove why.

Corporate applications include KYC/AML compliance (customer onboarding with identity verification, sanctions screening, and adverse media checks — regulators can demand proof that screening was performed for any customer), vendor risk assessment (financial health, litigation history, and regulatory actions), and grant or insurance claims assessment. Screening decisions are legally consequential in both directions — false negatives create liability, false positives cause harm. The audit trail makes every decision reconstructable.

### Discover & Surface

*Ingest documents → chunk → embed → index → RAG-powered retrieval → conversational Q&A*

A department indexes all its briefings, policy documents, and Hansard references. When a minister faces Question Time, advisors RAG-query the knowledge base: "What has our position been on X over the last 12 months?" The answer cites specific documents with dates. The RAG transform records which chunks were retrieved, their relevance scores, and which ones the LLM used — if the answer is challenged, the evidence chain is intact.

Corporate equivalents include internal research portals (product, legal, and strategy teams querying across organisational documents with citations), customer support knowledge bases (product documentation and resolution records surfaced to support staff), and regulatory compliance libraries (indexing all applicable regulations, policies, and interpretations so compliance officers can query for obligations under new legislation). RAG systems without provenance are a liability — they generate confident answers that may be wrong. With ELSPETH's audit model, every retrieved fact traces back to its source document.

## How It Gets Built

The work is structured in four phases, each independently shippable:

**Phase 1** delivers the web research pipeline end-to-end: browser-capable scraping, OpenSearch integration (source, sink, and RAG), and report generation. This validates the architecture with a real workflow — scraping target sites, extracting data with LLMs, indexing for analysis, and producing formatted reports.

**Phase 2** adds optimised vector search (Qdrant), notification delivery, and RSS/Atom feed monitoring. This enables the alerting and monitoring use case.

**Phase 3** builds the knowledge management foundation: document ingestion, chunking, embedding, and a lightweight search backend (Meilisearch) for development workflows.

**Phase 4** adds scheduled execution, change detection, entity extraction, and dashboard output for continuous analytical workflows.

Each phase builds on the previous without modifying core abstractions. The search protocol defined in Phase 1 accommodates every subsequent provider. The browser mode added in Phase 1 serves every scraping use case in later phases. No rework, no breaking changes — the architecture scales by addition, not modification.

## Why This Matters

The differentiator is not the individual capabilities — web scraping, search indexing, and report generation exist in many tools. The differentiator is that ELSPETH does all of them under a single audit model. When a research report says "competitor X raised prices by 12% this quarter," ELSPETH can prove it: here is the scrape that captured the page, here is the LLM call that extracted the price, here is the index query that compared it against the baseline, here is the row that entered the report. Every claim is traceable to evidence. Every step is reproducible.

For organisations operating in regulated environments, under formal inquiry, or simply wanting to trust their analytical outputs — that traceability transforms research automation from a convenience into a defensible process.
