# ClinPGx-QA: Roadmap to a Strong AI-for-Science Benchmark

This document outlines concrete enhancements to elevate ClinPGx-QA into a robust “AI for Science” benchmark focused on trustworthy, evidence-grounded reasoning over pharmacogenomic literature.

## Goals

- Evidence-grounded, full-text reasoning with reproducible, fair evaluation.
- Strong generalization claims via robust splits and perturbation tests.
- Clear, actionable leaderboard with defensible metrics and governance.

## ICLR LLM Reasoning Workshop Alignment

- Positioning: Frame ClinPGx-QA as a reasoning benchmark requiring multi-step, verifiable, evidence-grounded scientific inference over long contexts, with process- and outcome-level evaluation.
- Novelty: Full-text PGx focus, composite end-to-end metric, span-grounded verification, inequality/CI-aware numeric grading, and OOD/entity/time-based splits.
- Claims to support: (1) Verifiable reasoning improves reliability (process-level metrics correlate with outcome accuracy). (2) Robustness suites reveal brittleness in numeric/evidence handling. (3) Generalization gaps persist under OOD/entity-holdout/time splits.
- Artifacts: Public dataset release (non-hidden splits), sealed hidden test with eval server, reference baselines, reproducibility pack, and documented governance.

## High-Impact Priorities

- Evidence grounding: require extractive evidence spans (character offsets) for every supported claim; score with span IoU; allow multiple gold spans.
- OOD/generalization splits: time-based split (post-cutoff papers) and entity-holdouts (unseen variants/genes/drugs/phenotypes).
- Multi-document synthesis: add 2–5 paper tasks (agreement/contradiction, meta-summary) with citations to specific spans.
- Ontology normalization: standardize outputs to HGVS (cDNA/protein), PharmVar star-allele IDs/functions, DrugBank IDs, MedDRA/MeSH/HPO for phenotypes; evaluate exact and canonical matches.
- Robustness suite: paraphrases; numeric/format perturbations (1e‑3 vs 0.001 vs <.001); synonym swaps; rsID digit transpositions; near-miss star alleles; HLA formatting variants; “none of the options” ablations.
- Hidden test + eval server: private test set, submission API, fixed seeds, token budgets to prevent overfitting and ensure comparability.
- Human QC: expand expert annotations (32 → 200+), double-annotate with adjudication; report inter-annotator agreement per task.

## Reasoning-Focused Enhancements (Workshop Fit)

- Process supervision: add intermediate labels for chained tasks (e.g., which sentence supports the claim, which numeric value/inequality, which table cell), and score process steps separately from final answers.
- Justification fidelity: require citing evidence spans; evaluate answer–evidence entailment and penalize hallucinated citations.
- Counterfactual and negation control: expand negative/flip families and measure sensitivity to controlled edits (variant/drug/phenotype swaps, effect direction flips) with fine-grained tags.
- Tool-use track: optional programmatic reasoning (regex/table parsers) or retrieval-augmented chains; compare to pure LLM track under identical constraints.
- Consistency metrics: within-paper cross-question consistency checks and contradiction penalties across turns.

## Dataset & Labels

- Scale/diversity: broaden PMC OA coverage; include supplements where allowed; ensure representation of major PGx genes and drug classes.
- Unanswerable detection: explicitly include “not reported/insufficient evidence” with strong negatives and evaluate abstention.
- Table/figure fidelity: preserve table structure (markdown/HTML) and cell coordinates; optionally figure OCR for critical elements.
- Context metadata: enrich with study type, cohort size, ancestry, multiple testing correction; evaluate extraction and downstream use.

## Tasks

- Structured claim extraction: require tuples (variant, gene, drug, phenotype, effect direction, effect magnitude/estimate, p‑value/CI, study type, population, evidence spans).
- Study quality appraisal: risk-of-bias/style features (randomization, blinding, power) with categorical labels and short supporting spans.
- Cross-paper contradiction detection: for a set of papers, identify agreement vs conflict and summarize with citations and strength of evidence.
- Temporal update reasoning: given older vs newer papers, decide if guidance should change; cite both sides.

## Evaluation & Metrics

- Multi-level scoring: micro/macro by paper and by entity type; hierarchical credit (family→exact, e.g., CYP2C19*2 vs CYP2C19*2A).
- Numeric grading: inequality/CI-aware parsers with calibrated tolerance; unit normalization; partial credit for close matches.
- Calibration/abstention: coverage–accuracy curves, ECE/Brier; allow “abstain” with penalties; require per-answer confidence.
- Consistency checks: penalize intra-document contradictions across turns and cross-question inconsistencies.
- Efficiency: report tokens, latency, and cost; include Pareto front for accuracy vs cost.
- Variance: mandate ≥3 runs with fixed seeds; report mean±std on leaderboard.

## Submission Package (Workshop Repro Checklist)

- Code: inference + scoring scripts; pinned env (`pixi.lock`), model configs, seeds, and hardware details.
- Data: training/public dev splits, hidden test interface; documented licenses and data card.
- Prompts/parsers: versioned prompt templates; strict parsers; grader definitions and unit tests.
- Baselines: open small (e.g., Llama‑3.1‑8B‑Long), mid, reasoning model; deterministic rule-based numeric baseline.
- Reports: evaluation logs, bootstrap CIs, calibration curves, and cost/latency metrics.
- Ethics & limitations: broader impact statement; non-clinical-use disclaimer; provenance and bias notes.

## Robustness & Generalization

- Lexical/format adversaries: programmatic perturbations for numbers, unicode, Greek letters, hyphens/dashes, separators.
- Near-miss entities: curated distractors (rs123456 vs rs123465; HLA‑B*58:01 vs HLA‑B*58:1; CYP2C9*3 vs CYP2C9*13).
- OOD entity splits: hold out novel drugs/variants; evaluate compositional generalization (new drug × known gene).
- Tracks: closed-book (paper context only) vs tool-augmented; disallow web access in closed-book.

## Reproducibility & Governance

- Sealed prompts/parsers: version all prompts and graders; publish templates; pin env via `pixi.lock`.
- Data cards: document licenses, construction process, coverage, and known biases/limitations.
- Ethics: clinical non-use disclaimer; confirm no PHI; respect PMC OA licenses.
- Leaderboard policy: submission schema, compute caps, disclosure of tools, prohibition on training on hidden test.

## Baselines & Artifacts

- Diverse baselines: include a small open baseline (e.g., Llama‑3.1‑8B‑Long), a mid-size, and a reasoning model; add a rule-based numeric extractor as a sanity baseline.
- Reference implementations: canonical prompts, strict parsers, and a lightweight RAG baseline; scripts to run per-track and produce standardized summaries.
- Error analysis pack: auto-export failure cases with paper snippets, predicted vs gold spans, and diffs.

## Suggested Phased Plan

### Phase 1 (2–3 weeks)

- Add evidence-span fields to schemas and span-IoU scoring for claim tasks.
- Upgrade numeric graders (inequalities, CIs, units) and introduce abstention + calibration metrics.
- Create sealed test split; finalize leaderboard format and submission schema.

### Phase 2 (4–6 weeks)

- Build OOD/time-based/entity-holdout splits and robustness perturbation suite.
- Add ontology normalization to HGVS/PharmVar/DrugBank/MedDRA; expand annotations to ~100 papers with dual-review adjudication.

### Phase 3 (6–10 weeks)

- Introduce multi-document synthesis tasks and study quality appraisal.
- Add contradiction detection and temporal update reasoning.
- Implement evaluation server; report cost/latency alongside accuracy.

## Milestones to Workshop Deadline (Relative)

- T−12 weeks: Release v0.2 (evidence spans, upgraded graders), announce tracks and submission schema; publish small dev leaderboard with baselines.
- T−8 weeks: Ship OOD/time/entity-holdout splits; release robustness suite; add process-supervision scoring for chains; update docs and data card.
- T−4 weeks: Freeze prompts/parsers; open eval server for dry runs; publish baseline variance and calibration results; finalize ethics/limitations.
- T−2 weeks: Cut candidate submission; run full baselines on hidden test; prepare paper with ablations and confidence intervals.
- T−0: Submit workshop paper + artifacts; keep hidden test sealed; continue accepting leaderboard submissions post-workshop.

## Immediate Next Steps (Engineering Checklist)

- Schema updates: add `evidence_spans` to relevant tasks; version dataset (`v0.2`).
- Scorers: implement span-IoU; refactor numeric parsers to shared utils with unit handling.
- Split generators: time-split by publication year; entity-holdout by (gene, drug, variant, phenotype); OOD config YAMLs.
- Perturbation scripts: numeric/format/synonym/near-miss generators; save paired original/perturbed items.
- Leaderboard stub: JSON submission schema; result validator; basic static site or README-based board.
- Annotation expansion: recruit 2nd annotator; set up adjudication protocol; compute and report IAA.

## Ownership & Versioning

- Track changes in `docs/CHANGELOG.md` and tag dataset releases (e.g., `clinpgx-qa-v0.2`).
- Freeze prompts and graders per release; keep hidden test stable across leaderboard seasons.
