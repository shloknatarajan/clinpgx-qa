"""NeurIPS multi-turn QA chain generator for ClinPGx pharmacogenomics dataset.

Generates ~100k high-quality, transferable multi-turn QA chains across five
chain families: claim verification, hard negatives, summary aggregation,
cross-field consistency, and temporal reasoning.

Evidence localization is provenance/modality/aggregation-level,
NOT sentence-level spans.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel

# ═════════════════════════════════════════════════════════════════════════════
# Type Aliases & Constants
# ═════════════════════════════════════════════════════════════════════════════

ReasoningType = Literal[
    "claim_verification",
    "evidence_provenance_localization",
    "statistical_extraction",
    "counterfactual_evaluation",
    "objective_evaluation",
    "cross_field_consistency",
    "evidence_aggregation",
    "temporal_reasoning",
]

EvidenceGranularity = Literal[
    "document", "modality", "aggregation", "temporal", "db_record", "heuristic",
]

NegativeTypeLit = Literal[
    "syntactic_flip",
    "not_reported_variant_swap",
    "not_reported_drug_swap",
    "not_reported_population_swap",
    "not_reported_phenotype_swap",
    "contradiction_from_structured_fields",
]

ChainFamily = Literal[
    "A_claim→modality→stat→eval",
    "B_claim→presence_absence",
    "C_summary→aggregate→strength",
    "D_cross_field_consistency",
    "E_temporal_summary_history",
]

AmbiguityRisk = Literal["low", "medium", "high"]

FAMILY_PROPORTIONS: dict[str, float] = {
    "A": 0.60, "B": 0.20, "C": 0.15, "D": 0.03, "E": 0.02,
}

EVAL_STAT_FIELDS = [
    "P Value", "Ratio Stat", "Confidence Interval", "Study Cases", "Study Controls",
]

# ═════════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═════════════════════════════════════════════════════════════════════════════


class Turn(BaseModel):
    turn: int
    reasoning_type: ReasoningType
    question: str
    answer: str | bool | int | float | list[str]
    answer_source_fields: list[str]
    evidence_required: bool
    evidence_granularity: EvidenceGranularity
    negative_type: NegativeTypeLit | None = None
    ambiguity_risk: AmbiguityRisk = "low"
    metadata: dict[str, Any] = {}


class QuestionChain(BaseModel):
    chain_id: str
    chain_family: ChainFamily
    pmid: str | None = None
    variant_annotation_id: str | None = None
    study_parameters_id: str | None = None
    summary_annotation_id: str | None = None
    source_tables: list[str]
    capability_tags: list[str]
    num_turns: int
    has_negative: bool
    turns: list[Turn]
    context: str | None = None


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═════════════════════════════════════════════════════════════════════════════


def load_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    logger.info(f"Loaded {len(rows):,} rows from {path.name}")
    return rows


def build_index(rows: list[dict], key: str) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        k = r.get(key, "").strip()
        if k:
            idx[k].append(r)
    return dict(idx)


def build_pmid_paper_map(papers_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if not papers_dir.exists():
        logger.warning(f"Papers directory not found: {papers_dir}")
        return mapping
    pmid_pat = re.compile(r"\*\*PMID:\*\*\s*(\d+)")
    for md in sorted(papers_dir.glob("*.md")):
        try:
            head = md.read_text(encoding="utf-8")[:2000]
            m = pmid_pat.search(head)
            if m:
                mapping[m.group(1)] = md
        except Exception:
            continue
    logger.info(f"Mapped {len(mapping):,} PMIDs to paper files")
    return mapping


def load_all_data(base: Path) -> dict[str, Any]:
    vd = base / "variantAnnotations"
    sd = base / "summaryAnnotations"

    var_drug = load_tsv(vd / "var_drug_ann.tsv")
    var_pheno = load_tsv(vd / "var_pheno_ann.tsv")
    var_fa = load_tsv(vd / "var_fa_ann.tsv")
    sp = load_tsv(vd / "study_parameters.tsv")

    summaries = load_tsv(sd / "summary_annotations.tsv")
    evidence = load_tsv(sd / "summary_ann_evidence.tsv")
    history = load_tsv(sd / "summary_ann_history.tsv")

    all_anns: list[tuple[dict, str]] = (
        [(r, "var_drug_ann") for r in var_drug]
        + [(r, "var_pheno_ann") for r in var_pheno]
        + [(r, "var_fa_ann") for r in var_fa]
    )

    return {
        "all_anns": all_anns,
        "sp_index": build_index(sp, "Variant Annotation ID"),
        "summaries": summaries,
        "evidence_index": build_index(evidence, "Summary Annotation ID"),
        "history_index": build_index(history, "Summary Annotation ID"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Parsing Utilities
# ═════════════════════════════════════════════════════════════════════════════


def parse_p_value(raw: str) -> tuple[str, float | None]:
    raw = raw.strip()
    if not raw:
        return ("", None)
    for pfx in ("= ", "< ", "> ", "≤ ", "≥ "):
        if raw.startswith(pfx):
            try:
                return (pfx.strip(), float(raw[len(pfx) :]))
            except ValueError:
                return (pfx.strip(), None)
    try:
        return ("=", float(raw))
    except ValueError:
        return ("", None)


def is_p_significant(raw: str, alpha: float = 0.05) -> bool | None:
    comp, val = parse_p_value(raw)
    if val is None:
        return None
    if comp in ("<", "≤"):
        return val <= alpha
    if comp == "=":
        return val < alpha
    if comp in (">", "≥"):
        return False if val >= alpha else None
    return None


def parse_ci(start_s: str, stop_s: str) -> tuple[float | None, float | None]:
    def _f(s: str) -> float | None:
        s = s.strip()
        try:
            return float(s) if s else None
        except ValueError:
            return None

    return _f(start_s), _f(stop_s)


def safe_int(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def safe_float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Negative Generation
# ═════════════════════════════════════════════════════════════════════════════


def build_swap_pools(
    all_anns: list[tuple[dict, str]], sp_index: dict[str, list[dict]],
) -> dict[str, Any]:
    by_gene_pheno: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_gene_variant: dict[tuple[str, str], list[dict]] = defaultdict(list)
    phenotypes: set[str] = set()
    biogeo: set[str] = set()

    for ann, _ in all_anns:
        g = ann.get("Gene", "").strip()
        p = ann.get("Phenotype Category", "").strip()
        v = ann.get("Variant/Haplotypes", "").strip()
        if g and p:
            by_gene_pheno[(g, p)].append(ann)
        if g and v:
            by_gene_variant[(g, v)].append(ann)
        if p:
            phenotypes.add(p)

    for rows in sp_index.values():
        for sp in rows:
            bg = sp.get("Biogeographical Groups", "").strip()
            if bg and bg.lower() not in ("", "unknown"):
                biogeo.add(bg)

    return {
        "by_gene_pheno": dict(by_gene_pheno),
        "by_gene_variant": dict(by_gene_variant),
        "phenotypes": sorted(phenotypes),
        "biogeo": sorted(biogeo),
    }


def make_negative(
    ann: dict, source: str, pools: dict, rng: random.Random,
) -> tuple[str, NegativeTypeLit] | None:
    gene = ann.get("Gene", "").strip()
    pheno = ann.get("Phenotype Category", "").strip()
    variant = ann.get("Variant/Haplotypes", "").strip()
    drug = ann.get("Drug(s)", "").strip()
    pmid = ann.get("PMID", "").strip()
    sentence = ann.get("Sentence", "").strip()

    options: list[tuple[str, NegativeTypeLit]] = []

    # Variant swap
    gp_key = (gene, pheno)
    if gp_key in pools["by_gene_pheno"]:
        alts = list(
            {
                a["Variant/Haplotypes"].strip()
                for a in pools["by_gene_pheno"][gp_key]
                if a["Variant/Haplotypes"].strip() != variant
                and a.get("PMID", "").strip() != pmid
            }
        )
        if alts:
            alt = rng.choice(alts)
            if variant and variant in sentence:
                neg = sentence.replace(variant, alt, 1)
            else:
                neg = f"In PMID {pmid}, {alt} is associated with {pheno} outcomes for {drug}."
            options.append((neg, "not_reported_variant_swap"))

    # Drug swap
    gv_key = (gene, variant)
    if gv_key in pools["by_gene_variant"]:
        alt_drugs = list(
            {
                a["Drug(s)"].strip()
                for a in pools["by_gene_variant"][gv_key]
                if a["Drug(s)"].strip() and a["Drug(s)"].strip() != drug
            }
        )
        if alt_drugs:
            ad = rng.choice(alt_drugs)
            if drug and drug in sentence:
                neg = sentence.replace(drug, ad, 1)
            else:
                neg = f"In PMID {pmid}, {variant} is associated with {pheno} outcomes for {ad}."
            options.append((neg, "not_reported_drug_swap"))

    # Phenotype swap
    alt_phenos = [p for p in pools["phenotypes"] if p != pheno]
    if alt_phenos:
        ap = rng.choice(alt_phenos)
        neg = f"In PMID {pmid}, {variant} is associated with {ap} outcomes for {drug}."
        options.append((neg, "not_reported_phenotype_swap"))

    if not options:
        return None
    return rng.choice(options)


# ═════════════════════════════════════════════════════════════════════════════
# Helper: stat availability checks
# ═════════════════════════════════════════════════════════════════════════════


def _stat_available(sp: dict, field: str) -> bool:
    if field == "Confidence Interval":
        return bool(
            sp.get("Confidence Interval Start", "").strip()
            and sp.get("Confidence Interval Stop", "").strip()
        )
    if field == "Ratio Stat":
        return bool(
            sp.get("Ratio Stat", "").strip()
            and sp.get("Ratio Stat Type", "").strip()
        )
    return bool(sp.get(field, "").strip())


def _eval_possible(sp: dict, field: str) -> bool:
    if field == "P Value":
        return is_p_significant(sp.get("P Value", "").strip()) is not None
    if field in ("Ratio Stat", "Confidence Interval"):
        ci_s, ci_e = parse_ci(
            sp.get("Confidence Interval Start", ""),
            sp.get("Confidence Interval Stop", ""),
        )
        return ci_s is not None and ci_e is not None
    if field in ("Study Cases", "Study Controls"):
        return safe_int(sp.get(field, "")) is not None
    return False


def _get_label(ann: dict) -> str:
    is_assoc = ann.get("Is/Is Not associated", "").strip().lower()
    sig = ann.get("Significance", "").strip().lower()
    if "not associated" in is_assoc or sig == "no":
        return "contradicted"
    return "supported"


def _get_modality(source: str) -> str:
    return "functional_assay" if source == "var_fa_ann" else "clinical_association"


def _read_paper(
    pmid: str, paper_map: dict[str, Path], include_text: bool,
) -> str | None:
    if not include_text:
        return None
    p = paper_map.get(pmid)
    if p is None:
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Candidate Enumeration
# ═════════════════════════════════════════════════════════════════════════════


def enumerate_a(
    all_anns: list[tuple[dict, str]], sp_index: dict[str, list[dict]],
) -> list[tuple]:
    candidates = []
    for ann, source in all_anns:
        ann_id = ann.get("Variant Annotation ID", "").strip()
        sentence = ann.get("Sentence", "").strip()
        pmid = ann.get("PMID", "").strip()
        if not (ann_id and sentence and pmid):
            continue
        for sp in sp_index.get(ann_id, []):
            for field in EVAL_STAT_FIELDS:
                if _stat_available(sp, field) and _eval_possible(sp, field):
                    candidates.append((ann, source, sp, field))
    logger.info(f"Family A: {len(candidates):,} candidates")
    return candidates


def enumerate_b(
    all_anns: list[tuple[dict, str]], pools: dict, rng: random.Random,
) -> list[tuple]:
    candidates = []
    for ann, source in all_anns:
        sentence = ann.get("Sentence", "").strip()
        pmid = ann.get("PMID", "").strip()
        if not (sentence and pmid):
            continue
        result = make_negative(ann, source, pools, rng)
        if result is not None:
            neg_sent, neg_type = result
            candidates.append((ann, source, neg_sent, neg_type))
    logger.info(f"Family B: {len(candidates):,} candidates")
    return candidates


def enumerate_c(
    summaries: list[dict], evidence_index: dict[str, list[dict]],
) -> list[tuple]:
    candidates = []
    for s in summaries:
        sid = s.get("Summary Annotation ID", "").strip()
        loe = s.get("Level of Evidence", "").strip()
        pmid_count = s.get("PMID Count", "").strip()
        if not (sid and loe and pmid_count):
            continue
        ev_rows = evidence_index.get(sid, [])
        candidates.append((s, ev_rows, False))
        ev_types = {r.get("Evidence Type", "").strip() for r in ev_rows} - {""}
        if ev_types:
            candidates.append((s, ev_rows, True))
    logger.info(f"Family C: {len(candidates):,} candidates")
    return candidates


def enumerate_d(
    all_anns: list[tuple[dict, str]], sp_index: dict[str, list[dict]],
) -> list[tuple]:
    candidates = []
    for ann, source in all_anns:
        ann_id = ann.get("Variant Annotation ID", "").strip()
        sentence = ann.get("Sentence", "").strip()
        pmid = ann.get("PMID", "").strip()
        direction = ann.get("Direction of effect", "").strip().lower()
        if not (ann_id and sentence and pmid and direction):
            continue
        if direction not in ("increased", "decreased"):
            continue
        side_eff = ann.get("Side effect/efficacy/other", "").strip().lower()
        pd_pk = ann.get("PD/PK terms", "").strip().lower()
        has_risk = side_eff and ("risk of" in side_eff or "likelihood of" in side_eff)
        has_monotonic_pk = pd_pk and any(
            t in pd_pk
            for t in ("metabolism", "clearance", "exposure", "concentration", "response")
        )
        if not (has_risk or has_monotonic_pk):
            continue
        for sp in sp_index.get(ann_id, []):
            rs = sp.get("Ratio Stat", "").strip()
            rs_type = sp.get("Ratio Stat Type", "").strip()
            if rs and rs_type and rs_type in ("OR", "RR", "HR"):
                if safe_float(rs) is not None:
                    candidates.append((ann, source, sp))
    logger.info(f"Family D: {len(candidates):,} candidates")
    return candidates


def enumerate_e(
    summaries: list[dict],
    history_index: dict[str, list[dict]],
    rng: random.Random,
) -> list[tuple]:
    candidates = []
    for s in summaries:
        sid = s.get("Summary Annotation ID", "").strip()
        if not sid:
            continue
        h_rows = history_index.get(sid, [])
        if not h_rows:
            continue
        year = rng.randint(2018, 2022)
        candidates.append((s, h_rows, year))
    logger.info(f"Family E: {len(candidates):,} candidates")
    return candidates


# ═════════════════════════════════════════════════════════════════════════════
# Chain Builders
# ═════════════════════════════════════════════════════════════════════════════


def _build_stat_q(
    sp: dict, field: str, pmid: str, sp_id: str,
) -> tuple[str, str | int | float, list[str]]:
    if field == "P Value":
        return (
            f"What p-value is reported in Study Parameters {sp_id} (PMID {pmid}) for this association?",
            sp["P Value"].strip(),
            ["P Value"],
        )
    if field == "Ratio Stat":
        rs_type = sp["Ratio Stat Type"].strip()
        return (
            f"What {rs_type} is reported in Study Parameters {sp_id} (PMID {pmid})?",
            sp["Ratio Stat"].strip(),
            ["Ratio Stat", "Ratio Stat Type"],
        )
    if field == "Confidence Interval":
        cs = sp["Confidence Interval Start"].strip()
        ce = sp["Confidence Interval Stop"].strip()
        return (
            f"What confidence interval is reported in Study Parameters {sp_id} (PMID {pmid})?",
            f"{cs}\u2013{ce}",
            ["Confidence Interval Start", "Confidence Interval Stop"],
        )
    if field == "Study Cases":
        raw = sp["Study Cases"].strip()
        val = safe_int(raw)
        return (
            f"How many cases are reported in Study Parameters {sp_id} (PMID {pmid})?",
            val if val is not None else raw,
            ["Study Cases"],
        )
    if field == "Study Controls":
        raw = sp["Study Controls"].strip()
        val = safe_int(raw)
        return (
            f"How many controls are reported in Study Parameters {sp_id} (PMID {pmid})?",
            val if val is not None else raw,
            ["Study Controls"],
        )
    raise ValueError(f"Unknown stat field: {field}")


def _build_eval(sp: dict, field: str, turn_num: int, rng: random.Random) -> Turn:
    if field == "P Value":
        sig_05 = is_p_significant(sp["P Value"].strip(), 0.05)
        if rng.random() < 0.3:
            sig_01 = is_p_significant(sp["P Value"].strip(), 0.01)
            if sig_01 is not None:
                return Turn(
                    turn=turn_num,
                    reasoning_type="counterfactual_evaluation",
                    question="Would this association remain significant at alpha = 0.01?",
                    answer=sig_01,
                    answer_source_fields=["P Value"],
                    evidence_required=True,
                    evidence_granularity="db_record",
                    metadata={"alpha": 0.01},
                )
        return Turn(
            turn=turn_num,
            reasoning_type="objective_evaluation",
            question="Is this finding statistically significant at alpha = 0.05?",
            answer=sig_05,
            answer_source_fields=["P Value"],
            evidence_required=True,
            evidence_granularity="db_record",
        )
    if field in ("Ratio Stat", "Confidence Interval"):
        ci_s, ci_e = parse_ci(
            sp.get("Confidence Interval Start", ""),
            sp.get("Confidence Interval Stop", ""),
        )
        excludes = not (ci_s <= 1.0 <= ci_e)  # type: ignore[operator]
        rs_type = sp.get("Ratio Stat Type", "ratio").strip() or "ratio"
        return Turn(
            turn=turn_num,
            reasoning_type="objective_evaluation",
            question=f"Does the confidence interval for the {rs_type} exclude the null value of 1.0?",
            answer=excludes,
            answer_source_fields=["Confidence Interval Start", "Confidence Interval Stop"],
            evidence_required=True,
            evidence_granularity="db_record",
        )
    if field in ("Study Cases", "Study Controls"):
        n = safe_int(sp.get(field, ""))
        return Turn(
            turn=turn_num,
            reasoning_type="objective_evaluation",
            question=f"Does this meet the dataset minimum of N \u2265 30 for {field.lower()}?",
            answer=n >= 30,  # type: ignore[operator]
            answer_source_fields=[field],
            evidence_required=True,
            evidence_granularity="db_record",
        )
    raise ValueError(f"No eval for field: {field}")


def build_chain_a(
    cand: tuple,
    chain_id: str,
    rng: random.Random,
    paper_map: dict[str, Path],
    include_text: bool,
) -> QuestionChain:
    ann, source, sp, field = cand
    pmid = ann["PMID"].strip()
    ann_id = ann["Variant Annotation ID"].strip()
    sp_id = sp["Study Parameters ID"].strip()
    sentence = ann["Sentence"].strip()
    label = _get_label(ann)
    modality = _get_modality(source)

    t1 = Turn(
        turn=1,
        reasoning_type="claim_verification",
        question=f"Based on PMID {pmid}, is the following pharmacogenomic claim supported or contradicted: {sentence}",
        answer=label,
        answer_source_fields=["Sentence", "Is/Is Not associated", "Significance"],
        evidence_required=True,
        evidence_granularity="document",
        metadata={"Variant Annotation ID": ann_id},
    )
    t2 = Turn(
        turn=2,
        reasoning_type="evidence_provenance_localization",
        question="What type of evidence does this annotation represent: clinical association or functional assay?",
        answer=modality,
        answer_source_fields=["source_file"],
        evidence_required=False,
        evidence_granularity="modality",
    )
    q3, a3, src3 = _build_stat_q(sp, field, pmid, sp_id)
    t3 = Turn(
        turn=3,
        reasoning_type="statistical_extraction",
        question=q3,
        answer=a3,
        answer_source_fields=src3,
        evidence_required=True,
        evidence_granularity="db_record",
        metadata={"Study Parameters ID": sp_id},
    )
    t4 = _build_eval(sp, field, 4, rng)

    tags = ["claim_verification", "evidence_provenance", "numeric_extraction"]
    if t4.reasoning_type == "counterfactual_evaluation":
        tags.append("counterfactual_reasoning")
    else:
        tags.append("objective_statistical_reasoning")

    return QuestionChain(
        chain_id=chain_id,
        chain_family="A_claim\u2192modality\u2192stat\u2192eval",
        pmid=pmid,
        variant_annotation_id=ann_id,
        study_parameters_id=sp_id,
        source_tables=[source, "study_parameters"],
        capability_tags=tags,
        num_turns=4,
        has_negative=False,
        turns=[t1, t2, t3, t4],
        context=_read_paper(pmid, paper_map, include_text),
    )


def build_chain_b(
    cand: tuple,
    chain_id: str,
    rng: random.Random,
    paper_map: dict[str, Path],
    include_text: bool,
) -> QuestionChain:
    ann, source, neg_sentence, neg_type = cand
    pmid = ann["PMID"].strip()
    ann_id = ann.get("Variant Annotation ID", "").strip()

    t1 = Turn(
        turn=1,
        reasoning_type="claim_verification",
        question=(
            f"Based on PMID {pmid}, is the following pharmacogenomic claim "
            f"supported, contradicted, or not reported: {neg_sentence}"
        ),
        answer="not_reported",
        answer_source_fields=["Sentence"],
        evidence_required=True,
        evidence_granularity="document",
        negative_type=neg_type,
        metadata={"Variant Annotation ID": ann_id, "original_pmid": pmid},
    )
    t2 = Turn(
        turn=2,
        reasoning_type="evidence_provenance_localization",
        question=f"Does PMID {pmid} report any quantitative statistic (p-value/OR/CI) for this claim?",
        answer="no",
        answer_source_fields=["study_parameters"],
        evidence_required=False,
        evidence_granularity="db_record",
    )

    return QuestionChain(
        chain_id=chain_id,
        chain_family="B_claim\u2192presence_absence",
        pmid=pmid,
        variant_annotation_id=ann_id,
        source_tables=[source],
        capability_tags=["claim_verification", "evidence_provenance", "negation_scope"],
        num_turns=2,
        has_negative=True,
        turns=[t1, t2],
        context=_read_paper(pmid, paper_map, include_text),
    )


def build_chain_c(cand: tuple, chain_id: str, rng: random.Random) -> QuestionChain:
    summary, ev_rows, with_turn4 = cand
    sid = summary["Summary Annotation ID"].strip()
    variant = summary.get("Variant/Haplotypes", "").strip()
    gene = summary.get("Gene", "").strip()
    drug = summary.get("Drug(s)", "").strip()
    loe = summary["Level of Evidence"].strip()
    pmid_count_raw = summary["PMID Count"].strip()
    pmid_count = safe_int(pmid_count_raw) or 0

    desc = f"{variant} ({gene}) and {drug}" if drug else f"{variant} ({gene})"

    t1 = Turn(
        turn=1,
        reasoning_type="evidence_aggregation",
        question=f"What is the ClinPGx Level of Evidence (1A\u20134) for the summary annotation on {desc}?",
        answer=loe,
        answer_source_fields=["Level of Evidence"],
        evidence_required=False,
        evidence_granularity="aggregation",
        metadata={"Summary Annotation ID": sid},
    )
    t2 = Turn(
        turn=2,
        reasoning_type="evidence_aggregation",
        question=f"How many PMIDs support the summary annotation for {desc}?",
        answer=pmid_count,
        answer_source_fields=["PMID Count"],
        evidence_required=False,
        evidence_granularity="aggregation",
    )
    t3 = Turn(
        turn=3,
        reasoning_type="objective_evaluation",
        question="Is this summary annotation supported by more than one PMID?",
        answer=pmid_count > 1,
        answer_source_fields=["PMID Count"],
        evidence_required=False,
        evidence_granularity="aggregation",
    )

    turns = [t1, t2, t3]
    tags = ["aggregation_reasoning"]

    if with_turn4:
        ev_types = sorted(
            {r.get("Evidence Type", "").strip() for r in ev_rows} - {""}
        )
        type_map = {
            "Variant Drug Annotation": "clinical association",
            "Variant Phenotype Annotation": "clinical association",
            "Variant Functional Assay Annotation": "functional assay",
            "Guideline Annotation": "guideline",
            "Label Annotation": "label",
        }
        mapped = sorted({type_map.get(t, t) for t in ev_types})
        t4 = Turn(
            turn=4,
            reasoning_type="evidence_provenance_localization",
            question=f"Which evidence types contribute to this summary annotation for {desc}?",
            answer=mapped,
            answer_source_fields=["Evidence Type"],
            evidence_required=False,
            evidence_granularity="modality",
        )
        turns.append(t4)
        tags.append("evidence_provenance")

    return QuestionChain(
        chain_id=chain_id,
        chain_family="C_summary\u2192aggregate\u2192strength",
        summary_annotation_id=sid,
        source_tables=["summary_annotations", "summary_ann_evidence"],
        capability_tags=tags,
        num_turns=len(turns),
        has_negative=False,
        turns=turns,
    )


def build_chain_d(
    cand: tuple,
    chain_id: str,
    rng: random.Random,
    paper_map: dict[str, Path],
    include_text: bool,
) -> QuestionChain:
    ann, source, sp = cand
    pmid = ann["PMID"].strip()
    ann_id = ann["Variant Annotation ID"].strip()
    sp_id = sp["Study Parameters ID"].strip()
    direction = ann["Direction of effect"].strip().lower()
    rs_val = safe_float(sp["Ratio Stat"].strip())
    rs_type = sp["Ratio Stat Type"].strip()

    t1 = Turn(
        turn=1,
        reasoning_type="statistical_extraction",
        question=f"What {rs_type} is reported in Study Parameters {sp_id} (PMID {pmid})?",
        answer=f"{rs_val} ({rs_type})",
        answer_source_fields=["Ratio Stat", "Ratio Stat Type"],
        evidence_required=True,
        evidence_granularity="db_record",
        metadata={"Study Parameters ID": sp_id},
    )
    t2 = Turn(
        turn=2,
        reasoning_type="statistical_extraction",
        question=f"What direction of effect is annotated for this association in PMID {pmid}?",
        answer=direction,
        answer_source_fields=["Direction of effect"],
        evidence_required=True,
        evidence_granularity="db_record",
    )

    if rs_type in ("OR", "RR", "HR"):
        implies_increased = rs_val > 1.0  # type: ignore[operator]
    else:
        implies_increased = rs_val > 1.0  # type: ignore[operator]
    annotated_increased = direction == "increased"
    consistent = implies_increased == annotated_increased

    t3 = Turn(
        turn=3,
        reasoning_type="cross_field_consistency",
        question=(
            f"Is the annotated direction of effect ('{direction}') consistent "
            f"with the {rs_type} of {rs_val}?"
        ),
        answer=consistent,
        answer_source_fields=["Direction of effect", "Ratio Stat", "Ratio Stat Type"],
        evidence_required=False,
        evidence_granularity="db_record",
        ambiguity_risk="medium",
    )

    return QuestionChain(
        chain_id=chain_id,
        chain_family="D_cross_field_consistency",
        pmid=pmid,
        variant_annotation_id=ann_id,
        study_parameters_id=sp_id,
        source_tables=[source, "study_parameters"],
        capability_tags=["consistency_reasoning", "numeric_extraction"],
        num_turns=3,
        has_negative=False,
        turns=[t1, t2, t3],
        context=_read_paper(pmid, paper_map, include_text),
    )


def build_chain_e(cand: tuple, chain_id: str, rng: random.Random) -> QuestionChain:
    summary, h_rows, threshold_year = cand
    sid = summary["Summary Annotation ID"].strip()
    variant = summary.get("Variant/Haplotypes", "").strip()
    gene = summary.get("Gene", "").strip()

    latest_str = summary.get("Latest History Date (YYYY-MM-DD)", "").strip()
    if latest_str:
        try:
            latest_year = int(latest_str[:4])
        except ValueError:
            latest_year = 0
    else:
        dates = []
        for h in h_rows:
            d = h.get("Date (YYYY-MM-DD)", "").strip()
            if d:
                try:
                    dates.append(int(d[:4]))
                except ValueError:
                    pass
        latest_year = max(dates) if dates else 0

    updated_after = latest_year > threshold_year

    update_count = sum(
        1
        for h in h_rows
        if h.get("Type", "").strip().lower() in ("update", "correction")
    )

    desc = f"{variant} ({gene})"
    t1 = Turn(
        turn=1,
        reasoning_type="temporal_reasoning",
        question=f"Was the summary annotation for {desc} updated after {threshold_year}?",
        answer=updated_after,
        answer_source_fields=["Latest History Date (YYYY-MM-DD)"],
        evidence_required=False,
        evidence_granularity="temporal",
        metadata={"Summary Annotation ID": sid, "threshold_year": threshold_year},
    )
    t2 = Turn(
        turn=2,
        reasoning_type="temporal_reasoning",
        question=f"How many updates or corrections are recorded for the summary annotation on {desc}?",
        answer=update_count,
        answer_source_fields=["Type"],
        evidence_required=False,
        evidence_granularity="temporal",
    )

    return QuestionChain(
        chain_id=chain_id,
        chain_family="E_temporal_summary_history",
        summary_annotation_id=sid,
        source_tables=["summary_annotations", "summary_ann_history"],
        capability_tags=["temporal_reasoning"],
        num_turns=2,
        has_negative=False,
        turns=[t1, t2],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Sampling & Orchestration
# ═════════════════════════════════════════════════════════════════════════════


def allocate_targets(
    counts: dict[str, int], total_target: int,
) -> dict[str, int]:
    targets: dict[str, int] = {}
    shortfall = 0
    for fam, prop in FAMILY_PROPORTIONS.items():
        ideal = int(total_target * prop)
        avail = counts.get(fam, 0)
        actual = min(ideal, avail)
        targets[fam] = actual
        shortfall += ideal - actual

    if shortfall > 0:
        redistribute_to = ["A", "B", "C"]
        redis_total = sum(FAMILY_PROPORTIONS[f] for f in redistribute_to)
        for fam in redistribute_to:
            share = int(shortfall * FAMILY_PROPORTIONS[fam] / redis_total)
            avail = counts.get(fam, 0)
            targets[fam] = min(targets[fam] + share, avail)

    return targets


def generate_all_chains(
    data: dict[str, Any],
    seed: int,
    target_chains: int,
    include_text: bool,
    paper_map: dict[str, Path],
    include_summary: bool = True,
    include_temporal: bool = True,
) -> list[QuestionChain]:
    rng = random.Random(seed)

    all_anns = data["all_anns"]
    sp_index = data["sp_index"]
    summaries = data["summaries"]
    evidence_index = data["evidence_index"]
    history_index = data["history_index"]

    pools = build_swap_pools(all_anns, sp_index)

    logger.info("Enumerating candidates...")
    cands_a = enumerate_a(all_anns, sp_index)
    cands_b = enumerate_b(all_anns, pools, rng)
    cands_c = enumerate_c(summaries, evidence_index) if include_summary else []
    cands_d = enumerate_d(all_anns, sp_index)
    cands_e = enumerate_e(summaries, history_index, rng) if include_temporal else []

    counts = {
        "A": len(cands_a),
        "B": len(cands_b),
        "C": len(cands_c),
        "D": len(cands_d),
        "E": len(cands_e),
    }
    logger.info(f"Candidate counts: {counts}")

    targets = allocate_targets(counts, target_chains)
    logger.info(f"Allocated targets: {targets}")

    chains: list[QuestionChain] = []
    chain_counter = 0

    def _sample_and_build(cands, target_n, builder, needs_paper=True):
        nonlocal chain_counter
        if not cands or target_n <= 0:
            return
        if target_n <= len(cands):
            selected = rng.sample(cands, target_n)
        else:
            selected = cands[:]
            while len(selected) < target_n:
                selected.extend(rng.choices(cands, k=min(target_n - len(selected), len(cands))))
            selected = selected[:target_n]

        for c in selected:
            cid = f"chain_{chain_counter:06d}"
            if needs_paper:
                chain = builder(c, cid, rng, paper_map, include_text)
            else:
                chain = builder(c, cid, rng)
            chains.append(chain)
            chain_counter += 1

    _sample_and_build(cands_a, targets["A"], build_chain_a, needs_paper=True)
    _sample_and_build(cands_b, targets["B"], build_chain_b, needs_paper=True)
    _sample_and_build(cands_c, targets["C"], build_chain_c, needs_paper=False)
    _sample_and_build(cands_d, targets["D"], build_chain_d, needs_paper=True)
    _sample_and_build(cands_e, targets["E"], build_chain_e, needs_paper=False)

    rng.shuffle(chains)
    for i, c in enumerate(chains):
        c.chain_id = f"chain_{i:06d}"

    logger.info(f"Generated {len(chains):,} chains total")
    return chains


# ═════════════════════════════════════════════════════════════════════════════
# Validation
# ═════════════════════════════════════════════════════════════════════════════


def validate_chains(chains: list[QuestionChain]) -> None:
    errors: list[str] = []
    for c in chains:
        # Contiguous turn numbers
        expected = list(range(1, c.num_turns + 1))
        actual = [t.turn for t in c.turns]
        if actual != expected:
            errors.append(f"{c.chain_id}: turns {actual} != expected {expected}")

        # Family B: no statistical_extraction turns
        if c.chain_family == "B_claim\u2192presence_absence":
            for t in c.turns:
                if t.reasoning_type == "statistical_extraction":
                    errors.append(f"{c.chain_id}: Family B has statistical_extraction turn")

        # Evaluation turns must have source fields
        for t in c.turns:
            if t.reasoning_type in ("objective_evaluation", "counterfactual_evaluation"):
                if not t.answer_source_fields:
                    errors.append(f"{c.chain_id} turn {t.turn}: eval with no source fields")

        # Negative chains
        if c.has_negative:
            neg_found = any(t.negative_type is not None for t in c.turns)
            if not neg_found:
                errors.append(f"{c.chain_id}: has_negative=True but no negative_type in turns")

        # PMID required for document grounding
        for t in c.turns:
            if t.evidence_granularity == "document" and not c.pmid:
                errors.append(f"{c.chain_id} turn {t.turn}: document grounding but no PMID")

    if errors:
        msg = f"Validation failed with {len(errors)} errors:\n" + "\n".join(errors[:20])
        raise ValueError(msg)
    logger.info(f"Validation passed for {len(chains):,} chains")


# ═════════════════════════════════════════════════════════════════════════════
# Stats & Output
# ═════════════════════════════════════════════════════════════════════════════


def compute_stats(chains: list[QuestionChain]) -> dict[str, Any]:
    family_dist = Counter(c.chain_family for c in chains)
    turns_dist = Counter(c.num_turns for c in chains)
    neg_type_dist: Counter[str] = Counter()
    cap_counter: Counter[str] = Counter()
    uses_full_text_count = 0

    for c in chains:
        for t in c.turns:
            if t.negative_type:
                neg_type_dist[t.negative_type] += 1
        for tag in c.capability_tags:
            cap_counter[tag] += 1
        if c.context is not None:
            uses_full_text_count += 1

    total = len(chains)
    return {
        "total_chains": total,
        "distribution_by_chain_family": dict(family_dist),
        "distribution_by_num_turns": dict(sorted(turns_dist.items())),
        "negative_type_breakdown": dict(neg_type_dist),
        "capability_tag_percent": {
            k: round(v / total * 100, 2) for k, v in cap_counter.most_common()
        },
        "uses_full_text_count": uses_full_text_count,
        "uses_full_text_percent": round(uses_full_text_count / total * 100, 2) if total else 0,
    }


def save_output(
    chains: list[QuestionChain],
    jsonl_path: Path,
    stats_path: Path,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in chains:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(chains):,} chains to {jsonl_path}")

    stats = compute_stats(chains)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved stats to {stats_path}")

    print(f"\n{'=' * 60}")
    print(f"NeurIPS Chain Generation Summary")
    print(f"{'=' * 60}")
    print(f"Total chains: {stats['total_chains']:,}")
    print(f"\nBy family:")
    for k, v in stats["distribution_by_chain_family"].items():
        print(f"  {k}: {v:,}")
    print(f"\nBy turns:")
    for k, v in stats["distribution_by_num_turns"].items():
        print(f"  {k} turns: {v:,}")
    print(f"\nNegative type breakdown:")
    for k, v in stats["negative_type_breakdown"].items():
        print(f"  {k}: {v:,}")
    print(f"\nCapability tags (% of chains):")
    for k, v in stats["capability_tag_percent"].items():
        print(f"  {k}: {v}%")
    print(f"\nUses full text: {stats['uses_full_text_count']:,} ({stats['uses_full_text_percent']}%)")
    print(f"{'=' * 60}\n")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NeurIPS multi-turn QA chains from ClinPGx data",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_chains", type=int, default=100_000)
    parser.add_argument("--include_summary", type=bool, default=True)
    parser.add_argument("--include_temporal", type=bool, default=True)
    parser.add_argument("--include_full_text", type=bool, default=True)
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="data/neurips_chained_questions.jsonl",
    )
    parser.add_argument(
        "--output_stats",
        type=str,
        default="data/neurips_chained_questions_stats.json",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
    )
    parser.add_argument(
        "--papers_dir",
        type=str,
        default="data/papers",
    )
    args = parser.parse_args()

    data = load_all_data(Path(args.data_dir))
    paper_map = build_pmid_paper_map(Path(args.papers_dir))

    chains = generate_all_chains(
        data=data,
        seed=args.seed,
        target_chains=args.target_chains,
        include_text=args.include_full_text,
        paper_map=paper_map,
        include_summary=args.include_summary,
        include_temporal=args.include_temporal,
    )

    validate_chains(chains)
    save_output(chains, Path(args.output_jsonl), Path(args.output_stats))


if __name__ == "__main__":
    main()
