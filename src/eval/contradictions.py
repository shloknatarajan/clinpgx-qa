"""
Detect self-contradictions in MCQ model responses within the same PMCID.

Three contradiction types:
  1. Standard vs NOTA: same annotation, paired standard/none_of_the_above questions
     disagree (e.g. model picks entity in standard but fails to pick "None" in NOTA).
  2. Distractor-as-correct: within the same PMCID + MCQ type, model picks entity X
     for question Q_i (where X is wrong) while X is the correct answer for a different
     question Q_j with a different sentence context — implying contradictory associations.
  3. Cross-type: same annotation_id across variant/drug/phenotype MCQ files — model's
     answers reconstruct an inconsistent pharmacogenomic association.

Usage:
    python src/eval/contradictions.py \\
        --run-dir runs/20260213_020814_mc_study_all_models \\
        --model gpt-5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

LETTERS = ["a", "b", "c", "d"]


def load_eval_results(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def get_selected_entity(record: dict) -> str | None:
    """Return the text of the option the model selected."""
    pred = record.get("prediction_parsed")
    if pred and pred in LETTERS:
        return record.get(f"option_{pred}")
    return None


def get_correct_entity(record: dict) -> str | None:
    """Return the text of the correct option."""
    ans = record.get("correct_answer", "").strip().lower()
    if ans in LETTERS:
        return record.get(f"option_{ans}")
    return None


# ---------------------------------------------------------------------------
# Type 1: Standard vs NOTA contradictions
# ---------------------------------------------------------------------------

def detect_standard_nota_contradictions(records: list[dict]) -> list[dict]:
    """Find contradictions between standard and none_of_the_above question pairs.

    For the same annotation_id, the standard question has the correct entity as an
    option, while the NOTA question does not. Contradictions:
      - Model picks the correct entity in standard but picks a distractor (not "None")
        in NOTA — partially consistent, but NOTA answer implies wrong entity.
      - Model picks a distractor in standard but picks "None" in NOTA — model
        correctly identifies absence in NOTA but picked wrong entity when present.
      - Model picks wrong entity X in standard; in NOTA, model also picks X
        (or another distractor) — doubly wrong but may still be contradictory
        if X appears in NOTA and model assigns it to the same context.
    """
    # Group by annotation_id
    by_annotation: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_annotation[r["annotation_id"]][r["question_type"]] = r

    contradictions = []
    for ann_id, pair in by_annotation.items():
        if "standard" not in pair or "none_of_the_above" not in pair:
            continue

        std = pair["standard"]
        nota = pair["none_of_the_above"]

        std_pred = std.get("prediction_parsed")
        nota_pred = nota.get("prediction_parsed")
        if not std_pred or not nota_pred:
            continue

        std_entity = get_selected_entity(std)
        nota_entity = get_selected_entity(nota)
        std_correct = std["correct"]
        nota_correct = nota["correct"]

        # Find which option in NOTA is "None of the options"
        nota_none_letter = None
        for letter in LETTERS:
            if nota.get(f"option_{letter}", "").lower().strip() in (
                "none of the options", "none of the above"
            ):
                nota_none_letter = letter
                break

        nota_picked_none = nota_pred == nota_none_letter

        # Contradiction case 1: correct in standard, wrong in NOTA
        # Model knew the entity but didn't recognize its absence
        if std_correct and not nota_correct:
            contradictions.append({
                "type": "standard_vs_nota",
                "subtype": "correct_standard_wrong_nota",
                "annotation_id": ann_id,
                "pmcid": std["pmcid"],
                "blanked_sentence": std["blanked_sentence"],
                "std_question_id": std["question_id"],
                "nota_question_id": nota["question_id"],
                "std_selected": std_entity,
                "nota_selected": nota_entity,
                "nota_picked_none": nota_picked_none,
                "correct_entity": get_correct_entity(std),
            })

        # Contradiction case 2: wrong in standard, correct in NOTA
        # Model recognized absence but picked wrong entity when present
        if not std_correct and nota_correct:
            contradictions.append({
                "type": "standard_vs_nota",
                "subtype": "wrong_standard_correct_nota",
                "annotation_id": ann_id,
                "pmcid": std["pmcid"],
                "blanked_sentence": std["blanked_sentence"],
                "std_question_id": std["question_id"],
                "nota_question_id": nota["question_id"],
                "std_selected": std_entity,
                "nota_selected": nota_entity,
                "nota_picked_none": nota_picked_none,
                "correct_entity": get_correct_entity(std),
            })

    return contradictions


# ---------------------------------------------------------------------------
# Type 2: Distractor-as-correct-answer contradictions
# ---------------------------------------------------------------------------

def detect_distractor_contradictions(records: list[dict]) -> list[dict]:
    """Find cases where the model picks entity X for question Q_i (wrong) while
    X is the correct answer for a different question Q_j in the same PMCID,
    and Q_i and Q_j have different blanked sentences (different contexts).

    This means the model is assigning the same entity to two different contexts,
    creating contradictory pharmacogenomic claims.
    """
    # Only look at standard questions (NOTA has different dynamics)
    standard = [r for r in records if r["question_type"] == "standard"]

    # Group by pmcid
    by_pmcid: dict[str, list[dict]] = defaultdict(list)
    for r in standard:
        by_pmcid[r["pmcid"]].append(r)

    # Build mapping: (pmcid, entity_text) -> list of questions where it's correct
    correct_entity_map: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in standard:
        entity = get_correct_entity(r)
        if entity:
            correct_entity_map[(r["pmcid"], entity)].append(r)

    contradictions = []
    for pmcid, questions in by_pmcid.items():
        for q in questions:
            if q["correct"]:
                continue  # Only interested in wrong answers

            selected = get_selected_entity(q)
            if not selected:
                continue

            # Check if the entity the model wrongly picked is the correct answer
            # for another question in the same PMCID
            key = (pmcid, selected)
            if key not in correct_entity_map:
                continue

            for other_q in correct_entity_map[key]:
                if other_q["annotation_id"] == q["annotation_id"]:
                    continue  # Skip same annotation (that's the std/nota pair)
                if other_q["blanked_sentence"] == q["blanked_sentence"]:
                    continue  # Same context isn't contradictory

                contradictions.append({
                    "type": "distractor_as_correct",
                    "pmcid": pmcid,
                    "entity": selected,
                    "wrong_question_id": q["question_id"],
                    "wrong_annotation_id": q["annotation_id"],
                    "wrong_context": q["blanked_sentence"],
                    "wrong_correct_entity": get_correct_entity(q),
                    "right_question_id": other_q["question_id"],
                    "right_annotation_id": other_q["annotation_id"],
                    "right_context": other_q["blanked_sentence"],
                    "other_q_correct": other_q["correct"],
                })

    return contradictions


# ---------------------------------------------------------------------------
# Type 3: Cross-type contradictions
# ---------------------------------------------------------------------------

def detect_cross_type_contradictions(
    variant_records: list[dict],
    drug_records: list[dict],
    phenotype_records: list[dict],
) -> list[dict]:
    """Find contradictions across MCQ types for the same annotation_id.

    Each annotation generates questions in multiple MCQ files. The model's answers
    should reconstruct a consistent association. If the variant Q says the answer is
    rs123 but the drug Q for the same annotation implies a different drug than what
    rs123 is associated with, that's a cross-type contradiction.

    We detect: at least one type correct and at least one type wrong for the same
    annotation, which means the model partially knows the association but is
    inconsistent about it.
    """
    # Only standard questions
    def index_by_ann(records):
        idx = {}
        for r in records:
            if r["question_type"] == "standard":
                idx[r["annotation_id"]] = r
        return idx

    v_idx = index_by_ann(variant_records)
    d_idx = index_by_ann(drug_records)
    p_idx = index_by_ann(phenotype_records)

    # Find annotation_ids present in multiple types
    all_ann_ids = set(v_idx) | set(d_idx) | set(p_idx)

    contradictions = []
    for ann_id in all_ann_ids:
        results_for_ann = {}
        if ann_id in v_idx:
            results_for_ann["variant"] = v_idx[ann_id]
        if ann_id in d_idx:
            results_for_ann["drug"] = d_idx[ann_id]
        if ann_id in p_idx:
            results_for_ann["phenotype"] = p_idx[ann_id]

        if len(results_for_ann) < 2:
            continue

        correct_types = [t for t, r in results_for_ann.items() if r["correct"]]
        wrong_types = [t for t, r in results_for_ann.items() if not r["correct"]]

        if correct_types and wrong_types:
            detail = {}
            for mcq_type, r in results_for_ann.items():
                detail[mcq_type] = {
                    "question_id": r["question_id"],
                    "correct": r["correct"],
                    "selected": get_selected_entity(r),
                    "expected": get_correct_entity(r),
                    "blanked_sentence": r["blanked_sentence"],
                }

            contradictions.append({
                "type": "cross_type",
                "annotation_id": ann_id,
                "pmcid": list(results_for_ann.values())[0]["pmcid"],
                "correct_types": correct_types,
                "wrong_types": wrong_types,
                "details": detail,
            })

    return contradictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_model_files(run_dir: Path, model: str) -> dict[str, Path | None]:
    """Find eval result files for a given model in a run directory."""
    model_slug = model.replace("/", "_")
    result = {}
    for mcq_type in ("variant", "drug", "phenotype"):
        # Try exact prefix match first to avoid e.g. gpt-5 matching gpt-5.2
        exact = run_dir / f"{model_slug}_mcq_{mcq_type}_eval_results.jsonl"
        if exact.exists():
            result[mcq_type] = exact
            continue
        # Try without model slug (single-model run dirs)
        plain = run_dir / f"mcq_{mcq_type}_eval_results.jsonl"
        if plain.exists():
            result[mcq_type] = plain
            continue
        result[mcq_type] = None
    return result


def _format_contradiction_detail(c: dict) -> list[str]:
    """Format a single contradiction as human-readable lines."""
    lines = []
    if c["type"] == "standard_vs_nota":
        label = (
            "Correct standard, wrong NOTA"
            if c["subtype"] == "correct_standard_wrong_nota"
            else "Wrong standard, correct NOTA"
        )
        lines.append(f"  [{label}]  annotation={c['annotation_id']}  pmcid={c['pmcid']}")
        lines.append(f"    Sentence: {c['blanked_sentence'][:120]}")
        lines.append(f"    Correct entity: {c['correct_entity']}")
        lines.append(f"    Standard picked: {c['std_selected']}  (Q{c['std_question_id']})")
        lines.append(f"    NOTA picked: {c['nota_selected']}  (Q{c['nota_question_id']}, picked_none={c['nota_picked_none']})")

    elif c["type"] == "distractor_as_correct":
        lines.append(f"  [Distractor-as-correct]  pmcid={c['pmcid']}  entity={c['entity']}")
        lines.append(f"    Wrong Q{c['wrong_question_id']}: {c['wrong_context'][:100]}")
        lines.append(f"      Model picked \"{c['entity']}\" but correct was \"{c['wrong_correct_entity']}\"")
        lines.append(f"    Right Q{c['right_question_id']}: {c['right_context'][:100]}")
        lines.append(f"      \"{c['entity']}\" is correct here (model got it: {c['other_q_correct']})")

    elif c["type"] == "cross_type":
        lines.append(
            f"  [Cross-type]  annotation={c['annotation_id']}  pmcid={c['pmcid']}  "
            f"correct={c['correct_types']}  wrong={c['wrong_types']}"
        )
        for mcq_type, d in c["details"].items():
            status = "CORRECT" if d["correct"] else "WRONG"
            lines.append(
                f"    {mcq_type:<12} Q{d['question_id']}: "
                f"selected={d['selected']}  expected={d['expected']}  [{status}]"
            )

    return lines


def run_analysis(run_dir: Path, model: str, output_path: Path | None = None):
    files = find_model_files(run_dir, model)
    print(f"Model: {model}")
    print(f"Run dir: {run_dir}")
    print()

    all_records: dict[str, list[dict]] = {}
    for mcq_type, path in files.items():
        if path and path.exists():
            all_records[mcq_type] = load_eval_results(path)
            print(f"  Loaded {len(all_records[mcq_type]):>6} {mcq_type} eval results from {path.name}")
        else:
            all_records[mcq_type] = []
            print(f"  No {mcq_type} eval results found")

    print()

    # Collect all contradictions
    all_contradictions = []
    summary_lines = []

    summary_lines.append("=" * 70)
    summary_lines.append(f"Contradiction Analysis  |  model={model}")
    summary_lines.append("=" * 70)

    # --- Type 1: Standard vs NOTA ---
    summary_lines.append("")
    summary_lines.append("## Type 1: Standard vs NOTA contradictions")
    summary_lines.append("-" * 50)

    type1_contras = []
    for mcq_type in ("variant", "drug", "phenotype"):
        records = all_records.get(mcq_type, [])
        if not records:
            continue
        contras = detect_standard_nota_contradictions(records)
        type1_contras.extend(contras)
        all_contradictions.extend(contras)

        n_total_pairs = len({
            r["annotation_id"]
            for r in records
            if r["question_type"] == "standard"
        })
        correct_std_wrong_nota = [c for c in contras if c["subtype"] == "correct_standard_wrong_nota"]
        wrong_std_correct_nota = [c for c in contras if c["subtype"] == "wrong_standard_correct_nota"]

        summary_lines.append(f"")
        summary_lines.append(f"  {mcq_type.upper()} MCQ ({n_total_pairs} annotation pairs):")
        summary_lines.append(
            f"    Correct standard, wrong NOTA:  {len(correct_std_wrong_nota):>5}  "
            f"({len(correct_std_wrong_nota) / max(n_total_pairs, 1):.1%})"
        )
        summary_lines.append(
            f"    Wrong standard, correct NOTA:  {len(wrong_std_correct_nota):>5}  "
            f"({len(wrong_std_correct_nota) / max(n_total_pairs, 1):.1%})"
        )
        summary_lines.append(
            f"    Total contradictions:          {len(contras):>5}  "
            f"({len(contras) / max(n_total_pairs, 1):.1%})"
        )

    # --- Type 2: Distractor-as-correct ---
    summary_lines.append("")
    summary_lines.append("## Type 2: Distractor-as-correct-answer contradictions")
    summary_lines.append("-" * 50)

    type2_contras = []
    for mcq_type in ("variant", "drug", "phenotype"):
        records = all_records.get(mcq_type, [])
        if not records:
            continue
        contras = detect_distractor_contradictions(records)
        type2_contras.extend(contras)
        all_contradictions.extend(contras)

        n_standard = len([r for r in records if r["question_type"] == "standard"])
        n_wrong = len([
            r for r in records
            if r["question_type"] == "standard" and not r["correct"]
        ])
        wrong_q_ids = {c["wrong_question_id"] for c in contras}

        summary_lines.append(f"")
        summary_lines.append(f"  {mcq_type.upper()} MCQ ({n_standard} standard questions, {n_wrong} wrong answers):")
        summary_lines.append(
            f"    Contradiction instances:        {len(contras):>5}"
        )
        summary_lines.append(
            f"    Unique wrong answers involved:  {len(wrong_q_ids):>5}"
        )
        if n_wrong:
            summary_lines.append(
                f"    Wrong answers that contradict:  "
                f"{len(wrong_q_ids) / n_wrong:.1%}"
            )

    # --- Type 3: Cross-type ---
    summary_lines.append("")
    summary_lines.append("## Type 3: Cross-type contradictions")
    summary_lines.append("-" * 50)

    type3_contras = []
    v = all_records.get("variant", [])
    d = all_records.get("drug", [])
    p = all_records.get("phenotype", [])

    if any([v, d, p]):
        contras = detect_cross_type_contradictions(v, d, p)
        type3_contras.extend(contras)
        all_contradictions.extend(contras)

        def get_standard_ann_ids(records):
            return {r["annotation_id"] for r in records if r["question_type"] == "standard"}

        v_anns = get_standard_ann_ids(v)
        d_anns = get_standard_ann_ids(d)
        p_anns = get_standard_ann_ids(p)

        multi_type_anns = set()
        for ann in v_anns | d_anns | p_anns:
            count = sum(1 for s in [v_anns, d_anns, p_anns] if ann in s)
            if count >= 2:
                multi_type_anns.add(ann)

        summary_lines.append(
            f"  Annotations in 2+ MCQ types:     {len(multi_type_anns):>5}"
        )
        summary_lines.append(
            f"  Cross-type contradictions:        {len(contras):>5}  "
            f"({len(contras) / max(len(multi_type_anns), 1):.1%})"
        )

        conflict_pairs = defaultdict(int)
        for c in contras:
            for ct in c["correct_types"]:
                for wt in c["wrong_types"]:
                    conflict_pairs[(ct, wt)] += 1

        if conflict_pairs:
            summary_lines.append("")
            summary_lines.append("  Conflict breakdown (correct_type -> wrong_type):")
            for (ct, wt), count in sorted(conflict_pairs.items(), key=lambda x: -x[1]):
                summary_lines.append(f"    {ct:>12} correct, {wt:<12} wrong: {count}")

    # --- Overall totals ---
    summary_lines.append("")
    summary_lines.append("=" * 70)
    total_questions = sum(len(r) for r in all_records.values())
    summary_lines.append(
        f"Total contradictions: {len(all_contradictions)}  "
        f"(across {total_questions} total questions)"
    )
    summary_lines.append("=" * 70)

    # --- All contradiction details ---
    for label, contras in [
        ("Type 1: Standard vs NOTA", type1_contras),
        ("Type 2: Distractor-as-correct", type2_contras),
        ("Type 3: Cross-type", type3_contras),
    ]:
        if not contras:
            continue
        summary_lines.append("")
        summary_lines.append("-" * 70)
        summary_lines.append(f"All {label} contradictions ({len(contras)})")
        summary_lines.append("-" * 70)
        for i, c in enumerate(contras, 1):
            summary_lines.append("")
            summary_lines.append(f"  #{i}")
            summary_lines.extend(_format_contradiction_detail(c))

    summary = "\n".join(summary_lines)
    print(summary)

    # Save summary to run directory
    if output_path is None:
        model_slug = model.replace("/", "_")
        output_path = run_dir / f"{model_slug}_contradictions_summary.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nSummary saved to {output_path}")

    return all_contradictions


def discover_models(run_dir: Path) -> list[str]:
    """Auto-detect model slugs from eval result filenames in a run directory."""
    models = set()
    for f in run_dir.glob("*_mcq_variant_eval_results.jsonl"):
        # Strip the suffix to get the model slug
        slug = f.name.replace("_mcq_variant_eval_results.jsonl", "")
        models.add(slug)
    # Also check drug/phenotype in case variant is missing
    for mcq_type in ("drug", "phenotype"):
        for f in run_dir.glob(f"*_mcq_{mcq_type}_eval_results.jsonl"):
            slug = f.name.replace(f"_mcq_{mcq_type}_eval_results.jsonl", "")
            models.add(slug)
    return sorted(models)


def main():
    parser = argparse.ArgumentParser(
        description="Detect self-contradictions in MCQ model responses"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the run directory containing eval results",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        help="Model identifier (e.g. 'gpt-5', 'anthropic/claude-opus-4-6')",
    )
    group.add_argument(
        "--all-models",
        action="store_true",
        help="Run contradiction analysis for all models found in the run directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for summary txt (default: <run_dir>/<model>_contradictions_summary.txt)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: run directory {run_dir} does not exist")
        sys.exit(1)

    if args.all_models:
        models = discover_models(run_dir)
        if not models:
            print(f"No models found in {run_dir}")
            sys.exit(1)
        print(f"Found {len(models)} models: {', '.join(models)}\n")
        for model_slug in models:
            # Convert slug back to model id (replace _ with / for anthropic models)
            model = model_slug
            print(f"\n{'#' * 70}")
            print(f"# {model}")
            print(f"{'#' * 70}\n")
            run_analysis(run_dir, model)
            print()
    else:
        output_path = Path(args.output) if args.output else None
        run_analysis(run_dir, args.model, output_path)


if __name__ == "__main__":
    main()
