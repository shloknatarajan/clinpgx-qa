"""
Reads var_drug_ann.tsv and var_pheno_ann.tsv from data/raw/variantAnnotations/,
groups variants by PMID, and creates data/variant_bench.jsonl.

Enriches each entry with pmcid and article_title from data/papers/ markdown files.

Each row in the jsonl file contains:
- pmcid: str
- pmid: str
- article_title: str
- variants: list[str] - processed variant strings (deduplicated and ungrouped)
- raw_variants: list[list[str]] - preserving the original groupings from annotations
"""

import csv
import json
from collections import defaultdict
from pydantic import BaseModel
from pathlib import Path
from loguru import logger


class SingleArticleVariants(BaseModel):
    """
    Represents a data model for storing variant information extracted from a single article.

    Attributes:
        pmcid (str): The PubMed Central ID of the article.
        pmid (str): The PubMed ID of the article.
        article_title (str): The title of the article.
        variants (list[str]): A list of processed variant strings found in the article.
        raw_variants (list[list[str]]): A list of lists preserving the original groupings from annotations.
    """

    pmcid: str
    pmid: str
    article_title: str
    variants: list[str]
    raw_variants: list[list[str]]


def _clean_title(title: str) -> str:
    """Normalize unicode characters in titles to their ASCII equivalents."""
    replacements = {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2009": " ",  # thin space
        "\u00a0": " ",  # non-breaking space
        "\u202f": " ",  # narrow no-break space
        "\u03b1": "alpha",  # α
        "\u03b2": "beta",  # β
        "\u03b3": "gamma",  # γ
        "\u03b4": "delta",  # δ
        "\u0394": "Delta",  # Δ
        "\u03b5": "epsilon",  # ε
        "\u03bc": "mu",  # μ
        "\u00b5": "mu",  # µ (micro sign)
        "\u03ba": "kappa",  # κ
        "\u03bb": "lambda",  # λ
        "\u00ef": "i",  # ï
        "\u00e9": "e",  # é
        "\u00e8": "e",  # è
        "\u00fc": "u",  # ü
        "\u00f6": "o",  # ö
        "\u00e4": "a",  # ä
        "\u00e1": "a",  # á
        "\u00ed": "i",  # í
        "\u00f1": "n",  # ñ
        "\u00e7": "c",  # ç
        "\u00df": "ss",  # ß
        "\u00e3": "a",  # ã
        "\u00bb": "",  # » (right guillemet)
        "\u00ab": "",  # « (left guillemet)
        "\u00ae": "",  # ® (registered)
        "\u2605": "*",  # ★
        "\u00b0": " degrees",  # °
        "\u2003": " ",  # em space
        "\u2002": " ",  # en space
        "\u2212": "-",  # minus sign
        "\u2192": "->",  # rightwards arrow
        "\u2217": "*",  # asterisk operator
        "\u2032": "'",  # prime
        "\u2033": '"',  # double prime
    }
    for old, new in replacements.items():
        title = title.replace(old, new)
    # Return None if title still contains non-ASCII (e.g. Chinese characters)
    if any(ord(c) > 127 for c in title):
        return None
    return title


def _build_pmid_metadata(papers_dir: Path) -> dict[str, dict[str, str]]:
    """
    Builds a PMID -> {pmcid, title} mapping by parsing markdown files in data/papers/.

    Returns:
        Dict mapping PMID strings to {"pmcid": ..., "title": ...}.
    """
    pmid_map: dict[str, dict[str, str]] = {}

    for md_file in papers_dir.glob("*.md"):
        pmcid = md_file.stem  # filename is e.g. PMC5508045.md
        title = ""
        pmid = ""

        # Only need to read the metadata header (first ~15 lines)
        with open(md_file, "r") as f:
            for i, line in enumerate(f):
                if i == 0 and line.startswith("# "):
                    title = _clean_title(line[2:].strip())
                if line.startswith("**PMID:**"):
                    pmid = line.split("**PMID:**")[1].strip()
                if i > 15:
                    break

        if pmid and title is not None:
            pmid_map[pmid] = {"pmcid": pmcid, "title": title}

    return pmid_map


def get_all_tsv_variants(
    deduplicate: bool = True,
    ungroup: bool = True,
    save_jsonl: bool = True,
) -> list[SingleArticleVariants]:
    """
    Reads var_drug_ann.tsv and var_pheno_ann.tsv from data/raw/variantAnnotations/,
    groups variants by PMID, and returns one SingleArticleVariants per article.

    Enriches each entry with pmcid and article_title from data/papers/ markdown files
    when available.

    Does NOT include var_fa_ann.

    Args:
        deduplicate: If True, removes duplicate variants per article.
        ungroup: If True, splits comma-separated grouped variants into individual variants.
        save_jsonl: If True, saves to data/variant_bench.jsonl.

    Returns:
        List of SingleArticleVariants, one per unique PMID.
    """
    tsv_dir = Path("data/raw/variantAnnotations")
    tsv_files = ["var_drug_ann.tsv", "var_pheno_ann.tsv"]

    # Build PMID -> metadata lookup from papers
    papers_dir = Path("data/papers")
    pmid_metadata = _build_pmid_metadata(papers_dir) if papers_dir.exists() else {}
    logger.info(
        f"Built metadata lookup for {len(pmid_metadata)} PMIDs from {papers_dir}"
    )

    # Collect raw variant strings grouped by PMID
    pmid_variants: dict[str, list[str]] = defaultdict(list)

    for tsv_name in tsv_files:
        tsv_path = tsv_dir / tsv_name
        if not tsv_path.exists():
            logger.warning(f"TSV file not found: {tsv_path}")
            continue
        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                pmid = str(row["PMID"]).strip()
                variant = row["Variant/Haplotypes"].strip()
                if pmid and variant:
                    pmid_variants[pmid].append(variant)

    logger.info(
        f"Read {sum(len(v) for v in pmid_variants.values())} variant annotations "
        f"across {len(pmid_variants)} unique PMIDs from {tsv_files}"
    )

    results: list[SingleArticleVariants] = []
    skipped = 0
    for pmid, variants in pmid_variants.items():
        raw_variants: list[list[str]] = [
            [v.strip() for v in variant.split(",")] for variant in variants
        ]

        processed = list(variants)
        if deduplicate:
            processed = list(set(processed))
        if ungroup:
            processed = [v.strip() for variant in processed for v in variant.split(",")]

        meta = pmid_metadata.get(pmid)
        if not meta:
            skipped += 1
            continue

        results.append(
            SingleArticleVariants(
                pmcid=meta["pmcid"],
                pmid=pmid,
                article_title=meta["title"],
                variants=processed,
                raw_variants=raw_variants,
            )
        )

    logger.info(
        f"Kept {len(results)} PMIDs with paper metadata, "
        f"skipped {skipped} without a matching paper"
    )

    if save_jsonl:
        output_path = Path("data") / "variant_bench.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for item in results:
                f.write(json.dumps(item.model_dump()) + "\n")
        logger.info(f"Saved {len(results)} articles to {output_path}")

    return results


if __name__ == "__main__":
    all_variants = get_all_tsv_variants(save_jsonl=True)
    print(f"Found {len(all_variants)} articles from raw TSVs")

    with open("data/variant_bench.jsonl", "r") as f:
        loaded = [json.loads(line) for line in f]
    print(f"Loaded JSONL with {len(loaded)} rows")
    print(f"First entry: {json.dumps(loaded[0], indent=2)}")
