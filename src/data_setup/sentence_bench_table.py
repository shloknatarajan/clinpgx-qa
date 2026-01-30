"""
Creates a dataset of (pmcid, variant, summary sentence, association_type)

Saves to data/benchmark_v2/sentence_bench.jsonl
"""

import json
from pydantic import BaseModel
from pathlib import Path
from loguru import logger


class SingleSentenceEntry(BaseModel):
    """
    Represents a single sentence entry with its associated variant and article info.

    Attributes:
        pmcid (str): The PubMed Central ID of the article.
        pmid (str): The PubMed ID of the article.
        variant (str): The variant/haplotype associated with this sentence.
        sentence (str): The summary sentence from the annotation.
        annotation_type (str): The type of annotation (var_drug_ann, var_pheno_ann, or var_fa_ann).
    """

    pmcid: str
    pmid: str
    variant: str
    sentence: str
    annotation_type: str


class GroupedSentenceEntry(BaseModel):
    """
    Represents grouped sentences for a (pmcid, variant) pair.

    Attributes:
        pmcid (str): The PubMed Central ID of the article.
        pmid (str): The PubMed ID of the article.
        variant (str): The variant/haplotype associated with these sentences.
        sentences (list[str]): All sentences associated with this (pmcid, variant) pair.
    """

    pmcid: str
    pmid: str
    variant: str
    sentences: list[str]


def get_file_sentences(file_path: Path | str) -> list[SingleSentenceEntry]:
    """
    Extracts all sentence entries from a single JSON article file.

    This function reads a JSON file and extracts (variant, sentence) pairs from
    'var_drug_ann', 'var_pheno_ann', and 'var_fa_ann' sections.

    Args:
        file_path (Path | str): The path to the JSON file containing article annotations.

    Returns:
        list[SingleSentenceEntry]: A list of sentence entries extracted from the file.
                                   Returns an empty list if the file cannot be processed.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Warning: Could not process file {file_path}: {e}")
        return []

    entries: list[SingleSentenceEntry] = []
    pmcid = data["pmcid"]
    pmid = data["pmid"]

    annotation_types = ["var_drug_ann", "var_pheno_ann", "var_fa_ann"]

    for ann_type in annotation_types:
        for item in data.get(ann_type, []):
            variant = item.get("Variant/Haplotypes", "")
            sentence = item.get("Sentence", "")

            if variant and sentence:
                # Handle grouped variants (comma-separated)
                individual_variants = [v.strip() for v in variant.split(",")]
                for individual_variant in individual_variants:
                    entries.append(
                        SingleSentenceEntry(
                            pmcid=pmcid,
                            pmid=pmid,
                            variant=individual_variant,
                            sentence=sentence,
                            annotation_type=ann_type,
                        )
                    )

    return entries


def get_dir_sentences(dir_path: str) -> list[SingleSentenceEntry]:
    """
    Processes all JSON article files within a specified directory to extract sentences.

    Args:
        dir_path (str): The path to the directory containing JSON annotation files.

    Returns:
        list[SingleSentenceEntry]: A list of all sentence entries from all files.

    Raises:
        ValueError: If the provided `dir_path` does not exist or is not a directory.
    """
    directory = Path(dir_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")

    all_entries: list[SingleSentenceEntry] = []

    for file in directory.glob("*.json"):
        try:
            file_entries = get_file_sentences(file)
            all_entries.extend(file_entries)
        except Exception as e:
            logger.warning(f"Warning: Could not process file {file}: {e}")

    return all_entries


def group_sentences_by_pmcid_variant(
    entries: list[SingleSentenceEntry],
) -> list[GroupedSentenceEntry]:
    """
    Groups sentence entries by (pmcid, variant) pair.

    Args:
        entries (list[SingleSentenceEntry]): List of individual sentence entries.

    Returns:
        list[GroupedSentenceEntry]: List of grouped entries where each entry contains
                                    all sentences for a unique (pmcid, variant) pair.
    """
    from collections import defaultdict

    # Group by (pmcid, variant)
    grouped: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"pmid": "", "sentences": []}
    )

    for entry in entries:
        key = (entry.pmcid, entry.variant)
        grouped[key]["pmid"] = entry.pmid
        grouped[key]["sentences"].append(entry.sentence)

    # Convert to GroupedSentenceEntry objects
    result = []
    for (pmcid, variant), data in grouped.items():
        result.append(
            GroupedSentenceEntry(
                pmcid=pmcid,
                pmid=data["pmid"],
                variant=variant,
                sentences=data["sentences"],
            )
        )

    return result


def get_benchmark_sentences(save_jsonl: bool = True) -> list[GroupedSentenceEntry]:
    """
    Retrieves and processes sentences from the benchmark annotation directory.

    This function specifically targets the 'data/benchmark_annotations' directory.
    Sentences are grouped by (pmcid, variant) pair to handle cases where multiple
    sentences exist for the same pair.

    Args:
        save_jsonl (bool, optional): If True, saves the results to a JSONL file
                                     at 'data/benchmark_v2/sentence_bench.jsonl'. Defaults to True.

    Returns:
        list[GroupedSentenceEntry]: A list of `GroupedSentenceEntry` objects
                                    representing the benchmark sentences grouped by (pmcid, variant).
    """
    benchmark_dir = "data/benchmark_annotations"
    logger.info(f"Loading sentences from benchmark dir {benchmark_dir}")
    raw_sentences = get_dir_sentences(benchmark_dir)
    logger.info(f"Found {len(raw_sentences)} individual sentence entries")

    # Group sentences by (pmcid, variant) pair
    grouped_sentences = group_sentences_by_pmcid_variant(raw_sentences)
    logger.info(f"Grouped into {len(grouped_sentences)} unique (pmcid, variant) pairs")

    if save_jsonl:
        output_path = Path("data") / "benchmark_v2" / "sentence_bench.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for entry in grouped_sentences:
                f.write(json.dumps(entry.model_dump()) + "\n")
        logger.info(f"Saved {len(grouped_sentences)} grouped entries to {output_path}")

    return grouped_sentences


if __name__ == "__main__":
    benchmark_sentences = get_benchmark_sentences(save_jsonl=True)
    print(f"Found {len(benchmark_sentences)} grouped (pmcid, variant) entries")

    # Show stats on entries with multiple sentences
    multi_sentence_entries = [e for e in benchmark_sentences if len(e.sentences) > 1]
    print(f"Entries with multiple sentences: {len(multi_sentence_entries)}")

    # Verify the JSONL file is loadable
    with open("data/benchmark_v2/sentence_bench.jsonl", "r") as f:
        loaded_sentences = [json.loads(line) for line in f]
    print(f"Loaded JSONL with {len(loaded_sentences)} rows")
    print(f"First entry: {loaded_sentences[0]}")
