"""
Takes benchmark annotations from data/benchmark_annotations and creates a jsonl file data/benchmark_variants.jsonl
Each row in the jsonl file contains:
- pmcid: str
- pmid: str
- article_title: str
- variants: list[str] - processed variant strings (deduplicated and ungrouped)
- raw_variants: list[list[str]] - preserving the original groupings from annotations
"""

import json
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


def get_file_variants(
    file_path: Path | str, deduplicate: bool = True, ungroup: bool = True
) -> SingleArticleVariants:
    """
    Extracts all mentioned variants from a single JSON article file.

    This function reads a JSON file, extracts variant/haplotype information from
    'var_drug_ann', 'var_pheno_ann', and 'var_fa_ann' sections. It can also
    deduplicate and ungroup variants based on the provided flags.

    Args:
        file_path (Path | str): The path to the JSON file containing article annotations.
        deduplicate (bool, optional): If True, removes duplicate variants. Defaults to True.
        ungroup (bool, optional): If True, splits comma-separated grouped variants into
                                   individual variants. Defaults to True.

    Returns:
        SingleArticleVariants: An object containing the article's metadata and
                               the extracted variants. Returns an empty
                               SingleArticleVariants object if the file cannot
                               be processed, logging a warning.
    """
    # from json file, extract all the variants from variant/haplotypes
    # Convert file to path
    if isinstance(file_path, str):
        file_path = Path(file_path)
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Warning: Could not process file {file_path}: {e}")
        return SingleArticleVariants(
            pmcid="", pmid="", article_title="", variants=[], raw_variants=[]
        )
    variants: list[str] = []
    pmcid = data["pmcid"]
    pmid = data["pmid"]
    article_title = data["title"]
    for item in data["var_drug_ann"]:
        variants.append(item["Variant/Haplotypes"])
    for item in data["var_pheno_ann"]:
        variants.append(item["Variant/Haplotypes"])
    for item in data["var_fa_ann"]:
        variants.append(item["Variant/Haplotypes"])

    # Preserve original groupings as list of lists (split each group by comma)
    raw_variants: list[list[str]] = [
        [v.strip() for v in variant.split(",")] for variant in variants
    ]

    if deduplicate:
        variants = list(set(variants))
    if ungroup:
        variants = [variant.split(",") for variant in variants]
        variants = [variant.strip() for sublist in variants for variant in sublist]
    return SingleArticleVariants(
        pmcid=pmcid,
        pmid=pmid,
        article_title=article_title,
        variants=variants,
        raw_variants=raw_variants,
    )


def get_dir_variants(
    dir_path: str, deduplicate: bool = True, ungroup: bool = True
) -> list[SingleArticleVariants]:
    """
    Processes all JSON article files within a specified directory to extract variants.

    This function iterates through all JSON files in the given directory,
    applies `get_file_variants` to each, and aggregates the results.

    Args:
        dir_path (str): The path to the directory containing JSON annotation files.
        deduplicate (bool, optional): If True, variants extracted from each file
                                     will be deduplicated. Defaults to True.
        ungroup (bool, optional): If True, splits comma-separated grouped variants
                                  into individual variants for each file. Defaults to True.

    Returns:
        list[SingleArticleVariants]: A list of `SingleArticleVariants` objects, each
                                     representing the variants found in one article file.

    Raises:
        ValueError: If the provided `dir_path` does not exist or is not a directory.
    """
    # Validate path
    directory = Path(dir_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")
    variant_list: list[SingleArticleVariants] = []

    # Loop through json files in the directory
    for file in directory.glob("*.json"):
        try:
            article_variants = get_file_variants(
                file, deduplicate=deduplicate, ungroup=ungroup
            )
            variant_list.append(article_variants)
        except Exception as e:
            logger.warning(f"Warning: Could not process file {file}: {e}")
    return variant_list


def get_benchmark_variants(save_jsonl: bool = True) -> list[SingleArticleVariants]:
    """
    Retrieves and processes variants from the benchmark annotation directory.

    This function specifically targets the 'data/benchmark_annotations' directory,
    deduplicating and ungrouping variants by default.

    Args:
        save_jsonl (bool, optional): If True, saves the results to a JSONL file
                                     at 'data/benchmark_variants.jsonl'. Defaults to True.

    Returns:
        list[SingleArticleVariants]: A list of `SingleArticleVariants` objects
                                     representing the processed benchmark variants.
    """
    benchmark_dir = "data/benchmark_annotations"
    logger.info(f"Loading variants from benchmark dir {benchmark_dir}")
    benchmark_variants = get_dir_variants(benchmark_dir, deduplicate=True, ungroup=True)

    if save_jsonl:
        output_path = Path("data") / "benchmark_v2" / "variant_bench.jsonl"
        with open(output_path, "w") as f:
            for variant in benchmark_variants:
                f.write(json.dumps(variant.model_dump()) + "\n")
        logger.info(f"Saved {len(benchmark_variants)} articles to {output_path}")

    return benchmark_variants


if __name__ == "__main__":
    benchmark_variants = get_benchmark_variants(save_jsonl=True)
    print(f"Found {len(benchmark_variants)} articles with variants")

    # Verify the JSONL file is loadable
    with open("data/benchmark_v2/variant_bench.jsonl", "r") as f:
        loaded_variants = [json.loads(line) for line in f]
    print(f"Loaded JSONL with {len(loaded_variants)} rows")
    print(f"First entry: {loaded_variants[0]}")
