# PGxWA: A Benchmark for Long-Context Scientific Claim Extraction from Pharmacogenomic Literature

## Abstract

Modern AI systems struggle to reliably extract mechanistic scientific claims from full-length research articles while reconciling inconsistent biomedical nomenclature. Existing benchmarks largely emphasize short-context question answering, abstract-level reasoning, or curated knowledge bases, leaving a critical gap between model capabilities and the demands of real-world scientific discovery.

We introduce **PGxWA**, a large-scale dataset designed to evaluate long-context reasoning, terminology normalization, and structured association extraction from pharmacogenomic literature. The dataset requires models to interpret full-text articles, resolve synonymous biomedical entities into standardized ontology representations, and produce grounded scientific claims supported by evidence spans.

PGxWA contains **[N] full-text articles**, **[M] annotated associations**, and spans **[K] unique genes, variants, drugs, and phenotypes**. We benchmark state-of-the-art language models and find substantial performance gaps relative to expert annotations, particularly in evidence grounding and normalization.

PGxWA provides a rigorous testbed for scientific AI systems and enables research toward automated knowledge graph construction, literature-scale hypothesis generation, and trustworthy biomedical reasoning.

---

# 1. Introduction

Scientific progress increasingly depends on the ability to synthesize claims from a rapidly expanding body of literature. While large language models demonstrate strong performance on general reasoning benchmarks, they remain unreliable at extracting structured scientific knowledge from full-length research papers.

Current biomedical datasets predominantly rely on:

- abstracts rather than full texts  
- multiple-choice or exam-style questions  
- curated knowledge triples  
- short-context reasoning  

However, real scientific interpretation requires models to:

- integrate evidence across long documents  
- resolve synonymous terminology  
- distinguish mechanistic from statistical claims  
- ground outputs in textual evidence  

To address this gap, we introduce **PGxWA**, a benchmark for evaluating long-context scientific reasoning through structured claim extraction.

### Contributions

- **Long-context scientific task:** Questions grounded in full-length research articles.
- **Structured outputs:** Models must generate standardized association tuples.
- **Terminology normalization:** Outputs must align with biomedical ontologies.
- **Evidence grounding:** Predictions require supporting spans.
- **Expert-aligned evaluation:** Performance is compared against curated annotations.

---

# 2. Why a New Dataset Is Needed

Despite rapid progress in scientific language modeling, existing benchmarks fail to capture the complexity of real literature interpretation.

| Dataset | Full Text | Scientific Claims | Normalization | Multi-hop Reasoning |
|--------|------------|------------------|---------------|---------------------|
| PubMedQA | ❌ | Limited | ❌ | ❌ |
| MedMCQA | ❌ | Exam-style | ❌ | ❌ |
| SciFact | Partial | Claim verification | ❌ | Limited |

Most benchmarks emphasize **fact retrieval** rather than **claim extraction**.

We argue that the next frontier for AI in science is the transition from:

> fact recall → claim synthesis → structured knowledge generation.

PGxWA is designed to operationalize this transition.

---

# 3. Task Definition

We formalize scientific claim extraction as a structured prediction task.

## Input

- Full research article  
- Query requiring extraction of a pharmacogenomic association  

## Output

Models must produce a tuple: