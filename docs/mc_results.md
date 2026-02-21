# ClinPGx-QA: Multiple-Choice & Study Parameter Results

Results from benchmark-filtered evaluation runs (32 expert-annotated PMCIDs, n=284-474 per model per task).

**Run:** `20260220_160735_mc_study_all_models`

---

## 1. Drug MCQ (n=300)

| Model | Overall | Standard | NOTA | Parse Failures |
|-------|:-------:|:--------:|:----:|:--------------:|
| Claude Opus 4.6 | **93.7%** | 97.3% | 90.0% | 0.0% |
| GPT-5.2 | 91.0% | 94.7% | 87.3% | 0.0% |
| GPT-4o-mini | 86.0% | 99.3% | 72.7% | 0.0% |
| Claude Haiku 4.5 | 80.0% | 97.3% | 62.7% | 1.0% |
| Claude Sonnet 4.5 | 73.0% | 98.7% | 47.3% | 4.7% |

Standard accuracy is near-perfect across all models (94-99%). The spread comes entirely from NOTA questions, where models must recognize the correct drug is absent. Opus leads at 90% NOTA; Sonnet struggles at 47%.

---

## 2. Variant MCQ (n=300)

| Model | Overall | Standard | NOTA | Parse Failures |
|-------|:-------:|:--------:|:----:|:--------------:|
| Claude Opus 4.6 | **79.7%** | 89.3% | 70.0% | 0.0% |
| GPT-5.2 | 75.3% | 80.7% | 70.0% | 0.0% |
| Claude Haiku 4.5 | 65.7% | 72.7% | 58.7% | 1.3% |
| GPT-4o-mini | 62.3% | 68.0% | 56.7% | 0.0% |
| Claude Sonnet 4.5 | 53.0% | 75.3% | 30.7% | 4.3% |

Variant MCQ is harder than drug MCQ, with standard accuracy dropping to 68-89%. Opus and GPT-5.2 lead. Sonnet's NOTA performance (30.7%) is notably weak.

---

## 3. Phenotype MCQ (n=284)

| Model | Overall | Standard | NOTA | Parse Failures |
|-------|:-------:|:--------:|:----:|:--------------:|
| Claude Opus 4.6 | **40.5%** | 54.2% | 26.8% | 0.0% |
| GPT-5.2 | 40.5% | 54.9% | 26.1% | 0.0% |
| GPT-4o-mini | 31.0% | 53.5% | 8.5% | 0.0% |
| Claude Sonnet 4.5 | 30.6% | 50.7% | 10.6% | 2.1% |
| Claude Haiku 4.5 | --- | --- | --- | (credit limit) |

Phenotype MCQ is the hardest task. Even the best models only reach ~40% overall. NOTA accuracy drops to single digits for GPT-4o-mini and Sonnet, indicating models almost always guess rather than abstain on phenotype questions.

---

## 4. Study Parameter Extraction

Study parameter extraction requires models to extract p-values and determine statistical significance from full-text articles. Modified questions replace the drug, variant, or phenotype with a non-existent entity (expected answer: "not found") to test for hallucination.

### 4.1 Overall Results (n=300-474)

| Model | P-value Acc | Significance Acc | Both Correct | N |
|-------|:-----------:|:----------------:|:------------:|:---:|
| Claude Sonnet 4.5 | **60.0%** | **65.7%** | **58.0%** | 300 |
| GPT-4o-mini | 54.0% | 54.8% | 49.5% | 374 |
| Claude Opus 4.6 | 48.1% | 46.4% | 39.9% | 474 |
| Claude Haiku 4.5 | 43.0% | 47.3% | 40.7% | 300 |
| GPT-5.2 | 38.3% | 42.7% | 36.0% | 300 |

### 4.2 By Question Type (Claude Sonnet 4.5, n=300)

| Question Type | P-value Acc | Significance Acc | Both Correct | N |
|---------------|:-----------:|:----------------:|:------------:|:---:|
| Correct (real association) | 53.7% | 74.4% | 46.3% | 82 |
| Modified drug | 75.3% | 75.3% | 75.3% | 81 |
| Modified variant | 65.9% | 65.9% | 65.9% | 82 |
| Modified phenotype | 38.2% | 38.2% | 38.2% | 55 |

### 4.3 By Question Type (GPT-4o-mini, n=374)

| Question Type | P-value Acc | Significance Acc | Both Correct | N |
|---------------|:-----------:|:----------------:|:------------:|:---:|
| Correct (real association) | 46.0% | 49.0% | 29.0% | 100 |
| Modified drug | 63.0% | 63.0% | 63.0% | 100 |
| Modified variant | 58.0% | 58.0% | 58.0% | 100 |
| Modified phenotype | 47.3% | 47.3% | 47.3% | 74 |

### 4.4 By Question Type (Claude Opus 4.6, n=474)

| Question Type | P-value Acc | Significance Acc | Both Correct | N |
|---------------|:-----------:|:----------------:|:------------:|:---:|
| Correct (real association) | 65.6% | 61.6% | 36.8% | 125 |
| Modified drug | 46.4% | 44.8% | 44.8% | 125 |
| Modified variant | 41.6% | 40.8% | 40.8% | 125 |
| Modified phenotype | 36.4% | 36.4% | 36.4% | 99 |

Sonnet leads overall (58%) driven by strong hallucination detection on modified questions (65-75%). Opus has the highest accuracy on real associations (65.6% p-value) but is weaker at detecting non-existent associations, dragging down its overall score.

---

## 5. Contradiction Analysis

Contradictions occur when a model answers a standard MCQ correctly but fails the corresponding NOTA version (or vice versa).

### 5.1 Variant MCQ (120 annotation pairs)

| Model | Std correct, NOTA wrong | Std wrong, NOTA correct | Total Rate |
|-------|:-----------------------:|:-----------------------:|:----------:|
| Claude Haiku 4.5 | 32.5% | 20.0% | 52.5% |
| Claude Sonnet 4.5 | 45.8% | 3.3% | 49.2% |
| GPT-4o-mini | 18.3% | 10.8% | 29.2% |
| GPT-5.2 | 17.5% | 8.3% | 25.8% |
| Claude Opus 4.6 | 20.8% | 1.7% | 22.5% |

### 5.2 Drug MCQ (120 annotation pairs)

| Model | Std correct, NOTA wrong | Std wrong, NOTA correct | Total Rate |
|-------|:-----------------------:|:-----------------------:|:----------:|
| Claude Sonnet 4.5 | 40.0% | 0.8% | 40.8% |
| Claude Haiku 4.5 | 32.5% | 3.3% | 35.8% |
| GPT-4o-mini | 23.3% | 0.0% | 23.3% |
| GPT-5.2 | 7.5% | 0.8% | 8.3% |
| Claude Opus 4.6 | 5.8% | 0.0% | 5.8% |

### 5.3 Phenotype MCQ (114 annotation pairs)

| Model | Std correct, NOTA wrong | Std wrong, NOTA correct | Total Rate |
|-------|:-----------------------:|:-----------------------:|:----------:|
| GPT-5.2 | 40.4% | 16.7% | 57.0% |
| Claude Opus 4.6 | 36.8% | 14.0% | 50.9% |
| GPT-4o-mini | 46.5% | 1.8% | 48.2% |
| Claude Sonnet 4.5 | 39.5% | 5.3% | 44.7% |

Contradiction rates are high across all tasks and models, especially on phenotype (45-57%). Sonnet shows a distinctive pattern: very high "standard correct, NOTA wrong" rates with near-zero reverse contradictions, suggesting systematic over-commitment rather than random errors.
