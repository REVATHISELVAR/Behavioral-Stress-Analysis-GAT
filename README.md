## Behavioral-Stress-Analysis-GAT 

Predicting depression/stress severity (`Minimal` → `Severe`) by combining
standard PHQ-9 clinical questionnaire responses with everyday behavioral
factors — sleep quality, academic/study pressure, and financial pressure —
using a Random Forest baseline and a GATv2 Graph Attention Network, with
SHAP-based explainability for every individual prediction.

---

## Table of Contents
- [Why This Project Is Different](#why-this-project-is-different)
- [Application Domain](#application-domain)
- [Technology Stack](#technology-stack)
- [Results](#results)
- [Novelty / Contribution](#novelty--contribution)
- [A Note on Data Leakage](#a-note-on-data-leakage-important)
- [Repo Contents](#repo-contents)
- [Methodology](#methodology)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)
- [Team & Contributions](#team--contributions)
- [Data Source / License](#data-source--license)

---

## Why This Project Is Different

The standard PHQ-9 test scores depression severity using **only** 9 fixed
questions, added up and bucketed into a category — that's arithmetic, not
machine learning.

This project asks: **what if we also factor in everyday life stressors —
sleep quality, study pressure, financial pressure — alongside the clinical
questions, and let a model learn the combined pattern instead of just
summing scores?**

The result is a model that:
- Learns from clinical + behavioral signals together, not the questionnaire
  total alone
- Explains *why* it made each individual prediction (which specific factors
  mattered most for that person)
- Is built and evaluated carefully enough to avoid the most common silent
  mistake in projects like this — see [Data Leakage](#a-note-on-data-leakage-important)

---

## Application Domain

- **Mental health screening / digital health informatics** — early,
  low-friction risk-flagging based on a mix of clinical self-report and
  everyday behavioral/contextual stressors.
- **Student wellness** — the dataset's "Study Pressure" and "Financial
  Pressure" features make this particularly relevant to student/academic
  populations, where these are common, under-measured stress contributors.
- **Explainable AI (XAI) in healthcare-adjacent ML** — demonstrates
  per-individual model interpretability (SHAP) rather than a black-box
  score, which is a practical requirement for any tool that touches mental
  health.
- **Graph-based learning on tabular survey data** — applies a Graph
  Attention Network (typically used on naturally graph-structured data like
  social networks or molecules) to a tabular survey dataset by constructing
  a similarity graph between respondents.

---

## Technology Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.12 |
| Data handling | pandas, NumPy |
| Classical ML | scikit-learn (Random Forest, train/test split, metrics) |
| Class imbalance handling | imbalanced-learn (SMOTE) |
| Deep learning | PyTorch |
| Graph neural networks | PyTorch Geometric (GATv2Conv) |
| Explainability | SHAP (TreeExplainer) |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |

---

## Results

| Model | Accuracy | Macro F1 | Cohen's Kappa | AUC (macro, OvR) |
|---|---|---|---|---|
| Random Forest | 85.4% | 0.85 | 0.81 | 0.97 |
| GATv2 (Graph Attention Network) | 83.5% | 0.82 | 0.79 | 0.97 |

682 respondents, stratified train/test splits, seed = 42. Full per-class
precision/recall/F1 breakdown is in the notebook (Sections 3 and 5).

---

## Novelty / Contribution

This project's novelty is **multi-domain feature fusion for explainable
severity prediction**, specifically:

1. **Combining clinical + behavioral signals in one model.** Rather than
   scoring the 9 PHQ-9 items in isolation (as the standard PHQ-9 cutoff
   does), this model is trained on the PHQ-9 items *together with*
   sleep quality, study pressure, financial pressure, age, and gender —
   plus two engineered interaction terms (`Sleep × Study`,
   `Study × Financial`) that capture compounding stress effects two
   factors can have together.
2. **Graph-based representation of respondents.** A k-nearest-neighbor
   similarity graph is built over respondents (cosine similarity on
   standardized features), and a GATv2 model is trained
   node-classification-style on this graph — so a respondent's predicted
   severity is informed by similar respondents in the dataset, not just
   their own answers in isolation.
3. **Per-prediction explainability.** Every prediction — whether from the
   worked examples or the interactive quiz — comes with a SHAP-based
   breakdown of exactly which factors pushed that specific person's score
   up or down, not just a global feature-importance ranking.

This is **not** a claim of a new model architecture — GATv2 and SHAP are
established techniques. The contribution is the careful combination of
clinical + lifestyle features, the leakage-safe evaluation methodology, and
the explainability layer wired through to live predictions.

---

## A Note on Data Leakage (important)

The raw dataset includes a `PHQ_Total` column (the sum of the 9 PHQ-9 item
scores). The label this project predicts, `PHQ_Severity`, is **directly
calculated from `PHQ_Total`** using fixed clinical cutoffs:

| Severity | PHQ_Total range |
|---|---|
| Minimal | 0–4 |
| Mild | 5–9 |
| Moderate | 10–14 |
| Moderately Severe | 15–19 |
| Severe | 20–27 |

These ranges never overlap. A model given `PHQ_Total` as an input feature
doesn't need to learn anything — it can reach ~100% accuracy simply by
reading off which bucket the number falls into. That is target leakage, not
a genuine result.

**`PHQ_Total` and `PHQ_Severity` are excluded from the model's input
features in this project.** The model only ever sees the 9 individual
PHQ-9 answers plus sleep/study/financial pressure, age, and gender — never
the precomputed total. The accuracies reported above are the honest result
of that.

---

## Repo Contents

| File | Description |
|---|---|
| `Behavioral_stress_detection.ipynb` | Full pipeline: preprocessing, Random Forest + SHAP, GATv2 + SHAP, predict & explain functions, interactive quiz, limitations |
| `PHQ-9_Dataset_5th_Edition.csv` | Dataset used (682 rows, unmodified) |
| `README.md` | This file |

---

## Methodology

1. **Preprocessing** — PHQ-9 items mapped to a 0–3 scale; sleep/study/
   financial pressure mapped to a 0–3 scale; gender binarized. All mappings
   are case/whitespace-normalized before lookup. Missing values are never
   silently filled with `0` (since `0` is itself a valid category code) —
   the pipeline raises an error instead of masking gaps.
2. **Random Forest** — 300 trees, max depth 10, stratified 80/20 train/test
   split.
3. **GATv2** — a k-nearest-neighbor graph (k=8, cosine similarity) is built
   over standardized respondent feature vectors; a 2-layer GATv2 with
   residual connections and layer normalization is trained
   node-classification-style. SMOTE oversampling balances minority severity
   classes within the training split before the graph is built.
4. **Explainability** — SHAP `TreeExplainer` on the Random Forest provides
   both global feature importance and per-prediction explanations (used as
   a fast, exact proxy explanation alongside the GATv2 prediction — not a
   direct decomposition of GATv2's internal attention weights, which is
   called out explicitly in the notebook).
5. **Predicting a new respondent** — `predict_severity(...)` and
   `explain_prediction(...)` take explicit keyword arguments and run
   unattended; an interactive quiz (Section 7b) wraps both with `input()`
   prompts for a live, typed Q&A experience that prints the prediction and
   its explanation together.

---

## How to Run

```bash
pip install torch torch_geometric shap imbalanced-learn scikit-learn pandas matplotlib
jupyter notebook Behavioral_stress_detection.ipynb
```

Run all cells top-to-bottom for the full pipeline and results. To try the
interactive quiz, run **Section 7b's cell on its own** (not via "Run All") —
it will prompt you for answers one at a time and print the prediction plus
SHAP explanation.

---

## Limitations

- 682 respondents is a modest sample for a 5-class problem; the `Severe`
  class has the fewest examples, so its metrics are the least stable.
- All data is self-reported at a single point in time — results show
  association, not causation (e.g. "financial pressure was linked to higher
  predicted severity," not "financial pressure causes depression").
- SHAP explanations for the GATv2 predictions use the Random Forest's view
  of the same input as a fast proxy explanation, not an exact decomposition
  of GATv2's internal attention mechanism.
- This is a research/academic prototype, not a validated clinical or
  diagnostic tool.

---

## Disclaimer

This project is intended **for academic and research purposes only**. It is
**not** a certified medical or diagnostic tool and must not be used as a
substitute for professional clinical assessment, diagnosis, or treatment.
PHQ-9 severity bands are a screening heuristic used in clinical practice to
guide further evaluation — they are not, by themselves, a diagnosis of
depression or any other mental health condition.

If you or someone you know is struggling with depression, anxiety, or
suicidal thoughts, please reach out to a licensed mental health
professional or a crisis helpline in your country. This project's authors
and contributors are not medical professionals and accept no liability for
decisions made based on this model's output.

---

## Team & Contributions

| Name | Role | Contribution |
|---|---|---|
| **Revathi** | Data & Modeling | Dataset preprocessing and cleaning, feature engineering (PHQ-9 mappings, interaction terms), Random Forest baseline model, GATv2 graph model design and training, model evaluation (accuracy, F1, Kappa, AUC) |
| **Sivajith** | Explainability & Application | SHAP explainability integration (global + per-prediction), interactive prediction interface, results visualization, documentation and repo structuring |

*(Adjust the split above to match what each of you actually worked on —
this is a starting template. If contributions overlapped, you can phrase
shared sections as "Revathi & Sivajith" instead of assigning to one name.)*

---

## Data Source / License

Check the original source and license terms of the PHQ-9 dataset before
redistributing it further — questionnaire-based datasets sometimes carry
usage or attribution restrictions even when shared publicly. If this
dataset was obtained from a public repository (e.g. Kaggle), credit the
original uploader/source here.