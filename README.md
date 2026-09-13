# Grounded AI Customer Support Assistant (Hiver SDE Intern Assignment)

An evidence-grounded, reproducible AI Customer Support system built on the *Customer Support on Twitter* dataset (`twcs.csv`) for `@AmazonHelp`.

---

## 🌟 Overview

This system automates and assists tier-1 customer support operations with four core capabilities:
1. **Production Intent Classification:** Classifies customer inquiries into 7 data-derived intents with calibrated confidence. The production default is the validated **TF-IDF + Logistic Regression** model (89.00% Accuracy / 89.41% Macro F1), with Sentence Transformers available as a baseline and embedding provider.
2. **Historical Resolution Retrieval:** Retrieves top semantically similar historical resolutions using strict candidate filtering by predicted intent, strict conversation-level leakage masking, and regex PII/name sanitization.
3. **Grounded Reply Generation:** Drafts concise, empathetic customer support responses constrained strictly by historical evidence (zero hallucination of policies or credits, no customer name leakage, 100% compliant with Twitter's 280-character limit).
4. **Intelligent Escalation Routing:** Deterministically routes inquiries between `AUTO_HANDLE` and `ESCALATE_TO_HUMAN` based on confidence thresholds ($\ge 0.65$), retrieval evidence similarity ($\ge 0.60$), sensitive keywords, and security policies.

---

## 📊 Final Measured Results Table

All metrics below are 100% measured on the held-out Golden Evaluation Set ($N=200$):

| System / Model | Intent Accuracy | Intent Macro F1 | Intent Precision | Intent Recall | Mean Quality (1-5) | Groundedness (1-5) | Escalation Auto-Handle Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Baseline** | 12.50% | 3.17% | 1.79% | 14.29% | N/A *(Trivial)* | N/A | N/A |
| **TF-IDF + Logistic Regression (Champion)** | **89.00%** | **89.41%** | **91.78%** | **88.83%** | N/A *(Classifier only)* | N/A | N/A |
| **Proposed Semantic Pipeline** | **83.00%** | **82.78%** | **84.02%** | **82.51%** | **3.88 / 5.0** | **4.48 / 5.0** | **53.0%** *(47.0% Escalated)* |

* **Twitter 280-Character Limit Compliance:** **100.0%**
* **Unsupported Claims / Hallucination Rate:** **0.0%** (0 hallucinated policies across 200 evaluated queries)
* **Unit Test Suite:** **28 / 28 tests passing (100% PASS)**

---

## 🚀 Quickstart & Reproduction Guide (< 15 Minutes)

### 1. Environment Setup

Clone repository and activate the Python virtual environment:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

*(Optional)* Configure API credentials for live LLM generation/evaluation:
```powershell
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY or GEMINI_API_KEY
```
*(Note: If no API key is provided, the system seamlessly uses its deterministic offline synthesis and auditing engine.)*

---

### 2. Run the Full Reproducible Evaluation Suite

To run all 5 evaluation modules sequentially (Intents, Groundedness, Escalation, LLM Judge, Human Agreement):

```powershell
python run_pipeline.py --evaluate
```

Output reports will be automatically refreshed in `reports/`:
* `reports/intent_evaluation_results.json`
* `reports/reply_evaluation_results.json`
* `reports/escalation_evaluation_results.json`
* `reports/llm_judge_evaluations.json`
* `reports/human_judge_agreement_results.json`

---

### 3. Run Single Query or Interactive CLI

**Process sample customer inquiries:**
```powershell
python run_pipeline.py --query "Where is my package? It was supposed to arrive yesterday."
python run_pipeline.py --query "I was charged twice for the same order."
python run_pipeline.py --query "I want to return this item and get a refund."
python run_pipeline.py --query "I cannot log into my Amazon account."
python run_pipeline.py --query "My Prime subscription was charged unexpectedly."
python run_pipeline.py --query "Something went wrong with my account and I need help."
```

**Launch Interactive CLI REPL:**
```powershell
python run_pipeline.py --interactive
```

---

### 4. Launch the Interactive Streamlit Web Demo

```powershell
streamlit run app/app.py
```
Open your browser at `http://localhost:8501` to test custom customer support queries and view real-time intent confidence, retrieved historical evidence cards, grounded replies, and escalation routing badges.

---

### 5. Run the Automated Unit Test Suite

```powershell
pytest -q
```
Executes all 28 unit tests covering preprocessing, graph thread reconstruction, intent classification, retrieval with leakage masking, escalation routing, reply generation, evaluation entrypoints, and the end-to-end pipeline.

---

## 🏗️ Project Architecture & Directory Structure

```
hiver-sde-ai-support-agent/
│
├── data/
│   ├── raw/                               # Expected location for twcs.csv (optional for evaluation)
│   ├── processed/
│   │   ├── customer_brand_pairs.parquet   # 1,261,888 verified Customer -> Brand pairs
│   │   ├── conversation_threads.parquet   # 798,197 reconstructed conversation trees
│   │   └── brand_candidate_metrics.csv    # Empirical volume & latency metrics for 108 brands
│   ├── splits/                            # Conversation-level leak-free splits (70/15/15)
│   │   ├── train_pairs.parquet            # 118,246 training pairs
│   │   ├── val_pairs.parquet              # 25,103 validation pairs
│   │   └── test_pairs.parquet             # 25,465 test pairs
│   └── golden/
│       └── golden_set.csv                 # 200 held-out evaluated customer queries
│
├── configs/
│   ├── config.yaml                        # Master system thresholds and hyperparameters
│   └── intent_taxonomy.yaml               # 7 empirical MECE intent definitions
│
├── src/
│   ├── preprocessing.py                   # Text cleaning and RFC 2822 date parsing
│   ├── conversation_builder.py            # Graph path compression & dialogue thread builder
│   ├── brand_selection.py                 # Multi-brand empirical evaluator
│   ├── intent_classifier.py               # Majority, TF-IDF + LR, and Semantic Classifiers
│   ├── retrieval.py                       # Dense vector retrieval with leakage masking & PII filter
│   ├── reply_generator.py                 # Grounded response generator with guardrails
│   ├── escalation.py                      # Multi-rule escalation decision router
│   └── pipeline.py                        # Unified end-to-end SupportAgentPipeline
│
├── evaluation/
│   ├── evaluate_intents.py                # Classifier benchmarking across Golden Set
│   ├── evaluate_replies.py                # Token overlap, semantic similarity & hallucination check
│   ├── evaluate_escalation.py             # Escalation routing & difficulty stratification
│   ├── llm_judge.py                       # 5-dimension quality audit engine
│   └── human_judge_agreement.py           # Spearman correlation & MAE inter-rater agreement
│
├── app/
│   └── app.py                             # Streamlit interactive demonstration UI
│
├── reports/
│   ├── phase2_data_quality.md             # Audited Phase 2 data cleaning report
│   ├── brand_selection.md                 # Evidence-based brand selection report
│   ├── intent_taxonomy.md                 # Empirical intent discovery report
│   ├── failure_analysis.md                # Top 5 real failure modes and mitigations
│   ├── decision_log.md                    # 14 detailed architectural decisions
│   ├── hiver_requirements_checklist.md   # Complete requirements compliance audit
│   └── final_report.md                    # Comprehensive submission report
│
├── tests/
│   ├── test_preprocessing.py              # Text normalization & timestamp unit tests
│   ├── test_conversation_builder.py       # Graph tree reconstruction unit tests
│   ├── test_intent_classifier.py          # Classifier prediction unit tests
│   ├── test_retrieval.py                  # Retrieval & leakage exclusion unit tests
│   ├── test_escalation.py                 # Escalation threshold & rule unit tests
│   ├── test_pipeline.py                   # End-to-end pipeline integration tests
│   └── test_evaluation_entrypoints.py     # Evaluation script import & execution tests
│
├── .env.example                           # Template for optional API credentials
├── .gitignore                             # Protects secrets, cache, and raw binaries
├── requirements.txt                       # Project dependencies
├── run_pipeline.py                        # Unified CLI runner
└── README.md                              # This document
```

---

## 🛡️ Anti-Leakage & Safety Architecture

1. **Conversation-Level Partitioning:** Dataset splitting was conducted strictly at the `conversation_id` tree level (0% conversation overlap between train, val, and test splits).
2. **Query Conversation Retrieval Masking:** The retrieval engine actively masks out all historical turns sharing the query's conversation ID, ensuring test queries never retrieve their own dialogue turns.
3. **Pre-Ranking Intent Masking & PII Stripping:** Candidate pools are filtered by predicted intent before similarity selection. Responses are sanitized to strip other customer names (`Hi, Amy!`), handles, agent initials, and tweet fragment numbers.
4. **Data Bundling Note:** All processed parquet splits (`train_pairs.parquet`, `test_pairs.parquet`) and the 200-sample Golden Set are pre-bundled in `data/`, so reviewers do NOT need to download the full 3M-row `twcs.csv` raw dataset to run the complete pipeline, tests, and evaluation.

---

## 🚫 What Was NOT Built (Intentional Exclusions)

To maintain focus, avoid bloat, and ensure total reproducibility, the following components were intentionally excluded:
* **Live Twitter/X Scraping API:** Avoided due to rate limits and external dependency failures.
* **Autonomous Public Tweet Posting:** Automated agents should never post publicly without human oversight.
* **Distributed Cloud Infrastructure (Kubernetes / AWS ECS / PostgreSQL):** Unnecessary for evaluating core NLP and routing mechanics.
* **External Paid Vector DBs (Pinecone / Milvus):** Replaced with memory-efficient dense vector indexing using Scikit-Learn / NumPy.
* **Autonomous Financial Transaction Execution:** The system provides guidance and links; it does not directly refund bank accounts.

---

## 📚 Citations & Acknowledgments

* **Dataset:** *Customer Support on Twitter* (`twcs.csv`), Kaggle by ThoughtVector (2017).
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Reimers & Gurevych, 2019).
* **Machine Learning:** `scikit-learn` (Pedregosa et al., 2011).

