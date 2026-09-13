# Grounded AI Customer Support Assistant for Twitter: Engineering, Evaluation, and Architectural Report

**Author:** SDE Intern Candidate  
**Project Repository:** `hiver-sde-ai-support-agent`  
**Target Brand:** `AmazonHelp` (@AmazonHelp)  
**Dataset:** Customer Support on Twitter (`twcs.csv`, Kaggle: `thoughtvector/customer-support-on-twitter`)  
**Status:** Complete, Audited & Submission-Ready  

---

## 1. Problem Framing

Customer support operations on public social platforms such as Twitter/X face unique operational challenges:
1. **High Query Velocity & Response Latency Sensitivity:** Customers expect immediate initial contact (median response time for top brands is under 15 minutes).
2. **Extreme Text Sparsity & High Noise:** Tweets contain heavy abbreviations, emotional venting, incomplete context, and non-standard syntax.
3. **Severe Hallucination & Reputation Risk:** An automated agent that invents refund promises, promises unrealistic delivery dates, or hallucinates company policies causes direct financial and legal liability.

The goal of this project is to build an explainable, evidence-grounded AI Customer Support Agent that:
* Classifies incoming customer messages into a compact, empirical intent taxonomy.
* Retrieves semantically relevant historical brand resolutions.
* Generates concise, grounded replies strictly adhering to brand support policies.
* Routes ambiguous, high-risk, or unsupported inquiries to human agents (`ESCALATE_TO_HUMAN`) while safely resolving standard inquiries (`AUTO_HANDLE`).

---

## 2. What "Good" Means for This System

In an enterprise customer support setting, high accuracy alone does not guarantee business viability. A successful system must satisfy four operational criteria:

1. **Factual Groundedness (Zero Hallucination):** The model must never invent monetary credits, guarantee specific delivery times, or claim an action was taken without backend confirmation.
2. **Calibrated Escalation (Safety-First Routing):** Ambiguous requests, sensitive legal inquiries, or novel edge cases must be safely escalated to human specialists rather than guessing an answer.
3. **Explainability & Transparency:** Every routing decision must produce a clear, human-readable justification and confidence score for auditing.
4. **Reproducibility & Computational Efficiency:** The system must run on standard developer hardware (CPU inference) in under 15 minutes without expensive infrastructure overhead.

---

## 3. Dataset Characteristics & Audit

The underlying data is the *Customer Support on Twitter* dataset (`twcs.csv`), comprising **2,811,774 customer-support tweets** across **108 global brands**.

* **Total Raw Records:** 2,811,774
* **Cleaned Records:** 2,811,774 (0 empty rows, 0 exact duplicate rows)
* **Inbound (Customer) Messages:** 1,537,843 (54.7%)
* **Outbound (Brand) Messages:** 1,273,931 (45.3%)
* **Unique Authors:** 702,777 (702,669 customers, 108 brand handles)
* **Date Range:** October 10, 2017 to November 30, 2017

---

## 4. Data Preparation & Conversation Reconstruction

Raw Twitter records are individual tweets linked only by `tweet_id` and `in_response_to_tweet_id`. To convert these isolated turns into structured support dialogues:

1. **Graph Path Compression:** Built an adjacency graph of parent-child tweet pointers, traversing upwards to identify the canonical root tweet of every dialogue tree.
2. **Reconstructed Conversation Trees:** Reconstructed **798,197 distinct conversation trees** (254,775 multi-turn conversations; 543,422 single-turn interactions).
3. **Customer $\rightarrow$ Brand Pair Extraction:** Mapped every inbound customer turn to the immediate subsequent brand reply, producing **1,261,888 verified Customer $\rightarrow$ Brand interaction pairs**.
4. **Leakage-Free Partitioning:** Grouped pairs by `conversation_id` into a strict **70% Train (118,246 pairs)**, **15% Validation (25,103 pairs)**, and **15% Test (25,465 pairs)** split with **0% conversation overlap**.

---

## 5. Evidence-Based Brand Selection (`AmazonHelp`)

Rather than arbitrarily picking a company, candidate brands were evaluated using empirical metrics from `brand_candidate_metrics.csv`:

| Brand | Total Support Pairs | Multi-Turn % | Avg Turns / Thread | Median Latency (min) | Selection Rationale |
|---|---|---|---|---|---|
| **`AmazonHelp`** | **168,814** | **49.8%** | **4.53** | **11.5** | **SELECTED:** Highest volume, deep multi-turn context, diverse e-commerce intents. |
| `AppleSupport` | 106,719 | 43.2% | 3.89 | 14.2 | Excluded: Skewed toward iOS software updates & hardware battery diagnostics. |
| `Uber_Support` | 56,269 | 38.1% | 3.12 | 8.9 | Excluded: Heavy reliance on in-app driver trip dispute links. |
| `SpotifyCares` | 43,118 | 36.4% | 2.95 | 16.0 | Excluded: Narrow domain focused primarily on music streaming playback errors. |

`AmazonHelp` was selected because it represents a complete, diverse e-commerce customer support workload with sufficient data for training, semantic retrieval indexing, and evaluation.

---

## 6. Empirically Derived Intent Taxonomy

Through n-gram frequency analysis and semantic clustering on 168,814 `AmazonHelp` inquiries, customer requests grouped cleanly into **7 Mutually Exclusive & Collectively Exhaustive (MECE)** intents:

1. `delivery_tracking_delay`: Package status, carrier delays, missing tracking numbers, late shipments.
2. `refund_return_cancellation`: Return labels, refund status, order cancellation requests.
3. `damaged_defective_item`: Broken goods, incorrect items delivered, quality defects, replacements.
4. `account_access_security`: Locked accounts, password resets, unauthorized charges, 2FA issues.
5. `payment_billing_issue`: Double billing, payment method failures, gift card balance errors, invoices.
6. `prime_subscription_services`: Prime video/music streaming, membership cancellations, student trials.
7. `general_inquiry_support`: General feedback, store availability, international portal inquiries.

---

## 7. System Architecture

The AI Support Agent operates as a modular, 4-stage pipeline:

```
[ Incoming Customer Message ]
             │
             ▼
┌─────────────────────────────────────────┐
│ 1. Intent Classification Head           │  ──> Intent Label + Calibrated Confidence
│    Sentence Transformer (all-MiniLM-L6) │
└─────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 2. Historical Retrieval Engine          │  ──> Top-3 Historical Support Pairs
│    Cosine Similarity + Leakage Masking  │      (Query Conversation Masked)
└─────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 3. Escalation Decision Router           │  ──> AUTO_HANDLE  vs.  ESCALATE_TO_HUMAN
│    Confidence / Evidence / Risk Rules   │      + Detailed Justification
└─────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 4. Grounded Reply Generator             │  ──> Evidence-Constrained Support Reply
│    Anti-Hallucination & Policy Guardrails│      (<280 chars, No AI self-reference)
└─────────────────────────────────────────┘
```

---

## 8. Baselines

To establish rigorous benchmarks, two baselines were implemented and evaluated on the exact same Golden Set ($N=200$):

1. **Majority-Class Baseline:** Naively predicts the most frequent intent (`general_inquiry_support`). Serves as the minimal performance floor.
2. **TF-IDF + Logistic Regression Baseline:** Traditional machine learning baseline using sublinear TF-IDF word and character n-grams ($1-2$) with balanced class weighting and $L_2$ regularization.

---

## 9. Proposed Semantic Approach

The proposed **Semantic Intent Classifier** utilizes `all-MiniLM-L6-v2` to map customer text into 384-dimensional dense semantic vectors, feeding a calibrated multinomial logistic regression classification head with temperature-scaled confidence output.

* **Advantages:** Continuous semantic representations capture out-of-vocabulary synonyms, typo variations, and emotional expressions without requiring rigid keyword matches.
* **Calibrated Confidence:** Probability outputs reflect true prediction reliability, directly driving the downstream escalation router.

---

## 10. Historical Grounding Engine

The retrieval engine maintains an indexed vector store of 10,000 historical `AmazonHelp` support interactions.

* **Retrieval Leakage Prevention:** During evaluation, the retriever enforces an `exclude_conversation_id` constraint, guaranteeing that a test query cannot retrieve its own dialogue turns or near-duplicate sibling turns from the same conversation tree.
* **Evidence Ranking:** Queries are matched via normalized cosine similarity. Matches with similarity $<0.60$ trigger evidence-insufficiency escalations.

---

## 11. Escalation Decision Logic

The `EscalationRouter` executes four deterministic, safety-first routing rules:

1. **Sensitive / Legal / Fraud Trigger:** Explicit detection of high-risk keywords (`lawyer`, `attorney`, `police`, `fraud`, `human agent`, `supervisor`) immediately routes to `ESCALATE_TO_HUMAN`.
2. **High-Risk Security Policy:** All inquiries classified under `account_access_security` require human verification and are escalated.
3. **Low Intent Confidence:** If intent confidence is $<0.65$, query is marked ambiguous and escalated.
4. **Weak Retrieval Evidence:** If top historical evidence similarity is $<0.60$, query is escalated due to lack of historical precedent.
5. **Default Auto-Handle:** When confidence $\ge 0.65$ and retrieval evidence $\ge 0.60$, the system permits `AUTO_HANDLE`.

---

## 12. Evaluation Methodology & Golden Set

Evaluation was conducted on an independently verified **Golden Evaluation Set of 200 real customer inquiries** (`data/golden/golden_set.csv`) drawn from the held-out test split:

* **Intent Stratification:** Balanced representation across all 7 intents (~28-30 per category).
* **Difficulty Stratification:** Categorized into **Easy** (30), **Medium** (122), and **Hard** (48) based on lexical ambiguity, sarcasm, and multi-issue complexity.
* **Zero Leakage:** Complete isolation from training and retrieval sets.

---

## 13. Results Table

All metrics below were measured directly on the 200-sample Golden Evaluation Set:

| System / Model | Intent Accuracy | Intent Macro F1 | Intent Precision | Intent Recall | Reply Quality (1-5) | Groundedness (1-5) | Escalation Auto-Handle Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority-Class Baseline** | 12.50% | 3.17% | 1.79% | 14.29% | N/A *(Trivial)* | N/A | N/A |
| **TF-IDF + Logistic Regression** | **89.00%** | **89.41%** | **91.78%** | **88.83%** | N/A *(Classifier only)* | N/A | N/A |
| **Proposed Semantic Pipeline** | **83.00%** | **82.78%** | **84.02%** | **82.51%** | **3.88 / 5.0** | **4.48 / 5.0** | **53.0%** *(47.0% Escalated)* |

*Note on Comparison:* TF-IDF + Logistic Regression achieves higher lexical accuracy (89.00%) due to exact keyword matching on specific e-commerce terms (*"refund"*, *"damaged"*), whereas the Semantic Transformer (83.00%) provides smoother embeddings and calibrated probability confidence scores necessary for reliable escalation routing.

---

## 14. LLM-as-Judge & Human Agreement Validation

Evaluation of generated replies across the Golden Set using the 5-dimension rubric:

* **Correctness:** 4.05 / 5.0
* **Groundedness:** 4.48 / 5.0
* **Helpfulness:** 3.48 / 5.0
* **Relevance:** 3.65 / 5.0
* **Tone:** 3.75 / 5.0
* **Overall Quality:** **3.88 / 5.0**
* **Unsupported Claims Rate:** **0.0%** (0 hallucinated claims across 200 evaluated queries)

### Human vs. LLM Judge Agreement ($N=50$)
* **Exact Point Agreement:** **28.0%**
* **Adjacent Agreement ($\pm 1$ point):** **78.4%**
* **Mean Absolute Error (MAE):** **0.936 points**
* **Finding:** The automated judge is slightly stricter on helpfulness and relevance than human annotators, prioritizing exact DM actionability over generic empathy.

---

## 15. Top 5 Failure Modes

Auditing real failure cases on the Golden Set identified 5 key failure modes:

1. **Multi-Intent Compound Queries (6.5%):** Customers stating a damaged item while asking about review rejection (`GOLDEN_001`). *Mitigation: Multi-label classification head.*
2. **Multilingual Language Drift (3.0%):** Non-English inquiries to international Amazon accounts (`GOLDEN_012`). *Mitigation: FastText language pre-filter.*
3. **Stale Shortened URLs in Retrievals (8.0%):** Historical 2017 `https://t.co/...` links in retrieved evidence. *Mitigation: Canonical URL regex replacement.*
4. **Over-Escalation on Colloquial Slang (9.5%):** Sarcasm or unusual punctuation lowering confidence below 0.65 (`GOLDEN_002`). *Mitigation: Isotonic threshold calibration on validation split.*
5. **Context Omission in Follow-Up Turns (5.0%):** Tweets like *"Sent DM"* lacking prior problem context. *Mitigation: Thread concatenation from dialogue history.*

---

## 16. What is misleading about my headline number?

Our strongest measured metric is an **89.00% Intent Accuracy** on the TF-IDF baseline and an **83.00% Intent Accuracy** on the Semantic Transformer, alongside an impressive **0.0% Hallucination Rate** and **53.0% Auto-Handle Rate**.

However, treating these headline numbers as proof of production readiness is misleading for five critical reasons:

1. **Held-Out Golden Set Size ($N=200$):** While representative and stratified, a 200-sample evaluation set has a 95% confidence interval margin of approximately $\pm 5.0\%$. Real-world deployment involves millions of queries with long-tail distributions.
2. **Intent Accuracy Overestimates End-to-End Satisfaction:** Classifying a message as `delivery_tracking_delay` is relatively easy; successfully calming an angry customer whose package was stolen requires backend CRM integration, live logistics tracking, and personalized resolution that an offline model cannot perform.
3. **Single-Brand Specialization Bias:** High performance on `@AmazonHelp` does not transfer directly to other brands (e.g., airline baggage policies or telecom network outages) where intent definitions and resolution protocols differ fundamentally.
4. **Historical Dataset Staleness (2017 Twitter Constraints):** The dataset originates from 2017 when Twitter had a 140/280 character limit and distinct DM routing workflows. Modern support interactions often occur on WhatsApp, live web chat, or in-app messaging with multi-paragraph descriptions.
5. **Conservative Escalation Inflation:** Our 0.0% hallucination rate was achieved partly through conservative escalation rules (escalating 47.0% of queries). In a commercial contact center, escalating nearly half of all incoming volume would impose substantial human labor costs.

---

## 17. What Was NOT Built (Intentional Exclusions)

To maintain focus, avoid bloat, and ensure total reproducibility, the following production components were intentionally excluded:
* **Live Twitter/X Scraping API:** Avoided due to Twitter API rate limits and breaking API changes.
* **Autonomous Public Tweet Posting:** Automated bots should never post live tweets without human agent review.
* **Distributed Cloud Infrastructure (Kubernetes / AWS ECS):** Unnecessary for evaluating core NLP and routing mechanics.
* **External Paid Vector DBs (Pinecone / Milvus):** Replaced with memory-efficient dense vector indexing using Scikit-Learn / NumPy.
* **Autonomous Financial Transaction Execution:** The system provides guidance and links; it does not directly refund bank accounts.

---

## 18. One-Week Next Steps

A prioritized 7-day engineering improvement plan:

| Priority | Task Description | Impact | Effort | Risk |
|---|---|---|---|---|
| **P1** | **URL Sanitizer:** Replace historical `t.co` links in retrieved evidence with canonical `amazon.com/help` URLs. | High | Low | Low |
| **P2** | **Multi-Turn Context Ingestion:** Prepend previous conversation turns from `thread_context` into query embeddings. | High | Medium | Low |
| **P3** | **Threshold Calibration:** Use validation split cross-validation to tune escalation thresholds ($0.65 \rightarrow 0.60$). | Medium | Low | Medium |
| **P4** | **Multi-Label Intent Support:** Migrate softmax head to binary cross-entropy sigmoid for compound query support. | High | Medium | Medium |
| **P5** | **Language Pre-Filter:** Integrate `langdetect` to automatically route non-English queries to specialized language queues. | Medium | Low | Low |

---

## 19. Limitations & Citations

### System Limitations
* **Monolingual Focus:** Optimized for English customer support inquiries.
* **Static Retrieval Index:** Indexed historical interactions do not update dynamically in real time.
* **Single-Turn Preference:** Standalone evaluation treats messages independently unless conversation context is explicitly passed.

### Citations & Attribution
1. **Dataset:** *Customer Support on Twitter* (`twcs.csv`), Kaggle dataset by ThoughtVector (2017).
2. **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Reimers & Gurevych, 2019).
3. **Machine Learning Framework:** `scikit-learn` (Pedregosa et al., 2011).
4. **Evaluation Tooling:** `scipy` (Virtanen et al., 2020), `pytest`, `pandas`, `pyarrow`.
5. **Demonstration UI:** `Streamlit` (2024).
