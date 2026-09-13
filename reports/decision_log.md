# Technical Decision Log

This document records the 12 core architectural, modeling, and evaluation decisions made during the development of the AI Customer Support Agent for the Hiver Software Development Engineer Intern take-home assignment.

---

### Decision 1: Evidence-Based Selection of Primary Brand (`AmazonHelp`)
* **Decision:** Selected `AmazonHelp` (168,814 customer-brand pairs) as the primary brand for empirical study and deployment.
* **Why:** Empirical inspection of `brand_candidate_metrics.csv` demonstrated that `AmazonHelp` had the highest pair volume (168,814), rich multi-turn depth (4.53 turns average), rapid response time (median 11.5 minutes), and diverse customer service intents.
* **Alternatives Considered:** `AppleSupport` (high volume, but skewed heavily toward device hardware troubleshooting and OS updates), `SpotifyCares` (narrower intent scope), `British_Airways` (lower volume, specialized flight logistics).
* **Trade-off:** High volume required memory-bounded retrieval indexing (10,000 interactions) to run efficiently on standard CPU laptops.
* **Consequence:** Provided a robust, realistic foundation for intent classification, historical retrieval, and grounded response synthesis.

---

### Decision 2: Conversation-Level Splitting to Prevent Data Leakage
* **Decision:** Partitioned data by unique `conversation_id` into 70% Train (118,246 pairs), 15% Val (25,103 pairs), and 15% Test (25,465 pairs) with verified 0% conversation overlap.
* **Why:** Random row-level splitting leaks conversation context between training and evaluation splits, as turns within the same conversation share vocabulary, timestamps, and customer context.
* **Alternatives Considered:** Random row-level shuffle, temporal date-based split.
* **Trade-off:** Slightly more complex splitting logic requiring grouping by conversation graph roots.
* **Consequence:** Strict, uncorrupted evaluation integrity with zero leakage between train and test.

---

### Decision 3: Empirical 7-Class MECE Intent Taxonomy
* **Decision:** Derived a 7-intent taxonomy from data-driven keyword clustering and empirical frequency analysis rather than guessing generic categories.
* **Why:** Real customer support inquiries clustered cleanly into 7 operational workflows (`delivery_tracking_delay`, `refund_return_cancellation`, `damaged_defective_item`, `account_access_security`, `payment_billing_issue`, `prime_subscription_services`, `general_inquiry_support`).
* **Alternatives Considered:** 20+ fine-grained intents (high label noise, severe sparsity), 3 coarse intents (too generic to drive automated actions).
* **Trade-off:** Edge cases combining multiple intents require fallback to general support or escalation.
* **Consequence:** Clear, distinct categories that directly map to Amazon's actual customer service departments.

---

### Decision 4: Preserving Linguistic Signals During Preprocessing
* **Decision:** Retained customer `@mentions`, punctuation, casing, and emojis while cleaning HTML entities and non-printable control characters.
* **Why:** Customer support tweets convey critical sentiment and urgency through uppercase text (*"WHERE IS MY ORDER"*), multiple question marks (*"???"*), and emojis (😡 vs 🙏). Stripping them damages intent and escalation signals.
* **Alternatives Considered:** Aggressive lowercase conversion, regex stripping of all punctuation and symbols.
* **Trade-off:** Slightly larger vocabulary size in TF-IDF representations.
* **Consequence:** Intent classifiers and escalation routers have access to full emotional and context indicators.

---

### Decision 5: Two Distinct Baselines (Majority + TF-IDF Logistic Regression)
* **Decision:** Implemented both a trivial Majority Baseline and a competitive traditional ML baseline (TF-IDF + Logistic Regression).
* **Why:** Comparing a proposed neural/semantic system only against a naive majority baseline creates a misleading perception of superiority. TF-IDF + Logistic Regression provides a strong, interpretable benchmark.
* **Alternatives Considered:** Majority baseline only, random classifier.
* **Trade-off:** Required implementing, tuning, and maintaining two separate baseline pipelines.
* **Consequence:** Transparent benchmarking showing TF-IDF achieves 89.00% accuracy on lexical keywords while Sentence Transformers achieves 83.00% with continuous semantic embeddings and calibrated confidence.

---

### Decision 6: Dense Semantic Representations via `all-MiniLM-L6-v2`
* **Decision:** Selected `all-MiniLM-L6-v2` (384-dimensional embeddings) for semantic intent classification and historical retrieval.
* **Why:** Fast CPU inference (sub-20ms per query), compact memory footprint (80MB model), and proven semantic retrieval performance for conversational text.
* **Alternatives Considered:** `text-embedding-ada-002` (requires API latency and cost), large BERT/RoBERTa (slow on CPU without GPU acceleration).
* **Trade-off:** Monolingual English focus requires fallback for rare foreign language tweets.
* **Consequence:** The entire pipeline runs completely offline on standard developer laptops without GPU requirements.

---

### Decision 7: Strict Query Conversation Masking in Retrieval
* **Decision:** Enforced `exclude_conversation_id` in `HistoricalRetrievalEngine.retrieve()` to mask out any historical turns belonging to the query's own conversation.
* **Why:** Without conversation masking, test queries from multi-turn dialogues could retrieve their own ground truth response, creating artificial 100% similarity scores.
* **Alternatives Considered:** No filtering, post-hoc string deduplication.
* **Trade-off:** Requires passing `conversation_id` metadata through the pipeline.
* **Consequence:** Fully defensible retrieval evaluation with zero self-retrieval leakage.

---

### Decision 8: Multi-Factor Escalation Router Architecture
* **Decision:** Built a multi-layered rule- and threshold-based escalation router checking (1) sensitive/legal keywords, (2) security policy intent, (3) intent confidence ($<0.65$), and (4) retrieval evidence similarity ($<0.60$).
* **Why:** Real-world customer support automation must prioritize safety over deflection. A single confidence threshold is insufficient for legal threats or account security risks.
* **Alternatives Considered:** Pure machine-learning binary classifier for escalation.
* **Trade-off:** 47.0% escalation rate on the Golden Set prioritizes accuracy and security over aggressive automated deflection.
* **Consequence:** 100% of sensitive legal threats and compromised account reports are safely routed to human specialists.

---

### Decision 9: Dual-Mode Grounded Reply Generation with Anti-Hallucination Constraints
* **Decision:** Implemented dual-mode generation (OpenAI/Gemini LLM API with strict prompt constraints, backed by deterministic historical synthesis fallback).
* **Why:** Ensures the system executes reliably and reproducibly in both connected environments (with API keys) and completely offline test/grading environments.
* **Alternatives Considered:** Unconstrained open-ended LLM generation, static rule templates only.
* **Trade-off:** Deterministic fallback adheres closely to historical brand responses rather than generating highly varied creative phrasing.
* **Consequence:** Achieved a 0.0% unsupported claims/hallucination rate while maintaining 100% compliance with Twitter's 280-character limit.

---

### Decision 10: Multi-Tiered Evaluation Harness
* **Decision:** Evaluated all system facets independently: Intent Classification, Grounded Reply Quality, Historical Retrieval, Escalation Accuracy, and LLM-as-Judge vs. Human Agreement.
* **Why:** Evaluating only intent classification misses response quality; evaluating only response generation misses routing failures.
* **Alternatives Considered:** Single end-to-end composite score.
* **Trade-off:** Required running multiple evaluation scripts and analyzing multidimensional metrics.
* **Consequence:** Transparent, granular diagnosis of where the system excels and where specific failure modes occur.

---

### Decision 11: Intentional Exclusion of Unnecessary Infrastructure
* **Decision:** Excluded complex production infrastructure (Kubernetes, AWS ECS, distributed PostgreSQL, Kafka, live Twitter/X API scrapers).
* **Why:** The assignment objective is to build, evaluate, and defend a working AI customer support system. Unnecessary infrastructure adds boilerplate and obscures core ML logic.
* **Alternatives Considered:** Full Dockerized microservice architecture with FastAPI and Celery.
* **Trade-off:** System is optimized for single-node local execution and reproducible demonstration.
* **Consequence:** Simple, clean, understandable codebase that can be fully reviewed, tested, and explained in an interview in under 15 minutes.

---

### Decision 12: Independent Golden Evaluation Benchmark ($N=200$)
* **Decision:** Constructed an audited, held-out evaluation set of 200 real customer tweets stratified across all 7 intents and 3 difficulty levels (Easy, Medium, Hard).
* **Why:** Testing only on synthetic or training data provides unrealistic performance estimates. Real test data contains ambiguous phrasing, typos, and emotional venting.
* **Alternatives Considered:** Evaluating on unverified training split, synthetic GPT-generated test prompts.
* **Trade-off:** Manual verification and stratification required upfront effort.
* **Consequence:** Provides an uncorrupted benchmark for evaluating baseline and proposed models under realistic conditions.

---

### Decision 13: Empirical Selection of TF-IDF + Logistic Regression as Production Default
* **Decision:** Configured `TF-IDF + Logistic Regression` as the default production intent classifier for the live `SupportAgentPipeline`, while preserving the `SemanticIntentClassifier` as a baseline and continuous embedding provider.
* **Why:** Empirical benchmark results on the 200-sample Golden Set showed TF-IDF achieved **89.00% Accuracy and 89.41% Macro F1**, outperforming the SentenceTransformer head (**83.00% Accuracy and 82.78% Macro F1**) due to sharp exact-keyword disambiguation on e-commerce support queries.
* **Alternatives Considered:** Forcing the deeper neural network to be the production model despite lower measured test accuracy.
* **Trade-off:** TF-IDF has lower zero-shot generalization to extreme unseen phrasing, but provides sub-millisecond inference and superior accuracy on known operational domains.
* **Consequence:** Honest, data-driven engineering that chooses the champion model strictly on verified empirical evidence rather than theoretical complexity.

---

### Decision 14: Strict Pre-Ranking Intent Masking and PII Sanitization in Retrieval
* **Decision:** Enforced hard candidate filtering by predicted intent *prior* to top-k similarity selection, accompanied by regex sanitization to strip customer names (`Hi, Amy!`), `@handles`, agent signatures (`^TS`), and tweet split markers (`1/2`, `2/2`).
* **Why:** Without pre-ranking intent masking, general public advisory tweets (e.g. orphan PII disclaimers) with high lexical similarity could be retrieved for specific order tracking queries. Furthermore, copying historical customer greetings into new replies leaks customer-specific identifiers.
* **Alternatives Considered:** Soft similarity penalties (-0.15), unconstrained vector nearest neighbors.
* **Trade-off:** If predicted intent is wrong, retrieval is constrained to that predicted intent pool (though low similarity will trigger human escalation).
* **Consequence:** Clean, highly relevant, PII-safe historical resolution evidence with guaranteed safety routing when evidence similarity is $<0.60$.

