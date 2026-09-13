# Top 5 Failure Mode Analysis

**Domain:** E-Commerce Customer Support Automation (`@AmazonHelp`)  
**Evaluation Benchmark:** 200 Real Customer Inquiries from Held-Out Test Split (`data/golden/golden_set.csv`)  
**Methodology:** Error auditing across intent classification, historical retrieval ranking, grounding synthesis, and escalation routing.

---

## 1. Executive Summary of Failure Modes

While the proposed AI Support Pipeline achieves strong overall intent accuracy (**83.00%** on semantic classification; **89.00%** on TF-IDF baseline) and a **0.0% unsupported claims rate**, systematic failure auditing on real evaluation examples revealed 5 primary failure modes:

| # | Failure Mode | Primary Root Cause | Frequency in Eval (%) | Severity | Recommended Mitigation |
|---|---|---|---|---|---|
| **1** | Multi-Intent Compound Queries | Single-label classification constraint | 6.5% | Medium | Multi-label sigmoid head + intent decomposition |
| **2** | Non-English & Multilingual Inquiries | Monolingual English embedding bias | 3.0% | High | Multilingual pre-routing (`langdetect` + `paraphrase-multilingual-MiniLM-L12-v2`) |
| **3** | Non-Deterministic Brand URL Formatting | Historical tweets contain expired/shortened links | 8.0% | Low | Canonical URL template substitution (`amazon.com/your-orders`) |
| **4** | Over-Escalation on Subtle Slang & Frustration | High lexical dispersion lowers semantic confidence | 9.5% | Medium | Escalation threshold recalibration via validation split |
| **5** | Context-Dependent Dialogue Coreference | Missing turn history in standalone tweets | 5.0% | High | Turn concatenation using thread builder context |

---

## 2. In-Depth Failure Case Audits

### Failure Mode 1: Multi-Intent Compound Queries
* **Failure Description:** A customer expresses two distinct problems in a single tweet (e.g., received a damaged item AND asks about the refund turnaround time).
* **Real Evaluation Example:** `GOLDEN_001` (Conversation ID: `1267628`)
  > *"@AmazonHelp Item was already returned. I refused to accept damaged product. Pls read original tweet. Issue is that @115850 rejected my product review."*
* **Expected Behavior:** System recognizes the primary issue is a product review policy dispute (`general_inquiry_support` or review moderation) while acknowledging the damaged product background.
* **Actual Pipeline Behavior:** Classifier predicted `damaged_defective_item` due to high keyword salience of *"damaged product"* and *"returned"*.
* **Why It Failed:** Single-label softmax forces mutually exclusive categorization, missing the nuanced primary grievance.
* **Hypothesis:** Customer support tweets frequently conflate past events (e.g., *"item was damaged"*) with current requests (e.g., *"why did you reject my review?"*).
* **Proposed Improvement:** Implement intent decomposition or multi-label classification allowing compound intent tagging.

---

### Failure Mode 2: Multilingual Language Drift
* **Failure Description:** Customer tweets in French, German, Spanish, or Hindi targeting international Amazon sub-brands.
* **Real Evaluation Example:** `GOLDEN_012` (Conversation ID: `2891852`)
  > *"@120533 quand tu contactes amazon par mail pour un problème et que une heure après ils te rappellent !! Leur SAV est top!!"*
* **Expected Behavior:** Language identification triggers auto-escalation or routes to a native language prompt.
* **Actual Pipeline Behavior:** Classified into `general_inquiry_support` with low confidence (0.41), correctly triggering escalation, but generating an English fallback draft.
* **Why It Failed:** Embedding model `all-MiniLM-L6-v2` is primarily trained on English text, resulting in lower semantic similarity when encoding French customer praise/feedback.
* **Hypothesis:** Language-agnostic routing prevents nonsensical cross-lingual template responses.
* **Proposed Improvement:** Integrate lightweight fasttext language detection in the preprocessing tier.

---

### Failure Mode 3: Stale Shortened URLs in Historical Retrievals
* **Failure Description:** Historical tweets frequently contain Twitter URL shorteners (`https://t.co/...`) or expired direct links that are non-functional today.
* **Real Evaluation Example:** `GOLDEN_013` (Conversation ID: `544458`)
  > *"@115830 not particularly happy paying a prime membership to be waiting weeks for deliveries. 3 of my parcels say this! https://t.co/vDMbTXAMub"*
* **Expected Behavior:** Response directs customer to the current, canonical customer support portal (`amazon.com/help` or `amazon.com/orders`).
* **Actual Pipeline Behavior:** Top retrieved historical example contained raw `https://t.co/...` string from 2017.
* **Why It Failed:** Retrieval directly copied raw historical response text without link sanitation.
* **Hypothesis:** Historical support data ages over time; URL endpoints change while operational workflows remain constant.
* **Proposed Improvement:** Implement link regex sanitization that replaces legacy short links with verified canonical support URLs.

---

### Failure Mode 4: Over-Escalation on Informal Colloquial Frustration
* **Failure Description:** Customers using creative slang, heavy sarcasm, or rhetorical hyperbole receive low confidence scores and get escalated even when the core issue is a routine tracking inquiry.
* **Real Evaluation Example:** `GOLDEN_002` (Conversation ID: `277066`)
  > *"@AmazonHelp 'October 11, 2017 11:06 am Picked Up by Shipping Partner, USPS Awaiting Item, TRACY, CA 95304' That's all they've got."*
* **Expected Behavior:** System auto-handles with standard tracking reassurance and DM invitation.
* **Actual Pipeline Behavior:** Confidence score was 0.61 (below the 0.65 threshold), triggering `ESCALATE_TO_HUMAN`.
* **Why It Failed:** Heavy quotation marks and fragmented sentence structure reduced lexical overlap with standard tracking queries.
* **Hypothesis:** Threshold of 0.65 is conservative and prioritizes safety over deflection.
* **Proposed Improvement:** Calibrate confidence thresholds using isotonic regression on validation data to optimize the precision-recall trade-off.

---

### Failure Mode 5: Ambiguity from Missing Multi-Turn Dialogue Context
* **Failure Description:** Follow-up tweets that omit the original problem context (e.g., *"I did that already, still nothing"*) cannot be categorized accurately in isolation.
* **Real Evaluation Example:** `GOLDEN_065`
  > *"Done. Sent DM with the details you asked for."*
* **Expected Behavior:** System identifies this as an ongoing dialogue continuation and notifies the assigned agent.
* **Actual Pipeline Behavior:** Standalone classification maps to `general_inquiry_support` without recognizing the active DM transition.
* **Why It Failed:** Single-turn processing lacks previous conversation turns when evaluating isolated tweets.
* **Hypothesis:** Dialogue state tracking is required for multi-turn Twitter customer support threads.
* **Proposed Improvement:** Pass full thread context reconstructed by `conversation_builder.py` directly into the embedding encoder.
