# Golden Evaluation Dataset & Annotation Methodology

**Dataset Location:** `data/golden/golden_set.csv`  
**Total Examples:** `200`  
**Brand:** `AmazonHelp`  
**Split Source:** 100% sampled from the held-out **Test Split** (`data/splits/test_pairs.parquet`)  
**Data Leakage Prevention:** Verified 0% conversation overlap between Train, Validation, and Test/Golden sets.

---

## 1. Objective & Golden Set Principles

The Golden Evaluation Set serves as the ground-truth benchmark for evaluating:
1. Intent Classification Accuracy, Macro F1, Precision, and Recall across all 3 models (Majority Baseline, TF-IDF + Logistic Regression, Proposed Semantic Classifier).
2. Grounded Response Generation Quality and Hallucination Prevention.
3. Auto-Handle vs Human Escalation Decision Routing.

To guarantee empirical validity:
- **No Synthetic or Fabricated Queries:** 100% of the customer messages are genuine, historical customer tweets from `twcs.csv`.
- **Strict Held-Out Isolation:** Every query belongs to an independent `conversation_id` that never appears in the training partition or historical retrieval database.

---

## 2. Intent Distribution in Golden Set

| Intent ID | Display Name | Example Count | Percentage |
| :--- | :--- | :---: | :---: |
| `delivery_tracking_delay` | Delivery Tracking & Delays | 40 | 20.0% |
| `refund_return_cancellation` | Refunds, Returns & Cancellations | 35 | 17.5% |
| `damaged_defective_item` | Damaged, Defective or Wrong Item | 25 | 12.5% |
| `payment_billing_issue` | Payment & Billing Inquiries | 25 | 12.5% |
| `account_access_security` | Account Access & Security | 25 | 12.5% |
| `prime_subscription_services` | Prime & Digital Subscriptions | 25 | 12.5% |
| `general_inquiry_support` | General Policy & Agent Escalation | 25 | 12.5% |
| **Total** | | **200** | **100.0%** |

---

## 3. Difficulty Stratification

| Difficulty Level | Count | Share | Characteristics |
| :--- | :---: | :---: | :--- |
| **Easy** | 30 | 15.0% | Explicit keywords (e.g., *"where is my package tracking"*, *"need a refund"*). |
| **Medium** | 122 | 61.0% | Conversational nuances, multi-sentence queries, or implicit issues. |
| **Hard** | 48 | 24.0% | Highly ambiguous phrasing, multi-intent overlaps, or brief emotional expressions. |

---

## 4. Annotation Guidelines & Decision Boundaries

- **Rule 1 (Security Over Billing):** If a customer reports unauthorized charges due to an account takeover, label as `account_access_security`.
- **Rule 2 (Physical Damage Over Delay):** If a package arrived late AND the contents are broken, label as `damaged_defective_item`.
- **Rule 3 (Returns Over Tracking):** If an item is already delivered and the customer wants to send it back, label as `refund_return_cancellation`.

---

## 5. Label Provenance & Verification Process

1. **Sampling Frame:** 200 customer-inbound messages were deterministically sampled from the strictly held-out Test split (`data/splits/test_pairs.parquet`).
2. **Conversation Isolation:** None of the 200 `conversation_id`s exist in the training split (`data/splits/train_pairs.parquet`) or validation split (`data/splits/val_pairs.parquet`), guaranteeing 0% train/test leakage.
3. **Taxonomy Assignment:** Ground-truth intent labels were assigned and audited against the 7 predefined MECE intents in `configs/intent_taxonomy.yaml` using the structured annotation guidelines above.
4. **Authenticity:** All 200 customer messages represent genuine historical customer tweets from the Twitter Customer Support dataset (`twcs.csv`) without synthetic generation.

