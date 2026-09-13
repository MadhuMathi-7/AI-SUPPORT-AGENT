# Phase 2 Data Quality & Conversation Reconstruction Report

**Generated:** 2026-09-12 17:32:00 UTC  
**Source Dataset:** `C:\Users\HP\OneDrive\Desktop\hiver-sde-ai-support-agent\data\raw\twcs.csv`  
**Pipeline Execution Status:** Completed & Audited (100% Measured Values)

---

## 1. Executive Summary

This report documents the verified empirical results of **Phase 2: Data Cleaning and Conversation Reconstruction** for the Customer Support on Twitter dataset.

Using graph-based path compression on Twitter parent-child relationships (`tweet_id` $\rightarrow$ `in_response_to_tweet_id`), the raw corpus of **2,811,774 tweets** was cleaned and reconstructed into **798,197 distinct conversation trees** and **1,261,888 structured Customer Query $\rightarrow$ Brand Response pairs** across **108 brands**.

---

## 2. Dataset Cleaning & Record Accounting

| Metric | Measured Value | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **Raw Input Rows** | **2,811,774** | 100.00% | Total records in raw `twcs.csv` |
| **Exact Duplicate Rows Removed** | **0** | 0.00% | Full-row duplicates |
| **Empty / Missing Text Removed** | **0** | 0.00% | Records with no text content |
| **Invalid Timestamps Removed** | **0** | 0.00% | Failed RFC 2822 datetime parsing |
| **Self-Referencing Links Sanitized** | **0** | 0.00% | Self-loop references where parent = child |
| **Cleaned Usable Records** | **2,811,774** | **100.0%** | Fully validated and normalized records |

---

## 3. Inbound vs Outbound Message Accounting

| Direction | Message Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **Inbound (Customer)** | **1,537,843** | **54.69%** | Inbound messages sent by customers (`inbound=True`) |
| **Outbound (Brand)** | **1,273,931** | **45.31%** | Responses sent by customer service handles (`inbound=False`) |
| **Total Cleaned Messages** | **2,811,774** | **100.00%** | Directionality verified from dataset boolean flag |

### Outbound Message Mathematical Accounting

The 1,273,931 outbound brand messages partition into:

$$\text{Total Outbound (1,273,931)} = \text{Brand Roots (6,989)} + \text{Outbound with Parent (1,266,942)}$$

Within the **1,266,942** outbound messages that reply to a parent tweet:

$$\text{Outbound with Parent (1,266,942)} = \text{Customer-Brand Pairs (1,261,888)} + \text{Brand-to-Brand Replies (3,393)} + \text{Dangling References (1,661)}$$

| Outbound Subcategory | Count | % of Parented Outbound | % of Total Outbound | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Valid Customer $\rightarrow$ Brand Pairs** | **1,261,888** | **99.60%** | **99.05%** | Brand response replying to an inbound customer message |
| **Brand-to-Brand Replies** | **3,393** | **0.27%** | **0.27%** | Brand tweet replying to another brand tweet (internal/escalation) |
| **Dangling Parent References** | **1,661** | **0.1311%** | **0.1304%** | Brand response replying to a tweet not present in the dataset |
| **Brand Roots / Broadcasts** | **6,989** | — | **0.55%** | Unsolicited / broadcast tweets initiated by brands (`in_response_to_tweet_id` is null) |

---

## 4. Conversation Reconstruction Metrics

Conversation threads are reconstructed by identifying root tweets (ancestor tweets with no parent in the dataset) and grouping all descendant turns.

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Total Reconstructed Conversation Trees** | **798,197** | Distinct connected dialogue components resolved |
| **Conversations in Customer-Brand Pairs** | **798,080** | Conversations containing $\ge 1$ customer-brand pair (99.99%) |
| **Conversations without Customer-Brand Pairs** | **117** | Conversations with only customer-customer or brand-brand turns (0.01%) |
| **Mean Conversation Length** | **3.52 turns** | Average number of turns per dialogue |
| **Median Conversation Length** | **2 turns** | Median dialogue length |
| **Minimum Conversation Length** | **2 turns** | Minimum dialogue length |
| **Maximum Conversation Length** | **1,390 turns** | Longest multi-turn thread |
| **Total Brands Represented** | **108** | Unique company customer support handles |

### Conversation Length Distribution

All 798,197 conversation trees are partitioned below (Total Sum = 798,197):

| Turn Count Category | Conversation Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **2 turns (Query + Reply)** | **435,398** | **54.55%** | Single interaction customer-brand exchanges |
| **3–4 turns** | **216,856** | **27.17%** | Short back-and-forth resolutions |
| **5–9 turns** | **119,211** | **14.94%** | Multi-turn customer issue troubleshooting |
| **10–19 turns** | **22,716** | **2.85%** | Extended support discussions |
| **20+ turns** | **4,016** | **0.50%** | Complex long-running customer dialogues |
| **Total Sum** | **798,197** | **100.00%** | Exact sum of all conversation categories |

---

## 5. Customer $\rightarrow$ Brand Pairs & Response Time Metrics

A Customer-Brand pair represents an immediate inbound customer query and its direct outbound brand response.

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Total Brand Responses Evaluated** | **1,266,942** | Outbound brand tweets with parent links |
| **Valid Customer-Brand Pairs** | **1,261,888** | Successfully paired customer query + brand response |
| **Brand-to-Brand Replies Excluded** | **3,393** | Internal brand-to-brand follow-ups |
| **Dangling Parent References Excluded** | **1,661** | Parent tweet missing from corpus (0.1311%) |
| **Negative Response Time Anomalies** | **0** | No temporal contradictions detected ($t_{brand} < t_{cust}$ count = 0) |
| **Unique Customers in Pairs** | **667,666** | Unique anonymized customer author IDs |
| **Unique Conversations in Pairs** | **798,080** | Distinct conversation threads represented |
| **Minimum Response Time** | **0.00 minutes** | Minimum response latency |
| **Median Response Time** | **21.18 minutes** | Median response latency across all brands |
| **Mean Response Time** | **293.67 minutes** | Arithmetic mean response latency |
| **95th Percentile Response Time** | **1183.31 minutes** | 95% of replies sent within ~19.7 hours |

---

## 6. Data Leakage Prevention Strategy

To support sound evaluation in subsequent phases (Phase 4 Intent Classification and Phase 5 Retrieval/Generation):
1. **Conversation-Level Splitting:** Splits between Train, Validation, and Test sets will **strictly be performed by `conversation_id`**, rather than randomly sampling individual tweet pairs.
2. **Context Isolation:** Because multi-turn pairs share contextual turns within the same conversation thread, grouping by `conversation_id` ensures that no training dialogue context leaks into evaluation sets.
3. **Temporal Partitioning Support:** Timestamps (`customer_created_at`, `brand_created_at`) are preserved to allow temporal out-of-time evaluation if required.

---

## 7. Candidate Brand Statistics (Evidence for Phase 3)

The table below displays the **top 25 brands** ranked by verified customer-brand pair volume:

| Rank | Brand | Brand Responses | Interaction Pairs | Distinct Conversations | Avg Conv Length | Median Response Time (min) | Multi-Turn Context Pct |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **AmazonHelp** | 169,840 | 168,814 | 82,556 | 4.53 | 11.5 min | 49.8% |
| 2 | **AppleSupport** | 106,860 | 106,646 | 80,717 | 2.96 | 71.0 min | 29.8% |
| 3 | **Uber_Support** | 56,270 | 56,160 | 41,923 | 3.07 | 8.8 min | 28.6% |
| 4 | **SpotifyCares** | 43,265 | 43,092 | 28,277 | 3.25 | 43.8 min | 37.4% |
| 5 | **Delta** | 42,253 | 42,114 | 26,165 | 3.36 | 10.2 min | 32.3% |
| 6 | **Tesco** | 38,573 | 38,468 | 16,722 | 4.38 | 96.7 min | 35.2% |
| 7 | **AmericanAir** | 36,764 | 36,531 | 26,386 | 3.32 | 10.7 min | 32.8% |
| 8 | **TMobileHelp** | 34,317 | 34,215 | 22,820 | 3.63 | 2.8 min | 41.1% |
| 9 | **comcastcares** | 33,031 | 32,921 | 24,063 | 3.04 | 29.2 min | 27.4% |
| 10 | **British_Airways** | 29,361 | 29,290 | 16,450 | 3.69 | 180.5 min | 33.0% |
| 11 | **SouthwestAir** | 28,977 | 28,828 | 21,634 | 2.99 | 7.0 min | 27.1% |
| 12 | **VirginTrains** | 27,817 | 27,416 | 14,844 | 4.43 | 3.3 min | 47.5% |
| 13 | **Ask_Spectrum** | 25,860 | 25,617 | 18,529 | 3.22 | 18.7 min | 32.4% |
| 14 | **XboxSupport** | 24,557 | 23,235 | 13,429 | 4.29 | 80.8 min | 46.0% |
| 15 | **sprintcare** | 22,381 | 22,209 | 13,560 | 3.99 | 9.2 min | 42.6% |
| 16 | **hulu_support** | 21,872 | 21,681 | 14,952 | 3.30 | 450.6 min | 34.5% |
| 17 | **sainsburys** | 19,466 | 19,399 | 10,934 | 4.00 | 81.8 min | 44.4% |
| 18 | **GWRHelp** | 19,364 | 19,237 | 10,714 | 4.36 | 7.7 min | 47.4% |
| 19 | **AskPlayStation** | 19,098 | 18,675 | 12,533 | 3.44 | 31.4 min | 38.9% |
| 20 | **ChipotleTweets** | 18,749 | 18,599 | 14,392 | 2.91 | 12.2 min | 25.4% |
| 21 | **VerizonSupport** | 17,966 | 17,805 | 8,447 | 5.44 | 3.3 min | 59.4% |
| 22 | **UPSHelp** | 17,817 | 17,762 | 15,522 | 2.87 | 14.0 min | 19.3% |
| 23 | **ATVIAssist** | 17,650 | 17,514 | 11,112 | 4.35 | 363.3 min | 41.0% |
| 24 | **O2** | 16,212 | 16,069 | 9,556 | 3.82 | 78.9 min | 43.0% |
| 25 | **idea_cares** | 15,724 | 15,591 | 7,522 | 4.63 | 29.9 min | 46.7% |

---

## 8. Processed Data Artifacts & File Parity

The following processed artifacts are stored in `data/processed/` and have been verified for 100% row and schema parity:

1. **`data/processed/customer_brand_pairs.parquet`** (452.54 MB):
   - High-performance binary storage containing all 1,261,888 customer-brand pairs.
2. **`data/processed/customer_brand_pairs.csv`** (935.07 MB):
   - Universal CSV format containing all 1,261,888 records (exact parity with Parquet).
3. **`data/processed/conversation_threads.parquet`** (656.83 MB) / **`.csv`** (1272.12 MB):
   - Complete record of all 2,811,774 chronological turns, speaker roles, and dialogue metadata (exact parity with Parquet).
4. **`data/processed/brand_candidate_metrics.csv`** (0.01 MB):
   - Measured support volume, response latency, and conversation metrics across all 108 brands.

---

## 9. Known Dataset Limitations

1. **Twitter Character Limit / Formatting:** Tweets reflect Twitter's character constraints, resulting in concise language and frequent redirection to Private Messages (DM).
2. **Anonymized Customer IDs:** Customer IDs are anonymized numeric identifiers (`115712`), preventing cross-platform customer profile linking.
3. **External Dangling References:** Exactly 1,661 brand responses (0.1311% of parented brand tweets) reply to tweets originating prior to the collection window or deleted from Twitter, which are safely excluded during pair extraction.

---

## 10. Phase 2 Validation Summary

| Audit Check | Status | Verification Detail |
| :--- | :---: | :--- |
| **Numerical Consistency** | **PASS** | Outbound equation ($1,261,888 + 3,393 + 1,661 = 1,266,942$) and length category sum ($= 798,197$) match exactly. |
| **Dataset Integrity** | **PASS** | `data/raw/twcs.csv` is 100% untouched (516,508,641 bytes, 2,811,774 rows). Zero null/duplicate issues. |
| **Conversation Reconstruction Validation** | **PASS** | 798,197 conversation trees reconstructed with path compression; chronological ordering and turn numbering verified. |
| **Customer-Brand Pair Validation** | **PASS** | 1,261,888 valid pairs extracted; 0 negative response times detected; 108 unique brands represented. |
| **Processed File Consistency** | **PASS** | Parquet and CSV files for pairs (1,261,888 rows) and threads (2,811,774 rows) have 100% row and schema parity. |
| **Leakage-Prevention Readiness** | **PASS** | Unique `conversation_id` present on every pair and thread record, enabling leak-free conversation-level partitioning. |
