# Evidence-Based Brand Selection Report

**Selected Brand:** `AmazonHelp`  
**Primary Domain:** E-Commerce / Retail Customer Support  
**Empirical Volume:** 168,814 Customer $\rightarrow$ Brand Pairs  
**Distinct Conversations:** 82,556 Threads  
**Average Conversation Depth:** 4.53 Turns  
**Median Response Latency:** 11.5 Minutes  
**Pairs with Historical Context:** 49.8%

---

## 1. Objective & Decision Criteria

Rather than arbitrarily picking a brand, candidate brands from the 108 companies represented in `twcs.csv` were evaluated against five quantitative and qualitative criteria:

1. **Statistical Power & Pair Volume:** Sufficient volume to construct clean, leak-free training, retrieval, and evaluation partitions ($\ge 20,000$ pairs).
2. **Conversational Dialogue Depth:** Multi-turn conversation length ($\ge 3.0$ turns avg) to test context-aware retrieval and multi-turn response generation.
3. **Intent Richness & Domain Separability:** Broad, realistic customer-support problem space (e.g., shipping, returns, damaged items, payments, account access, subscription services).
4. **Historical Response Latency:** Fast median response times ($\le 15$ minutes) indicating an active, standardized customer support workflow.
5. **Interview Explainability & Defensibility:** A relatable domain with practical, universally understood resolution procedures.

---

## 2. Comparative Analysis of Top Candidate Brands

The table below contrasts the top 10 candidate brands from `data/processed/brand_candidate_metrics.csv`:

| Rank | Candidate Brand | Domain | Pair Count | Conversation Count | Avg Length | Median Latency | Context Pct | Primary Strength / Limitation |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **AmazonHelp** | E-Commerce | 168,814 | 82,556 | 4.53 turns | 11.5 min | 49.8% | Highest pair volume and conversational depth (4.53 turns). |
| 2 | **AppleSupport** | Tech / Hardware | 106,646 | 80,717 | 2.96 turns | 71.0 min | 29.8% | High volume, but replies heavily redirect to external support links. |
| 3 | **Uber_Support** | Ride-Hailing | 56,160 | 41,923 | 3.07 turns | 8.8 min | 28.6% | Fast response, but limited multi-turn back-and-forth depth. |
| 4 | **SpotifyCares** | Digital Media | 43,092 | 28,277 | 3.25 turns | 43.8 min | 37.4% | Moderate volume, slower response time (43.8 min). |
| 5 | **Delta** | Airlines | 42,114 | 26,165 | 3.36 turns | 10.2 min | 32.3% | Airline domain has high flight cancellation noise. |
| 6 | **Tesco** | Supermarket | 38,468 | 16,722 | 4.38 turns | 96.7 min | 35.2% | Slower median response time (96.7 min). |
| 7 | **AmericanAir** | Airlines | 36,531 | 26,386 | 3.32 turns | 10.7 min | 32.8% | High cancellation seasonality. |
| 8 | **TMobileHelp** | Telecom | 34,215 | 22,820 | 3.63 turns | 2.8 min | 41.1% | Telecom accounts require heavy private SMS/PIN authentication. |
| 9 | **comcastcares** | Cable / Internet | 32,921 | 24,063 | 3.04 turns | 29.2 min | 27.4% | Frequent network outage reports. |
| 10 | **British_Airways** | Airlines | 29,290 | 16,450 | 3.69 turns | 180.5 min | 33.0% | High response latency (180.5 min). |

---

## 3. Why `AmazonHelp` Was Selected

1. **Superior Dialogue Depth (Rank 1 among High-Volume Brands):**
   - `AmazonHelp` achieves an average of **4.53 turns per conversation**, with **49.8%** of pairs containing prior conversational history. This is significantly higher than AppleSupport (2.96 turns) and Uber_Support (3.07 turns).
2. **Rich, Realistic Support Taxonomy:**
   - E-commerce support represents the canonical customer service benchmark. Customer issues naturally partition into distinct, actionable categories: Order Tracking, Delivery Delays, Refund/Return Inquiries, Damaged/Missing Items, Account Access/Security, Prime Subscription Issues, and Payment Problems.
3. **High Operational Responsiveness:**
   - With a median response time of **11.5 minutes**, historical Amazon agents provide prompt, structured resolutions suitable for few-shot grounded retrieval.
4. **Leakage-Safe Partitioning Scale:**
   - With **82,556 distinct conversations**, we can construct generous, completely isolated conversation-level train, validation, and test splits without data sparsity.

---

## 4. Trade-Offs & Mitigations

| Trade-Off | Description | Mitigation in Pipeline |
| :--- | :--- | :--- |
| **Private Information (DM) Redirections** | In sensitive order inquiries, agents often ask customers to send a Direct Message. | The reply generator is explicitly grounded to advise DM/account lookup only when order numbers or private links are required. |
| **High Overall Volume** | Processing 168k pairs repeatedly in development is unnecessary. | We implement a reproducible sampling strategy with fixed random seed for fast (<15 min) local execution. |
| **Global Marketplace Variability** | Inquiries occasionally reference Amazon UK, US, or India domains. | The preprocessing module cleans regional URL variations while preserving domain semantics. |
