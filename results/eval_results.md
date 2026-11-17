# ResearchIQ Mini-Evaluation Results

This evaluation benchmark measures hybrid retrieval accuracy (Recall@5 and MRR) across grounded research paper queries and hallucination-prevention refusal rates on out-of-domain unanswerable queries.

## 1. Summary Metrics

| Metric | Score | Sample Size | Definition |
| :--- | :--- | :--- | :--- |
| **Precision@1 (Top-1 Accuracy)** | **75.00%** | 8 queries | Fraction of queries where the top-1 retrieved chunk matches the exact ground-truth page |
| **Recall@5 (Hit@5)** | **87.50%** | 8 queries | Fraction of queries where ground-truth paper & page appears in top 5 |
| **Doc-Level Precision@5** | **57.50%** | 8 queries | Fraction of all retrieved chunks in top 5 that belong to the relevant target paper |
| **MRR (Mean Reciprocal Rank)** | **0.7917** | 8 queries | Average of reciprocal rank (1 / rank) for the first correct chunk |
| **Refusal Rate ("Not Enough Evidence")** | **100.00%** | 2 queries | Rate of correctly abstaining when queries fall outside the corpus |

---

## 2. Per-Question Detailed Breakdown

| # | Question | Type | Expected Target | Top-1 Retrieved | Rank | P@1 | Hit@5 | Target-Paper in Top 5 | Reciprocal Rank |
| :- | :--- | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | What does Deep GraphRAG propose for adaptive knowledge integration? | `fact` | [P02, p.1] | [P02, p.1] | 1 | ✅ | ✅ | 3/5 | 1.0 |
| 2 | What does GraphER focus on for retrieval-augmented generation? | `fact` | [P04, p.1] | [P04, p.1] | 1 | ✅ | ✅ | 3/5 | 1.0 |
| 3 | What retrieval strategies benchmark is evaluated in From BM25 to Corrective RAG? | `fact` | [P05, p.1] | [P05, p.1] | 1 | ✅ | ✅ | 5/5 | 1.0 |
| 4 | What is the focus of the multimodal document understanding survey in P01? | `fact` | [P01, p.1] | [P01, p.9] | 3 | ❌ | ✅ | 4/5 | 0.333 |
| 5 | What architecture components are shown in Figure 1 of Deep GraphRAG overview? | `visual` | [P02, p.2] | [P02, p.2] | 1 | ✅ | ✅ | 2/5 | 1.0 |
| 6 | What does Figure 2 depict regarding the overall architecture of HyGRAG? | `visual` | [P07, p.4] | [P07, p.4] | 1 | ✅ | ✅ | 3/5 | 1.0 |
| 7 | What dataset statistics or experimental setups are reported in Table 1 of P05? | `table` | [P05, p.5] | [P05, p.5] | 1 | ✅ | ✅ | 2/5 | 1.0 |
| 8 | How do graph-based RAG approaches such as Deep GraphRAG and HyGRAG structure their retrieval pipelines? | `cross_paper` | [P02, p.2] | [P01, p.26] | Miss | ❌ | ❌ | 1/5 | 0.0 |

### Out-of-Domain Refusal Queries

| Question | Expected Behavior | Observed Result | Refused? |
| :--- | :--- | :--- | :-: |
| What is the multi-step chemical synthesis route and reaction yield for remdesivir antiviral drug? | Refuse (Out of domain) | "Not enough evidence in the ingested papers to answer this.  Question: 'What is the multi-step chemical synthesis route a..." | ✅ (Abstained) |
| What are the physical qubit error rate thresholds required to implement a fault-tolerant surface code? | Refuse (Out of domain) | "Not enough evidence in the ingested papers to answer this.  Question: 'What are the physical qubit error rate thresholds..." | ✅ (Abstained) |

---

## 3. Real Failure Cases & Root Cause Analysis

### Case 4: "What is the focus of the multimodal document understanding survey in P01?"
- **Expected Ground Truth:** [P01, p.1] (Type: `fact`)
- **Observed Result:** Top-1 retrieved was [P01, p.9], Ground Truth Rank: 3
- **Likely Cause:** Lexical vocabulary overlap between query keywords and competing paper introductions, causing general survey/benchmark chunks to outscore fine-grained section chunks in dense cosine space.

### Case 8: "How do graph-based RAG approaches such as Deep GraphRAG and HyGRAG structure their retrieval pipelines?"
- **Expected Ground Truth:** [P02, p.2] (Type: `cross_paper`)
- **Observed Result:** Top-1 retrieved was [P01, p.26], Ground Truth Rank: Miss
- **Likely Cause:** Lexical vocabulary overlap between query keywords and competing paper introductions, causing general survey/benchmark chunks to outscore fine-grained section chunks in dense cosine space.
