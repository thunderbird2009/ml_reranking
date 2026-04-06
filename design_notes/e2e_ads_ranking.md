# End-to-End Sponsored Search Ads Ranking

## Table of Contents

1. [Overview](#1-overview)
2. [Objectives, Labels, and Their Semantics](#2-objectives-labels-and-their-semantics)
   - [2.1. Click labels](#21-click-labels)
   - [2.2. Conversion labels](#22-conversion-labels)
   - [2.3. Human relevance labels](#23-human-relevance-labels)
   - [2.4. How labels map to model heads across stages](#24-how-labels-map-to-model-heads-across-stages)
3. [Relevance: Labeling, Encoding, and Distillation](#3-relevance-labeling-encoding-and-distillation)
   - [3.1. Rubric design and biases](#31-rubric-design-and-biases)
   - [3.2. Annotation process](#32-annotation-process)
   - [3.3. Loss encoding for graded relevance](#33-loss-encoding-for-graded-relevance)
   - [3.4. Scaling relevance via LLM teacher distillation](#34-scaling-relevance-via-llm-teacher-distillation)
   - [3.5. Relevance in the scoring pipeline](#35-relevance-in-the-scoring-pipeline)
4. [Lightweight Retrieval Ranking](#4-lightweight-retrieval-ranking)
   - [4.1. Two-tower embedding model](#41-two-tower-embedding-model)
   - [4.2. Multi-channel candidate generation](#42-multi-channel-candidate-generation)
   - [4.3. Candidate scoring and fusion](#43-candidate-scoring-and-fusion)
   - [4.4. Evaluation](#44-evaluation)
5. [Multi-Tower, Multi-Task Model for Pre-Ranking](#5-multi-tower-multi-task-model-for-pre-ranking)
   - [5.1. Relevance as a pre-ranking task](#51-relevance-as-a-pre-ranking-task)
   - [5.2. Can one model support both retrieval and pre-ranking?](#52-can-one-model-support-both-retrieval-and-pre-ranking)
   - [5.3. Multi-task outputs](#53-multi-task-outputs)
   - [5.4. Pre-ranking score](#54-pre-ranking-score)
   - [5.5. Do we need extra tasks to support retrieval?](#55-do-we-need-extra-tasks-to-support-retrieval)
6. [Feature-Interaction Network for Final Ads Ranking](#6-feature-interaction-network-for-final-ads-ranking)
   - [6.1. Why use a feature-interaction network?](#61-why-use-a-feature-interaction-network)
   - [6.2. Inputs](#62-inputs)
   - [6.3. Outputs](#63-outputs)
   - [6.4. Final ad score](#64-final-ad-score)
   - [6.5. Reference architectures](#65-reference-architectures)
   - [6.6. Position bias debiasing](#66-position-bias-debiasing)
   - [6.7. Calibration](#67-calibration)
7. [Re-Ranking for Mixing Ads into the Organic Result List](#7-re-ranking-for-mixing-ads-into-the-organic-result-list)
   - [7.1. Why a separate re-ranking stage is needed](#71-why-a-separate-re-ranking-stage-is-needed)
   - [7.2. Listwise model role](#72-listwise-model-role)
   - [7.3. Training](#73-training)
   - [7.4. Constraint enforcement](#74-constraint-enforcement)
8. [Multi-Task Training: Balancing Competing Objectives](#8-multi-task-training-balancing-competing-objectives)
   - [8.1. The problem](#81-the-problem)
   - [8.2. Loss weighting strategies](#82-loss-weighting-strategies)
   - [8.3. Gradient-level methods](#83-gradient-level-methods)
   - [8.4. Pareto multi-task optimization](#84-pareto-multi-task-optimization)
   - [8.5. Architecture-level separation](#85-architecture-level-separation)
   - [8.6. Two levels of multi-objective tradeoff](#86-two-levels-of-multi-objective-tradeoff)
   - [8.7. Practical recommendations](#87-practical-recommendations)
9. [What Can Be Shared with Organic Search?](#9-what-can-be-shared-with-organic-search)
   - [9.1. Sharing principle and decision criteria](#91-sharing-principle-and-decision-criteria)
   - [9.2. Risks of sharing](#92-risks-of-sharing)
   - [9.3. Sharing matrix by pipeline stage](#93-sharing-matrix-by-pipeline-stage)
   - [9.4. Summary](#94-summary)
10. [Closing: Interfaces, Recommendations, and Open Questions](#10-closing-interfaces-recommendations-and-open-questions)
    - [10.1. Stage interfaces](#101-stage-interfaces)
    - [10.2. Practical recommendations](#102-practical-recommendations)
    - [10.3. Cross-cutting concerns not yet covered](#103-cross-cutting-concerns-not-yet-covered)
    - [10.4. Open design decisions](#104-open-design-decisions)

---

## 1. Overview

This note describes an end-to-end sponsored search ranking stack, from low-latency candidate retrieval to final re-ranking into the mixed organic results page.

The full serving pipeline is:

1. lightweight retrieval ranking inside each search shard
2. pre-ranking over the merged ad candidate set
3. final ads ranking over a much smaller candidate pool
4. listwise re-ranking that mixes ads into the organic results under hard business constraints

Each stage solves a different problem and therefore uses a different model form.

| Stage | Typical candidate count | Latency budget | Main goal |
| --- | --- | --- | --- |
| Lightweight retrieval ranking | thousands to millions per shard | ~1–5ms per shard | fast recall with weak but decomposable scoring |
| Pre-ranking | 100 to 1000 | ~5–15ms | remove weak candidates while preserving likely winners |
| Final ads ranking | 30 to 100 | ~20–50ms | maximize per-ad quality and monetization |
| Mixed-list re-ranking | top organic list + top ads | ~10–30ms | choose the final page under structural constraints |

The key design principle is to push expensive interaction modeling later in the stack, while keeping the early stages simple enough to run at scale.

### Data flow

```
                       Search Query + User + Context
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  ┌───────────┐         ┌───────────┐
                  │ Semantic   │         │ Lexical   │
                  │ ANN search │         │ BM25      │
                  │ cos(e_r,e_i)│        │           │
                  └─────┬──────┘        └─────┬─────┘
                        │                     │
                        └──────────┬──────────┘
                                   ▼
                     ┌───────────────────────┐
                     │  Union + Dedup        │  ~1000s candidates/shard
                     │  Lightweight Score    │  Section 4
                     └───────────┬───────────┘
                 s = w_r·cos(e_r,e_i) + w_bm25·BM25(q,i) + w_qual·Q(i)
                 relevance: implicit via embedding geometry (no filter)
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  Pre-Ranking          │  ~100-1000 candidates
                     │  Multi-task model     │  CTR / CVR / Rel heads
                     └───────────┬───────────┘  Section 5
                   filter: drop if p_rel < τ_floor (kill grade-0 ads)
                   score:  s = p_ctr · p_cvr · bid · f(p_rel)
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  Final Ads Ranking    │  ~30-100 candidates
                     │  Feature-interaction  │  DCN-v2 / DLRM / DIN
                     │  + debiasing + calib  │  Section 6
                     └───────────┬───────────┘
                   filter: optional tighter floor with better model
                   score:  s = p_ctr · p_cvr · bid · q_factor
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │  Mixed-List Re-Ranking    │  top ads + organic list
                   │  Listwise optimization    │  Section 7
                   │  under hard constraints   │
                   │  (cgr/ , cgr_w_cvr/)      │
                   └───────────┬───────────────┘
                R(A) = Σ_i (revenue_i + engagement_i + profit_i - penalty_i)
                               │
                               ▼
                      ┌──────────────┐
                      │  Final SERP  │
                      │  (mixed page)│
                      └──────────────┘

Labels & objectives (Section 2) feed into all model stages.
Relevance details (Section 3) cover labeling, encoding, and distillation.
Multi-task balancing (Section 8) applies to Sections 5 and 6.
Stage interfaces (Section 10.1) define the contract at each arrow.
```

---

## 2. Objectives, Labels, and Their Semantics

Every model in the pipeline predicts one or more user outcomes. This section defines the three core label types, where they come from, their biases, and how they map to model heads across stages.

### 2.1. Click labels

**Source**: logged impression data. Each (query, ad, position, clicked?) tuple from serving logs produces a binary label.

**Scale**: billions of examples per day on a major platform. This is the most abundant signal.

**Biases**:

- **Position bias**: users click higher-positioned results more regardless of quality. Click labels conflate "the user examined and liked this ad" with "the user clicked because it was at the top." Must be debiased (see Section 6.6).
- **Presentation bias**: ad creative, image, price shown in snippet all affect CTR independently of underlying relevance.
- **Selection bias**: only ads that passed earlier pipeline stages are ever shown. The model never sees negatives from the full candidate space, only from the served set.

**Positive rate**: typically 2–5% in sponsored search. The extreme class imbalance (95–98% negatives) means that most training examples provide only "this ad was not clicked" signal. Standard mitigations include negative downsampling (train on a random subset of negatives) and calibration correction to recover true CTR after downsampling. Focal loss can also help by down-weighting easy negatives, but is less common in production ads systems than downsampling.

**Semantics**: click ≈ "the user was interested enough to engage." It is a noisy proxy for relevance, strongly confounded by position and presentation. Useful at scale but should not be the only signal.

### 2.2. Conversion labels

**Source**: post-click tracking. A conversion event (purchase, sign-up, add-to-cart) is logged when the user completes an action after clicking an ad. Joined back to the original impression via attribution windows (typically 1–30 days).

**Scale**: orders of magnitude sparser than clicks. A 2% CTR with a 5% post-click conversion rate means ~1 conversion per 1000 impressions.

**Biases**:

- **Delayed feedback**: conversions can arrive hours or days after the click. Training on recent data will systematically under-count conversions, biasing $p_{cvr}$ downward. Standard mitigations include waiting for the attribution window to close before using an example, or using importance-weighted corrections for fresh data.
- **Price and inventory effects**: conversion depends on factors outside the model's control (product price, stock availability, competing offers). This makes CVR inherently noisier than CTR.
- **Selection bias**: only clicked ads can convert, so CVR training data is conditioned on click, which is itself biased.

**Positive rate**: extremely low. Among clicked ads, only 0.1–1% lead to a conversion (varying by vertical — e-commerce tends higher, lead-gen lower). This is compounded by the fact that CVR training data is already conditioned on click (itself 2–5%), so conversions represent roughly 1 in 1,000 to 1 in 50,000 impressions. Negative downsampling is standard, with careful calibration correction. Some systems also use multi-task learning with CTR as an auxiliary signal to provide denser gradients to the shared backbone, partially compensating for CVR sparsity.

**Semantics**: conversion ≈ "the user completed a valuable action." This is the label most directly tied to advertiser value, but too sparse to be the sole training signal for early-stage models.

### 2.3. Human relevance labels

**Source**: human annotation. Trained raters judge (query, ad) pairs on a graded relevance scale (0 = irrelevant, 1 = weak match, 2 = relevant, 3 = highly relevant).

**Scale**: expensive. Typically 10K–100K judged pairs, refreshed periodically. Orders of magnitude smaller than click data.

**Semantics**: relevance ≈ "this ad is a semantically appropriate result for this query." It is the cleanest signal for semantic match quality, free from position and presentation confounds. It captures something that click and conversion labels cannot: whether an ad *should* have been shown, not just whether users happened to engage with it.

For rubric design, annotation process, loss encoding, and scaling via LLM teacher distillation, see Section 3.

### 2.4. How labels map to model heads across stages

| Stage | Click (CTR) | Conversion (CVR) | Relevance | Role of relevance |
| --- | --- | --- | --- | --- |
| Retrieval (Section 4) | primary contrastive loss on click pairs | not used | auxiliary loss on judged pairs | regularizes embeddings toward semantic quality |
| Pre-ranking (Section 5) | pointwise CTR head | pointwise CVR head | ordinal classification head | separate head; feeds into $q_{rel}$ quality factor |
| Final ranking (Section 6) | pointwise CTR head (debiased) | pointwise CVR head | ordinal classification head | separate head; feeds into $q_{factor}$ |
| Mixed-list reranking (Section 7) | exposure/click head (listwise) | optional conversion head | not a direct head | relevance quality baked into upstream ad scores |

Key design points:

- **Click labels are used everywhere** but must be debiased at ranking stages (Section 6.6).
- **Conversion labels are used from pre-ranking onward**, where the candidate set is small enough that sparsity is manageable.
- **Relevance labels serve dual duty**: as an auxiliary loss for retrieval (to correct for click bias in the embedding space) and as a standalone prediction head for pre-ranking and final ranking (to gate low-relevance ads via the $q_{rel}$ / $q_{factor}$ multiplier).
- **Relevance labels should not dominate any objective.** In sponsored search, semantic match is necessary but not sufficient. Weight the relevance loss at 0.1–0.2 relative to the primary click/contrastive loss and tune on downstream metrics.

---

## 3. Relevance: Labeling, Encoding, and Distillation

Human relevance is the scarcest but cleanest supervision signal in the pipeline. This section covers how to define it, annotate it, encode it as a training loss, scale it via LLM distillation, and apply it in the scoring formula at each stage.

### 3.1. Rubric design and biases

**Recommended scale**:

| Grade | Meaning |
| --- | --- |
| 0 | Irrelevant — ad has no meaningful connection to the query |
| 1 | Weak match — tangentially related but not what the user likely wants |
| 2 | Relevant — reasonable match for the query intent |
| 3 | Highly relevant — strong semantic and intent match |

**Rubric principle**: the rubric should focus on **query-item semantic match**, not on commercial value, historical click behavior, or ad quality. "Is this ad a good answer to this query?" is the question, not "will this ad get clicked?" or "is this a good ad?"

**Biases**:

- **Rater subjectivity**: inter-annotator agreement is imperfect, especially at the boundary between grades 1 and 2. Mitigate with clear rubrics, multiple raters per pair, and adjudication for disagreements.
- **Coverage bias**: only a sample of (query, ad) pairs are judged. The judged set may not represent the full traffic distribution, especially for long-tail queries.
- **Staleness**: relevance judgments reflect the state of the query and ad at annotation time. They do not update as ad content, landing pages, or user expectations change.

### 3.2. Annotation process

A practical annotation pipeline:

1. **Sampling**: sample (query, ad) pairs from serving logs, stratified by query frequency and ad category. Include both served and near-miss candidates (ads that almost made it into the served set) to cover the decision boundary.
2. **Annotation**: each pair is judged by 2–3 trained raters using the graded scale above. Provide the query, ad title, ad description, and landing page URL. Do not show the ad's position, CTR, or bid.
3. **Adjudication**: for pairs where raters disagree by more than 1 grade, use a senior rater or majority vote.
4. **Quality control**: embed known "gold" pairs with established labels into the annotation queue. Monitor rater accuracy on gold pairs and retrain or remove raters who drift.
5. **Refresh cadence**: re-annotate a rolling sample (e.g. quarterly) to keep the label set current.

### 3.3. Loss encoding for graded relevance

The 0/1/2/3 grades are ordinal — grade 3 > 2 > 1 > 0, and predicting 0 when truth is 3 is worse than predicting 2. The choice of loss function must respect this structure. Three main options:

| Approach | How it works | Respects ordering? | Produces calibrated score? | Complexity |
| --- | --- | --- | --- | --- |
| Ordinal regression (cumulative threshold) | Learn a latent score $z$ and 3 thresholds $\theta_1 < \theta_2 < \theta_3$. Predict $P(grade \geq k) = sigmoid(z - \theta_k)$ | Yes — misranking by 2 grades is penalized more than by 1 | Yes — $z$ is a continuous relevance score usable directly | Moderate |
| Pairwise ranking loss | For each query, form (higher-grade item, lower-grade item) pairs, train with margin loss | Yes — by construction | No — only relative ordering, no absolute $p_{rel}$ | Moderate |
| Regression (MSE/Huber on grade) | Treat 0–3 as a continuous target | Yes — larger errors penalized more | Partially — produces a score, but assumes equal spacing between grades | Simple |

**Recommendation**: use **ordinal regression** for stages that need a calibrated relevance score (pre-ranking and final ranking), because:

- It respects grade ordering — a 2-grade error produces a larger loss than a 1-grade error.
- It outputs a single continuous score $z$ that can be used directly as $p_{rel}$ (after sigmoid transformation) or mapped to $q_{rel}$ via a monotone function.
- The learned thresholds $\theta_k$ adapt to the actual grade boundary locations in the embedding space, accommodating unequal spacing between grades.
- It is more sample-efficient than pairwise loss (each labeled example contributes directly, rather than needing to form pairs).

For the retrieval stage (Section 4.1.3), **pairwise ranking loss** is a reasonable alternative because retrieval only needs relative ordering in the embedding space, not a calibrated probability.

Regression is the simplest option but assumes equal grade spacing, which may not reflect the true semantic gap between "irrelevant" and "weak match" versus "relevant" and "highly relevant."

### 3.4. Scaling relevance via LLM teacher distillation

Human annotation (Section 3.2) produces 10K–100K judged pairs — clean but far too few to be the primary training signal for models that see billions of examples. LLM teacher distillation bridges this gap by producing **millions of soft relevance labels** at a fraction of the cost of human annotation.

#### 3.4.1. Teacher model

Fine-tune a large language model on the human relevance labels as a **cross-encoder** over (query, ad text). The teacher can be an encoder model (BERT, DeBERTa) or a decoder-only model (LLaMA, Mistral) with a regression head — decoder-only models are now competitive as reranking cross-encoders and offer easier scaling:

- **Input**: concatenation of query, ad title, ad description, and optionally landing page text.
- **Output**: relevance grade prediction (ordinal regression head, same as Section 3.3).
- **Architecture**: a cross-encoder is appropriate here because the teacher runs **offline** — it does not need to meet serving latency. It can afford to jointly attend over query and ad text, unlike the two-tower retrieval model which must decompose for ANN search.

Train on the full human-labeled set with standard train/validation splits. Validate on held-out human judgments to ensure the teacher's predictions align with the rubric.

#### 3.4.2. Distillation pipeline

Once the teacher is trained:

1. **Score at scale**: run the teacher over millions of (query, ad) pairs sampled from serving logs. This produces a **soft relevance score** (continuous $z$ from the ordinal regression head) for each pair.
2. **Use as training targets**: the soft scores become distillation targets for the lightweight pipeline models:
   - At **retrieval** (Section 4.1.3): pairwise or ordinal loss on teacher-scored pairs, supplementing the small human-labeled set. This gives the embedding space a much denser relevance signal.
   - At **pre-ranking and final ranking** (Sections 5, 6): the relevance head can be trained on teacher soft labels in addition to (or instead of) the sparse human labels.
3. **Refresh**: re-run the teacher periodically (e.g. monthly) on new (query, ad) pairs as ad inventory changes. This keeps the relevance signal current without requiring new human annotations.

#### 3.4.3. Soft vs. hard labels

The teacher outputs continuous scores, not hard grades. This is more informative than the 4-grade scale:

- A hard label says "this is grade 2." A soft label says "this is 0.73 on the relevance scale — closer to grade 2 than grade 3, but with some uncertainty."
- The student model learns from the teacher's confidence, not just the argmax. Items near grade boundaries get appropriately uncertain targets.
- Soft labels also reduce the impact of rater disagreement — the teacher's prediction is a smoothed average of the patterns in the human data, not a noisy single-rater judgment.

#### 3.4.4. Calibration and validation

The teacher can drift from the human rubric, especially on query types or ad formats underrepresented in the training set. Mitigations:

- **Periodic validation**: sample teacher predictions and have human raters judge them. Track agreement rate over time.
- **Confidence thresholding**: only use teacher predictions where the model is confident (e.g. predicted grade is far from a threshold boundary). Flag low-confidence predictions for human review.
- **Domain-specific monitoring**: track teacher accuracy by query category, ad vertical, and query frequency tier. The teacher may perform well on head queries but poorly on long-tail.

#### 3.4.5. LLM-assisted annotation

A related but distinct use: use a strong general-purpose LLM (e.g. GPT-4, Claude) as a **first-pass annotator** in the human annotation pipeline (Section 3.2). The LLM judges (query, ad) pairs, and human raters audit a sample. This can increase annotation throughput 5–10x while maintaining quality, provided the LLM judgments are validated against the human rubric.

This is different from the teacher distillation above: LLM-assisted annotation produces **hard grades** that enter the human-labeled pool. Teacher distillation produces **soft scores** at massive scale for direct model training.

### 3.5. Relevance in the scoring pipeline

Relevance is consumed differently at each pipeline stage:

| Stage | Relevance source | Filter | Multiplier |
| --- | --- | --- | --- |
| Retrieval (Section 4) | Implicit via embedding geometry | No — ANN top-K cutoff acts as implicit threshold | No — no eCPM at this stage |
| Pre-ranking (Section 5) | Dedicated $p_{rel}$ head | Yes — hard floor $\tau_{floor}$, drop grade-0 ads | Yes — $q_{rel} = f(p_{rel})$ in eCPM |
| Final ranking (Section 6) | More accurate $p_{rel}$ head | Optional tighter floor | Yes — relevance folded into $q_{factor}$ |

At pre-ranking and final ranking, relevance is applied as **both a hard floor and a multiplicative factor** — not one or the other. The hard floor is a user-experience safety net (no irrelevant ad regardless of bid). The multiplicative factor ensures higher-relevance ads rank above lower-relevance ads, all else equal. See Sections 5.4 and 6.4 for details.

---

## 4. Lightweight Retrieval Ranking

This stage runs inside each search shard. Its goals are:

- keep recall high across diverse query types
- keep compute extremely low (tower outputs are precomputed or cached; scoring is a dot product + weighted sum)
- produce a candidate set with enough headroom for later stages to correct mistakes

This stage should not try to solve the full ranking problem. It is a retrieval filter, not the final decision maker.

It has three parts: a two-tower embedding model, multi-channel candidate generation, and lightweight scoring over the merged candidate set.

### 4.1. Two-tower embedding model

The retrieval model is a **two-tower (dual-encoder) architecture** with independent towers for different entity types. Each tower produces a dense embedding, and retrieval is based on cosine similarity between them.

#### 4.1.1. Tower structure

- **Item tower**: encodes item features (title, category, attributes, historical quality signals) into a dense embedding $e_i$. Computed **offline or nearline** and indexed for fast lookup.
- **Request tower**: encodes query, user, and context features into a dense embedding $e_r = f(query, user, context)$. Computed **online** at request time. Combining query, user, and context into a single tower avoids the need for separate cosine heads with unclear standalone supervision (see Section 2.1).

Each tower is a small MLP or shallow transformer that maps heterogeneous input features to a fixed-dimensional vector (typically 64-256 dimensions).

**Optional personalization tower**: if non-search interaction data is available (browse history, past purchases), a separate user tower producing $e_u$ can be trained independently on (user, interacted_item) pairs. This gives a clean user-item affinity signal not confounded by query, and enables a dedicated personalization ANN channel (Section 4.2).

#### 4.1.2. Cosine similarity heads

The primary retrieval score is:

- $\cos(e_r, e_i)$ — request-item relevance (combines query semantics, user context, and session context)

If a separate personalization tower exists:

- $\cos(e_u, e_i)$ — user-item affinity from non-search behavior

These scores are cheap to compute because each tower runs independently — there is no cross-attention or feature interaction between towers. This decomposability is what makes the model suitable for retrieval at scale.

#### 4.1.3. Training

The two-tower model is trained with a **multi-signal approach** using the label types defined in Section 2:

**Primary loss — contrastive on click pairs** (weight ~0.7–0.8):

The dominant training signal comes from (query, clicked_item) positive pairs from logged impressions, trained with a retrieval-aligned contrastive loss:

- **In-batch negative sampling**: treat other items in the same mini-batch as negatives. Simple and effective for large batch sizes.
- **Sampled softmax**: sample negatives from the full item corpus according to a frequency-weighted distribution.
- **Hard negative mining**: mix random negatives with "near-miss" items that are similar but not relevant, to sharpen the embedding boundary.

**Choosing a negative sampling strategy**: start with in-batch negatives (simplest, effective at batch sizes ≥1024). Add sampled softmax if the item corpus is highly skewed (popular items dominate batches). Introduce hard negative mining only after the model has converged with random negatives — mining too-hard negatives early causes training instability and embedding collapse. A practical schedule is: random negatives for the first pass, then mix in 10–20% hard negatives mined from the previous checkpoint's top-K.

A pointwise CTR loss should **not** be used here. Retrieval needs an embedding space with good nearest-neighbor geometry, which contrastive losses provide and pointwise losses do not.

**Auxiliary loss — relevance on judged and distilled pairs** (weight ~0.1–0.2):

Human relevance labels (Section 2.3) and LLM teacher distillation targets (Section 3.4) provide a secondary training signal that corrects for click bias in the embedding space. Applied as a pairwise ranking loss or ordinal loss over (query, item, relevance_grade) triples.

This auxiliary loss is valuable because click data is confounded by position and presentation bias — the retrieval stage has no position features to debias with. Relevance labels give a cleaner notion of semantic match, pulling the embedding space toward "this ad is a good answer to this query" rather than "this ad was clicked because it was shown at position 1."

Teacher-distilled soft labels (Section 3.4) are especially valuable here: they provide orders of magnitude more relevance-labeled pairs than the human annotation pool alone, giving the embedding space a much denser relevance signal.

**Optional — personalization tower loss**:

If a separate user tower exists (Section 4.1.1), train it with a contrastive loss on (user, interacted_item) pairs from non-search behavior (browse, purchase history). This signal is independent of query and gives clean user-item affinity.

### 4.2. Multi-channel candidate generation

Retrieval should be the **union** of multiple independent channels, each optimized for different recall characteristics:

| Channel | Method | Strength |
| --- | --- | --- |
| Semantic | ANN search over $\cos(e_r, e_i)$ | Recall on paraphrased or intent-similar queries |
| Personalized | ANN search over $\cos(e_u, e_i)$ (if separate user tower exists) | Recall on user-specific preferences |
| Lexical | Inverted index with BM25 or term matching | Precision on exact-match and rare-term queries |

Each channel produces its own top-K candidate set. The final retrieval candidate set is their **union** (deduplicated by item ID). This hybrid approach ensures:

- semantic recall from embedding ANN — catches relevant items even when query wording differs from item titles
- lexical precision from BM25 — catches exact-match queries where embedding models may under-recall (especially for long-tail or product-specific queries)
- personalization from user-item ANN — catches items aligned with user history even if the query is generic

The ANN infrastructure (e.g. HNSW, ScaNN, FAISS) runs per-shard over pre-indexed item embeddings from the item tower.

### 4.3. Candidate scoring and fusion

After candidate generation, a lightweight scoring pass ranks the merged union candidate set. The score must be decomposable and cheap:

$$s_{retrieve}(r, i) = w_r \cdot \cos(e_r, e_i) + w_u \cdot \cos(e_u, e_i) + w_{bm25} \cdot BM25(q, i) + w_{qual} \cdot Q(i)$$

where:

- $\cos(e_r, e_i)$ is the request-item similarity from the primary two-tower model
- $\cos(e_u, e_i)$ is the user-item affinity from the personalization tower (0 if no separate user tower exists)
- $BM25(q, i)$ is the lexical relevance score (0 if the item was not retrieved by the lexical channel)
- $Q(i)$ is a lightweight item-quality prior

A simple form for the quality prior is:

$$
Q(i) = a_1 \cdot sale(i) + a_2 \cdot page\_quality(i) + a_3 \cdot merchant\_quality(i)
$$

where the quality terms should be normalized so that they do not swamp the similarity and BM25 terms.

The weights $w_r, w_u, w_{bm25}, w_{qual}$ can be hand-tuned initially and later learned from logged data via a lightweight LTR model.

**Relevance handling at retrieval**: there is no separate relevance filter or relevance multiplier at this stage. Relevance is handled **implicitly** through the embedding geometry — the auxiliary relevance loss in training (Section 4.1.3) shapes the embedding space so that semantically relevant items have higher $\cos(e_r, e_i)$. The ANN top-K cutoff acts as an implicit relevance threshold. Explicit relevance filtering begins at pre-ranking (Section 5.4), which is the first stage with a dedicated relevance prediction head.

### 4.4. Evaluation

Retrieval evaluation is complicated by **selection bias**: click labels only exist for items that were served, so you cannot directly measure whether a new model's novel retrievals are good or bad.

#### 4.4.1. Model-level evaluation

These metrics assess whether the two-tower model is learning well, independent of the retrieval system built on top of it.

**Held-out loss tracking (per-signal)**: track the click contrastive loss and relevance loss separately on validation sets. If click loss improves but relevance loss degrades, the two training signals are in gradient conflict. If relevance loss plateaus early, oversampling may be causing overfitting on the small judged set. Both improving together indicates healthy multi-signal training.

**Contrastive accuracy**: for held-out (query, clicked_item) pairs, compute: is the clicked item ranked in the top K by cosine similarity among a random set of N candidates? This is essentially Recall@K measured directly on the embedding space before ANN approximation — it isolates model quality from index quality.

**Pairwise relevance accuracy**: for held-out judged pairs where $grade(A) > grade(B)$ for the same query, how often does the model score $z_A > z_B$? This directly measures whether the auxiliary relevance loss is working.

**Embedding space health**: two-tower contrastive models can degenerate in well-known ways:

- **Collapse**: all embeddings converge to the same region. Measure by computing average pairwise cosine similarity between random items — it should stay well below 1 (ideally near 0 for a uniformly distributed space).
- **Dimensional collapse**: embeddings use only a few dimensions of the available space. Measure via the singular value spectrum of a batch of embeddings — a healthy space has a relatively flat spectrum, not one dominated by 2–3 singular values.
- **Alignment vs. uniformity** (Wang & Isola, 2020): alignment measures how close positive pairs are; uniformity measures how spread out all embeddings are on the hypersphere. Good training improves both. If alignment improves but uniformity degrades, the model is collapsing.

**ANN approximation gap**: the serving system uses approximate nearest neighbor search (HNSW, ScaNN), not exact cosine search. Measure ANN recall: $|\text{ANN top-K} \cap \text{exact top-K}| / K$. If this drops below ~95%, the ANN index parameters need tuning. A model change can shift the embedding distribution enough to break a previously well-tuned index.

**Embedding stability across retraining**: when the model is retrained, item embeddings change and the ANN index must be rebuilt. Measure average cosine similarity between old and new embeddings for the same items. Large shifts mean costly full re-indexing; small shifts may allow incremental updates.

#### 4.4.2. Offline retrieval metrics

**Recall@K** is the primary metric — of all relevant items for a query, what fraction appears in the top K retrieved?

$$Recall@K = \frac{|\text{relevant items in top-K}|}{|\text{all relevant items}|}$$

What counts as "relevant" depends on the label source:

| Relevance definition | Strength | Weakness |
| --- | --- | --- |
| Clicked in held-out logs | Large-scale, automatic | Position-biased — misses relevant items that were never shown |
| Human-judged grade $\geq$ 2 (Section 2.3) | Unbiased by position | Small scale — only covers the judged pool |
| Purchased / converted | Strongest commercial signal | Extremely sparse |

Best practice: report **both** click-based Recall@K on a large held-out set (for statistical power) and judgment-based Recall@K on the annotated pool (for unbiased signal).

Additional metrics:

- **MRR (Mean Reciprocal Rank)**: average of $1/rank$ of the first relevant item. Measures how quickly the model surfaces a good candidate.
- **NDCG@K**: uses graded relevance — a grade-3 item ranked first matters more than a grade-1 item. Only computable on the judged subset.
- **Coverage**: fraction of queries where at least one relevant ad is retrieved. A model that fails to find any ads for a significant share of queries has a coverage problem, even if Recall@K is high on the remaining queries.

**Channel-level evaluation** (for Section 4.2 multi-channel retrieval): measure the **marginal recall contribution** of each channel by computing Recall with and without each channel. If a channel adds less than ~1–2% marginal recall, it may not justify its infrastructure cost.

#### 4.4.3. Online evaluation

Offline recall does not tell you whether better candidates actually improve the final page. End-to-end A/B tests should measure:

- **Downstream CTR/CVR**: do better retrieval candidates lead to more clicks and conversions after the full pipeline runs?
- **Ad revenue**: does improved retrieval translate to higher eCPM in the final auction?
- **Query-level coverage**: fraction of queries with at least one ad shown. Retrieval failures propagate — if no good ads are retrieved, no later stage can fix it.
- **Latency**: retrieval runs per-shard under tight latency budgets. A model that improves recall by 2% but adds 5ms p99 latency may not be worth it.

#### 4.4.4. The counterfactual problem

The hardest part of retrieval evaluation: you cannot measure what you did not retrieve. If the current model retrieves items {1, 2, 3} and a new model retrieves {1, 4, 5}, you have click labels for items 1, 2, 3 from serving logs but none for items 4 and 5.

Approaches to address this:

- **Randomized retrieval**: inject random items into the candidate set for a small fraction of traffic. Gives unbiased labels for items outside the normal retrieval set, at the cost of some user experience degradation on that traffic slice.
- **Human evaluation on model diff**: identify queries where the old and new model disagree, send the disagreement items to human raters. Directly measures whether novel retrievals are relevant.
- **Interleaving experiments**: merge candidates from both models, serve the combined set, and credit the model that contributed clicked items. More sample-efficient than standard A/B tests for detecting retrieval quality differences.

---

## 5. Multi-Tower, Multi-Task Model for Pre-Ranking

This stage is primarily a pre-ranking model, but it can share representation layers with retrieval if the architecture is designed carefully.

**Latency constraint**: pre-ranking typically has a hard p99 budget (5–15ms for 100–1000 candidates). This limits model depth — shallow MLPs or small MMoE/PLE configurations are standard. Techniques like early-exit inference, quantization, and distillation from the final ranker can help recover quality within the latency envelope.

### 5.1. Relevance as a pre-ranking task

Human-labeled relevance (defined in Section 2.3, with details in Section 3) should be treated as a **separate prediction head**, not folded into CTR or CVR.

The pre-ranking model trains an ordinal classification head (Section 3.3) on the graded relevance labels. This head predicts $p_{rel}$, which is then used as a multiplicative quality gate $q_{rel}$ in the pre-ranking score (Section 5.4). Teacher-distilled soft labels (Section 3.4) can supplement the sparse human labels to provide denser training signal for this head.

Why a separate head matters:

- CTR is biased by position and presentation (Section 2.1). Relevance is not.
- CVR is even sparser and strongly affected by price, inventory, and user intent maturity.
- Keeping relevance separate lets the system answer "is this ad relevant?" independently from "will it get clicked?" and "will it convert?"

**Practical caution**: do not let the relevance head dominate the multi-task objective. In sponsored search, perfect semantic match is necessary but not sufficient. Weight the relevance loss below the CTR and CVR losses and calibrate against business outcomes.

### 5.2. Can one model support both retrieval and pre-ranking?

In principle yes, but this is **not standard practice at major platforms** and carries known risks.

The practical pattern, if sharing is desired, is:

1. use shared tower encoders for item, query, user, and context
2. export a lightweight retrieval score that depends only on separately computed embeddings
3. add a deeper pre-ranking head on top of the tower outputs for the merged candidate set

That means the same backbone can support both stages, but the scoring path is not identical.

#### 5.2.1. Architecture if sharing

- **Item tower:** produces item embeddings offline or nearline
- **Request tower:** produces request embedding online (query + user + context)
- **Retrieval head:** cosine similarity over tower outputs
- **Pre-ranking head:** shallow multi-layer perceptron (MLP) over concatenated tower outputs and a small set of scalar features

In other words:

- retrieval uses the towers directly
- pre-ranking uses the same towers plus an extra interaction layer

#### 5.2.2. Risks and caveats

Retrieval and pre-ranking optimize for fundamentally different objectives:

- **Retrieval** optimizes for recall over the full candidate space, typically via contrastive loss, sampled softmax, or in-batch negatives.
- **Pre-ranking** optimizes for pointwise CTR/CVR prediction on the much smaller merged candidate set.

Joint training creates **gradient conflict**: the contrastive loss pushes embeddings to separate positives from the full negative space, while the CTR loss pushes embeddings to separate clicks from impressions. These can pull tower representations in different directions.

Industry evidence:

- Alibaba's **COLD** system (CIKM 2020) explicitly designs pre-ranking as a separate lightweight model, not a shared-tower derivative.
- Meta trains retrieval (EBR) and ranking models separately.
- Google uses distinct models for ad retrieval and ad ranking.

#### 5.2.3. Recommendation

If infrastructure cost is not the binding constraint, **train separate models**. If sharing is necessary, the safest approach is:

1. Train the retrieval towers first with a retrieval-aligned loss.
2. **Freeze the tower weights** and train only the pre-ranking head on top.
3. This avoids gradient conflict while still reusing the tower embeddings.

Do not jointly train a single model end-to-end for both retrieval and pre-ranking unless you have empirical evidence that it works for your specific feature distribution and traffic pattern.

### 5.3. Multi-task outputs

For pre-ranking, a reasonable task set is:

- CTR (see Section 2.1 for label semantics)
- CVR (see Section 2.2)
- relevance (see Section 3.3; ordinal classification head)
- optional post-click value target such as expected margin or GMV

### 5.4. Pre-ranking score

The pre-ranking score should follow the standard **eCPM (expected cost per mille)** formulation used in ad auctions. The score must be proportional to expected revenue for the auction to be incentive-compatible:

$$
s_{prerank} = p_{ctr} \cdot p_{cvr} \cdot bid \cdot q_{rel}
$$

where:

- $p_{ctr}$ is the predicted click-through rate
- $p_{cvr}$ is the predicted conversion rate (set to 1 for CPC-only campaigns)
- $bid$ is the advertiser's bid
- $q_{rel}$ is a relevance quality factor derived from $p_{rel}$, used as a multiplicative gate

For CPC campaigns, the score simplifies to $p_{ctr} \cdot bid \cdot q_{rel}$.

Relevance should be applied as **both a hard floor and a multiplicative factor** — not one or the other:

**Hard floor (filter)**: before scoring, drop any ad with $p_{rel} < \tau_{floor}$. This is a user-experience safety net that guarantees no clearly irrelevant ad can appear regardless of bid. Set $\tau_{floor}$ conservatively — it should only kill grade-0 (irrelevant) ads, roughly filtering ~5–10% of candidates. A filter alone has problems: it creates a cliff effect at the threshold boundary (noisy predictions near $\tau_{floor}$ cause arbitrary keep/drop decisions) and it discards information (a grade-3 ad is treated identically to a grade-1 ad once past the gate).

**Multiplicative factor (ranking)**: among surviving ads, relevance enters the score as $q_{rel} = f(p_{rel})$ where $f$ is a smooth, monotone function. Options for $f$:

- Linear: $f(p_{rel}) = p_{rel}$ — simple, full dynamic range
- Clipped linear: $f(p_{rel}) = \max(p_{rel}, \epsilon)$ — avoids near-zero multiplication
- Power: $f(p_{rel}) = p_{rel}^\alpha$ where $\alpha < 1$ compresses the range (softer penalty for moderate relevance) and $\alpha > 1$ amplifies it

A multiplicative factor alone also has problems: a sufficiently high bid can compensate for low relevance ($bid = \$100$ with $p_{rel} = 0.05$ scores the same as $bid = \$5$ with $p_{rel} = 1.0$), allowing irrelevant ads through. The hard floor prevents this.

The key design constraint is that relevance must act as a **multiplicative quality adjustment**, not an additive term. An additive combination like $\alpha \cdot p_{ctr} + \beta \cdot p_{rel}$ has no economic justification — it allows a high-relevance zero-bid ad to outrank a relevant ad with a real bid, breaking auction incentive properties.

This two-layer approach (hard floor + soft multiplier) follows the standard industry pattern — Google Ad Rank uses a quality threshold below which ads are ineligible, and above that threshold, quality score multiplies into the auction rank.

### 5.5. Do we need extra tasks to support retrieval?

Usually not in the form of explicit tasks that regress the cosine similarities themselves.

What retrieval actually needs is an embedding space in which query, user, context, and item vectors have the right geometry for nearest-neighbor search or cheap similarity scoring. That is better enforced through a retrieval-aligned training objective such as:

- contrastive loss
- sampled softmax
- pairwise ranking loss
- in-batch negative training

So the cleaner design is:

- use CTR, CVR, relevance, and value-style heads for pre-ranking
- add a retrieval loss on the shared embedding towers if retrieval quality matters enough
- keep cosine similarity as the serving-time retrieval score, not as a separate prediction target

If retrieval is only a lightweight first-pass filter, the shared towers may already be good enough. If retrieval quality is critical, then adding a dedicated retrieval loss is more defensible than adding tasks whose only job is to imitate cosine similarity.

---

## 6. Feature-Interaction Network for Final Ads Ranking

This stage ranks the top ad candidates after pre-ranking, typically around 30 to 100 items.

### 6.1. Why use a feature-interaction network?

At this stage, the system can afford richer feature interaction between:

- query terms
- user features and behavioral history
- context features
- item attributes
- auction features
- business features

Unlike a multi-tower model, a feature-interaction network (such as DCN-v2, DLRM, or DIN/DIEN) is not restricted to separately encoded representations. It can learn fine-grained cross-feature interactions over a wide feature set — not just text, but hundreds of dense and sparse signals including behavioral history, bid, budget, advertiser quality, and query-ad match features.

A key advantage of this stage over pre-ranking is access to **real-time features**: remaining daily budget, recent user behavior in the same session, live inventory signals, and fresh auction context. These features change too fast for pre-ranking's latency budget but are critical for final ranking quality.

Note: "cross-encoder" in the NLP sense (BERT over concatenated text) is too narrow for this role. Ads ranking depends heavily on non-textual features. Text relevance is one input signal to the feature-interaction network, not the model itself.

### 6.2. Inputs

- dense features from query, item, user, and context
- behavioral features such as historical CTR, past impressions, and engagement signals
- lexical and semantic matching features
- auction features such as bid, budget, remaining daily budget, and campaign metadata
- quality and policy features

### 6.3. Outputs

- CTR (position-debiased — see Section 6.6)
- CVR
- relevance
- optional value estimate such as expected margin, order value, or long-term value

All label types are defined in Section 2.

### 6.4. Final ad score

The final ad score must follow the **eCPM** formulation to maintain auction incentive compatibility:

$$
s_{final} = p_{ctr} \cdot p_{cvr} \cdot bid \cdot q_{factor}
$$

where $q_{factor}$ is a composite quality multiplier that may incorporate:

- semantic relevance ($p_{rel}$ as a multiplicative gate)
- landing page quality
- policy penalties (multiplicative suppression for policy-violating ads)
- calibration adjustment

For CPC campaigns (no conversion optimization), $p_{cvr}$ is set to 1, giving:

$$
s_{final}^{CPC} = p_{ctr} \cdot bid \cdot q_{factor}
$$

For CPA campaigns, the full form applies and $bid$ represents the target cost-per-acquisition.

As at pre-ranking (Section 5.4), relevance is applied as **both a filter and a multiplier**. The final ranking model has a more accurate relevance prediction than the pre-ranker, so an optional **tighter floor** can be applied here — catching borderline-irrelevant ads that slipped past the pre-ranking filter. The multiplicative relevance component is folded into $q_{factor}$.

The important design point is that this stage ranks ads against other ads. It still does not choose the final mixed page layout.

### 6.5. Reference architectures

Common architectures for this stage include:

- **DCN-v2** (Google, 2021): explicit cross-feature interaction layers combined with deep MLP
- **DLRM** (Meta, 2019): embedding lookups for sparse features + MLP interaction with dense features
- **DIN / DIEN** (Alibaba, 2018-2019): attention over user behavioral history for interest modeling

These networks typically take 200-500+ input features and produce calibrated multi-task outputs.

### 6.6. Position bias debiasing

Click-through rate is strongly confounded by display position — users click top results more regardless of relevance. If $p_{ctr}$ is trained on logged click data (Section 2.1) without position debiasing, estimates will be inflated for historically top-served ads and deflated for lower-served ads. This corrupts both ranking quality and auction fairness.

Standard debiasing approaches:

1. **Position as a training feature with serving-time dropout**: include display position as a feature during training, but set it to a fixed default (e.g. position 1 or a neutral value) at serving time. This lets the model learn position effects but predict as if all ads were in the same position.

2. **Inverse propensity weighting (IPW)**: weight each training example by the inverse of the probability of being shown at that position. This corrects for the fact that higher-positioned ads generate more clicks regardless of quality.

3. **PAL (Position-Aware Learning)**: factorize $p_{click} = p_{exam}(position) \cdot p_{relevance}(query, ad)$. Train the examination model on position-randomized data, then use only the relevance component for ranking.

Position debiasing should be applied to CTR prediction in both the final ads ranker (this stage) and the pre-ranker (Section 5). The mixed-list reranker (Section 7) models position effects explicitly through its position embeddings and autoregressive attention.

### 6.7. Calibration

Calibration is mission-critical in ads because predicted probabilities directly determine advertiser charges. In a CPC auction, the charge to the advertiser is approximately:

$$
CPC = \frac{s_{next}}{p_{ctr} \cdot q_{factor}}
$$

If $p_{ctr}$ is miscalibrated (e.g. systematically 20% too high), advertisers are undercharged, platform revenue leaks. If too low, advertisers are overcharged, they reduce bids, and auction efficiency drops.

#### 6.7.1. When to calibrate

Calibration should be applied:

1. **After each model that produces probabilities used in scoring** — both the pre-ranker and the final ranker should output calibrated $p_{ctr}$ and $p_{cvr}$.
2. **As a post-processing step**, not baked into the model loss. This separation makes it easier to recalibrate when serving conditions change (new traffic mix, new ad inventory) without retraining the model.

#### 6.7.2. Standard methods

- **Platt scaling**: fit a logistic regression ($\sigma(a \cdot z + b)$) on a held-out calibration set, where $z$ is the model logit. Simple and effective for global calibration.
- **Isotonic regression**: non-parametric monotone function fit on held-out data. Better for correcting non-linear miscalibration but can overfit with small calibration sets.
- **Field-aware calibration** (Meta, 2021): segment predictions by key features (country, device, ad category) and calibrate per segment. Handles the reality that a model can be well-calibrated globally but miscalibrated within important subgroups.
- **Temperature scaling**: divide logits by a learned temperature $T$ before sigmoid. A special case of Platt scaling with $a = 1/T, b = 0$.

#### 6.7.3. Monitoring

Calibration drifts over time as user behavior and ad inventory change. Standard practice is to monitor:

- expected vs. observed CTR by decile (reliability diagram)
- expected calibration error (ECE) tracked daily
- segment-level calibration for high-revenue ad categories

When calibration drifts beyond a threshold, recalibrate on recent data without retraining the full model.

---

## 7. Re-Ranking for Mixing Ads into the Organic Result List

This is the last stage and the one most closely related to the code in this workspace.

The input is:

- an already-ranked organic result list
- a small set of top ad candidates from the final ads ranker
- hard structural constraints such as max ads, spacing, position bounds, and density limits

The output is the final mixed list shown to the user.

### 7.1. Why a separate re-ranking stage is needed

Final ads ranking scores ads independently. But the page decision is listwise.

The system must answer questions such as:

- should we insert zero, one, or two ads?
- where should they go?
- is a slightly weaker ad better if it keeps the page less crowded?
- how should ad revenue trade off against user engagement and organic value?

Those are list-level questions, not per-item ranking questions.

### 7.2. Listwise model role

A listwise re-ranker predicts position-dependent outcomes for a candidate mixed list:

- exposure probability
- click probability
- conversion probability

Those predictions combine into a list-level reward:

$$
R(A) = \sum_i (revenue_i + engagement_i + profit_i - penalty_i)
$$

subject to hard constraints on layout feasibility.

This is where the `cgr/` and `cgr_w_cvr/` directories in this workspace fit:

- `cgr/` models exposure and click for mixed-list reranking
- `cgr_w_cvr/` extends the same framework with conversion and profit-margin terms

### 7.3. Training

The listwise model is trained on **logged mixed pages** — each training example is a full SERP with observed per-position outcomes (impressions, clicks, conversions). Key considerations:

- **Data**: each training example is a (page layout, per-slot outcomes) tuple from serving logs. The model sees organic results and ad placements together.
- **Loss**: typically a combination of per-slot prediction losses (exposure CE, click CE, optional CVR CE) summed over the list positions. The model learns position-dependent effects (e.g., an ad at position 3 gets lower attention than position 1).
- **Counterfactual correction**: the model only observes outcomes for the layout that was actually served. Layouts that *could have been* shown have no labels. Importance weighting or off-policy correction (e.g., IPS on the layout policy) helps mitigate this.

### 7.4. Constraint enforcement

Hard business constraints (max ad count, minimum spacing between ads, position eligibility, ad density caps) should be enforced **outside** the model, not learned implicitly:

- **Candidate enumeration**: generate feasible layout candidates that satisfy all constraints, then score each with the listwise model. For small candidate sets this is exact; for larger sets, beam search or greedy insertion with constraint checking is standard.
- **Why not learn constraints?** Learned models can violate hard constraints at inference time. A constraint violation (e.g., showing 5 ads when the policy is max 3) has direct business and user-experience consequences that no soft penalty can adequately prevent.

---

## 8. Multi-Task Training: Balancing Competing Objectives

Models at the pre-ranking (Section 5), final ranking (Section 6), and mixed-list reranking (Section 7) stages all train multiple heads jointly — CTR, CVR, relevance, and possibly value targets. These tasks compete for shared representation capacity, and their gradients can conflict. This section covers techniques for managing that tension.

### 8.1. The problem

Multi-task learning promises positive transfer: a shared backbone learns richer representations than any single-task model. But in practice, tasks often interfere. The root cause is that the tasks being jointly trained differ dramatically in sample volume, label type, and value scale.

#### 8.1.1. Asymmetries across tasks

| Task | Typical daily sample count | Label type | Positive rate | Loss type | Gradient magnitude |
| --- | --- | --- | --- | --- | --- |
| CTR | ~billions (all impressions) | binary (0/1) | 2–5% | binary cross-entropy | large (high volume, frequent updates) |
| CVR | ~10–100× fewer (clicked items only) | binary (0/1) | 0.1–1% of clicks | binary cross-entropy | small (sparse, noisy) |
| Relevance | ~10,000–100,000× fewer (human-annotated) | ordinal (0–3) | N/A (graded, not binary) | ordinal CE or pairwise | very small (rare updates) |
| Value (GMV/margin) | similar to CVR (conversion events) | continuous, heavy-tailed | N/A | MSE or Huber | variable (heavy tail causes spikes) |

These asymmetries create three concrete problems:

- **Sample imbalance**: CTR produces gradient updates on nearly every mini-batch. CVR updates only on batches containing clicked items. Relevance updates only on the small fraction of examples with human labels. In naive training, the CTR head receives thousands of gradient steps for every one that the relevance head receives, pulling the shared backbone toward CTR-optimal representations.

- **Label scale and distribution mismatch**: CTR and CVR are binary, relevance is ordinal, and value targets are continuous and potentially heavy-tailed (a few high-value conversions dominate). When losses are summed without normalization, the task with the largest raw loss magnitude dominates the gradient, independent of sample count. A single high-GMV example can produce a gradient spike that overwhelms hundreds of CTR updates.

- **Convergence rate differences**: tasks with more data converge faster. If the CTR head reaches near-optimal performance while the relevance head is still early in training, continued CTR gradients can push shared representations away from where the relevance head needs them. This is the "training rate" asymmetry that GradNorm (Section 8.2) is designed to address.

#### 8.1.2. Symptoms

These asymmetries manifest as:

- **Gradient conflict**: the CTR gradient may push shared weights in a direction that hurts CVR prediction, and vice versa. When task gradients point in opposing directions, naive summation of losses leads to oscillation or convergence to a solution that is mediocre for all tasks.
- **Task dominance**: the task with the most data or the largest gradient magnitude dominates the shared representation. Without balancing, sparse heads (relevance, CVR) get starved.
- **Negative transfer**: in the worst case, a jointly trained model underperforms single-task models on one or more heads.

### 8.2. Loss weighting strategies

The simplest intervention is to weight the per-task losses (in mini-batch gradient descent):

$$
L_{total} = \sum_t w_t \cdot L_t
$$

**Static weights** (hand-tuned): set $w_{ctr}, w_{cvr}, w_{rel}$ manually. Easy to implement, but fragile — optimal weights change as data distributions shift and models are retrained. A reasonable starting point is to normalize by task label frequency (see the sample count column in Section 8.1.1) so that sparse tasks are not drowned out. For value targets with heavy-tailed distributions, clipping or using Huber loss before weighting prevents gradient spikes from dominating.

**Uncertainty weighting** (Kendall et al., 2018): learn task weights as a function of homoscedastic uncertainty. Each task's weight is $w_t = \frac{1}{2\sigma_t^2}$, where $\sigma_t$ is a learnable parameter. Tasks with higher predictive uncertainty are automatically down-weighted. Adds one learnable scalar per task — negligible overhead.

**GradNorm** (Chen et al., 2018): dynamically adjust $w_t$ during training to equalize the gradient norms across tasks relative to a reference training rate. If one task is learning faster than others, its weight is reduced. This prevents a single task from dominating the shared backbone.

### 8.3. Gradient-level methods

Rather than weighting losses, these methods operate directly on the per-task gradients:

**PCGrad** (Yu et al., 2020): when two task gradients conflict (negative cosine similarity), project one onto the normal plane of the other to remove the conflicting component. This preserves the non-conflicting part of each gradient. Cost: requires computing per-task gradients separately, roughly $T$× the backward pass cost for $T$ tasks.

**CAGrad** (Liu et al., 2021): find the gradient direction within a neighborhood of the average gradient that maximizes the minimum task improvement. More conservative than PCGrad — it guarantees that no task gets worse, at the cost of slower overall progress.

**Practical note**: gradient-level methods are effective but expensive. For a model with 3–4 tasks, the overhead of separate backward passes is manageable. For many tasks (>6), the cost may be prohibitive in production training pipelines.

### 8.4. Pareto multi-task optimization

A multi-task model sits on a **Pareto front** if no task's loss can be improved without degrading another. Pareto-based methods explicitly seek parameter updates that move toward this front:

**MGDA** (Multiple Gradient Descent Algorithm, Sener & Koltun, 2018): at each step, find the minimum-norm element in the convex hull of the per-task gradients. This is the steepest descent direction that improves all tasks simultaneously. When tasks conflict, MGDA automatically reduces the step size toward the conflicting directions.

**Nash-MTL** (Navon et al., 2022): frame multi-task optimization as a bargaining game where each task is a player. Find the Nash bargaining solution — the gradient direction that maximizes the product of per-task improvements. This gives a fairer balance than MGDA when tasks have very different scales.

**Strengths**: theoretically principled — guarantees convergence toward Pareto-optimal solutions. No need to hand-tune task weights.

**Limitations**: computationally expensive. Both methods require per-task gradient computation plus a convex optimization step at each training iteration. MGDA scales as $O(T^2 \cdot d)$ per step where $T$ is the number of tasks and $d$ is the parameter count. This is practical for offline experimentation and model selection but rarely used in production training loops on billion-scale data.

**When to use**: Pareto methods are most valuable during **model development** — to diagnose whether tasks are fundamentally conflicting and to find a good region of the Pareto front. The discovered weight ratios can then be frozen as static weights for production training.

### 8.5. Architecture-level separation

Instead of fighting gradient conflict in the optimizer, these methods reduce it structurally by giving tasks partially separate parameters:

**Shared-bottom with task-specific towers**: a shared feature-extraction backbone feeds into per-task MLPs. Simple and widely used. The shared backbone still suffers from gradient conflict, but the task-specific towers can compensate.

**MMoE** (Multi-gate Mixture-of-Experts, Ma et al., 2018): replace the shared backbone with multiple expert sub-networks. Each task has its own gating network that selects a soft mixture of experts. Tasks can route to different experts, reducing interference. Google reported significant gains over shared-bottom for multi-task ranking.

**PLE** (Progressive Layered Extraction, Tang et al., 2020): extends MMoE by adding task-specific experts alongside shared experts, with progressive extraction across layers. This is what the CGR model in this workspace uses (Section 7) — it has EXP-oriented and CLK-oriented attention blocks producing separate expert sets that are fused by task-specific gates.

**Comparison**:

| Method | Gradient conflict mitigation | Extra parameters | Implementation complexity |
| --- | --- | --- | --- |
| Shared-bottom + task towers | Low — conflict still in backbone | Low | Simple |
| MMoE | Medium — gating allows routing | Medium — multiple experts | Moderate |
| PLE | High — task-specific + shared experts | Higher | Moderate-high |

Architecture-level separation is the **most practical approach for production systems**: it adds a fixed parameter cost (not a per-step compute cost), is compatible with standard optimizers, and composes well with any loss-weighting scheme.

### 8.6. Two levels of multi-objective tradeoff

It is important to distinguish the **model-level** and **business-level** multi-objective problems:

**Model-level** (this section): how to train a model with multiple heads so that all heads achieve good predictive quality. The goal is accurate calibrated predictions for each objective. The techniques above (loss weighting, gradient methods, architecture separation) all address this.

**Business-level** (Sections 5.4, 6.4): how to combine the model's predictions into a final ranking score. The eCPM formula $s = p_{ctr} \cdot p_{cvr} \cdot bid \cdot q_{factor}$ is a business decision about how to trade off advertiser value, user experience, and platform revenue. Changing this formula changes what the platform optimizes for, but it does not change how the models are trained.

The key insight: model-level balancing aims for **accuracy on all tasks**, not for any particular business tradeoff. The business tradeoff is encoded in the scoring formula. If the model predicts well-calibrated $p_{ctr}$, $p_{cvr}$, and $p_{rel}$, the business can adjust how these combine without retraining.

### 8.7. Practical recommendations

1. **Start with architecture-level separation.** Use PLE (Progressive Layered Extraction) or MMoE (Multi-gate Mixture-of-Experts) for the pre-ranking and final ranking models. This provides the strongest baseline with no per-step overhead.

2. **Use task-specific learning rates** as a zero-overhead baseline. Give sparse tasks (relevance, CVR) a higher learning rate than CTR to compensate for fewer gradient updates. This is simpler than gradient surgery and often surprisingly effective.

3. **Add uncertainty weighting** as a lightweight second line of defense. One learnable scalar per task, negligible cost.

4. **Monitor per-task metrics independently.** Track CTR AUC, CVR AUC, and relevance accuracy/NDCG separately. If one task degrades while others improve, you have a gradient conflict problem.

5. **Use Pareto methods (MGDA, Nash-MTL) for offline diagnosis**, not production training. Run them on a subsample to map the Pareto front and find good static weight ratios.

6. **Do not use gradient surgery (PCGrad, CAGrad) in production training** unless per-step overhead is acceptable. Reserve for offline experiments.

7. **Weight relevance loss explicitly higher than its data volume suggests.** Relevance has 100–1000x fewer labels than CTR. Without upweighting, the relevance head will be dominated. A weight of 5–10x relative to its natural scale is a reasonable starting point.

8. **Re-evaluate task weights when data distributions shift.** New ad formats, traffic mix changes, or relevance label refreshes can change the optimal balance. Treat task weights as hyperparameters that require periodic tuning, not permanent settings.

---

## 9. What Can Be Shared with Organic Search?

Sponsored search and organic search often operate on the same item catalog, the same user request, and the same serving infrastructure. Organic search usually has a similar multi-stage ranking flow. Sharing the right components reduces cost and improves consistency. Sharing the wrong components blurs objectives and hurts both systems.

### 9.1. Sharing principle and decision criteria

**Principle**: share representations and infrastructure where the semantics are common; separate objectives where the product and economic goals diverge.

**Decision criteria** — before sharing a component, ask:

1. **Semantic alignment**: do organic and ads need the same notion of "good" for this component? Query understanding (spelling, intent) is the same; final scoring (user utility vs. auction revenue) is not.
2. **Data sufficiency**: does the ads system have enough data to train this component standalone? If not, sharing with organic gives a data advantage. If yes, independent training avoids objective contamination.
3. **Deployment coupling**: can the ads team iterate on this component without risking regressions in organic search? Shared models create release dependencies.
4. **Regulatory / policy constraints**: does mixing organic engagement data into ads training (or vice versa) create compliance issues? Some jurisdictions or platform policies restrict cross-system data use.

### 9.2. Risks of sharing

Sharing is not free. Real costs include:

- **Objective contamination**: a shared backbone trained on organic click data may learn patterns (e.g., favoring long-form content) that do not transfer to ad ranking, degrading the ads head.
- **Deployment coupling**: a shared model means coordinated releases. The ads team cannot ship a model update without regression-testing organic, and vice versa.
- **Metric attribution**: when a shared model improves one system's metrics, it is hard to tell whether the gain came from organic signal or ads signal. This makes experimentation slower.
- **Optimization horizon conflict**: organic search may optimize for long-term user retention; ads optimize for short-term auction revenue. A shared backbone will compromise between these.

### 9.3. Sharing matrix by pipeline stage

Shareability levels:
- **High** = can share the same binary/service with no ads-specific fork
- **Partial** = shared architecture or codebase, but separate trained instances or configuration
- **Low** = separate implementation; little direct reuse

#### 9.3.1. Platform and infrastructure

| Component | Shareability | What to share | Notes |
| --- | --- | --- | --- |
| Query understanding | High | spelling correction, rewrites, intent classification, taxonomy | semantic request-understanding; no objective divergence |
| Item understanding | High | text encoders, category/attribute embeddings, page quality, catalog normalization | both systems need consistent item representations |
| Lexical retrieval infra | High | inverted index, BM25, shard plumbing, dedup | infrastructure, not business logic |
| Feature platform | High | offline feature generation, online feature store, logging, monitoring | shared platform reduces duplicated engineering |
| Relevance labeling framework | High | annotation tooling, evaluation pipeline | both care about semantic match; rubric details may differ (see 9.3.3) |

#### 9.3.2. Retrieval and pre-ranking

| Component | Shareability | What to share | What to keep separate |
| --- | --- | --- | --- |
| Retrieval embeddings | Partial | item-side embeddings (item understanding is shared); query/user towers can be shared if retrieval objectives are compatible | if organic retrieval optimizes for engagement while ads retrieval optimizes for click+relevance, joint training may degrade both; evaluate empirically before committing |
| Pre-ranking model | Partial | embedding backbone or shared experts via transfer learning | training objectives (organic may use listwise engagement; ads use pointwise CTR/CVR), task heads, and loss weights |
| Model features | Partial | raw item/query/user features, page quality, semantic match signals | auction features (bid, budget, pacing), ads-only policy signals, organic-only engagement aggregates |

#### 9.3.3. Labels and relevance

| Component | Shareability | What to share | What to keep separate |
| --- | --- | --- | --- |
| Relevance labels | Partial | annotation workflow, tooling, some shared (query, item) judgment sets | rubric definitions — ads relevance asks "is this ad a good answer to this query?" while organic asks "is this page a good result?" The item types and user expectations differ, so rubrics should be independently defined even if the tooling is shared |
| Behavioral labels (clicks, conversions) | High at representation layer, Partial at prediction layer | Both ads and organic models benefit from training on each other's behavioral data. Organic clicks provide item quality and query-item relevance signal at massive volume; ads clicks provide commercial intent signal. Standard pattern: pre-train shared representations on combined data, then fine-tune task heads per domain. | Treat domain as a feature, not a data boundary. At the prediction layer, organic and ads have different base rates, position bias curves, presentation formats, and selection policies — so CTR/CVR heads need domain indicators or separate calibration. Raw label mixing without this distinction will miscalibrate predictions. |

#### 9.3.4. Final ranking and page optimization

| Component | Shareability | What to share | What to keep separate |
| --- | --- | --- | --- |
| Final ranking model | Low | some reusable feature transforms or shared sparse embeddings | ranking objectives, debiasing logic, scoring formulas, eCPM auction constraints |
| Calibration tooling | Partial | calibration pipelines, reliability dashboards | target definitions, segment choices, recalibration cadence |
| Page-level optimization | Partial | candidate generation APIs, constraint-checking framework | reward functions, monetization terms, organic utility terms, hard business constraints |
| Scoring objectives | Low | — | organic optimizes user utility; ads must include bid, advertiser value, and auction incentive compatibility |
| Auction logic / budget pacing | Low | — | purely sponsored concerns with no organic analogue |
| Charging / counterfactual eval | Low | — | sponsored systems require economic correctness that organic does not |

### 9.4. Summary

The pattern across the matrix: shareability decreases as you move down the pipeline. Infrastructure and representations at the top of the funnel are almost always shareable. Model instances become partially shareable at retrieval/pre-ranking (shared architecture, separate training). Final scoring, auction logic, and economic mechanisms should stay fully separate.

---

## 10. Closing: Interfaces, Recommendations, and Open Questions

### 10.1. Stage interfaces

Each stage should hand off a clean contract to the next.

**Retrieval → pre-ranking**: candidate item IDs, tower embeddings, lightweight retrieval scores, basic filtering metadata.

**Pre-ranking → final ads ranking**: top-N candidate ads, calibrated CTR/CVR/relevance estimates, a small set of distilled dense and sparse features.

**Final ads ranking → mixed-list reranking**: top-M ad candidates, bids and billing type, calibrated outcome predictions or features needed by the reranker, organic list from the organic ranking pipeline, request-level business constraints. The mixed-list reranker should not need to revisit the full ad corpus.

### 10.2. Practical recommendations

1. Use two-tower models with ANN for primary ad retrieval; fuse with lexical channels via a lightweight scoring function.
2. Train retrieval and pre-ranking models separately unless infrastructure cost forces sharing. If sharing, freeze retrieval towers and train only the pre-ranking head.
3. Train relevance as a separate task with human-graded labels (Section 2.3) rather than folding it into CTR. Use relevance labels as an auxiliary signal at retrieval (Section 4.1.3). Scale via LLM teacher distillation (Section 3.4).
4. Use a feature-interaction network (DCN-v2, DLRM, DIN) for final ad ranking, after heavy candidate reduction.
5. Use multiplicative eCPM-based scoring in both pre-ranking and final ranking. Relevance should be a multiplicative quality factor, not an additive term.
6. Apply position bias debiasing to CTR prediction in both pre-ranking and final ranking.
7. Calibrate predicted probabilities as a post-processing step. Monitor calibration drift daily.
8. Treat mixed-list page construction as a separate listwise optimization problem, not as a byproduct of per-ad ranking.
9. Enforce hard structural constraints (ad load, spacing, position bounds, density) in the final reranking stage rather than trying to learn them implicitly.
10. Use architecture-level multi-task separation (PLE or MMoE) as the primary defense against gradient conflict. Add task-specific learning rates and uncertainty weighting as lightweight supplements (Section 8).

### 10.3. Cross-cutting concerns not yet covered

The following topics cut across multiple pipeline stages and are not detailed in this document but are significant engineering concerns:

**Serving architecture**: how models are deployed and served — model serving infrastructure (TF Serving, Triton, TorchServe), request batching, model versioning and rollback, A/B testing framework, and canary deployment. These choices affect every stage.

**Training data pipelines**: how impression logs become training examples — join logic for delayed conversions (Section 2.2), negative sampling at data generation time, feature logging and backfill, data freshness and staleness monitoring. A broken training pipeline silently degrades all models.

**Cold-start**: new ads, new advertisers, and new query patterns have no historical signals. This affects every model in the pipeline. Common mitigations include content-based fallback features, explore/exploit strategies (e.g., Thompson sampling on ad impressions), and warm-starting from similar ads or advertisers.

**Feature engineering and management**: feature stores, feature freshness guarantees, feature drift monitoring, and the lifecycle of feature additions and deprecations across stages. In practice, feature management consumes more engineering effort than model architecture.

### 10.4. Open design decisions

The main choices that still need product and platform alignment:

- the exact relevance rubric and annotation process (Section 3.2)
- validating ordinal regression vs. simpler alternatives on actual annotation data (Section 3.3)
- whether to deploy LLM teacher distillation (Section 3.4) and how to validate teacher quality
- how much value modeling should happen in final ad ranking versus mixed-list reranking
- whether the mixed-list reranker consumes calibrated predictions from the ads ranker or learns its own outcome heads end to end
- the auction mechanism (GSP vs. VCG vs. first-price) and how it constrains the scoring function
- budget pacing strategy and at which stages it influences candidate selection
- serving infrastructure choices and latency budgets per stage
- cold-start strategy for new ads and advertisers
- whether to split relevance into a separate model at final ranking — start with a relevance head inside the shared multi-task model at both pre-ranking and final ranking; consider splitting into a separate model at final ranking if relevance quality becomes a bottleneck, the relevance label refresh cadence outpaces full model retrains, or the team wants to use a cross-encoder architecture optimized for semantic matching (the final ranking latency budget can accommodate a second model; pre-ranking's cannot)
