# CGR With CVR

This directory contains a CVR-extended variant of the paper-inspired CGR implementation from arXiv:2603.04227, "Constraint-Aware Generative Re-ranking for Multi-Objective Optimization in Advertising Feeds".

Original paper: https://arxiv.org/abs/2603.04227

At a high level, the repository answers one question:

> Given an already-ranked organic feed, a small set of ad candidates, and a set of structural business rules as **hard constraints**, insert ads to the ranked list of organic feed to optimize predefined reward, often based on ads reveue, ads penalty, profit margin of selling an item and engagement.

The code follows the paper's overall decomposition, and tried to faithfully implement the following key algorithmic features, though some components are intentionally simplified and may differ materially from the paper.

- structured item representation over pre-computed upstream embeddings
- dual hierarchical attention with local structural bias
- Multi-Channel Mixture-of-Experts (MoE) via PLE
- Cross-Attention for Position Preference
- Unified Sequence Reward Modeling
- two-stage constrain-aware reward pruning over space of ad insertions

In addition, it extends with conversion prediction and profit-margin optimization, as well as beam-search based inference. 

- multi-task exposure, click, and conversion prediction
- list-level reward computation with ad revenue and per-item profit margin
- beam-search inference as an alternative heuristic for larger ad budgets

More work in progress: Pareto Efficiency on individual metrics in the total reward

- Feedback-Loop Tuning (PID Control). For ex: A target of Ad Load at exactly 20%
   - If Actual Ad Load < 20%: Increase the Revenue weight ($\alpha$) or decrease the Ad Penalty ($\delta$).
   - If Actual Ad Load > 20%: Do the opposite.
   - Outcome: The system automatically "pushes" into the Pareto front to find the most profitable sequences that still satisfy the 20% constraint.


## Table Of Contents

- [How To Read This Repo](#how-to-read-this-repo)
- [What The Model Actually Predicts](#what-the-model-actually-predicts)
- [Current Repository Surface](#current-repository-surface)
- [Main Components](#main-components)
- [Inference Paths](#inference-paths)
- [Paper-to-Code Mapping](#paper-to-code-mapping)
- [Important Differences From The Paper](#important-differences-from-the-paper)
- [Training Note](#training-note)
- [Inference Note](#inference-note)
- [CGR vs. SORT-Gen](#cgr-vs-sort-gen)
- [Practical Reading Of This Directory](#practical-reading-of-this-directory)

## How To Read This Repo

The easiest way to understand the code is to follow the serving-time flow:

1. Start with a request-level candidate set: an organic list plus ad candidates.
2. Turn each candidate item into a sequence representation using pre-computed user, context, and item embeddings.
3. Predict per-position exposure and click values for a specific ordered list.
4. Convert those predictions into a list-level reward using bids, engagement scores, and ad penalties.
5. Search over feasible mixed lists and return the best sequence found by the implemented search method.

That is the central design idea in this directory: the model scores complete candidate lists directly, rather than training a separate list generator and an independent evaluator.

## What The Model Actually Predicts

The model does not directly output a final mixed list. Instead, for a candidate ordered sequence it predicts:

- `p_exp`: per-position exposure probability
- `p_clk`: per-position click probability
- `p_cvr`: per-position conversion probability

Those probabilities are then combined with business-side scalars:

- bid
- engagement score
- ad penalty weight
- profit margin
- billing mode flag

to compute a list-level reward. That reward is what the search procedures compare across different feasible lists.

So a useful mental model is:

- model = "how good does this particular list look?"
- inference = "which feasible list should we try, compare, and keep?"

## Current Repository Surface

The main user-facing pieces in this directory are:

- [data_types.py](data_types.py): typed feed objects, ad metadata, and structural constraint checks
- [model.py](model.py): `CGRModel`, `HierarchicalAttentionBlock`, `PLEFusion`, and `ExposureClickCVRHead`
- [train.py](train.py): `TrainConfig` and `CGRTrainer` for weighted multi-task EXP/CLK/CVR training
- [inference.py](inference.py): Stage I insertion, Stage II bounded decoding, `cgr_inference`, and `beam_search_inference`
- [demo.ipynb](demo.ipynb): repaired executable walkthrough of the current implementation
- [demo_executed.ipynb](demo_executed.ipynb): stored executed artifact of the demo notebook
- [example.py](example.py): lightweight script entry point for package usage examples

## Main Components

### 1. Data And Constraints

[data_types.py](data_types.py) defines the objects that move through the system:

- `Item`: one organic or ad candidate together with embeddings and scalar business signals
- `CandidateSet`: the request-level organic list plus ad candidates
- `AdConstraints`: structural rules such as max ads, minimum spacing, position bounds, large-ad cap, and density limit

These constraints are about placement structure, not upstream delivery concerns like budget pacing or user-level frequency capping.

### 2. Sequence Model

[model.py](model.py) contains `CGRModel`.

The model takes an ordered list and builds a hidden representation for each slot using:

- projected item embeddings
- projected user embeddings
- projected context embeddings
- learned list-position embeddings
- learned item-type embeddings

It then runs several attention experts:

- shared self-attention
- exposure-oriented self-attention
- click-oriented self-attention
- cross-attention for user/item position preference
- local self-attention with small windows

Those expert outputs are fused by `PLEFusion`, and `ExposureClickCVRHead` produces `p_exp`, `p_clk`, and `p_cvr`.

### 3. Training

[train.py](train.py) trains the model from logged impression sequences.

The trainer uses:

- `CGRModel.forward_logits` during optimization
- `BCEWithLogitsLoss` for numerical stability
- `click_positive_class_weight` to compensate for sparse click positives
- `cvr_positive_class_weight` to compensate for even sparser conversion positives

This means the training path uses logits internally, while inference still works in probability space through `CGRModel.forward`.

### 4. Inference

[inference.py](inference.py) contains the search procedures that build mixed lists.

There are two separate inference paths in this repo:

- `cgr_inference`: the main small-`K` bounded-search path
- `beam_search_inference`: a heuristic alternative for larger search spaces

They use the same reward model, but they explore candidate lists differently.

## Inference Paths

### Two-Stage Bounded Search

`cgr_inference` is the repository's default search path.

It has two steps:

1. Stage I tries feasible single-ad insertions and keeps the best intermediate result.
2. Stage II enumerates the implemented list families around the organic base list, applies hard-constraint filtering, optionally uses pruning, and returns the best sequence found.

This is the path most closely inspired by the paper's small-`K` story.

### Beam Search

`beam_search_inference` is also implemented in this repository and should be treated as a first-class alternative, not just a side note.

Beam search works by:

1. starting from the pure organic list
2. inserting ads step by step
3. keeping only the top `beam_width` partial candidates at each step
4. returning the best full candidate found during expansion

Why it matters:

- it can handle larger ad budgets than the small-`K` bounded-search path
- it is heuristic, so it can miss strong solutions
- in this repository, it can also sometimes beat `cgr_inference`, which is one reason the docs avoid overclaiming optimality

## Paper-to-Code Mapping

| Paper section / equation | Code | Notes |
| --- | --- | --- |
| Section 3.1 candidate set | [data_types.py](data_types.py), `CandidateSet` | Represents organic items plus ad candidates for one request. |
| Section 3.4 constraints, Eq. 6-7 | [data_types.py](data_types.py), `AdConstraints` | Load, spacing, large-ad count, and density are implemented. |
| Section 4.1 item representation, Eq. 10 | [model.py](model.py), `CGRModel.encode_items` | Uses projected item/user/context embeddings plus learned position and item-type embeddings. |
| Section 4.2 hierarchical attention, Eq. 11-13 | [model.py](model.py), `HierarchicalAttentionBlock` | Includes shared/task-specific causal attention, cross-attention, and local attention with windows 4 and 6, used in dual EXP/CLK streams. |
| Section 4.3 PLE fusion, Eq. 14-15 | [model.py](model.py), `PLEFusion` | Implemented as a simplified PLE-style fusion rather than the paper's full expert layout. |
| Section 4.4 reward heads, Eq. 16-20 (extended) | [model.py](model.py), `ExposureClickCVRHead` | Predicts `p_exp`, `p_clk`, and `p_cvr`, then computes list reward from bids, engagement, ad penalty, and per-item profit margin. |
| Training-time logits path | [model.py](model.py), `CGRModel.forward_logits` | Exposes logits directly so training can use `BCEWithLogitsLoss` instead of probability-space BCE. |
| Training objective | [train.py](train.py), `CGRTrainer` | Weighted multi-task BCE in logits space over exposure and click labels. |
| Stage I constrained insertion | [inference.py](inference.py), `stage1_constrained_insertion` | Enumerates single-ad insertions, but with implementation differences listed below. |
| Stage II bounded decoding | [inference.py](inference.py), `stage2_bounded_decoding` | Enumerates several feasible sequence types, but not the full paper-described feasible set. |
| Full inference pipeline | [inference.py](inference.py), `cgr_inference` | Orchestrates Stage I and Stage II. |
| Alternative search path | [inference.py](inference.py), `beam_search_inference` | Heuristic search that supports larger `K` than the small-`K` bounded-search path. |
| End-to-end walkthrough | [demo.ipynb](demo.ipynb) | Demonstrates setup, reward computation, training, two-stage inference, pruning comparison, and beam search. |

## Important Differences From The Paper

The current code is best understood as a faithful high-level interpretation of the model family, not a paper-faithful reproduction.

1. PLE fusion is simplified.
   The paper describes a larger expert layout for the EXP and CLK branches. This code uses a reduced set of unique expert outputs to keep the implementation compact.

2. Cross-attention is simplified.
   The paper describes user-side and item-side position-aware encoders. This code keeps the same high-level idea, but implements it more compactly through `user_repr = u + p` and `item_repr = v + p + t`, followed by standard multi-head cross-attention.

3. Stage I differs from the paper.
   The paper says single-ad insertion includes large ads and e-commerce ads. This implementation skips large ads in Stage I.

4. Stage II does not enumerate every paper-described feasible combination.
   The code rebuilds from the organic base sequence and considers large-ad-only and regular double-ad variants, but it does not enumerate mixed large-ad plus regular-ad combinations.

5. Pruning is heuristic.
   The paper describes upper-bound pruning on partial sequences. This code uses an optimistic completion heuristic for already constructed candidates. It is useful as a speedup, but it should not be read as a proof-preserving implementation of the paper's theorem.

6. Constraint verification is partial.
   `AdConstraints.is_feasible` checks load, spacing, large-ad count, and density, but it does not currently re-check all position-bound rules from the paper.

7. Beam search is repository-specific.
   `beam_search_inference` is useful for experimentation when `K > 2`, but it is an added heuristic interface in this repo rather than part of the paper's core two-stage guarantee story.

8. The demo notebook is intentionally implementation-accurate, not theorem-accurate.
   [demo.ipynb](demo.ipynb) has been updated to describe the current code path as a bounded-search implementation and to avoid claiming paper-backed optimality that the repository does not currently establish.

## Training Note

The paper describes the training objective at the probability level: predict `p_exp` and `p_clk`, then apply BCE to those predicted probabilities.

This repository intentionally trains one step earlier, on the pre-sigmoid logits, using `BCEWithLogitsLoss` rather than plain probability-space BCE.

That choice matters for two reasons:

1. Logit-space BCE is numerically more stable than applying sigmoid first and then using `BCELoss`.
2. Click and conversion labels are much sparser than exposure labels, so the trainer exposes `click_positive_class_weight` and `cvr_positive_class_weight` to upweight positive examples without changing the inference-time reward formula.

Task balance still happens at two levels:

- `lambda_exp`, `lambda_clk`, and `lambda_cvr` balance the three prediction tasks.
- `click_positive_class_weight` and `cvr_positive_class_weight` balance positive vs negative labels within the sparse downstream tasks.

At inference time, the model still returns probabilities through `CGRModel.forward`, so this logits-space training choice is an implementation detail for optimization stability, not a change to the serving-time reward formula or search interfaces.

## Inference Note

This repo currently exposes two different search styles:

- `cgr_inference`: the repository's main small-`K` path, built from Stage I constrained insertion plus Stage II bounded decoding
- `beam_search_inference`: a heuristic search alternative that can explore larger ad budgets

The practical implication is important: in this repository, beam search can sometimes return a higher-scoring feasible list than `cgr_inference`. That is one reason the docs and notebook now describe the implementation as paper-inspired and bounded-search-based rather than theorem-backed globally optimal.

If you want a runnable explanation of both paths, [demo.ipynb](demo.ipynb) now covers:

- setup and candidate construction
- reward computation
- training on synthetic logged data
- Stage I and Stage II inference
- pruning vs. no pruning
- beam search with different settings

## CGR vs. SORT-Gen

The relevant SORT-Gen paper is "A Generative Re-ranking Model for List-level Multi-objective Optimization at Taobao": https://arxiv.org/abs/2505.07197. SORT-Gen and CGR are adjacent ideas, but they are not solving exactly the same problem.

The shortest comparison is:

- SORT-Gen is stronger as a one-call list-level reranker for general multi-objective optimization.
- CGR is stronger when explicit organic/ad mixing and hard ad-placement feasibility are the central problem.

### What SORT-Gen Optimizes Well

SORT-Gen is built for efficient list-level multi-objective reranking in large e-commerce recommendation systems.

Its main strengths are:

- one serving-time model call rather than repeated online model invocation loops
- list-level optimization instead of pure item-level scalar sorting
- tensorized sequential generation inside the inference graph
- integrated diversity handling through its MMR-style mechanism

In other words, SORT-Gen is trying to make generative reranking practical for large-scale serving.

### What CGR Focuses On

The CGR paper is centered on a narrower but harder industrial problem:

- mixing organic and ad items in one list
- satisfying hard structural ad rules
- optimizing monetization and engagement jointly under those rules

That is why CGR is organized around:

- explicit feasibility constraints
- bounded constrained decoding
- reward-based comparison of feasible mixed lists

### Why The Difference Matters

SORT-Gen is close to a one-call list generator at the serving level, even though it still performs internal sequential selection logic.

This repository's CGR implementation behaves more like:

- learned reward model + bounded constrained search

than:

- direct one-shot final-list generator

That distinction matters if the goal is specifically "generate the final mixed organic/ad list directly in one fast serving call". In that framing, SORT-Gen is the more natural reference point. If the goal is "guarantee feasible ad placement under hard structural rules", CGR is the more natural reference point.

### Latency Comparison

The paper-reported latency numbers are directionally useful, but they are not an apples-to-apples benchmark because the systems, workloads, and deployment settings differ.

- SORT-Gen reports end-to-end latency of about 19 ms after its Mask-Driven Fast Generation optimization.
- The CGR paper reports deployment under a strict latency SLA of <40 ms P99.
- The CGR paper also claims inference-latency reduction of over 85% relative to its prior production baseline.

The practical reading is:

- SORT-Gen emphasizes one-call serving efficiency and reports a very concrete absolute latency number.
- CGR emphasizes constraint-aware feasible decoding under production ad-feed rules and reports that it stays within production latency budgets while materially reducing latency versus a Generator-Evaluator baseline.

So if you compare them purely on headline latency language:

- SORT-Gen presents the cleaner "fast one-call reranking" story.
- CGR presents the stronger "constraint-aware mixed-feed optimization within industrial SLA" story.

### Recommendation-Level Takeaway

If you want:

- efficient one-call multi-objective reranking for a general list problem, SORT-Gen is a strong direction.
- strict mixed organic/ad placement with hard structural feasibility, CGR is the more directly relevant design.

If you wanted the best of both, the natural future direction would be a hybrid:

- SORT-Gen-style fast serving mechanics
- plus CGR-style constraint-aware masking or constrained decoding for ad-feasibility guarantees

## Practical Reading Of This Directory

Use this code if you want:

- a compact CGR-style architecture for experimentation with profit-margin-aware reranking
- a readable scaffold for adapting the paper to your own reranking stack
- a demonstration of how unified reward modeling can replace a separate evaluator while adding a third CVR task

Do not use this directory as evidence that every equation, decoding guarantee, or theorem in arXiv:2603.04227 has been implemented exactly as written.
