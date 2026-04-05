# CGR

This directory contains a paper-inspired reference implementation of CGR from arXiv:2603.04227, "Constraint-Aware Generative Re-ranking for Multi-Objective Optimization in Advertising Feeds".

It follows the paper's overall decomposition:

- structured item representation over pre-computed upstream embeddings
- hierarchical attention with local structural bias
- multi-task exposure and click prediction
- list-level reward computation
- two-stage constrained inference over ad insertions

It does not claim to be an exact reproduction of the paper. Several components are intentionally simplified, and some inference details differ materially from the algorithm described in the paper.

## Paper-to-Code Mapping

| Paper section / equation | Code | Notes |
| --- | --- | --- |
| Section 3.1 candidate set | [data_types.py](data_types.py), `CandidateSet` | Represents organic items plus ad candidates for one request. |
| Section 3.4 constraints, Eq. 6-7 | [data_types.py](data_types.py), `AdConstraints` | Load, spacing, large-ad count, and density are implemented. |
| Section 4.1 item representation, Eq. 10 | [model.py](model.py), `CGRModel.encode_items` | Uses projected item/user/context embeddings plus learned position and item-type embeddings. |
| Section 4.2 hierarchical attention, Eq. 11-13 | [model.py](model.py), `HierarchicalAttentionBlock` | Includes shared/task-specific causal attention, cross-attention, and local attention with windows 4 and 6. |
| Section 4.3 PLE fusion, Eq. 14-15 | [model.py](model.py), `PLEFusion` | Implemented as a simplified PLE-style fusion rather than the paper's full expert layout. |
| Section 4.4 reward heads, Eq. 16-20 | [model.py](model.py), `ExposureClickHead` | Predicts `p_exp` and `p_clk`, then computes list reward from bids, engagement, and penalty scalars. |
| Training objective | [train.py](train.py), `CGRTrainer` | Weighted multi-task BCE over exposure and click labels. |
| Stage I constrained insertion | [inference.py](inference.py), `stage1_constrained_insertion` | Enumerates single-ad insertions, but with implementation differences listed below. |
| Stage II bounded decoding | [inference.py](inference.py), `stage2_bounded_decoding` | Enumerates several feasible sequence types, but not the full paper-described feasible set. |
| Full inference pipeline | [inference.py](inference.py), `cgr_inference` | Orchestrates Stage I and Stage II. |

## Important Differences From The Paper

The current code is best understood as a faithful high-level interpretation of the model family, not a paper-faithful reproduction.

1. PLE fusion is simplified.
   The paper describes a larger expert layout for the EXP and CLK branches. This code uses a reduced set of unique expert outputs to keep the implementation compact.

2. Cross-attention is simplified.
   The paper describes user-side and item-side position-aware encoders. This code uses cross-attention over the already fused sequence representation.

3. Stage I differs from the paper.
   The paper says single-ad insertion includes large ads and e-commerce ads. This implementation skips large ads in Stage I.

4. Stage II does not enumerate every paper-described feasible combination.
   The code rebuilds from the organic base sequence and considers large-ad-only and regular double-ad variants, but it does not enumerate mixed large-ad plus regular-ad combinations.

5. Pruning is heuristic.
   The paper describes upper-bound pruning on partial sequences. This code uses an optimistic completion heuristic for already constructed candidates. It is useful as a speedup, but it should not be read as a proof-preserving implementation of the paper's theorem.

6. Constraint verification is partial.
   `AdConstraints.is_feasible` checks load, spacing, large-ad count, and density, but it does not currently re-check all position-bound rules from the paper.

## Training Note

The trainer uses `BCEWithLogitsLoss` rather than plain probability-space BCE.
That choice matters for two reasons:

1. Logit-space BCE is numerically more stable than applying sigmoid first and then using `BCELoss`.
2. Click labels are typically much sparser than exposure labels, so the trainer exposes `click_positive_class_weight` to upweight positive click examples without changing the inference-time reward formula.

Task balance still happens at two levels:

- `lambda_exp` vs `lambda_clk` balances the exposure task against the click task.
- `click_positive_class_weight` balances positive vs negative labels within the click task itself.

## Practical Reading Of This Directory

Use this code if you want:

- a compact CGR-style architecture for experimentation
- a readable scaffold for adapting the paper to your own reranking stack
- a demonstration of how unified reward modeling can replace a separate evaluator

Do not use this directory as evidence that every equation, decoding guarantee, or theorem in arXiv:2603.04227 has been implemented exactly as written.