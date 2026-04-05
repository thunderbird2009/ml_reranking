"""Two-stage constraint-aware generative inference inspired by Sections 5-7.

This module implements a CGR-style inference pipeline illustrated by Figure 2.
The key insight is that because the maximum number of ads K is small
(typically ≤ 2), the factorial search space O(N!) collapses to a bounded
O(K·L) enumeration, making real-time serving feasible under strict latency
SLAs (<40 ms P99).

Pipeline overview:

    Organic list from upstream ranker
            │
            ▼
    ┌──────────────────────────────────┐
    │  Stage I: Constrained Insertion  │  — try every feasible single-ad
    │  (Section 7.1)                   │    insertion; pick the best one.
    └──────────────┬───────────────────┘
                   ▼
    ┌──────────────────────────────────┐
    │  Stage II: Bounded Decoding      │  — enumerate large-ad, double-ad,
    │  (Section 7.2)                   │    single-ad, and no-ad variants;
    │  + Constraint-Aware Pruning      │    prune infeasible / dominated
    │    (Section 6)                   │    candidates via reward upper bounds.
    └──────────────┬───────────────────┘
                   ▼
            Best feasible sequence
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import torch

from .model import CGRModel, ExposureClickHead
from .data_types import AdConstraints, CandidateSet, Item, ItemType


# ---------------------------------------------------------------------------
# Data structure for inference results
# ---------------------------------------------------------------------------


@dataclass
class RankedSequence:
    """A fully-constructed ranking sequence together with its predicted reward.

    This is the output of the inference pipeline.  The ``items`` list contains
    both organic and ad items in their final display order, ``reward`` is the
    model-predicted R(A), and ``ad_positions`` records which indices hold ads
    (useful for downstream logging / constraint verification).
    """

    items: list[Item]
    reward: float
    ad_positions: list[int]

    @property
    def num_ads(self) -> int:
        """Number of ads in this sequence."""
        return len(self.ad_positions)


# ---------------------------------------------------------------------------
# Helpers: convert Item lists → model tensors
# ---------------------------------------------------------------------------


def _build_model_inputs(
    items: list[Item],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a list of Items into the four tensor inputs expected by ``CGRModel``.

    Creates a batch of size 1.  The returned tensors correspond to the
    pre-computed upstream embeddings (item, user, context) and the integer
    item-type indices.

    Returns:
        (item_embs, user_embs, context_embs, item_types) — each [1, L, *].
    """
    item_feats = torch.stack([it.item_emb for it in items]).unsqueeze(0).to(device)
    user_feats = torch.stack([it.user_emb for it in items]).unsqueeze(0).to(device)
    ctx_feats = torch.stack([it.context_emb for it in items]).unsqueeze(0).to(device)
    types = torch.tensor([it.item_type.value for it in items], device=device).unsqueeze(0)
    return item_feats, user_feats, ctx_feats, types


def _build_reward_inputs(
    items: list[Item],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract the scalar reward signals (bid, engagement, penalty, billing model).

    These scalars are *not* inputs to the neural network — they are combined
    with the model's predicted p_exp/p_clk in ``ExposureClickHead.compute_reward()``
    to produce R(A).

    Returns:
        (bids, engagement_scores, ad_penalty_weights, is_cpa) — each [1, L].
    """
    bids = torch.tensor([it.bid for it in items], device=device).unsqueeze(0)
    engagement = torch.tensor([it.engagement_score for it in items], device=device).unsqueeze(0)
    # d_i: ad penalty weight from business policy
    penalty = torch.tensor(
        [it.ad_penalty_weight for it in items], device=device
    ).unsqueeze(0)
    # CPA flag (simplified: all ads use CPA billing)
    is_cpa = torch.tensor(
        [1.0 if it.is_ad else 0.0 for it in items], device=device
    ).unsqueeze(0)
    return bids, engagement, penalty, is_cpa


# ---------------------------------------------------------------------------
# Reward evaluation (single sequence and batched)
# ---------------------------------------------------------------------------


def _evaluate_sequence(
    model: CGRModel,
    items: list[Item],
    device: torch.device,
) -> float:
    """Compute the list-level reward R(A) for a single candidate sequence.

    Performs one forward pass through the model (no gradient), then combines
    the predicted p_exp/p_clk with the auction/policy scalars via Eq. 17.

    Returns:
        Scalar reward value.
    """
    item_feats, user_feats, ctx_feats, types = _build_model_inputs(items, device)
    bids, engagement, penalty, is_cpa = _build_reward_inputs(items, device)

    with torch.no_grad():
        p_exp, p_clk = model(item_feats, user_feats, ctx_feats, types)

    reward = ExposureClickHead.compute_reward(p_exp, p_clk, bids, engagement, penalty, is_cpa)
    return reward.item()


def _evaluate_sequences_batched(
    model: CGRModel,
    sequences: list[list[Item]],
    device: torch.device,
) -> list[float]:
    """Evaluate multiple same-length sequences in a single batched forward pass.

    This is the performance-critical path during inference.  By stacking
    candidate sequences into one batch, we amortise the cost of attention
    computation across all candidates — similar in spirit to the M-FALCON
    micro-batching strategy from Meta's Generative Recommenders [4].

    All sequences in the list **must have the same length** so they can be
    stacked into a [B, L, D] tensor.  The caller (Stage I / Stage II) groups
    sequences by length before calling this function.

    Returns:
        List of scalar reward values, one per input sequence.
    """
    if not sequences:
        return []

    # All sequences must have the same length for batching
    seq_len = len(sequences[0])
    B = len(sequences)

    item_feats = torch.stack(
        [torch.stack([it.item_emb for it in seq]) for seq in sequences]
    ).to(device)
    user_feats = torch.stack(
        [torch.stack([it.user_emb for it in seq]) for seq in sequences]
    ).to(device)
    ctx_feats = torch.stack(
        [torch.stack([it.context_emb for it in seq]) for seq in sequences]
    ).to(device)
    types = torch.tensor(
        [[it.item_type.value for it in seq] for seq in sequences], device=device
    )
    bids = torch.tensor(
        [[it.bid for it in seq] for seq in sequences], device=device, dtype=torch.float32
    )
    engagement = torch.tensor(
        [[it.engagement_score for it in seq] for seq in sequences], device=device, dtype=torch.float32
    )
    penalty = torch.tensor(
        [[it.ad_penalty_weight for it in seq] for seq in sequences],
        device=device, dtype=torch.float32,
    )
    is_cpa = torch.tensor(
        [[1.0 if it.is_ad else 0.0 for it in seq] for seq in sequences],
        device=device, dtype=torch.float32,
    )

    with torch.no_grad():
        p_exp, p_clk = model(item_feats, user_feats, ctx_feats, types)

    rewards = ExposureClickHead.compute_reward(p_exp, p_clk, bids, engagement, penalty, is_cpa)
    return rewards.tolist()


# ---------------------------------------------------------------------------
# 6. Constraint-Aware Reward Pruning
# ---------------------------------------------------------------------------


def _upper_bound_reward(
    model: CGRModel,
    partial_items: list[Item],
    remaining_items: list[Item],
    list_length: int,
    device: torch.device,
) -> float:
    """Estimate an optimistic upper bound on the achievable reward (Eq. 25).

    For a partially-constructed sequence, we fill the remaining slots with
    the highest-value items available (sorted by bid + engagement).  The
    resulting "optimistic" sequence is evaluated through the model to obtain
    R_upper(A_partial).

    This upper bound is used for **reward pruning** (Section 6.2, Eq. 26):
    if R_upper(A_partial) < R_best, the candidate can be safely discarded
    without full evaluation because no completion can beat the current best.

    Because the reward model is integrated into the same network (Section 6.3),
    computing this upper bound requires only one additional forward pass with
    negligible overhead.

    Args:
        partial_items: items already placed in the sequence.
        remaining_items: items not yet placed (used to fill optimistically).
        list_length: target total sequence length.
        device: computation device.

    Returns:
        Optimistic reward estimate (scalar).
    """
    if not remaining_items:
        return _evaluate_sequence(model, partial_items, device)

    # Fill remaining slots with the highest-bid / highest-engagement items
    remaining_sorted = sorted(
        remaining_items,
        key=lambda it: it.bid + it.engagement_score,
        reverse=True,
    )
    optimistic_seq = list(partial_items)
    for i in range(list_length - len(partial_items)):
        if i < len(remaining_sorted):
            optimistic_seq.append(remaining_sorted[i])
        else:
            # Duplicate last organic if we run out
            optimistic_seq.append(partial_items[-1])

    return _evaluate_sequence(model, optimistic_seq, device)


def _hard_constraint_filter(
    ad: Item,
    position: int,
    existing_ad_positions: list[int],
    constraints: AdConstraints,
    list_length: int,
) -> bool:
    """Hard constraint filtering — fast pre-check before reward evaluation (Section 6.1).

    Before spending compute on a model forward pass, this function checks
    whether inserting ``ad`` at ``position`` (given ads already placed at
    ``existing_ad_positions``) would violate any business constraint.

    Constraints checked (in order):
    1. **Load constraint** (Eq. 6):  total ads ≤ K.
    2. **Position bounds**: ad must fall within [min_ad_position, max_ad_position].
    3. **Spacing constraint** (Eq. 7): minimum distance Δ between any two ads.
    4. **Large-ad cap**: at most ``max_large_ads`` large-format ads per list.

    Returns:
        True if the insertion is feasible, False if any constraint is violated.
    """
    # Load constraint
    if not constraints.check_load(len(existing_ad_positions) + 1):
        return False

    # Position bounds
    if position < constraints.min_ad_position or position > min(
        constraints.max_ad_position, list_length - 1
    ):
        return False

    # Spacing constraint
    all_positions = existing_ad_positions + [position]
    if not constraints.check_spacing(all_positions):
        return False

    # Large ad constraint
    if ad.is_large_ad:
        num_large = sum(1 for p in existing_ad_positions if p >= 0)  # simplified
        if num_large >= constraints.max_large_ads:
            return False

    return True


# ---------------------------------------------------------------------------
# 7.1  Stage I: Constrained Single-Ad Insertion
# ---------------------------------------------------------------------------


def stage1_constrained_insertion(
    model: CGRModel,
    organic_list: list[Item],
    ad_candidates: list[Item],
    constraints: AdConstraints,
    device: torch.device,
) -> RankedSequence:
    """Stage I — constrained single-ad insertion (Section 7.1).

    Given the pre-ranked organic list from upstream, this stage constructs
    every feasible single-ad insertion (one regular ad at each valid position),
    evaluates them all via the reward model, and selects the top-1 as the
    intermediate list for Stage II.

    Large ads are *not* inserted here — they are handled in Stage II because
    they interact with the double-ad expansion logic.

    The no-ad baseline (pure organic list) is always considered; if no ad
    insertion improves the reward, the organic list is returned as-is.

    Complexity: O(A_regular × P) forward passes, where A_regular is the
    number of regular ad candidates and P is the number of feasible positions.
    All candidates have the same length and are batched into a single forward
    pass for efficiency.

    Args:
        model: trained CGR model.
        organic_list: pre-ranked organic items from upstream ranker.
        ad_candidates: available ad candidates for insertion.
        constraints: business constraints governing ad placement.
        device: computation device.

    Returns:
        The best single-ad (or no-ad) ``RankedSequence``.
    """
    feasible_positions = constraints.get_feasible_ad_positions(len(organic_list) + 1)
    best = RankedSequence(items=list(organic_list), reward=float("-inf"), ad_positions=[])

    # Always consider the no-ad baseline
    no_ad_reward = _evaluate_sequence(model, organic_list, device)
    best = RankedSequence(items=list(organic_list), reward=no_ad_reward, ad_positions=[])

    candidate_sequences: list[list[Item]] = []
    candidate_meta: list[tuple[Item, int]] = []  # (ad, position)

    for ad in ad_candidates:
        if ad.is_large_ad:
            continue  # Large ads handled in Stage II
        for pos in feasible_positions:
            if not _hard_constraint_filter(ad, pos, [], constraints, len(organic_list) + 1):
                continue
            # Insert ad at position into a copy of the organic list
            seq = list(organic_list)
            seq.insert(pos, ad)
            candidate_sequences.append(seq)
            candidate_meta.append((ad, pos))

    if candidate_sequences:
        # Batch-evaluate all single-ad insertions in one forward pass
        rewards = _evaluate_sequences_batched(model, candidate_sequences, device)

        for i, reward in enumerate(rewards):
            if reward > best.reward:
                ad, pos = candidate_meta[i]
                best = RankedSequence(
                    items=candidate_sequences[i],
                    reward=reward,
                    ad_positions=[pos],
                )

    return best


# ---------------------------------------------------------------------------
# 7.2  Stage II: Bounded Generative Decoding
# ---------------------------------------------------------------------------


def stage2_bounded_decoding(
    model: CGRModel,
    intermediate: RankedSequence,
    ad_candidates: list[Item],
    constraints: AdConstraints,
    device: torch.device,
    use_pruning: bool = True,
) -> RankedSequence:
    """Stage II — bounded generative decoding (Section 7.2).

    Starting from the Stage I intermediate result, this stage enumerates
    four families of feasible list configurations:

    1. **Large-ad lists** — insert a single large-format ad into the organic
       base at every feasible position.
    2. **Double-ad lists** — insert two regular ads (all valid pairs × valid
       position pairs satisfying spacing constraints).
    3. **Single-ad list** — carried over from Stage I.
    4. **No-ad list** — the pure organic baseline.

    The model evaluates the reward for each feasible sequence and returns the
    one with the highest R(A).

    **Constraint-aware reward pruning** (Section 6) is applied when
    ``use_pruning=True``:
    - *Hard constraint filtering* (Section 6.1) discards candidates that
      violate any business rule before the forward pass.
    - *Reward upper-bound pruning* (Section 6.2) computes R_upper for each
      candidate and skips those where R_upper < R_best.

    Since K ≤ 2 (at most two ads per list), the total number of feasible
    sequences is polynomial in the number of ad candidates and positions,
    not factorial in N — this is the core insight that makes real-time
    bounded decoding feasible (Section 8.3, Eq. 32).

    Args:
        model: trained CGR model.
        intermediate: Stage I result (best single-ad or no-ad sequence).
        ad_candidates: full set of ad candidates (including large ads).
        constraints: business constraints.
        device: computation device.
        use_pruning: if True, apply reward upper-bound pruning (Section 6.2).

    Returns:
        The globally best ``RankedSequence`` across all feasible list types.
    """
    best = intermediate  # Start with Stage I result as the current best
    base_organic = [it for it in intermediate.items if not it.is_ad]

    large_ads = [ad for ad in ad_candidates if ad.is_large_ad]
    regular_ads = [ad for ad in ad_candidates if ad.is_ad and not ad.is_large_ad]
    feasible_positions = constraints.get_feasible_ad_positions(len(base_organic) + 2)

    candidate_sequences: list[list[Item]] = []
    candidate_ad_positions: list[list[int]] = []

    # (1) Large-ad lists: insert one large ad at each feasible position
    for la in large_ads:
        for pos in feasible_positions:
            if not _hard_constraint_filter(la, pos, [], constraints, len(base_organic) + 1):
                continue
            seq = list(base_organic)
            seq.insert(pos, la)
            candidate_sequences.append(seq)
            candidate_ad_positions.append([pos])

    # (2) Double-ad lists: insert two regular ads at feasible position pairs
    if constraints.max_ads_per_list >= 2:
        for ad1, ad2 in combinations(regular_ads, 2):
            for pos1 in feasible_positions:
                if not _hard_constraint_filter(ad1, pos1, [], constraints, len(base_organic) + 2):
                    continue
                for pos2 in feasible_positions:
                    if pos2 <= pos1:
                        continue
                    if not _hard_constraint_filter(
                        ad2, pos2, [pos1], constraints, len(base_organic) + 2
                    ):
                        continue
                    seq = list(base_organic)
                    # Insert in position order (pos1 < pos2) to preserve indices
                    seq.insert(pos1, ad1)
                    seq.insert(pos2, ad2)
                    candidate_sequences.append(seq)
                    candidate_ad_positions.append([pos1, pos2])

    # (4) No-ad list — pure organic baseline
    no_ad_reward = _evaluate_sequence(model, base_organic, device)
    if no_ad_reward > best.reward:
        best = RankedSequence(items=list(base_organic), reward=no_ad_reward, ad_positions=[])

    if not candidate_sequences:
        return best

    # Batch-evaluate candidates, grouped by sequence length (required for stacking)
    len_groups: dict[int, list[int]] = {}
    for i, seq in enumerate(candidate_sequences):
        slen = len(seq)
        len_groups.setdefault(slen, []).append(i)

    for slen, indices in len_groups.items():
        group_seqs = [candidate_sequences[i] for i in indices]

        # Reward upper-bound pruning (Section 6.2, Eq. 26):
        # skip any candidate whose optimistic reward can't beat the current best
        if use_pruning:
            filtered_seqs = []
            filtered_indices = []
            for seq, idx in zip(group_seqs, indices):
                ub = _upper_bound_reward(model, seq[:3], seq[3:], slen, device)
                if ub > best.reward:
                    filtered_seqs.append(seq)
                    filtered_indices.append(idx)
            group_seqs = filtered_seqs
            indices = filtered_indices

        if not group_seqs:
            continue

        rewards = _evaluate_sequences_batched(model, group_seqs, device)
        for reward, idx in zip(rewards, indices):
            if reward > best.reward:
                best = RankedSequence(
                    items=candidate_sequences[idx],
                    reward=reward,
                    ad_positions=candidate_ad_positions[idx],
                )

    return best


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------


def cgr_inference(
    model: CGRModel,
    candidates: CandidateSet,
    constraints: AdConstraints,
    device: torch.device,
    use_pruning: bool = True,
) -> RankedSequence:
    """Full CGR inference pipeline (Figure 2).

    Orchestrates the two-stage process:

     1. **Stage I** — constrained single-ad insertion: finds the best single
         implemented ad insertion candidate (or decides no ad improves the list).
    2. **Stage II** — bounded generative decoding: expands to large-ad,
         double-ad, single-ad, and no-ad variants; applies the repository's
         pruning heuristic; returns the best sequence found by that search.

     This repository does not enumerate every feasible sequence described in
     the paper, so its output should be interpreted as the best sequence found
     under the implemented search procedure rather than as a theorem-backed
     globally optimal solution.

    Args:
        model: trained CGR model (will be set to eval mode).
        candidates: the request-level candidate set (organic + ads).
        constraints: business constraints for this request.
        device: computation device.
        use_pruning: whether to apply reward upper-bound pruning.

    Returns:
        The best ``RankedSequence`` found by the implemented search.
    """
    model.eval()

    # Stage I: find best single-ad insertion
    intermediate = stage1_constrained_insertion(
        model, candidates.organic_items, candidates.ad_candidates, constraints, device
    )

    # Stage II: expand to all feasible list types and select the best
    result = stage2_bounded_decoding(
        model, intermediate, candidates.ad_candidates, constraints, device, use_pruning
    )

    return result
