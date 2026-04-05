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

from .model import CGRModel, ExposureClickCVRHead
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract the scalar reward signals used by the CVR-extended reward.

    These scalars are *not* inputs to the neural network — they are combined
    with the model's predicted p_exp/p_clk/p_cvr in
    ``ExposureClickCVRHead.compute_reward()``
    to produce R(A).

    Returns:
        (bids, engagement_scores, ad_penalty_weights, profit_margins,
        is_cpa) — each [1, L].
    """
    bids = torch.tensor([it.bid for it in items], device=device).unsqueeze(0)
    engagement = torch.tensor([it.engagement_score for it in items], device=device).unsqueeze(0)
    # d_i: ad penalty weight from business policy
    penalty = torch.tensor(
        [it.ad_penalty_weight for it in items], device=device
    ).unsqueeze(0)
    profit_margins = torch.tensor(
        [it.profit_margin for it in items], device=device
    ).unsqueeze(0)
    # CPA flag (simplified: all ads use CPA billing)
    is_cpa = torch.tensor(
        [1.0 if it.is_ad else 0.0 for it in items], device=device
    ).unsqueeze(0)
    return bids, engagement, penalty, profit_margins, is_cpa


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
    the predicted p_exp/p_clk/p_cvr with the auction/policy scalars.

    Returns:
        Scalar reward value.
    """
    item_feats, user_feats, ctx_feats, types = _build_model_inputs(items, device)
    bids, engagement, penalty, profit_margins, is_cpa = _build_reward_inputs(items, device)

    with torch.no_grad():
        p_exp, p_clk, p_cvr = model(item_feats, user_feats, ctx_feats, types)

    reward = ExposureClickCVRHead.compute_reward(
        p_exp,
        p_clk,
        p_cvr,
        bids,
        engagement,
        penalty,
        profit_margins,
        is_cpa,
    )
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
    profit_margins = torch.tensor(
        [[it.profit_margin for it in seq] for seq in sequences],
        device=device, dtype=torch.float32,
    )
    is_cpa = torch.tensor(
        [[1.0 if it.is_ad else 0.0 for it in seq] for seq in sequences],
        device=device, dtype=torch.float32,
    )
    with torch.no_grad():
        p_exp, p_clk, p_cvr = model(item_feats, user_feats, ctx_feats, types)

    rewards = ExposureClickCVRHead.compute_reward(
        p_exp,
        p_clk,
        p_cvr,
        bids,
        engagement,
        penalty,
        profit_margins,
        is_cpa,
    )
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
    existing_ads: Optional[list[Item]] = None,
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
    5. **Density constraint**: total ad fraction must not exceed ad_density_limit.

    Args:
        ad: the ad to insert.
        position: 0-indexed insertion position.
        existing_ad_positions: positions of ads already placed.
        constraints: business constraints.
        list_length: total sequence length after insertion.
        existing_ads: the actual ad Item objects already placed (for accurate
            large-ad counting).  If None, large-ad counting is skipped.

    Returns:
        True if the insertion is feasible, False if any constraint is violated.
    """
    num_ads_after = len(existing_ad_positions) + 1

    # Load constraint
    if not constraints.check_load(num_ads_after):
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

    # Large ad constraint — count actual large ads among existing placements
    if ad.is_large_ad:
        num_large = 0
        if existing_ads is not None:
            num_large = sum(1 for a in existing_ads if a.is_large_ad)
        if num_large >= constraints.max_large_ads:
            return False

    # Density constraint
    if num_ads_after / list_length > constraints.ad_density_limit:
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

    The organic list arrives pre-ranked and its order is never changed.
    Stage I only asks: "if I insert one regular ad into this fixed list,
    where should it go?"

    For example, with 8 organic items, 3 regular ad candidates, and 8
    feasible positions (1-8), this produces 3 × 8 = 24 candidate lists.
    Each candidate is the same 8 organic items with one ad inserted,
    giving a 9-item list.  Because all 24 candidates have the same
    length (9), they are stacked into a single [24, 9, D] tensor and
    evaluated in **one batched forward pass** — not 24 separate calls.

    The output is a single 9-item list (or the original 8-item organic
    list if no ad improves the reward).  This becomes the starting point
    for Stage II, which considers more complex configurations (large ads,
    two-ad lists) and uses the Stage I reward as a pruning lower bound.

    Large ads are *not* inserted here — they are handled in Stage II
    because they interact with the double-ad expansion logic.

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

    # Collect all feasible (ad, position) insertions.  Each produces a list
    # of len(organic_list)+1 items — e.g. 8 organic + 1 ad = 9 items.
    candidate_sequences: list[list[Item]] = []
    candidate_meta: list[tuple[Item, int]] = []  # (ad, position)

    for ad in ad_candidates:
        if ad.is_large_ad:
            continue  # Large ads handled in Stage II
        for pos in feasible_positions:
            if not _hard_constraint_filter(
                ad, pos, [], constraints, len(organic_list) + 1, existing_ads=[]
            ):
                continue
            # Insert ad at position into a copy of the organic list
            seq = list(organic_list)
            seq.insert(pos, ad)
            candidate_sequences.append(seq)
            candidate_meta.append((ad, pos))

    if candidate_sequences:
        # All candidates are the same length (organic + 1 ad), so they can be
        # stacked into a single [num_candidates, L+1, D] tensor and evaluated
        # in one batched forward pass — e.g. [24, 9, D] for 24 candidates.
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

    Stage II takes the Stage I winner (a 9-item single-ad list, or the
    8-item organic baseline) and uses its reward as a pruning lower bound.
    It then goes back to the original organic list and enumerates **all**
    feasible mixed-list configurations across four families:

    1. **Large-ad lists** (length 9) — insert one large-format ad at each
       feasible position into the 8-item organic base.
    2. **Double-ad lists** (length 10) — insert two regular ads at all
       valid position pairs satisfying spacing constraints.  With 3
       regular ads and 8 positions, this is C(3,2) × valid position
       pairs ≈ tens of candidates.
    3. **Single-ad lists** (length 9) — re-enumerate every feasible
       single-ad insertion (not just the Stage I winner), so the full
       feasible set is covered for Theorem 9.1 optimality.
    4. **No-ad list** (length 8) — the pure organic baseline.

    Because candidates have different lengths (8, 9, or 10 items), they
    cannot all be stacked into one tensor.  Instead, candidates are
    grouped by length, and each group is batched into one forward pass.
    Typically this means 2-3 batched forward passes total.

    **Constraint-aware reward pruning** (Section 6) is applied when
    ``use_pruning=True``:
    - *Hard constraint filtering* (Section 6.1) discards candidates that
      violate any business rule before the forward pass.
    - *Reward upper-bound pruning* (Section 6.2) computes R_upper for each
      candidate and skips those where R_upper < R_best (the Stage I
      reward serves as the initial R_best).

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
    # Use the Stage I reward as the initial lower bound for pruning.
    # Any Stage II candidate whose upper-bound reward < this is skipped.
    best = intermediate
    base_organic = [it for it in intermediate.items if not it.is_ad]

    large_ads = [ad for ad in ad_candidates if ad.is_large_ad]
    regular_ads = [ad for ad in ad_candidates if ad.is_ad and not ad.is_large_ad]

    candidate_sequences: list[list[Item]] = []
    candidate_ad_positions: list[list[int]] = []

    # (1) Large-ad lists: insert one large ad at each feasible position
    feasible_positions_1 = constraints.get_feasible_ad_positions(len(base_organic) + 1)
    for la in large_ads:
        for pos in feasible_positions_1:
            if not _hard_constraint_filter(
                la, pos, [], constraints, len(base_organic) + 1, existing_ads=[]
            ):
                continue
            seq = list(base_organic)
            seq.insert(pos, la)
            candidate_sequences.append(seq)
            candidate_ad_positions.append([pos])

    # (2) Double-ad lists: insert two regular ads at feasible position pairs
    feasible_positions_2 = constraints.get_feasible_ad_positions(len(base_organic) + 2)
    if constraints.max_ads_per_list >= 2:
        for ad1, ad2 in combinations(regular_ads, 2):
            for pos1 in feasible_positions_2:
                if not _hard_constraint_filter(
                    ad1, pos1, [], constraints, len(base_organic) + 2, existing_ads=[]
                ):
                    continue
                for pos2 in feasible_positions_2:
                    if pos2 <= pos1:
                        continue
                    if not _hard_constraint_filter(
                        ad2, pos2, [pos1], constraints, len(base_organic) + 2,
                        existing_ads=[ad1],
                    ):
                        continue
                    seq = list(base_organic)
                    # Insert in position order (pos1 < pos2) to preserve indices
                    seq.insert(pos1, ad1)
                    seq.insert(pos2, ad2)
                    candidate_sequences.append(seq)
                    candidate_ad_positions.append([pos1, pos2])

    # (3) All single-ad lists: re-enumerate every feasible single-ad insertion
    # so the feasible set is complete (Theorem 9.1 requires exhaustive coverage)
    for ad in regular_ads:
        for pos in feasible_positions_1:
            if not _hard_constraint_filter(
                ad, pos, [], constraints, len(base_organic) + 1, existing_ads=[]
            ):
                continue
            seq = list(base_organic)
            seq.insert(pos, ad)
            candidate_sequences.append(seq)
            candidate_ad_positions.append([pos])

    # (4) No-ad list — pure organic baseline
    no_ad_reward = _evaluate_sequence(model, base_organic, device)
    if no_ad_reward > best.reward:
        best = RankedSequence(items=list(base_organic), reward=no_ad_reward, ad_positions=[])

    if not candidate_sequences:
        return best

    # Candidates have different lengths: single-ad/large-ad lists are L+1 (e.g. 9),
    # double-ad lists are L+2 (e.g. 10).  Group by length so each group can be
    # stacked into one tensor and evaluated in a single batched forward pass.
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


# ---------------------------------------------------------------------------
# Alternative: Beam Search Inference
# ---------------------------------------------------------------------------


@dataclass
class _BeamCandidate:
    """Internal state for one partial sequence during beam search."""

    items: list[Item]
    ad_positions: list[int]         # positions where ads were inserted
    inserted_ad_ids: set[int]       # item_ids of ads already used
    visual_slot_count: int          # total visual slots consumed by ads


def _beam_constraint_check(
    ad: Item,
    position: int,
    candidate: _BeamCandidate,
    constraints: AdConstraints,
    list_length_after: int,
) -> bool:
    """Check whether inserting ``ad`` at ``position`` into a beam candidate is feasible.

    Similar to ``_hard_constraint_filter`` but works with the richer
    ``_BeamCandidate`` state, including visual slot tracking for large ads.
    """
    num_ads_after = len(candidate.ad_positions) + 1

    # Load constraint: total ads ≤ K
    if not constraints.check_load(num_ads_after):
        return False

    # Position bounds
    if position < constraints.min_ad_position or position > min(
        constraints.max_ad_position, list_length_after - 1
    ):
        return False

    # Spacing constraint: new ad must be ≥ Δ away from all existing ads.
    # Existing ad positions shift right by 1 if they are ≥ the insertion point.
    shifted_existing = [
        p + 1 if p >= position else p for p in candidate.ad_positions
    ]
    all_positions = shifted_existing + [position]
    if not constraints.check_spacing(all_positions):
        return False

    # Large-ad cap
    if ad.is_large_ad:
        existing_large = sum(
            1 for p in candidate.ad_positions
            if candidate.items[p].is_large_ad
        )
        if existing_large >= constraints.max_large_ads:
            return False

    # Density constraint (by visual slots for large-ad awareness)
    visual_after = candidate.visual_slot_count + ad.visual_slots
    total_visual = (list_length_after - num_ads_after) + visual_after
    if visual_after / total_visual > constraints.ad_density_limit:
        return False

    return True


def beam_search_inference(
    model: CGRModel,
    candidates: CandidateSet,
    constraints: AdConstraints,
    device: torch.device,
    beam_width: int = 10,
    max_ads: Optional[int] = None,
) -> RankedSequence:
    """Beam search inference — an alternative to the two-stage pipeline.

    Instead of exhaustively enumerating all feasible lists (which only works
    for K ≤ 2), beam search builds the mixed list incrementally:

    1. Start with the organic list as the single initial candidate.
    2. At each step (up to K steps), for every candidate in the beam:
       - Try inserting each unused ad at each feasible position.
       - Apply hard constraint checks (load, spacing, position, density).
       - Also keep the candidate as-is ("stop inserting ads").
    3. Score all expanded candidates and keep the top-B (beam width).
    4. After K steps (or when no more ads can be inserted), return the
       best candidate found across all steps.

    **When to use beam search vs. two-stage:**

    - Two-stage (``cgr_inference``): optimal for K ≤ 2 because it
      exhaustively covers the feasible set.  Guarantees Theorem 9.1
      optimality.  Does not scale to larger K.
    - Beam search: works for any K.  Approximate — it can miss the
      global optimum if the best partial sequence is pruned early.
      But practical for K > 2 where exhaustive enumeration is infeasible.

    **Large (double-slot) ads:** A large ad occupies 2 visual slots
    (``Item.visual_slots``), which is accounted for in density and
    spacing constraint checks.  In the list representation, a large ad
    is still one Item at one position.

    Args:
        model: trained CGR model.
        candidates: organic items + ad candidates.
        constraints: business constraints.
        device: computation device.
        beam_width: number of candidates to keep at each step (B).
            Larger B → better quality, more compute.  B=1 is greedy.
        max_ads: maximum number of insertion steps.  Defaults to
            ``constraints.max_ads_per_list``.

    Returns:
        The best ``RankedSequence`` found by the beam search.
    """
    model.eval()

    if max_ads is None:
        max_ads = constraints.max_ads_per_list

    ad_pool = [ad for ad in candidates.ad_candidates if ad.is_ad]
    organic = list(candidates.organic_items)

    # Initialize beam with the no-ad organic list
    initial = _BeamCandidate(
        items=organic,
        ad_positions=[],
        inserted_ad_ids=set(),
        visual_slot_count=0,
    )
    beam = [initial]

    # Track the global best across all steps (including intermediate steps
    # where a candidate chose to stop inserting early)
    best_reward = float("-inf")
    best_result = RankedSequence(items=organic, reward=best_reward, ad_positions=[])

    for _step in range(max_ads):
        # Collect all expansions: (candidate_list, ad_positions, metadata)
        expanded: list[_BeamCandidate] = []

        for cand in beam:
            # Option 1: stop inserting (keep this candidate as final)
            # Already tracked via scoring below.
            expanded.append(cand)

            # Option 2: try inserting each unused ad at each feasible position
            list_len_after = len(cand.items) + 1
            for ad in ad_pool:
                if ad.item_id in cand.inserted_ad_ids:
                    continue  # each ad can only be inserted once

                feasible = constraints.get_feasible_ad_positions(list_len_after)
                for pos in feasible:
                    if not _beam_constraint_check(
                        ad, pos, cand, constraints, list_len_after
                    ):
                        continue

                    # Build the new candidate with ad inserted
                    new_items = list(cand.items)
                    new_items.insert(pos, ad)
                    # Shift existing ad positions that are at or after the insertion point
                    new_ad_positions = [
                        p + 1 if p >= pos else p for p in cand.ad_positions
                    ] + [pos]
                    new_ids = cand.inserted_ad_ids | {ad.item_id}

                    expanded.append(_BeamCandidate(
                        items=new_items,
                        ad_positions=sorted(new_ad_positions),
                        inserted_ad_ids=new_ids,
                        visual_slot_count=cand.visual_slot_count + ad.visual_slots,
                    ))

        if not expanded:
            break

        # Score all expanded candidates.  Group by length for batched evaluation.
        len_groups: dict[int, list[int]] = {}
        for i, cand in enumerate(expanded):
            slen = len(cand.items)
            len_groups.setdefault(slen, []).append(i)

        rewards = [0.0] * len(expanded)
        for slen, indices in len_groups.items():
            group_seqs = [expanded[i].items for i in indices]
            group_rewards = _evaluate_sequences_batched(model, group_seqs, device)
            for idx, r in zip(indices, group_rewards):
                rewards[idx] = r

        # Update global best
        for i, r in enumerate(rewards):
            if r > best_reward:
                best_reward = r
                best_result = RankedSequence(
                    items=expanded[i].items,
                    reward=r,
                    ad_positions=expanded[i].ad_positions,
                )

        # Keep top-B candidates that still have room for more ads
        # (candidates at max ads or with no feasible insertions naturally
        # fall out of the beam in the next step)
        scored = sorted(
            zip(rewards, expanded), key=lambda x: x[0], reverse=True
        )
        beam = []
        for r, cand in scored:
            if len(beam) >= beam_width:
                break
            # Only keep candidates that can still expand
            if len(cand.ad_positions) < max_ads:
                beam.append(cand)

        if not beam:
            break  # all candidates are fully expanded

    return best_result
