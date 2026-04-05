"""Data types and constraint definitions for CGR (Section 3 of the paper).

This module defines the core domain objects that flow through the CGR system:

* **Item** — a single candidate (organic content or sponsored ad) carrying
  pre-computed embeddings from upstream towers plus scalar auction / policy
  signals.
* **AdConstraints** — the full set of business rules that govern where and
  how many ads may appear in a single page (load, spacing, density, etc.).
* **CandidateSet** — the request-level bundle of organic items and ad
  candidates that the inference pipeline receives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class ItemType(Enum):
    """Content type of a feed item.

    CGR treats different ad formats differently during constraint checking
    and during Stage II decoding (Section 7.2):

    * ORGANIC   — natural recommendation from the upstream ranking pipeline.
    * AD        — standard sponsored advertisement.
    * LARGE_AD  — a high-impact ad format that occupies more visual space and
                  has its own placement cap (max_large_ads).
    * ECOM_AD   — an e-commerce ad with purchase-intent signals; treated like
                  a regular ad for constraint purposes but may carry different
                  bid / penalty parameters.
    """

    ORGANIC = 0
    AD = 1
    LARGE_AD = 2
    ECOM_AD = 3


@dataclass
class Item:
    """A candidate item in the feed (organic content or advertisement).

    CGR is a reranking model at the end of a multi-stage pipeline:

        Matching → Pre-ranking → Ranking (DLRM) → **CGR Reranking**

    The embedding fields (item_emb, user_emb, context_emb) are **pre-computed
    dense vectors from upstream models**, not raw features.  CGR does not learn
    these representations — it only learns to fuse and rerank them.

    Mapping to paper notation (Eq. 10, Section 4.1):
        x_i = [u_i, c_i, v_i, p_i]

        v_i  →  item_emb      : item embedding from upstream DLRM / item tower
        u_i  →  user_emb      : user embedding from upstream user tower
        c_i  →  context_emb   : context embedding (time-of-day, device, session, etc.)
        p_i  →  (learned inside CGR via nn.Embedding, not stored on Item)

    Scalar fields come from other upstream systems:
        s_i  →  bid            : bid price, set by the advertiser / ad auction
        n_i  →  engagement_score : predicted engagement benefit from upstream model
        d_i  →  ad_penalty_weight: business-policy weight penalising ad exposure
    """

    item_id: int
    item_type: ItemType
    # --- Pre-computed embeddings from upstream pipeline ---
    item_emb: torch.Tensor       # v_i: dense item embedding from DLRM / item tower
    user_emb: torch.Tensor       # u_i: dense user embedding from user tower
    context_emb: torch.Tensor    # c_i: dense context embedding (time, device, session …)
    # --- Scalars from ad auction / business systems ---
    bid: float = 0.0             # s_i: bid price (only meaningful for ads)
    engagement_score: float = 0.0  # n_i: upstream engagement prediction
    ad_penalty_weight: float = 0.0  # d_i: policy weight penalising ad exposure

    @property
    def is_ad(self) -> bool:
        """True for any sponsored item (AD, LARGE_AD, ECOM_AD)."""
        return self.item_type != ItemType.ORGANIC

    @property
    def is_large_ad(self) -> bool:
        """True only for the high-impact large-ad format."""
        return self.item_type == ItemType.LARGE_AD


@dataclass
class AdConstraints:
    """Business constraints for ad insertion in feeds (Section 3.4).

    Industrial advertising feeds impose strict structural constraints that
    must hold for every served page.  CGR's bounded decoding (Section 5)
    exploits the fact that K is small (typically ≤ 2) to reduce the search
    space from O(N!) to O(K·L).

    The three families of constraints described in the paper:

    (1) **Load constraint** (Eq. 6):  ``#Ad(A) ≤ K``
        → ``max_ads_per_list``

    (2) **Position spacing constraint** (Eq. 7):
        ``|pos(a_i) - pos(a_j)| ≥ Δ``  for any two ads a_i, a_j
        → ``min_ad_spacing``

    (3) **Structural rules** (Section 3.4, bullet 3):
        - Valid insertion interval  →  ``min_ad_position``, ``max_ad_position``
        - Large-ad placement cap   →  ``max_large_ads``
        - User-level exposure frequency control  →  ``ad_density_limit``
    """

    max_ads_per_list: int = 2  # K: maximum number of ads per list
    min_ad_spacing: int = 3  # Delta: minimum positions between ads
    min_ad_position: int = 1  # earliest position an ad can appear (0-indexed)
    max_ad_position: int = 10  # latest position an ad can appear (0-indexed)
    max_large_ads: int = 1  # at most one large ad per list
    ad_density_limit: float = 0.3  # at most x% of the page can be ads

    def get_feasible_ad_positions(self, list_length: int) -> list[int]:
        """Return all positions where an ad could legally be inserted.

        Positions are 0-indexed and bounded by both ``min_ad_position`` /
        ``max_ad_position`` and the actual list length.  This is the starting
        point for both Stage I and Stage II candidate enumeration.
        """
        return [
            p
            for p in range(list_length)
            if self.min_ad_position <= p <= min(self.max_ad_position, list_length - 1)
        ]

    def check_spacing(self, ad_positions: list[int]) -> bool:
        """Check minimum spacing between every pair of ads (Eq. 7).

        Returns True when ``|pos(a_i) - pos(a_j)| ≥ min_ad_spacing`` holds
        for all pairs.  An empty or single-ad list trivially passes.
        """
        sorted_pos = sorted(ad_positions)
        for i in range(1, len(sorted_pos)):
            if sorted_pos[i] - sorted_pos[i - 1] < self.min_ad_spacing:
                return False
        return True

    def check_load(self, num_ads: int) -> bool:
        """Check the ad load constraint ``#Ad(A) ≤ K`` (Eq. 6)."""
        return num_ads <= self.max_ads_per_list

    def is_feasible(self, sequence: list[Item]) -> bool:
        """Validate that a fully-constructed sequence satisfies ALL constraints.

        Used as a post-hoc verification after inference to confirm correctness.

        This helper checks the constraints implemented in this repository.
        It should not be interpreted as a proof that all structural rules from
        the paper are being enforced exactly as written.
        """
        ad_positions = [i for i, item in enumerate(sequence) if item.is_ad]
        num_ads = len(ad_positions)
        num_large = sum(1 for i in ad_positions if sequence[i].is_large_ad)

        if not self.check_load(num_ads):
            return False
        if not self.check_spacing(ad_positions):
            return False
        if num_large > self.max_large_ads:
            return False
        if num_ads > 0 and num_ads / len(sequence) > self.ad_density_limit:
            return False
        return True


@dataclass
class CandidateSet:
    """Candidate pool X = {x_1, ..., x_N} for a single user request (Section 3.1).

    The organic items arrive **pre-ranked** by the upstream ranking stage
    (e.g. a DLRM).  CGR's job is to decide *where* (and *whether*) to insert
    ads from ``ad_candidates`` into that organic list while maximising the
    combined reward R(A) = Σ(V_i + N_i − P_i) subject to business constraints.
    """

    organic_items: list[Item]
    ad_candidates: list[Item]

    @property
    def all_items(self) -> list[Item]:
        """All items in the pool (organic + ads), unordered."""
        return self.organic_items + self.ad_candidates

    @property
    def num_organic(self) -> int:
        """Number of organic (non-sponsored) items."""
        return len(self.organic_items)

    @property
    def num_ads(self) -> int:
        """Number of candidate ads available for insertion."""
        return len(self.ad_candidates)
