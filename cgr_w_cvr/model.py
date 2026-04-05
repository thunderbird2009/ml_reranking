"""CGR-style generative ranking model inspired by Section 4 of the paper.

This module implements a compact approximation of the architecture described
in Section 4 of arXiv:2603.04227.  The model is a **unified listwise
autoregressive network** that jointly predicts per-position exposure
probability (p_exp), click probability (p_clk), and conversion probability
(p_cvr), then derives a list-level reward without needing a separate evaluator
network.

The implementation preserves the paper's main decomposition, but simplifies
some details, especially PLE fusion and the cross-attention pathway.

Architecture overview (Figure 1):

    Pre-computed embeddings  [v_i, u_i, c_i]
              │
              ▼
    ┌─────────────────────┐
    │  Adapter projections │  — align heterogeneous upstream dims → d_model
    │  + positional emb p_i│
    └─────────┬───────────┘
              ▼
    ┌─────────────────────┐
    │  Hierarchical Attn  │  — shared causal SA  +  task-specific SA
    │  with Local Bias    │    + cross-attention   +  LSA(4) + LSA(6)
    └─────────┬───────────┘
              ▼
    ┌─────────────────────┐
    │  PLE Fusion         │  — gated expert mix → h_exp, h_clk
    └─────────┬───────────┘
              ▼
    ┌─────────────────────┐
    │  Reward Heads       │  — MLP → p_exp, p_clk, p_cvr → R(A)
    └─────────────────────┘
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 4.2  Hierarchical Attention with Local Structural Bias
# ---------------------------------------------------------------------------


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Build a standard upper-triangular causal mask for autoregressive attention.

    Position j can only attend to positions ≤ j.  Implemented as an additive
    mask where future positions are −∞ (zeroed out after softmax).

    Returns:
        [seq_len, seq_len] float tensor usable as ``attn_mask`` in
        ``nn.MultiheadAttention``.
    """
    return torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)


def _band_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    """Build a causal band mask for Local Self-Attention (Eq. 13).

    Restricts each position to attend only within a symmetric window of size
    ``window`` *and* only to past-or-current positions (causal).  This encodes
    the locality of feed browsing behaviour — nearby items in the feed have
    stronger exposure-pattern correlations than distant ones.

    The paper uses two fixed window sizes (4 and 6) to capture browsing
    patterns at different granularities.

    Args:
        seq_len: length of the sequence (L).
        window: total window width; each position sees ±window//2 neighbours.
        device: target device.

    Returns:
        [seq_len, seq_len] additive attention mask.
    """
    idx = torch.arange(seq_len, device=device)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    mask = torch.where(diff <= window // 2, 0.0, float("-inf"))
    # Also enforce causality
    mask = mask + _causal_mask(seq_len, device)
    return mask


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with an optional additive mask.

    Used as the building block for three kinds of attention in Section 4.2:

    * **Global causal SA** (shared and task-specific experts, Eq. 11) — pass a
      causal mask so that the model is autoregressive.
    * **Local SA** (LSA, Eq. 13) — pass a band mask that restricts each
      position to a fixed-size local window while remaining causal.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: [B, L, D] input sequence.
            mask: optional [L, L] additive attention mask (−∞ blocks attention).

        Returns:
            [B, L, D] attended output (no residual — caller adds it).
        """
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


class CrossAttentionPositionPreference(nn.Module):
    """Cross-attention for user–item position preference (Eq. 12).

    Captures structural biases such as *first-position preference* (users pay
    more attention to the top of the feed) and *ad-slot sensitivity* (user
    engagement drops differently depending on where an ad is placed).

    Implemented as cross-attention where user-side embeddings query into
    item-side embeddings, both projected through separate linear layers so
    that the model can learn asymmetric user↔item positional interactions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.user_proj = nn.Linear(d_model, d_model)
        self.item_proj = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Compute cross-attention from user queries to item keys/values.

        Args:
            user_emb: [B, L, D] user-side position-aware representations (queries).
            item_emb: [B, L, D] item-side position-aware representations (keys/values).

        Returns:
            [B, L, D] cross-attended output.
        """
        u = self.user_proj(user_emb)
        v = self.item_proj(item_emb)
        out, _ = self.cross_attn(u, v, v)
        return out


class HierarchicalAttentionBlock(nn.Module):
    """One block of Hierarchical Attention with Local Structural Bias (Section 4.2).

    Each block produces six expert representations:

    1. **Shared self-attention** — global causal SA applied to all items.
       Captures universal inter-item dependencies.
    2. **EXP-specific self-attention** — a second causal SA layer on top of the
       shared output, specialised for the exposure-prediction task.
    3. **CLK-specific self-attention** — same as above but specialised for
       click prediction.
    4. **Cross-attention** — models user–item position preference using
       separate user-side and item-side representations (Eq. 12).
    5. **LSA-4** — local self-attention with window=4 to capture short-range
       feed browsing patterns.
    6. **LSA-6** — local self-attention with window=6 for slightly wider
       contextual patterns.

    All sub-layers use pre-norm residual connections.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # Shared self-attention (captures global item–item dependencies)
        self.shared_self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        # Task-specific self-attention (one per prediction objective)
        self.exp_self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.clk_self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        # Cross-attention for position preference (user queries into items)
        self.cross_attn = CrossAttentionPositionPreference(d_model, n_heads, dropout)
        # Local self-attention with windows 4 and 6
        self.lsa_4 = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.lsa_6 = MultiHeadSelfAttention(d_model, n_heads, dropout)
        # Layer norms (one per sub-layer, applied as post-norm residual)
        self.norm_shared = nn.LayerNorm(d_model)
        self.norm_exp = nn.LayerNorm(d_model)
        self.norm_clk = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_lsa4 = nn.LayerNorm(d_model)
        self.norm_lsa6 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run all six attention experts and return their outputs.

        Args:
            x: [B, L, D] input sequence (fused item representations).
            user_repr: [B, L, D] position-aware user-side representations
                for cross-attention (Eq. 12).
            item_repr: [B, L, D] position-aware item-side representations
                for cross-attention (Eq. 12).

        Returns:
            dict with keys ``"shared"``, ``"exp"``, ``"clk"``, ``"cross"``,
            ``"lsa4"``, ``"lsa6"`` — each mapping to a [B, L, D] tensor.
        """
        B, L, D = x.shape
        device = x.device
        causal = _causal_mask(L, device)
        band4 = _band_mask(L, 4, device)
        band6 = _band_mask(L, 6, device)

        # Global causal self-attention (shared → task-specific)
        h_shared = self.norm_shared(x + self.shared_self_attn(x, mask=causal))
        h_exp = self.norm_exp(h_shared + self.exp_self_attn(h_shared, mask=causal))
        h_clk = self.norm_clk(h_shared + self.clk_self_attn(h_shared, mask=causal))

        # Cross-attention: user queries into item keys/values (Eq. 12)
        h_cross = self.norm_cross(x + self.cross_attn(user_repr, item_repr))

        # Local self-attention with two window sizes
        h_lsa4 = self.norm_lsa4(x + self.lsa_4(x, mask=band4))
        h_lsa6 = self.norm_lsa6(x + self.lsa_6(x, mask=band6))

        return {
            "shared": h_shared,
            "exp": h_exp,
            "clk": h_clk,
            "cross": h_cross,
            "lsa4": h_lsa4,
            "lsa6": h_lsa6,
        }


# ---------------------------------------------------------------------------
# 4.3  Multi-Expert Fusion via PLE
# ---------------------------------------------------------------------------


class PLEGate(nn.Module):
    """A single Progressive Layered Extraction gate (Eqs. 14-15).

    PLE (from the MMoE/PLE family of multi-task learning architectures) fuses
    an arbitrary number of expert outputs through a learned soft-attention
    gate.  Each position in the sequence independently computes gate weights
    over the experts, enabling the model to adaptively weight global vs. local
    vs. position-preference signals at every slot in the feed.

    Gating mechanism:
        gate_input = mean(expert_outputs)          — [B, L, D]
        weights    = softmax(Linear(gate_input))   — [B, L, num_experts]
        output     = Σ_k  weights_k · expert_k     — [B, L, D]
    """

    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, expert_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Fuse a list of expert tensors into a single representation.

        Args:
            expert_outputs: list of ``num_experts`` tensors, each [B, L, D].

        Returns:
            [B, L, D] gated fusion of all experts.
        """
        stacked = torch.stack(expert_outputs, dim=-2)  # [B, L, num_experts, D]
        # Gate input: average across experts gives a position-level summary
        gate_input = stacked.mean(dim=-2)  # [B, L, D]
        weights = F.softmax(self.gate(gate_input), dim=-1)  # [B, L, num_experts]
        # Weighted combination via einsum: sum over expert dim
        fused = torch.einsum("blnd,bln->bld", stacked, weights)
        return fused


class PLEFusion(nn.Module):
    """PLE fusion layer producing task-specific representations (Section 4.3).

    The paper states:
    * The **EXP branch** integrates 12 experts — all 6 experts from both the
      EXP-oriented and CLK-oriented attention blocks.
    * The **CLK branch** integrates 8 experts — (shared, clk) from both blocks
      plus (cross, lsa4, lsa6) from the CLK-oriented block.

    Each branch has its own ``PLEGate`` so the learned mixing weights are
    task-specific, enabling cross-task knowledge transfer (via shared experts
    and cross-block experts) while preserving task specificity.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # EXP gate: 12 experts (6 from EXP block + 6 from CLK block)
        self.exp_gate = PLEGate(d_model, num_experts=12)
        # CLK gate: 8 experts (shared+clk from each block + cross+lsa4+lsa6 from CLK block)
        self.clk_gate = PLEGate(d_model, num_experts=8)

    def forward(
        self,
        exp_block_out: dict[str, torch.Tensor],
        clk_block_out: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce task-specific fused representations for EXP and CLK.

        Args:
            exp_block_out: dict from the EXP-oriented ``HierarchicalAttentionBlock``.
            clk_block_out: dict from the CLK-oriented ``HierarchicalAttentionBlock``.

        Returns:
            (h_exp, h_clk) — each [B, L, D], ready for the prediction heads.
        """
        # EXP branch: all 12 experts from both blocks
        h_exp = self.exp_gate([
            exp_block_out["shared"],
            exp_block_out["exp"],
            exp_block_out["clk"],
            exp_block_out["cross"],
            exp_block_out["lsa4"],
            exp_block_out["lsa6"],
            clk_block_out["shared"],
            clk_block_out["exp"],
            clk_block_out["clk"],
            clk_block_out["cross"],
            clk_block_out["lsa4"],
            clk_block_out["lsa6"],
        ])
        # CLK branch: shared+clk from each block, cross+lsa4+lsa6 from CLK block
        h_clk = self.clk_gate([
            exp_block_out["shared"],
            exp_block_out["clk"],
            clk_block_out["shared"],
            clk_block_out["clk"],
            clk_block_out["cross"],
            clk_block_out["lsa4"],
            clk_block_out["lsa6"],
            clk_block_out["exp"],
        ])
        return h_exp, h_clk


# ---------------------------------------------------------------------------
# 4.4  Unified Sequence Reward Modeling
# ---------------------------------------------------------------------------


class ExposureClickCVRHead(nn.Module):
    """Prediction heads for exposure, click, and CVR plus reward computation.

    Unlike traditional Generator–Evaluator systems that need a separate
    evaluator network, CGR integrates reward estimation directly into the
    generative model.  This module:

     1. Predicts per-position probabilities p_exp_i, p_clk_i, and p_cvr_i via
         small MLP heads.
     2. Combines them with scalar auction / policy signals (bid, engagement,
         ad penalty, profit margin) to compute the list-level reward.

    This design eliminates a separate evaluator forward pass during inference,
    which is critical for meeting the <40 ms P99 latency SLA (Section 11).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.exp_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.clk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.cvr_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward_logits(
        self,
        h_exp: torch.Tensor,
        h_clk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict per-position exposure, click, and conversion logits.

        Args:
            h_exp: [B, L, D] EXP-branch representation from PLE fusion.
            h_clk: [B, L, D] CLK-branch representation from PLE fusion.

        Returns:
            exp_logits: [B, L] exposure logits per position.
            clk_logits: [B, L] click logits per position.
            cvr_logits: [B, L] conversion logits per position.
        """
        exp_logits = self.exp_head(h_exp).squeeze(-1)  # [B, L]
        clk_logits = self.clk_head(h_clk).squeeze(-1)  # [B, L]
        # Reuse the click-oriented fused representation for CVR because
        # conversion is downstream of click in the interaction funnel.
        cvr_logits = self.cvr_head(h_clk).squeeze(-1)  # [B, L]
        return exp_logits, clk_logits, cvr_logits

    def forward(
        self,
        h_exp: torch.Tensor,
        h_clk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict per-position exposure, click, and CVR probabilities.

        The training code uses logits for numerical stability via
        ``BCEWithLogitsLoss``. The public forward path keeps returning
        probabilities because inference and reward computation operate on
        calibrated probabilities.

        Args:
            h_exp: [B, L, D] EXP-branch representation from PLE fusion.
            h_clk: [B, L, D] CLK-branch representation from PLE fusion.

        Returns:
            p_exp: [B, L] predicted exposure probability per position.
            p_clk: [B, L] predicted click probability per position.
            p_cvr: [B, L] predicted conversion probability per position.
        """
        exp_logits, clk_logits, cvr_logits = self.forward_logits(h_exp, h_clk)
        p_exp = torch.sigmoid(exp_logits)
        p_clk = torch.sigmoid(clk_logits)
        p_cvr = torch.sigmoid(cvr_logits)
        return p_exp, p_clk, p_cvr

    @staticmethod
    def compute_reward(
        p_exp: torch.Tensor,
        p_clk: torch.Tensor,
        p_cvr: torch.Tensor,
        bids: torch.Tensor,
        engagement_scores: torch.Tensor,
        ad_penalty_weights: torch.Tensor,
        profit_margins: torch.Tensor,
        is_cpa: torch.Tensor,
    ) -> torch.Tensor:
        """Compute list-level reward with ad revenue and conversion-linked profit.

                The reward balances four objectives per position:
        * **V_i — monetization value** (Eq. 18): ad revenue, depending on
          billing model (CPA vs. CPM).
        * **N_i — engagement benefit** (Eq. 19): predicted user engagement
          weighted by upstream engagement score.
                * **O_i — profit value**: expected profit margin,
                    modeled as exposure × click × conversion × margin.
        * **P_i — ad exposure penalty** (Eq. 20): penalises showing ads,
          controlled by business-policy weight d_i.

        Args:
            p_exp: [B, L] predicted exposure probabilities.
            p_clk: [B, L] predicted click probabilities.
                p_cvr: [B, L] predicted conversion probabilities.
                bids:  [B, L] advertiser bid prices s_i (0 for organic items).
            engagement_scores: [B, L] upstream engagement predictions n_i.
            ad_penalty_weights: [B, L] policy penalty weights d_i.
                profit_margins: [B, L] realized margin if the item converts.
            is_cpa: [B, L] 1.0 if the item uses CPA (cost-per-action) billing,
                    0.0 for CPM (cost-per-mille) or organic items.

        Returns:
            [B] scalar reward for each sequence in the batch.
        """
        # V_i: monetization value (Eq. 18)
        # CPA: revenue accrues only on click → p_clk * p_exp * bid
        # CPM / other: revenue accrues on exposure → p_exp * bid
        v_cpa = p_clk * p_exp * bids
        v_other = p_exp * bids
        v = torch.where(is_cpa.bool(), v_cpa, v_other)

        # N_i: engagement benefit (Eq. 19)
        n = p_exp * p_clk * engagement_scores

        # O_i: expected conversion-linked profit value.
        profit_value = p_exp * p_clk * p_cvr * profit_margins

        # P_i: ad exposure penalty (Eq. 20)
        p = ad_penalty_weights * p_exp

        reward = (v + n + profit_value - p).sum(dim=-1)
        return reward


# ---------------------------------------------------------------------------
# Full CGR Model
# ---------------------------------------------------------------------------


class CGRModel(nn.Module):
    """Constraint-Aware Generative Re-ranking Model.

    Unified listwise autoregressive model (Section 4) that combines:
    - Light adapter projections over pre-computed upstream embeddings
    - Hierarchical attention with local structural bias
    - PLE multi-expert fusion
    - Unified sequence reward modeling

    This is a paper-inspired implementation, not an exact reproduction of the
    full model described in arXiv:2603.04227.

    **Important:** CGR sits at the tail of a multi-stage pipeline.  The input
    embeddings (item, user, context) are **pre-computed dense vectors** from
    upstream models (e.g. a DLRM item tower, a user tower, a context encoder).
    The linear projections here are thin adapters that align heterogeneous
    upstream embedding spaces into a shared d_model space — they are NOT
    full feature encoders.

    Args:
        item_emb_dim: dimensionality of pre-computed item embeddings (v_i).
        user_emb_dim: dimensionality of pre-computed user embeddings (u_i).
        context_emb_dim: dimensionality of pre-computed context embeddings (c_i).
        d_model: internal hidden dimension used throughout the model.
        n_heads: number of attention heads in every MHA sub-layer.
        n_layers: number of stacked ``HierarchicalAttentionBlock`` layers.
        max_list_len: maximum sequence length L (paper uses 11).
        dropout: dropout rate for adapter fusion and attention.
    """

    def __init__(
        self,
        item_emb_dim: int,
        user_emb_dim: int,
        context_emb_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_list_len: int = 11,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_list_len = max_list_len

        # Adapter projections: align pre-computed upstream embeddings → d_model
        # x_i = [u_i, c_i, v_i, p_i]  (Eq. 10)
        self.item_adapter = nn.Linear(item_emb_dim, d_model)      # v_i projection
        self.user_adapter = nn.Linear(user_emb_dim, d_model)      # u_i projection
        self.context_adapter = nn.Linear(context_emb_dim, d_model)  # c_i projection
        # p_i: positional embedding — the only embedding learned *within* CGR
        self.position_emb = nn.Embedding(max_list_len, d_model)
        # Item type (organic/ad/large_ad/ecom_ad) — added to positional signal
        self.item_type_emb = nn.Embedding(4, d_model)

        # Fuse the four components [u_i, c_i, v_i, p_i] into a single vector
        self.input_fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Dual stacked hierarchical attention blocks (Section 4.3):
        # EXP-oriented and CLK-oriented blocks produce 6 experts each,
        # yielding 12 experts for the EXP PLE gate and 8 for the CLK gate.
        self.exp_attn_blocks = nn.ModuleList(
            [HierarchicalAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.clk_attn_blocks = nn.ModuleList(
            [HierarchicalAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # PLE fusion (produces task-specific h_exp, h_clk from 12/8 experts)
        self.ple_fusion = PLEFusion(d_model)

        # Exposure/click/conversion prediction heads (p_exp, p_clk, p_cvr → R(A))
        self.exposure_click_cvr_head = ExposureClickCVRHead(d_model)

    def encode_items(
        self,
        item_embs: torch.Tensor,
        user_embs: torch.Tensor,
        context_embs: torch.Tensor,
        item_types: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project pre-computed upstream embeddings and fuse them (Eq. 10).

        Each item's representation is formed by concatenating four d_model
        vectors — one per adapter — and projecting them down through a
        single-layer fusion MLP.

        Also returns the position-aware user-side and item-side representations
        used by cross-attention (Eq. 12).

        Args:
            item_embs: [B, L, item_emb_dim]  — from upstream item tower / DLRM.
            user_embs: [B, L, user_emb_dim]  — from upstream user tower.
            context_embs: [B, L, context_emb_dim] — from context encoder.
            item_types: [B, L] integer item type indices (see ``ItemType``).
            positions: [B, L] optional position indices; defaults to 0..L−1.

        Returns:
            (fused, user_repr, item_repr):
            - fused: [B, L, d_model] fused item representations for SA.
            - user_repr: [B, L, d_model] position-aware user-side (for cross-attn queries).
            - item_repr: [B, L, d_model] position-aware item-side (for cross-attn keys).
        """
        B, L, _ = item_embs.shape
        if positions is None:
            positions = torch.arange(L, device=item_embs.device).unsqueeze(0).expand(B, -1)

        v = self.item_adapter(item_embs)        # v_i → d_model
        u = self.user_adapter(user_embs)        # u_i → d_model
        c = self.context_adapter(context_embs)  # c_i → d_model
        p = self.position_emb(positions)        # p_i (learned in CGR)
        t = self.item_type_emb(item_types)      # type signal

        # Concatenate and fuse: [u_i, c_i, v_i, p_i] with type info added to position
        combined = torch.cat([u, c, v, p + t], dim=-1)  # [B, L, 4*d_model]
        fused = self.input_fusion(combined)  # [B, L, d_model]

        # Position-aware user/item representations for cross-attention (Eq. 12)
        user_repr = u + p  # user embedding + positional signal
        item_repr = v + p + t  # item embedding + positional + type signal

        return fused, user_repr, item_repr

    def forward(
        self,
        item_embs: torch.Tensor,
        user_embs: torch.Tensor,
        context_embs: torch.Tensor,
        item_types: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: embed → attend → fuse → predict.

        This is the single unified forward pass that replaces the traditional
        two-network Generator–Evaluator pipeline.  It produces both the
        sequence generation (via autoregressive attention) and the reward
        evaluation (via the prediction heads) in one shot.

        Args:
            item_embs: [B, L, item_emb_dim]  — pre-computed item embeddings.
            user_embs: [B, L, user_emb_dim]  — pre-computed user embeddings.
            context_embs: [B, L, context_emb_dim] — pre-computed context embeddings.
            item_types: [B, L] integer item type indices.
            positions: [B, L] optional position indices.

        Returns:
            p_exp: [B, L] per-position exposure probabilities.
            p_clk: [B, L] per-position click probabilities.
            p_cvr: [B, L] per-position conversion probabilities.
        """
        exp_logits, clk_logits, cvr_logits = self.forward_logits(
            item_embs, user_embs, context_embs, item_types, positions
        )
        return torch.sigmoid(exp_logits), torch.sigmoid(clk_logits), torch.sigmoid(cvr_logits)

    def forward_logits(
        self,
        item_embs: torch.Tensor,
        user_embs: torch.Tensor,
        context_embs: torch.Tensor,
        item_types: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass returning logits for stable binary losses.

        Training uses logits rather than probabilities because
        ``BCEWithLogitsLoss`` is numerically more stable and supports positive
        class weighting for sparse outcomes such as clicks.

        Args:
            item_embs: [B, L, item_emb_dim] pre-computed item embeddings.
            user_embs: [B, L, user_emb_dim] pre-computed user embeddings.
            context_embs: [B, L, context_emb_dim] pre-computed context embeddings.
            item_types: [B, L] integer item type indices.
            positions: [B, L] optional position indices.

        Returns:
            exp_logits: [B, L] per-position exposure logits.
            clk_logits: [B, L] per-position click logits.
            cvr_logits: [B, L] per-position conversion logits.
        """
        x, user_repr, item_repr = self.encode_items(
            item_embs, user_embs, context_embs, item_types, positions
        )

        # Apply dual hierarchical attention blocks sequentially.
        # Each block propagates via its shared expert output.
        x_exp = x
        x_clk = x
        exp_out: dict[str, torch.Tensor] = {}
        clk_out: dict[str, torch.Tensor] = {}
        for exp_block, clk_block in zip(self.exp_attn_blocks, self.clk_attn_blocks):
            exp_out = exp_block(x_exp, user_repr, item_repr)
            clk_out = clk_block(x_clk, user_repr, item_repr)
            x_exp = exp_out["shared"]
            x_clk = clk_out["shared"]

        # PLE fusion on the final layer's dual expert outputs (12 + 8)
        h_exp, h_clk = self.ple_fusion(exp_out, clk_out)

        # Prediction heads → per-position logits
        exp_logits, clk_logits, cvr_logits = self.exposure_click_cvr_head.forward_logits(h_exp, h_clk)
        return exp_logits, clk_logits, cvr_logits
