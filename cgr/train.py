"""Training loop for CGR with multi-task loss (Section 4, bottom of page 5).

The training objective is a weighted sum of two binary cross-entropy losses:

    L = λ_exp · L_exp  +  λ_clk · L_clk

where:
* L_exp = BCE(p_exp, y_exp) — exposure prediction loss.  y_exp is 1 when the
  item was actually shown to the user in the logged data.
* L_clk = BCE(p_clk, y_clk) — click prediction loss.  y_clk is 1 when the
  user clicked on the item.

The training data consists of *logged impression sequences*: for each served
page, we have the pre-computed upstream embeddings of every item together
with the binary exposure / click labels observed in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import CGRModel


@dataclass
class TrainConfig:
    """Hyper-parameters for CGR training.

    Attributes:
        lr: learning rate for Adam.
        weight_decay: L2 regularisation weight.
        lambda_exp: weight for the exposure BCE loss.
        lambda_clk: weight for the click BCE loss.
        click_positive_class_weight: positive-class weight for the click loss.
              This is useful because click labels are usually much
              sparser than exposure labels.
        epochs: number of full passes over the training data.
        batch_size: mini-batch size.
        grad_clip: maximum gradient norm (prevents training instability in
                   the autoregressive attention layers).
    """

    lr: float = 1e-3
    weight_decay: float = 1e-5
    lambda_exp: float = 1.0
    lambda_clk: float = 1.0
    click_positive_class_weight: float = 1.0
    epochs: int = 10
    batch_size: int = 32
    grad_clip: float = 1.0


class CGRTrainer:
    """Manages the training loop for a ``CGRModel``.

    The trainer wraps an Adam optimiser, the multi-task BCE loss, and gradient
    clipping.  It expects data as pre-computed embedding tensors (not raw
    features) — matching how CGR operates in production, where embeddings are
    dumped from the upstream DLRM pipeline alongside logged labels.

    Typical usage::

        trainer = CGRTrainer(model, TrainConfig(), device)
        for epoch in range(config.epochs):
            metrics = trainer.train_epoch(dataloader)
            print(metrics)
    """

    def __init__(self, model: CGRModel, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.loss_exp = nn.BCEWithLogitsLoss()
        # Click is still predicted for a specific item at a specific position,
        # but positive click labels are typically far rarer than negative ones.
        # Without pos_weight, the click head can minimise loss by leaning too
        # heavily toward predicting "no click" almost everywhere.
        self.loss_clk = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(config.click_positive_class_weight, device=device)
        )

    def train_step(
        self,
        item_embs: torch.Tensor,
        user_embs: torch.Tensor,
        context_embs: torch.Tensor,
        item_types: torch.Tensor,
        exp_labels: torch.Tensor,
        clk_labels: torch.Tensor,
    ) -> dict[str, float]:
        """Run a single gradient-update step on one mini-batch.

        The forward pass produces per-position logits for exposure and click.
        These are compared against the logged binary labels via
        ``BCEWithLogitsLoss``, weighted by λ_exp and λ_clk, and the combined
        loss is back-propagated.

        Args:
            item_embs: [B, L, item_emb_dim]  — pre-computed from upstream item tower.
            user_embs: [B, L, user_emb_dim]  — pre-computed from upstream user tower.
            context_embs: [B, L, ctx_emb_dim] — pre-computed from context encoder.
            item_types: [B, L] integer item type indices.
            exp_labels: [B, L] binary ground-truth exposure labels (from logs).
            clk_labels: [B, L] binary ground-truth click labels (from logs).

        Returns:
            dict with keys ``"loss"``, ``"loss_exp"``, ``"loss_clk"`` (all floats).
        """
        self.model.train()
        self.optimizer.zero_grad()

        exp_logits, clk_logits = self.model.forward_logits(
            item_embs.to(self.device),
            user_embs.to(self.device),
            context_embs.to(self.device),
            item_types.to(self.device),
        )

        exp_labels = exp_labels.to(self.device)
        clk_labels = clk_labels.to(self.device)

        # Multi-task loss: L = λ_exp · BCE(exp_logits, y_exp) + λ_clk · BCE(clk_logits, y_clk)
        loss_exp = self.loss_exp(exp_logits, exp_labels)
        loss_clk = self.loss_clk(clk_logits, clk_labels)
        loss = self.config.lambda_exp * loss_exp + self.config.lambda_clk * loss_clk

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "loss_exp": loss_exp.item(),
            "loss_clk": loss_clk.item(),
        }

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> dict[str, float]:
        """Train for one full epoch, returning average losses.

        Expects each batch from the dataloader to be a dict with keys::

            item_embs, user_embs, context_embs, item_types, exp_labels, clk_labels

        Returns:
            dict with epoch-averaged ``"loss"``, ``"loss_exp"``, ``"loss_clk"``.
        """
        total_loss = 0.0
        total_exp = 0.0
        total_clk = 0.0
        n_batches = 0

        for batch in dataloader:
            metrics = self.train_step(
                batch["item_embs"],
                batch["user_embs"],
                batch["context_embs"],
                batch["item_types"],
                batch["exp_labels"],
                batch["clk_labels"],
            )
            total_loss += metrics["loss"]
            total_exp += metrics["loss_exp"]
            total_clk += metrics["loss_clk"]
            n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "loss_exp": total_exp / max(n_batches, 1),
            "loss_clk": total_clk / max(n_batches, 1),
        }
