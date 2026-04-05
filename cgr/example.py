"""End-to-end CGR example with synthetic data.

Demonstrates:
1. Simulating pre-computed upstream embeddings (as would come from a DLRM pipeline)
2. Training the CGR model on synthetic exposure/click labels
3. Running two-stage constrained inference
4. Printing the final ranked list with reward
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from .inference import cgr_inference
from .model import CGRModel
from .train import CGRTrainer, TrainConfig
from .data_types import AdConstraints, CandidateSet, Item, ItemType

# Upstream embedding dimensions (would match the output of your DLRM / towers)
ITEM_EMB_DIM = 16   # e.g. output dim of item tower
USER_EMB_DIM = 8    # e.g. output dim of user tower
CTX_EMB_DIM = 4     # e.g. output dim of context encoder
D_MODEL = 32
MAX_LIST_LEN = 11


def make_synthetic_item(
    item_id: int,
    item_type: ItemType,
    bid: float = 0.0,
    engagement: float = 1.0,
    ad_penalty: float = 0.0,
) -> Item:
    """Simulate an item whose embeddings have been pre-computed by upstream models."""
    return Item(
        item_id=item_id,
        item_type=item_type,
        # In production these would come from upstream towers, not random:
        item_emb=torch.randn(ITEM_EMB_DIM),     # v_i from item tower
        user_emb=torch.randn(USER_EMB_DIM),      # u_i from user tower
        context_emb=torch.randn(CTX_EMB_DIM),    # c_i from context encoder
        # From ad auction / business systems:
        bid=bid,                                   # s_i
        engagement_score=engagement,               # n_i
        ad_penalty_weight=ad_penalty,              # d_i
    )


def make_synthetic_dataset(
    n_samples: int = 256,
    list_len: int = 8,
) -> TensorDataset:
    """Create a synthetic training dataset.

    In production, each row would contain pre-computed embeddings dumped from
    the upstream pipeline alongside logged exposure/click labels.
    """
    # Simulate pre-computed upstream embeddings
    item_embs = torch.randn(n_samples, list_len, ITEM_EMB_DIM)
    user_embs = torch.randn(n_samples, list_len, USER_EMB_DIM)
    context_embs = torch.randn(n_samples, list_len, CTX_EMB_DIM)
    # Mix of organic (0) and ad (1) items
    item_types = torch.zeros(n_samples, list_len, dtype=torch.long)
    item_types[:, 2] = 1  # ad at position 2
    item_types[:, 5] = 1  # ad at position 5

    # Logged labels: exposure ~ 0.7 base, click ~ 0.3 base
    exp_labels = (torch.rand(n_samples, list_len) > 0.3).float()
    clk_labels = (torch.rand(n_samples, list_len) > 0.7).float() * exp_labels

    return TensorDataset(item_embs, user_embs, context_embs, item_types, exp_labels, clk_labels)


def collate_fn(batch):
    item_e, user_e, ctx_e, types, exp, clk = zip(*batch)
    return {
        "item_embs": torch.stack(item_e),
        "user_embs": torch.stack(user_e),
        "context_embs": torch.stack(ctx_e),
        "item_types": torch.stack(types),
        "exp_labels": torch.stack(exp),
        "clk_labels": torch.stack(clk),
    }


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Build model
    model = CGRModel(
        item_emb_dim=ITEM_EMB_DIM,
        user_emb_dim=USER_EMB_DIM,
        context_emb_dim=CTX_EMB_DIM,
        d_model=D_MODEL,
        n_heads=4,
        n_layers=2,
        max_list_len=MAX_LIST_LEN,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 2. Train on synthetic (pre-computed embedding, logged label) pairs
    print("\n--- Training ---")
    config = TrainConfig(lr=1e-3, epochs=5, batch_size=32)
    trainer = CGRTrainer(model, config, device)

    dataset = make_synthetic_dataset(n_samples=256, list_len=8)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(config.epochs):
        metrics = trainer.train_epoch(loader)
        print(f"  Epoch {epoch+1}: loss={metrics['loss']:.4f}  "
              f"exp={metrics['loss_exp']:.4f}  clk={metrics['loss_clk']:.4f}")

    # 3. Run inference — simulate a real request with pre-computed embeddings
    print("\n--- Inference ---")
    organic_items = [
        make_synthetic_item(i, ItemType.ORGANIC, engagement=1.0 + 0.1 * i)
        for i in range(8)
    ]
    ad_candidates = [
        make_synthetic_item(100, ItemType.AD, bid=2.5, engagement=0.5, ad_penalty=0.3),
        make_synthetic_item(101, ItemType.AD, bid=3.0, engagement=0.3, ad_penalty=0.4),
        make_synthetic_item(102, ItemType.LARGE_AD, bid=5.0, engagement=0.8, ad_penalty=0.5),
        make_synthetic_item(103, ItemType.ECOM_AD, bid=1.5, engagement=0.6, ad_penalty=0.2),
    ]

    candidates = CandidateSet(organic_items=organic_items, ad_candidates=ad_candidates)
    constraints = AdConstraints(
        max_ads_per_list=2,
        min_ad_spacing=3,
        min_ad_position=1,
        max_ad_position=8,
    )

    result = cgr_inference(model, candidates, constraints, device, use_pruning=True)

    print(f"\nFinal list (reward={result.reward:.4f}):")
    print(f"  Ad positions: {result.ad_positions}")
    print(f"  Sequence ({len(result.items)} items):")
    for i, item in enumerate(result.items):
        marker = " [AD]" if item.is_ad else ""
        bid_str = f"  bid={item.bid:.1f}" if item.is_ad else ""
        print(f"    [{i}] item_{item.item_id} ({item.item_type.name}){bid_str}{marker}")

    # Verify constraints
    is_valid = constraints.is_feasible(result.items)
    print(f"\n  Constraints satisfied: {is_valid}")


if __name__ == "__main__":
    run()
