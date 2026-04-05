# ML Re-ranking Workspace

This workspace is about constraint-aware re-ranking for mixed feeds that contain both organic content and ads.

The core problem is:

> Given an already-ranked organic list, a small set of ad candidates, and hard placement rules, which final mixed list should we show?

In this setting, the system is not optimizing a single metric. It needs to balance several objectives at once, including ad revenue, user engagement, conversion value, and business constraints such as ad load, spacing, and position limits.

## Main Source Directories

### cgr/

The baseline paper-inspired CGR implementation.

- Predicts exposure and click signals
- Computes list reward from monetization, engagement, and ad-penalty terms
- Includes bounded two-stage inference and beam-search inference
- Best starting point if you want the simpler reference version

See [cgr/README.md](cgr/README.md) for the detailed package-level walkthrough.

### cgr_w_cvr/

An extended variant of the same setup with conversion prediction and profit-aware reward shaping.

- Predicts exposure, click, and conversion signals
- Adds per-item profit margin into the reward
- Keeps the same overall constrained reranking structure
- Best starting point if you want the CVR/profit-margin version

See [cgr_w_cvr/README.md](cgr_w_cvr/README.md) for the detailed package-level walkthrough.
