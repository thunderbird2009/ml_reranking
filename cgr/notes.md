# Notes on 2505.07197 for Mixed Organic/Ad List Generation

Paper: "A Generative Re-ranking Model for List-level Multi-objective Optimization at Taobao" (arXiv:2505.07197)

## Short Answer

The method in 2505.07197 is a sound approach for efficient one-call list-level multi-objective reranking.

It is not, by itself, a complete solution for constraint-heavy mixed organic/ad feed generation.

## What The Paper Is Good At

The paper addresses a real and important problem:

- list-level multi-objective optimization
- avoiding repeated online model calls
- generating a final ranked list efficiently in one serving call

Its main strengths are:

1. It models list-level value rather than only item-level scores.
2. It uses a generative reranking formulation rather than greedy sorting over one scalar score.
3. It compresses iterative item selection into tensor operations within one inference call.
4. It includes an integrated diversity mechanism instead of treating diversity as pure post-processing.

For large-scale e-commerce reranking, that is a credible and practical design.

## What "One Shot" Means Here

The method is one-shot at the serving-call level, not literally one-step direct generation.

Internally, SORT-Gen still performs sequential selection logic:

- maintain objective-specific candidate queues
- expand candidate sub-lists
- score next-step choices
- update masks / queue state
- repeat the above through tensorized operations

So the value proposition is:

- no repeated RPC or model invocation loop during serving

not:

- a single unconstrained network output that directly emits the final mixed list with no internal sequential structure

## Why It Is Not Enough For Mixed Organic/Ad Feeds

Mixed organic/ad ranking is harder than generic multi-objective reranking because the system usually must satisfy hard policy constraints, for example:

- max ads per page
- minimum spacing between ads
- legal ad insertion intervals
- large-ad caps
- business exposure penalties
- policy compliance requirements

2505.07197 is primarily about e-commerce list-level optimization over click, conversion, and GMV. It does not present a dedicated mechanism for strict ad-allocation feasibility in the style needed for industrial ad/organic feeds.

That means it is missing, or at least does not foreground, the following pieces that matter for ad mixing:

1. explicit hard feasibility constraints on ad placement
2. constraint-preserving generation logic
3. a guarantee that all generated lists satisfy ad policy rules
4. explicit treatment of ads as a structurally different item family rather than just another objective tradeoff

Its integrated MMR diversity term helps with homogeneity, but diversity control is not the same thing as ad-policy enforcement.

## Comparison With CGR

SORT-Gen and CGR are solving adjacent but different problems.

### SORT-Gen is stronger when:

- you want one-call efficient reranking
- you care about general list-level multi-objective optimization
- your constraints are soft or can be expressed indirectly through queue design and scoring

### CGR is stronger when:

- you must explicitly mix organic and ad items
- hard structural ad constraints matter
- feasible placement itself is the central optimization problem

### Current repo caveat

The CGR implementation in this repository should be read as a paper-inspired prototype, not as a strict reproduction of every important detail in the CGR paper.

There are two separate caveats here.

First, it is only partially faithful to the paper.

That means the repository follows the paper's high-level ideas:

- sequence-level modeling over a mixed list
- exposure/click prediction
- reward-based comparison of candidate lists
- constrained ad placement as the central problem

Several details that were initially simplified have since been addressed:

1. PLE fusion now uses dual attention blocks (EXP-oriented + CLK-oriented) producing 12 experts for the EXP gate and 8 for the CLK gate, matching the paper's Section 4.3
2. Cross-attention now uses separate user-side and item-side position-aware representations (Eq. 12) instead of the fused input
3. Stage II inference re-enumerates all single-ad insertions (not just the Stage I winner) for complete feasible set coverage (Theorem 9.1)
4. Hard constraint filtering now checks density limits and correctly counts large ads

Remaining simplifications:

1. the implementation is still a prototype, not a production-grade system
2. user-level exposure frequency control (cross-request state) is not implemented

So when this note compares SORT-Gen with CGR, it is comparing:

- the SORT-Gen paper
- the CGR paper idea

not:

- SORT-Gen paper versus a fully faithful production-grade CGR implementation in this repository

Second, the current repo still uses bounded candidate enumeration during inference.

That means the model does not directly output the final mixed organic/ad list in one shot. Instead, inference explicitly constructs a small set of feasible candidate lists and scores them.

In practical terms, the current CGR-style inference logic does something like this:

1. start from the organic list
2. try feasible ad insertions or small constrained list variants
3. evaluate each candidate mixed list with the model and reward function
4. select the best candidate found

This is called bounded candidate enumeration because:

- it does not search all permutations of all items
- it only searches a small constrained subset of possibilities
- the search space is kept manageable by assumptions such as very small ad count and strict placement constraints

That is much cheaper than brute-force permutation search, but it is still explicit search over candidate placements. It is not a pure direct generator that emits the final mixed list without constructing candidate alternatives.

This distinction matters because SORT-Gen's main appeal is one-call fast generation, whereas the current CGR repo still behaves more like:

- learned scorer + bounded constrained search

than:

- fully direct one-shot mixed-list generation

So the comparison in this note should be interpreted carefully:

- SORT-Gen is being evaluated as a candidate direction for one-call list generation
- CGR is being treated as the constraint-aware mixed-list idea it represents in the paper
- the local CGR code is useful for understanding that idea, but it is not the final word on what a fully realized CGR system would look like

## Practical Judgment

If the goal is:

- "generate a strong multi-objective ranked list in one serving call"

then 2505.07197 is a sound direction.

If the goal is:

- "generate a mixed organic/ad list in one shot while satisfying strict ad placement constraints"

then 2505.07197 is incomplete as-is.

## How To Adapt It For Mixed Organic/Ad Lists

To use the 2505.07197 style approach for ad/organic feeds, it would likely need additional structure such as:

1. typed candidate queues that distinguish organic, standard ad, large ad, and other ad families
2. constraint-aware mask updates that forbid invalid next-step selections
3. explicit feasibility state in the generator, such as current ad count, spacing state, and remaining legal positions
4. a reward/value definition that includes ad revenue, organic value, user engagement, and ad penalties
5. either hard masking or constrained decoding to guarantee policy compliance

At that point, the method starts to move toward a hybrid of SORT-Gen-style fast generation and CGR-style constrained decoding.

## Adding Organic Profit Margin

One natural extension beyond the current CGR reward is to let organic items contribute direct commerce profit, not just engagement.

In the current implementation, reward is mostly driven by:

- ad monetization
- engagement
- ad exposure penalty

That means organic items mainly matter through engagement terms. If the business goal is to optimize expected organic profit margin as well, then the missing quantity is usually conversion.

### Does This Require CVR?

Usually yes.

If organic profit is realized only when the user converts, then expected organic profit is naturally modeled as something like:

- exposure probability
- click probability
- conversion probability
- profit margin conditional on conversion

Conceptually:

- expected organic profit ≈ `p_exp * p_clk * p_cvr * organic_margin`

So if the system wants to rank by expected organic business value rather than just organic engagement, CVR is the clean missing prediction.

### Two Ways To Add It

There are two practical options.

1. Use upstream CVR.
	Reuse an existing ranking-stage CVR estimate and plug it into the reward function. This is the lowest-complexity way to test whether organic profit margin changes the preferred list in a useful way.

2. Add a third prediction head.
	Extend the listwise model from EXP/CLK to EXP/CLK/CVR and let the reranker learn list-aware conversion effects directly.

The first option is a good product experiment. The second option is the cleaner long-term modeling direction.

### Does This Make The Model More Complex?

Architecturally, only a bit.

Adding one more head is not the difficult part. The harder part is training.

CVR introduces:

- much sparser positive labels than click
- stronger class imbalance
- delayed and noisier supervision
- more conflict between tasks during multitask learning

So the real complexity increase is in optimization and calibration, not in the shape of the network.

### Pareto Tradeoffs In Multi-Task Training

Once EXP, CLK, and CVR are trained together, the training problem becomes a multi-objective optimization problem.

In practice, this usually appears as a weighted multitask loss rather than an explicit Pareto solver:

- `L = lambda_exp * L_exp + lambda_clk * L_clk + lambda_cvr * L_cvr`

That weighted-sum loss is already selecting one tradeoff point among competing tasks.

So Pareto thinking does matter here, but the usual production implementation is still:

- weighted losses
- task-specific heads
- partial parameter sharing or PLE-style expert separation

If CVR is added, task conflict becomes more important, not less.

### Recommended Order Of Attack

If the goal is to move carefully, the best progression is:

1. add upstream CVR into the reward first
2. validate that organic profit margin changes ranking in a useful way
3. only then add a learned CVR head if the upstream estimate is not sufficient

That separates two questions that should not be mixed too early:

- is organic profit margin actually a valuable ranking signal?
- do we need a more complex EXP/CLK/CVR multitask model to capture it well?

## Business Constraints

### Structural constraints enforced by CGR

These constrain where ads go in the list. They are the reason bounded decoding works — with K ≤ 2, the feasible set is small enough to enumerate exhaustively.

| Constraint | Config field | Paper ref | Example |
|---|---|---|---|
| Ad load cap | `max_ads_per_list` | Eq. 6 | ≤ 2 ads per page |
| Minimum spacing | `min_ad_spacing` | Eq. 7 | Ads must be ≥ 3 positions apart |
| Position bounds | `min_ad_position`, `max_ad_position` | §3.4 | Ads only in positions 1-8 (never at the very top) |
| Large-ad cap | `max_large_ads` | §3.4 | At most 1 large-format ad |
| Density limit | `ad_density_limit` | §3.4 | Ads ≤ 30% of page |

These are all hard constraints — every generated list must satisfy all of them. They are checked both during candidate generation (`_hard_constraint_filter`, beam search) and post-hoc (`AdConstraints.is_feasible()`).

### Non-structural constraints handled upstream

These are not CGR's responsibility. They are enforced by other systems in the ad serving pipeline before candidates reach CGR.

| Constraint | Why external |
|---|---|
| User-level frequency cap | Ad auction filters over-exposed ads before they reach CGR |
| Budget pacing | Advertiser daily budget enforced by the auction system |
| Category exclusion ("no competitor ads on same page") | Filtering before candidate set is formed |
| Regulatory / legal labeling ("ad must be labeled") | Enforced by the ad serving policy layer |
| Sensitive content exclusion | Content safety layer upstream of the ranking pipeline |

The paper's design relies on this separation: CGR optimises the structural placement problem (where to insert ads in a fixed organic list), while the upstream pipeline controls which ads are even eligible.

### Inference methods and constraint scaling

| Method | When to use | Guarantee |
|---|---|---|
| Two-stage (`cgr_inference`) | K ≤ 2 | Optimal — exhaustive over feasible set (Theorem 9.1) |
| Beam search (`beam_search_inference`) | Any K | Approximate — can miss global best if beam prunes it |

For K ≤ 2 (Bilibili's production setting), two-stage exhaustive decoding is both fast and optimal. For larger K (e.g. search result pages with many ad slots), beam search is the practical alternative.

## Bottom Line

2505.07197 is a sound method for efficient list-level reranking.

It is not a full one-shot answer to mixed organic/ad feed generation unless additional constraint-aware machinery is added.