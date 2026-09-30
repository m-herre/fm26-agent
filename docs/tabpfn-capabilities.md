# TabPFN capabilities: what we use, what's next

Status as of v1.6.0 (30 September 2026).

## Used today

| Capability | Where |
|---|---|
| Regression, no training loop or tuning | Potential model and market-value model, each fitted once on the raw table |
| Mixed raw inputs | 53 numbers, 5 categories (club, nation, foot, positions), 1 free-text column (traits); missing values left as they are |
| Full predictive distribution (quantiles) | One prediction gives the median estimate, the 80% range and the chance of reaching 160+; drives the ranking modes (expected, ceiling, safe) |
| Calibrated uncertainty | 10,000 unseen players: ranges hold the truth 82% of the time; star chances have AUC 0.943 and match what really happens |
| Fit once, predict many (`fit_with_cache`) | One fit per save, then whole candidate pools (4,000+ players) per search |
| Small context, strong result | 10,000 training players beat gradient boosting on the same data: 8.6 vs 10.7 average error |
| Local and hosted, same model | `tabpfn` on MPS or CUDA, or `tabpfn-client` at Prior Labs (`tabpfn_backend.py`); fitted state saved and reloaded |
| Imputation-style use | The value model fills in the market values the save lacks and checks itself on 2,000 held-out known values |
| Probabilistic conditions | Planning mode: every condition in an agreed objective needs a minimum chance ("potential 160+, at least 25% likely"), read off the target's predicted distribution |
| Cross-fitting | Fair value: each half of the priced players is valued by a fit on the other half, so no player is valued by a model that saw his price (39.4% typical error vs 51.6% for boosting) |
| Models made on demand | The agent defines its own task (16 hidden targets); TabPFN fits it mid-conversation, self-checks on 2,000 held-out players and reports a verdict (useful, weak, not predictable) |

## Predictive tasks

Two fixed models, up to 16 the agent builds, and fair value (two cross-fitted models):

1. Potential (PA) regression, the main model.
2. Market-value regression on log value, the second model.
3. Chance of reaching 160+: works like a classification, read off model 1's distribution (no separate classifier).
4. Agent-built regressions on any glossary target (current ability, room to grow, 5 hidden
   attributes, 8 personality traits), each with its own threshold chance. On the demo 7 are
   useful, 2 weak and 7 not predictable; TabPFN beats boosting on all 16 (docs/targets.json).

The ranges and the upside/safe rankings come from model 1's distribution, so they aren't separate tasks.

Tried and dropped: a wonderkid classifier (removed in V1), per-query fits on the matching slice
(worse: 10.4 vs 8.9 for midfielders under 21), and ensembles of five contexts (9.01 vs 9.09, not
worth it).

## Not shown yet

In the APIs we use (parameters checked):
- **Thinking mode** (`thinking_mode`, `thinking_effort`), hosted only. Named in the hackathon
  rules. On hold.
- **Time and group columns** (`time_col`, `group_col`) in the hosted client. No use without
  multi-season player history, which the save doesn't give us.
- **Classification** (`TabPFNClassifier`): not used.
- **Cost estimation** (`estimate_cost`): could show users what a hosted question costs.

From Prior Labs documentation and the extensions package (not yet checked against our installed
versions):
- **Interpretability** (SHAP-style attributions): real reasons for each pick, e.g. "vision and age
  explain most of his potential", instead of LLM wording.
- **Unsupervised** (outliers, density, synthetic data): outliers as "hidden gems"; a **synthetic
  demo dataset** instead of real-save players, which also settles the question of publishing
  game data.
- **Embeddings:** could replace the plain attribute cosine behind "players like X" (the local
  package may offer them; the hosted client doesn't).
- **Time-series forecasting** (TabPFN-TS): development curves; needs multi-season history.
- **Relational data** (TabPFN-Rel, on the hackathon page): players, clubs and competitions as
  linked tables; fits the "Showcase a harness" track. A big step.
- Fine-tuning and many-class extensions: not relevant here.

## Suggested next steps for the hackathon

0. Done in v1.6.0: **planning mode** (questions, objective card, "go", deterministic execution)
   and **fair value** for "undervalued".

1. **Explain each pick with interpretability.** The best showcase value for the effort.
2. **Bargain finder:** stored value vs the value model's estimate to find underpriced players. No
   new model needed.
3. **Synthetic demo dataset:** a further capability, and it fixes reproducibility and data rights.

Also open: the web demo (point 1, planned), making the repo public, and thinking mode (on hold).
