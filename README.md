# FM26 scouting agent

A terminal proof of concept. DeepSeek queries a static player database, asks a saved hosted
TabPFN model (Prior Labs) to score the matches, and explains a shortlist. No game automation.

## Supported saves

**Only Football Manager 26 saves from build `26.3.2+2329565` (the final FM26 update) can be read.**
Saves from FM25 or earlier are rejected. A save from an older FM26 update is stopped with an
explanation (`--allow-reader-warnings` tries it anyway, but the data may be wrong); try loading it in
the latest FM26 and saving again. This comes from [fmsave](https://pypi.org/project/fmsave/), which
reads saves by reverse-engineered layouts for that one build (currently 0.5.3, the latest release).
`doctor` states the limit, `prepare` prints the build it read, and `inspect_save()` in
`fm26_agent.extract` checks a file's game version without reading the players.

## Setup

All data the tool writes (databases, model references, caches, reports) lives inside this directory.
Configured paths that resolve outside the project (including through symlinks) are rejected, and
so is a `--save` file outside it.

```sh
cd "/Users/markusherre/Code/FM26 Agent"
conda env create -f environment.yml
conda activate fm26-agent
cp config.example.toml config.toml
export DEEPSEEK_API_KEY="your-key"     # or LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
export TABPFN_TOKEN="your-token"
```

Credentials are read only from environment variables. The token stays in-process, so
tabpfn-client does not cache it on disk. Put your `.fm` save in this directory.

```sh
fm26-agent doctor --save <save>.fm
fm26-agent prepare --save <save>.fm --preview   # sampling diagnostics only, no changes, no API
fm26-agent prepare --save <save>.fm             # extract, fit the classifier once; later calls reuse the fit
fm26-agent fit-regression                       # fit the potential regressor chat uses (uploads exact PA)
fm26-agent chat                                 # interactive, ranks by predicted potential
fm26-agent chat --query "Find me five central midfield wonderkids under 20 for at most €8M."
fm26-agent evaluate                             # agent-only vs agent + model on five fixed queries
```

`chat` ranks by **predicted potential** (regression) by default. Options: `--classifier` (rank by
wonderkid probability instead), `--agent-only` (no model), `--held-out` (only the reserved
evaluation players), `--known-values-only` (see below). `evaluate` also takes `--classifier`.

## Players with no market value

About 48% of players (23,931 of 50,202 in the current save) have no value. That is a property of
the save, not a bug in extraction. fmsave reads the raw field and reports its state:

| State | Players | Meaning |
|---|---|---|
| OK | 26,271 | A real stored value |
| ZERO | 23,072 | The save stores 0. Mostly players with no contract: free agents and players at clubs the game does not simulate |
| PLACEHOLDER | 540 | The game's 300,000,000 "unset" sentinel |
| UNSET | 319 | Stub records with no data |

FM works out market value on the fly, so these players do have an in-game value; the save just
does not cache it. The database stores them as `value_eur = NULL` ("unknown", never "worthless"),
and the model sees them as missing.

Value filters keep these players by default. They are returned with `value_known: false`, counted in
`unknown_value_count`, shown as "value unknown (not stored in save)", and the shortlist note says
how many shortlisted players could not be checked against the budget. Pass `--known-values-only`
to drop them instead. On the current save this grows the under-20, ≤€8M pools by about a third
(MC 1,860 → 2,545) without changing the number of true wonderkids in them.

### Euro values

The save stores values in internal units. `money.eur_per_internal_unit = 1.0` is an unverified
placeholder (`money.calibrated = false`), so euro budgets are provisional until you check them
against the game:

```sh
fm26-agent spotcheck                                   # players to look up in the game
fm26-agent calibrate 'Name=4.5M' 'Other Name=12M'      # compute the multiplier, add --apply to save it
```

The multiplier is applied only when results are filtered or shown. The database and both models
always use the raw internal units, so calibrating (or changing the multiplier later) needs **no
refit and no new prepare**. The `value_eur` and `wage_eur` columns keep their names so existing
fits stay valid, but they hold internal units.

## Data and leakage boundary

- `data/players.sqlite3` holds observable fields and the reference membership. Runtime tools use only this.
- `data/private/labels.sqlite3` holds exact PA, the binary target and the split. Only preparation and
  offline evaluation read it. CA is never stored.
- **10,000 reference players** are sampled from the exactly-labelled population (seed 42, natural
  target ratio, approximate marginal balance); all other labelled players are held out. Players with
  unknown PA are excluded from training and evaluation but can still be searched in demo mode.
- TabPFN sees **59 features**: 53 numeric, 5 string categories (club, nation id, natural and
  accomplished position sets, foot) and 1 text column of trait labels. Identity, PA/CA, reputation,
  personality and hidden attributes never enter the model input.
- No manual preprocessing is needed or done. TabPFN takes raw DataFrames: categories, text and
  missing values are handled by the model, with no encoding, imputation, scaling or outlier
  removal. The code only selects the allowed columns, marks strings as categories and leaves
  missing values missing.
- Fitting the classifier uploads the feature matrix and binary labels to Prior Labs. The regressor
  (`fit-regression`, the default for `chat`) also uploads **exact PA as the target**, which you have
  authorised explicitly.
  Prior Labs may retain either fitted context. Actual PA is never sent to DeepSeek or returned by a tool.
- The save, both databases and all run artifacts are gitignored and stay local.

## How the agent works

- Bounds are inclusive ("under 20" is `age_max=19`). A position matches natural **and** accomplished labels.
- `search_players` returns a `search_id`, the total count, `unknown_value_count` and up to 500 rows.
  Passing the `search_id` to the prediction tool scores **every** match in one request, reuses the
  saved fit and a local score cache, and returns only the global top-k. A failure never returns a partial shortlist.
- At most 8 tool rounds, then up to 2 format-only retries in JSON mode. Wrong constraints, invented
  IDs, ranking mistakes and eligibility violations are errors, never silently repaired.
- `chat` defaults to the full-save demo pool and labels training-reference players; `--held-out`
  and `evaluate` use only held-out players. Follow-up requests are independent.
- DeepSeek thinking is disabled by default (`[llm] thinking = true` to enable).

## What to expect from the model

The goal is a shortlist of players whose potential lands in the right range, not an exact PA for
everyone. Exact answers would take the discovery out of the game. On the held-out set (5,000
players, 60 true wonderkids) the current fits are well past that bar:

| Model | Result |
|---|---|
| Classifier | ROC AUC 0.97; its top 10 are all true wonderkids |
| Regressor | Average error about 9 PA points, R² 0.88 |

Expect the ranking to be good and individual estimates to be off by roughly ±10 PA. In narrow
pools (for example cheap under-20 central midfielders) even the best candidates sit below the
wonderkid line, and the agent says so rather than forcing a match.

## Evaluation and experiments

`evaluate` runs five fixed queries (MC, STC, DC, AML, GK; under 20, ≤€8M, top five) with and without
the model, plus a model-only ranking of each complete eligible pool. It reports ROC AUC, average
precision, Precision/Recall@5/10, true-wonderkid counts and average hidden PA. Pools with no true
wonderkids cannot show any improvement. These numbers measure within-save generalisation only.

```sh
fm26-agent fit-regression          # the regressor chat uses by default; same reference, classifier untouched
fm26-agent compare-models          # offline classifier vs regressor on identical held-out pools
```

Fits are reused; filter or agent changes never refit. `--refit` replaces only its own fit. Reports
go to `runs/`. A cost estimate prints before any new fit.

## Tests

```sh
pytest -m "not hosted and not integration"            # offline: synthetic fixtures, mocked APIs
FM26_TEST_SAVE=<save>.fm pytest -m integration        # reads the real save, never modifies it
FM26_RUN_HOSTED_TESTS=1 pytest -m hosted              # uses Prior Labs credits
```

pytest writes its temp files to `.pytest-tmp/` inside this directory.

## Toward a web app

The core is already separate from the terminal. `fm26_agent.runtime` (`open_runtime`, `scout`,
`write_report`) and `prepare` return data and report progress through callbacks (`trace`, `emit`)
instead of printing, so a web front end can call them directly. The intended shape is one
workspace directory per uploaded save, each with its own `config.toml`, databases and model
references; the containment check stops a workspace from reading or writing outside itself.

Planned: at the start a user provides their own DeepSeek and TabPFN API keys, or runs TabPFN
locally instead. That would take quota, credentials and upload consent off the operator.

Still open:

- `prepare` fits per save and takes long enough to need a job queue.
- Uploads must be validated up front (`inspect_save`), and the currency multiplier needs a per-save
  calibration story.
- Saves are large (about 600 MB here): upload limits, storage and cleanup.
- Per-user isolation of caches and model references, and request limits.

## Known limits

- PA comes from fmsave's reverse-engineered reader; spot-check a few players in an editor before
  treating metrics as evidence.
- The under-20, ≤€8M pools hold very few true wonderkids (MC 1, DC 3, STC 6, AML 12, GK 0 in the
  held-out set), so those queries cannot separate a good model from a poor one. Use larger pools
  for model comparisons.
