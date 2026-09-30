# FM26 Scout: a TabPFN-3.5 scouting agent for Football Manager 26

Every player in Football Manager has a hidden **potential ability** (PA, 1-200): how good they can
ever become. The game never shows it. Players spend hours scouting to guess it.

This project turns a Football Manager 26 save file into a table of 50,000 players and lets
**TabPFN-3.5** learn hidden potential from what you can see: 48 attributes, age, position, club,
contract, traits. An LLM agent turns plain questions into searches, and TabPFN scores **every**
matching player. You don't just get a number. For each pick you get:

- **an estimate**, the median of TabPFN's predictive distribution
- **a likely range**, the 10th to 90th percentile, which really holds the truth 83% of the time
- **a star chance**, the probability of reaching 160+ PA, read off the same distribution. On
  unseen players it matches what really happens (chart below).

All three come from **one TabPFN request per question**, for the whole candidate pool.

```text
What are you looking for? five wonderkid central midfielders under 21, max €20M
  Searching your save...
  Estimating potential for 3,939 players...
  Taking a closer look at the best candidates...

1. Sami Pavlovic · 19 · Brighton & Hove Albion · €18.7M
   Potential ≈ 179 (likely 164–183) · 93% chance of reaching 160+
   A 19-year-old left-footed midfielder who already looks close to the finished article:
   outstanding vision and technique, strong teamwork and stamina, and a habit of pushing forward.
...
5. Dario Delgado · 15 · GNK Dinamo Zagreb · €3.5M
   Potential ≈ 162 (likely 157–164) · 77% chance of reaching 160+
```

Ask for the players with the **most upside** and it ranks by the top of each range instead
("boom or bust"). Ask for **safe** picks and it ranks by the bottom. The same scores are used, so
no extra TabPFN call is needed:

```text
What are you looking for? same search, but the three with the biggest upside, boom or bust is fine
2. Dani Duarte · 17 · Wattener SG Tirol · €19.0M
   Potential ≈ 157 (likely 140–179) · 42% chance of reaching 160+
   ... The range is wide, so he could become a top midfielder or stall short of it.
Ranked by best case: the top of each player's range.
```

## Try it in five minutes (no Football Manager needed)

The repository includes `sample/players.csv.gz`, all 50,202 players of a real FM26 save with
made-up names, plus their real potential as the training target.

```sh
conda env create -f environment.yml && conda activate fm26-agent
# or: python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
fm26-agent --demo
```

The first run asks for two keys and remembers them in `.env`:
- a **TabPFN key** (Prior Labs, [platform.priorlabs.ai](https://platform.priorlabs.ai/account/api-keys)), which fits the model and scores players
- a **DeepSeek key** ([platform.deepseek.com](https://platform.deepseek.com/api_keys)), which understands your questions

Setup fits TabPFN once, which takes about 30 seconds. Then ask anything, for example:

> "best left-footed wingers under 21" · "goalkeepers at Real Madrid or Barcelona" ·
> "a striker on an expiring contract who could become world class" · "three safe centre-backs under 23"

To reproduce the numbers and charts below, run `python scripts/evaluate_model.py --demo`
(one TabPFN request).

With your own game, put an FM26 save (`.fm`, build `26.3.2+2329565`) in this folder and run
`fm26-agent`. It finds the save, reads it with [fmsave](https://pypi.org/project/fmsave/) and sets
it up.

## How TabPFN-3.5 is used

| Step | What TabPFN does | Why it matters here |
|---|---|---|
| **Setup** (once per save) | `TabPFNRegressor(v3.5)` is fitted on 10,000 representative players with `fit_with_cache` and saved | A foundation model needs no training loop or tuning. The raw table goes in as it is: 53 numbers, 5 categories (club, nation, foot, positions) and a free-text trait list via `text_handling="advanced"`. Missing values are left missing. |
| **Every question** | One `predict(output_type="quantiles")` call returns 19 percentiles for every matching player, often 4,000+ in one request | The estimate, 80% range and star chance all come from this one distribution. Results are cached per player, so re-ranking and follow-up questions cost nothing. |
| **Agent tools** | The LLM calls `predict_player_potential` with a `search_id` and a ranking mode (`expected`, `ceiling`, `safe`) | Uncertainty isn't just shown, it changes the ranking. "Upside" and "safe bet" are real scouting styles. |

The 10,000 training players are chosen to mirror the whole save: the same share of wonderkids
(PA 160+), and the same mix of every input (age, positions, value, club, attributes). Real potential is **only the training
target**, never an input. Current ability and the game's hidden attributes are never used.

## How good is it?

These are players the model never saw: 10,000 held out from the demo save. For comparison,
scikit-learn gradient boosting was fitted on the **same** 10,000 training players, with its own
quantile models for 80% ranges.

![TabPFN vs gradient boosting](docs/tabpfn-vs-boosting.svg)

| | TabPFN-3.5 | Gradient boosting |
|---|---|---|
| Average error | **8.75** points | 10.67 |
| Within 10 points | **68%** | 57% |
| R² | **0.876** | 0.836 |
| 80% range really holds the truth | **83%** | 70% |
| Tuning, feature engineering | none | frequency-encoded categories, hand-set trees |

Guessing the average potential for everyone would be off by 28.9 points.

**The star chances are honest.** Players TabPFN gave a 5–20% chance reached 160+ 11% of the time.
Those given 80–100% did so 97% of the time. AUC is 0.944, and the Brier score is half that of
always quoting the base rate.

![Star chance reliability](docs/star-chances.svg)

**Wider ranges mean real risk.** The model knows which players it is unsure about:

![Range width vs error](docs/range-width.svg)

More results from the same run:
- **Top picks:** the 25 players it rates highest (all ages) all really have 160+ potential,
  averaging 177.8 (the best possible 25 average 183.4). For under-22s, 8 of its top 10 are real
  wonderkids.
- **By age:** the average error is 8.2–9.2 in every age band (15–18: 9.2, 19–21: 8.2).
- **Missing values:** about half the players have no market value in the save. The error for them
  is 9.7, against 7.9 for players with one. They are kept in searches and flagged "value not in
  save".
- **Full numbers:** [docs/results.json](docs/results.json).

What didn't help: fitting a separate TabPFN per query on only the matching slice was worse (for
example 10.4 vs 8.9 average error for midfielders under 21), because fewer rows hurt more than
relevance helps. Averaging five fits on different 10,000-player contexts gave 9.01 vs 9.09. The
single fitted model is the right trade.

## How the agent stays honest

The LLM (DeepSeek, OpenAI-compatible) plans; code checks every answer before you see it:

1. It turns the question into filters: age, value, position, club (one or several), stronger foot
   and contract expiry. Anything it can't filter (nationality, league, wage) it says so instead of
   guessing.
2. `search_players` returns a handle for the **complete** matching pool.
   `predict_player_potential` scores all of it in one TabPFN request, not just the first page.
3. The final answer must be JSON matching a schema. Code then checks that the constraints match
   the search, every pick is eligible, the whole pool was scored, and the order matches the chosen
   ranking mode. A wrong answer is sent back to the agent with the reason, and never silently
   repaired.
4. Names, clubs, prices, estimates, ranges and this season's stats (games, goals, assists, rating)
   are printed by code, not by the LLM, so the numbers can't be made up.

```text
save.fm ──fmsave──▶ 50k players ─┬─▶ players.sqlite3 (what you can see + season stats)
                                 └─▶ labels.sqlite3  (real potential, training only)
                                          │
              10k stratified sample ──▶ TabPFN-3.5 fit (once) ──▶ model.json
                                                                     │
question ─▶ DeepSeek agent ─▶ search_players ─▶ predict_player_potential ─▶ checked shortlist
                                  (all matches)      (1 TabPFN call: median, range, star chance)
```

## Good to know

- **Prices:** the save stores values in its own units. Match them to your game with
  `fm26-agent calibrate 'Name=4.5M' 'Other Name=12M' --apply` (`fm26-agent spotcheck` suggests who
  to look up). Nothing has to be refitted.
- **Supported saves:** only FM26 build `26.3.2+2329565`, the final update. The tool checks this and
  explains if a save can't be read. A save from an older FM26 update can be loaded and saved again
  in the latest game.
- **Prospect questions:** potential only matters for players who are still developing, so
  "prospect" or "wonderkid" without an age means under 22.
- **Privacy:** your questions and the players discussed go to DeepSeek. The 10,000 training
  players go to Prior Labs. Everything else stays in this folder.
- **Cost:** one TabPFN request per new search. Repeat and re-ranked searches come from the local
  cache.

## For developers

```sh
pytest -m "not hosted and not integration"      # offline tests, all services mocked
python scripts/evaluate_model.py --demo         # accuracy report + charts (1 TabPFN call)
python scripts/evaluate_model.py --redraw       # redraw the charts from docs/results.json
python scripts/export_sample.py                 # rebuild sample/ from a prepared save
FM26_RUN_HOSTED_TESTS=1 FM26_RUN_AGENT_SMOKE=1 pytest -m hosted    # live services, needs both keys
FM26_TEST_SAVE=<save>.fm pytest -m integration                      # reads a real save, read-only
```

| File | Role |
|---|---|
| `extract.py` | Reads the save with fmsave. Checks the game build and that potential is plausible (never below current ability). Current ability is only compared, never stored. |
| `sampling.py`, `prepare.py` | Picks the 10,000 training players and fits TabPFN once. |
| `features.py`, `prediction.py` | The raw feature table. One quantile call gives the estimate, range and star chance. |
| `tools.py`, `agent.py` | The agent's tools, ranking modes and answer checks. |
| `app.py`, `cli.py` | The guided command-line experience. `runtime.py` is what a web front end would call. |
| `sample.py` | The portable demo dataset. |

Everything the tool writes stays inside this folder; `--demo` keeps its data in `data/demo`, so it
never replaces your own setup.

## Licence

Apache License 2.0, see [LICENSE](LICENSE). Football Manager is a trademark of Sports Interactive
and SEGA; this project is not affiliated with them.
