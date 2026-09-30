# FM26 scouting assistant

Ask for players in plain words and get a short list with the reasons, based on your own
Football Manager 26 save.

> "five young central midfielders under €8M"
> "best left-footed wingers under 21"
> "goalkeepers at Real Madrid or Barcelona"
> "a striker on an expiring contract who could become world class"

It works out each player's hidden **potential** from what you can see (attributes, age, position,
traits), so you can find prospects without opening every profile. Potential is an estimate,
off by about 9 points on average (scale 1-200): two thirds of players land within 10 points of
the real value, four in five within 15.

## Getting started

```sh
conda env create -f environment.yml    # once
conda activate fm26-agent
fm26-agent                             # then just follow the questions
```

The first run asks for two keys (it tells you where to get them and remembers them), finds your
save, and sets it up. That takes a minute or two, once per save. After that you can start asking.
Put your `.fm` save file in this folder, or let it offer to copy one in from the game.

You need:
- a **DeepSeek key**, which understands your questions (platform.deepseek.com)
- a **TabPFN key**, which estimates potential (platform.priorlabs.ai)

Keys are stored in a `.env` file in this folder. Keep that file private.

## Supported saves

**Only Football Manager 26 saves from build `26.3.2+2329565` (the final FM26 update) can be read.**
Saves from FM25 or earlier cannot. If your save comes from an older FM26 update, load it in the
latest FM26 and save it again. The tool checks this and explains if a save isn't supported.

## Good to know

- **Prices are approximate** until you match them to your game. The save stores values in its own
  units, so in the game compare two or three players' values with `fm26-agent spotcheck` and run
  `fm26-agent calibrate 'Name=4.5M' 'Other Name=12M' --apply` (setup also offers this). Nothing has
  to be redone afterwards.
- **Some players have no price.** About half the players in a save (free agents, clubs the game
  isn't simulating) have no stored market value. They are kept in budget searches and shown as
  "value not in save", so good prospects aren't lost.
- **What it can filter:** age, value, position, club (one or several), stronger foot and contract
  expiry. Nationality, league and wage can't be filtered; it says so instead of guessing.
- **Privacy:** your questions and the details of the players it discusses go to DeepSeek. The
  potential model is taught with the attributes and exact potential of 10,000 sample players sent
  to Prior Labs. Nothing else leaves your computer; the save, databases and logs stay in this folder.
- **Season stats:** when the save has them, each pick shows this season's games, goals, assists and
  average rating (clean sheets for goalkeepers). They are only shown, never used to rank, and many
  players, especially youth players and those at clubs the game isn't simulating, have none.
- Potential only matters for players who are still developing, so questions about "prospects" or
  "wonderkids" are treated as young players (under 22) unless you give an age.

Other commands: `fm26-agent doctor` (checks everything), `fm26-agent --query "..."` (one question),
`fm26-agent prepare --save file.fm --refit` (redo a setup).

## For developers

```sh
pytest -m "not hosted and not integration"                     # offline, mocked services
python scripts/evaluate_model.py                               # how accurate is potential? (needs the TabPFN key)
FM26_TEST_SAVE=<save>.fm pytest -m integration                 # reads a real save, never changes it
FM26_RUN_HOSTED_TESTS=1 FM26_RUN_AGENT_SMOKE=1 pytest -m hosted        # needs both keys
FM26_RUN_FULL_SETUP=1 FM26_TEST_SAVE=<save>.fm pytest -m "hosted and integration"   # a real first run
```

How it fits together:
- `extract.py` reads the save with [fmsave](https://pypi.org/project/fmsave/) and checks the game
  version and that potential is plausible (never below current ability, which the game guarantees).
  Current ability is never stored or modelled.
- `prepare.py` picks 10,000 representative players and fits a TabPFN regressor once
  (`prediction.py`). Potential is the training target only, never an input. TabPFN needs no manual
  preprocessing: it takes the raw table and handles categories, text and missing values itself.
- `tools.py` and `agent.py` let DeepSeek search the save and call the model. Every answer is
  checked in code: constraints, eligibility, completeness and ranking. Wrong answers are rejected,
  never silently repaired.
- `app.py` is the guided experience, `runtime.py` is what another front end (such as a web app)
  would call; neither `runtime.py` nor `prepare.py` prints or reads input.
- The euro multiplier is applied only when results are filtered or shown. The database and model
  always use the save's raw units.
- Everything the tool writes stays inside this folder; paths outside it are refused.

Planned: a web app where a user uploads a save and supplies their own keys, or runs TabPFN locally.
`inspect_save()` already checks a file's game version without reading any players.
