"""Developer tool: write the prepared players to sample/players.csv.gz for `fm26-agent --demo`.

Needs a save that is already set up (run fm26-agent first). Player names are replaced with
made-up ones unless --keep-names is given; real potential is included because it is what the
model learns from. Decide for yourself whether a file made with --keep-names may be shared.

    python scripts/export_sample.py
"""

from __future__ import annotations

import argparse

from fm26_agent.config import ensure_inside, load_settings
from fm26_agent.private_db import PrivateStore
from fm26_agent.sample import export_sample
from fm26_agent.visible_db import VisibleStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="sample/players.csv.gz")
    parser.add_argument("--keep-names", action="store_true", help="keep the real player names")
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    out = ensure_inside(settings.project_root, args.out, "--out")
    count = export_sample(
        VisibleStore(settings.data.visible_database),
        PrivateStore(settings.data.private_database),
        out,
        keep_names=args.keep_names,
    )
    print(f"Wrote {count:,} players to {out.relative_to(settings.project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
