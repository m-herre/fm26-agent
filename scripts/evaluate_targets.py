"""How predictable is each hidden value from what a scout can see?

For every target in the glossary (targets.py) this builds the same TabPFN task the agent would
build (one fit on the 10,000 reference players, checked on 2,000 held-out players; stored fits are
reused), and fits scikit-learn gradient boosting on exactly the same players for comparison.
Writes docs/targets.json and docs/targets.svg.

    python scripts/evaluate_targets.py --demo     # the included sample (local TabPFN: free)
    python scripts/evaluate_targets.py            # the save that is set up
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_model import INK, MUTED, OTHER, TABPFN, _svg, boosting  # noqa: E402

from fm26_agent.config import demo_settings, load_settings  # noqa: E402
from fm26_agent.custom_tasks import TaskLab, TaskSpec, check_sample  # noqa: E402
from fm26_agent.features import FeatureSchema  # noqa: E402
from fm26_agent.keys import load_env_file  # noqa: E402
from fm26_agent.private_db import PrivateStore  # noqa: E402
from fm26_agent.tabpfn_backend import use_project_weights  # noqa: E402
from fm26_agent.targets import TARGETS  # noqa: E402
from fm26_agent.visible_db import VisibleStore  # noqa: E402


def chart(rows: list[dict]) -> str:
    """Share of error removed compared with guessing the average, TabPFN vs boosting."""
    body, top, left, width = [], 70, 190, 330
    body.append(
        f'<text x="20" y="50" fill="{MUTED}">share of the error removed compared with guessing '
        "the average (held-out players)</text>"
    )
    y = top
    for row in rows:
        body.append(
            f'<text x="{left - 10}" y="{y + 15}" text-anchor="end" fill="{INK}">{row["label"]}</text>'
        )
        for offset, value, colour in (
            (0, row["tabpfn"]["better_than_guessing"], TABPFN),
            (12, row["gradient_boosting"]["better_than_guessing"], OTHER),
        ):
            bar = max(0.0, value) * width
            body.append(
                f'<rect x="{left}" y="{y + offset}" width="{bar:.1f}" height="10" rx="2" fill="{colour}"/>'
            )
            body.append(
                f'<text x="{left + bar + 6:.1f}" y="{y + offset + 9}" font-size="11" fill="{INK}">{value:.0%}</text>'
            )
        y += 32
    body += [
        f'<rect x="{left}" y="{y + 6}" width="12" height="10" fill="{TABPFN}"/>',
        f'<text x="{left + 18}" y="{y + 15}" fill="{INK}">TabPFN-3.5</text>',
        f'<rect x="{left + 120}" y="{y + 6}" width="12" height="10" fill="{OTHER}"/>',
        f'<text x="{left + 138}" y="{y + 15}" fill="{INK}">Gradient boosting</text>',
    ]
    return _svg(600, y + 30, body, "What the agent can predict, and how well")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--charts", default="docs")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--only", nargs="*", help="evaluate just these targets")
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.demo:
        settings = demo_settings(settings)
    load_env_file(settings.project_root)
    use_project_weights(settings.project_root)
    visible = VisibleStore(settings.data.visible_database)
    private = PrivateStore(settings.data.private_database)
    if not private.has_targets():
        print("This setup has no hidden targets; run fm26-agent prepare once.", file=sys.stderr)
        return 1
    seed = settings.training.random_seed
    lab = TaskLab(
        visible,
        private,
        FeatureSchema.load(settings.data.feature_schema),
        settings.tabpfn_backend,
        settings.data.model_reference.parent / "tasks",
        seed=seed,
        progress=lambda message: print(message, flush=True),
    )
    rows = []
    for name in args.only or lab.available():
        started = time.monotonic()
        report = lab.build(TaskSpec(name))
        seconds = time.monotonic() - started
        train = private.target_values(name, "train")
        held_out = private.target_values(name, "test")
        check = check_sample(held_out, seed)
        train_ids = sorted(train)
        y = np.array([train[i] for i in train_ids], dtype=float)
        actual = np.array([held_out[i] for i in check], dtype=float)
        point, low, high = boosting(visible.get_players(train_ids), y, visible.get_players(check))
        point = np.clip(point, *TARGETS[name].scale)
        error = float(np.mean(np.abs(point - actual)))
        guess = float(np.mean(np.abs(y.mean() - actual)))
        rows.append(
            {
                "target": name,
                "label": TARGETS[name].label,
                "scale": list(TARGETS[name].scale),
                "tabpfn": report,
                "tabpfn_seconds": round(seconds, 1),
                "gradient_boosting": {
                    "average_error": round(error, 2),
                    "better_than_guessing": round(1 - error / guess, 3) if guess else 0.0,
                    "range_coverage_80": round(
                        float(np.mean((actual >= low) & (actual <= high))), 3
                    ),
                },
            }
        )
        print(
            f"{name:>18}: TabPFN {report['average_error']:.2f} ({report['better_than_guessing']:.0%}"
            f" better than guessing, {report['verdict']}), boosting {error:.2f}",
            flush=True,
        )
    rows.sort(key=lambda row: -row["tabpfn"]["better_than_guessing"])
    charts = Path(args.charts)
    charts.mkdir(parents=True, exist_ok=True)
    result = {
        "source": "sample" if args.demo else visible.metadata().get("source"),
        "backend": settings.tabpfn_backend,
        "targets": rows,
    }
    (charts / "targets.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (charts / "targets.svg").write_text(chart(rows), encoding="utf-8")
    print(f"Wrote {charts}/targets.json and {charts}/targets.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
