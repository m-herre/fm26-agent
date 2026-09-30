from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def stratified_cap(rows: Sequence[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= maximum:
        return list(rows)
    from sklearn.model_selection import train_test_split

    _, sample = train_test_split(
        list(rows),
        test_size=maximum,
        random_state=seed,
        stratify=[int(row["wonderkid"]) for row in rows],
    )
    return sorted(sample, key=lambda row: int(row["player_id"]))
