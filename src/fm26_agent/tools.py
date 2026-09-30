from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import Currency
from .custom_tasks import CHANCE, ESTIMATE, HIGH, LOW, TaskLab, TaskSpec
from .prediction import CHANCE_FIELD, HIGH_FIELD, LOW_FIELD, SCORE_BOUNDS, SCORE_FIELD, Predictor
from .schema import VISIBLE_ATTRIBUTES
from .targets import glossary
from .visible_db import VisibleStore, scale_money

PREDICTION_TOOL = "predict_player_potential"
BUILD_TASK_TOOL = "build_prediction_task"
TASK_PREDICTION_TOOL = "predict_with_task"
RANKINGS = ("expected", "ceiling", "safe", "chance")


@dataclass
class TaskScores:
    """What one agent-built task has predicted so far in this request."""

    spec: TaskSpec
    report: dict[str, Any]
    scores: dict[int, float] = field(default_factory=dict)
    intervals: dict[int, tuple[float, float]] = field(default_factory=dict)
    chances: dict[int, float] = field(default_factory=dict)


SEARCH_PROPERTIES = {
    "age_min": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
    "age_max": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
    "value_min_eur": {"type": ["number", "null"], "minimum": 0},
    "value_max_eur": {"type": ["number", "null"], "minimum": 0},
    "position": {
        "type": ["string", "array", "null"],
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 4,
        "description": 'Canonical FM code such as MC, STC, DC, AML or GK, or a list to match ANY of them (e.g. ["AML", "AMR"] for wingers). Matches BOTH natural and accomplished labels, never natural-only.',
    },
    "height_min_cm": {"type": ["integer", "null"], "minimum": 150, "maximum": 210},
    "height_max_cm": {"type": ["integer", "null"], "minimum": 150, "maximum": 210},
    "min_attributes": {
        "type": ["object", "null"],
        "propertyNames": {"enum": list(VISIBLE_ATTRIBUTES)},
        "additionalProperties": {"type": "number", "minimum": 1, "maximum": 20},
        "maxProperties": 6,
        "description": 'Visible attributes (1-20) the player must have at least, e.g. {"pace": 16, "heading": 14}.',
    },
    "club": {
        "type": ["string", "array", "null"],
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "description": "A club name, or a list of names to match ANY of them. Partial names work.",
    },
    "preferred_foot": {
        "type": ["string", "null"],
        "enum": ["left", "right", "both", None],
        "description": "The player's stronger foot; 'both' means genuinely two-footed.",
    },
    "contract_ends_within_days": {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": 3650,
        "description": "Only players whose contract ends within this many days of the save date.",
    },
    "similar_to": {
        "type": ["integer", "null"],
        "description": "A player_id from find_player. Keeps only the 100 players whose visible attribute profile looks most like that player's and who play one of his natural positions; each match gets a profile_match score (0-1). Combine with other filters for e.g. a cheaper or younger version.",
    },
    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 500},
    "offset": {
        "type": "integer",
        "minimum": 0,
        "default": 0,
        "description": "Stable page offset. Use next_offset to retrieve the remaining matches.",
    },
}


def _function(
    name: str, description: str, properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def tool_schemas(
    predictions_enabled: bool = True,
    include_unknown_value: bool = True,
    task_targets: list[str] | None = None,
) -> list[dict[str, Any]]:
    unknown_value_policy = (
        "When the save stores no market value (value_known=false), budget filters use TabPFN's "
        "estimated value instead (estimated_value_count); players with neither are INCLUDED and "
        "flagged (unknown_value_count)."
        if include_unknown_value
        else "When the save stores no market value, budget filters use TabPFN's estimated value; "
        "players with neither fail budget filters."
    )
    tools = [
        _function(
            "get_database_summary",
            "Inspect save date, player counts, available filters and field coverage.",
            {},
        ),
        _function(
            "search_players",
            "Find players using inclusive bounds. Returns a query-scoped search_id and one page of at most 500 records, with next_offset, total matching_count and unknown_value_count. "
            + unknown_value_policy
            + f" For potential rankings pass search_id to {PREDICTION_TOOL}: it scores EVERY match, not just this page.",
            SEARCH_PROPERTIES,
        ),
        _function(
            "find_player",
            "Look a player up by name (exact, else partial; at most 10 results) to get his player_id, for example for similar_to.",
            {"name": {"type": "string", "minLength": 2, "maxLength": 80}},
            ["name"],
        ),
        _function(
            "get_player_details",
            "Inspect observable attributes of up to 25 players, plus this season's stats (season_stats: appearances, minutes, goals, assists, average_rating) when the save has them, otherwise null. No hidden variables are available.",
            {
                "player_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "maxItems": 25,
                    "uniqueItems": True,
                },
            },
            ["player_ids"],
        ),
    ]
    if predictions_enabled:
        tools.append(
            _function(
                PREDICTION_TOOL,
                "Estimate each player's potential ability (1-200) with the saved TabPFN model; never refits. Prefer search_id: score EVERY matching player in one operation and return the global leaders plus coverage counts. Alternatively player_ids scores only those explicit IDs (at most 500). Results are estimates of hidden potential, not facts. Cached scores are reused. Every result also carries potential_low and potential_high, the range the real value should fall in about 4 times out of 5, and star_chance, the chance (0-1) of reaching 160+ potential. rank_by picks the order of the leaders: expected (highest predicted_potential, the default), ceiling (highest potential_high, for upside and boom-or-bust) or safe (highest potential_low, for reliable picks).",
                {
                    "player_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 500,
                        "uniqueItems": True,
                    },
                    "search_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "An opaque search handle returned by this request's search_players call; scores the complete matching pool.",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 25, "default": 25},
                    "rank_by": {"type": "string", "enum": list(RANKINGS), "default": "expected"},
                },
            )
        )
        tools[-1]["function"]["parameters"]["anyOf"] = [
            {"required": ["player_ids"]},
            {"required": ["search_id"]},
        ]
    if task_targets:
        tools.extend(task_schemas(task_targets))
    return tools


def task_schemas(targets: list[str]) -> list[dict[str, Any]]:
    return [
        _function(
            BUILD_TASK_TOOL,
            "Define your own prediction task when the request is about something other than "
            "potential. TabPFN learns the chosen hidden target from the visible data of 10,000 "
            "reference players on the spot (seconds when it runs locally), checks itself on "
            "2,000 held-out players and returns a task_id plus a quality report: average_error, "
            "average_error_if_guessing, better_than_guessing (share of error removed) and a "
            "verdict (useful, weak, not predictable). You never see the training data. The same "
            "task is reused for free once built. With a threshold, every prediction also carries "
            "the chance of meeting it (at least the threshold, or at most for targets where low "
            "is good). Targets:\n" + glossary(targets),
            {
                "target": {"type": "string", "enum": targets},
                "threshold": {
                    "type": ["number", "null"],
                    "description": "Optional level that counts as good enough, on the target's scale.",
                },
            },
            ["target"],
        ),
        _function(
            TASK_PREDICTION_TOOL,
            "Score EVERY player in a search with a task from build_prediction_task and return the "
            "leaders in one operation. Each result has estimate, low and high (the range the true "
            "value falls in about 4 times out of 5) and, for tasks with a threshold, chance. "
            "rank_by: expected (best estimate, respecting whether high or low is good), ceiling "
            "(best case), safe (worst case) or chance (most likely to meet the threshold; needs a "
            "threshold).",
            {
                "task_id": {"type": "string", "minLength": 1},
                "search_id": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 25, "default": 25},
                "rank_by": {"type": "string", "enum": list(RANKINGS), "default": "expected"},
            },
            ["task_id", "search_id"],
        ),
    ]


class ScoutingTools:
    def __init__(
        self,
        store: VisibleStore,
        predictor: Predictor | None = None,
        *,
        include_unknown_value: bool = True,
        currency: Currency | None = None,
        star_level: int = 160,
        lab: TaskLab | None = None,
    ):
        self.store = store
        self.lab = lab
        self.tasks: dict[str, TaskScores] = {}
        self.currency = currency or Currency()
        self.scale = self.currency.eur_per_internal_unit
        self.predictor = predictor
        self.include_unknown_value = include_unknown_value
        self.searched_ids: set[int] = set()
        self.star_level = star_level
        self.scores: dict[int, float] = {}
        self.intervals: dict[int, tuple[float, float]] = {}
        self.chances: dict[int, float] = {}
        self.profile_matches: dict[int, float] = {}
        self.searches: list[dict[str, Any]] = []
        self.queries: dict[str, dict[str, Any]] = {}
        self.prediction_operations: list[dict[str, Any]] = []
        self.progress: Callable[[str], None] | None = None

    @property
    def schemas(self) -> list[dict[str, Any]]:
        targets = self.lab.available(include_value=False) if self.lab is not None else []
        return tool_schemas(
            self.predictor is not None,
            self.include_unknown_value,
            targets if self.predictor is not None and len(targets) > 1 else None,
        )

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        from jsonschema import Draft202012Validator

        schema = next(
            (
                tool["function"]["parameters"]
                for tool in self.schemas
                if tool["function"]["name"] == name
            ),
            None,
        )
        if schema is None:
            raise ValueError(f"Unknown or unavailable tool: {name}")
        errors = sorted(
            Draft202012Validator(schema).iter_errors(arguments), key=lambda error: str(error.path)
        )
        if errors:
            raise ValueError(errors[0].message)
        if name == "get_database_summary":
            result = self.store.summary()
            result["field_coverage"] = self.store.metadata().get("field_coverage", {})
            result["currency"] = "EUR"
            result["eur_per_internal_unit"] = self.currency.eur_per_internal_unit
            result["currency_calibrated"] = self.currency.calibrated
            result["unknown_value_policy"] = (
                "included_and_flagged" if self.include_unknown_value else "excluded"
            )
            return result
        if name == "find_player":
            return [
                {
                    "player_id": row["player_id"],
                    "name": row["name"],
                    "age": row["age"],
                    "club": row["club"],
                    "positions": row["natural_positions"] + row["accomplished_positions"],
                    "value_eur": row["value_eur"],
                    "value_known": row["value_eur"] is not None,
                }
                for row in self.store.get_players(
                    [match["player_id"] for match in self.store.lookup(arguments["name"])],
                    currency_scale=self.scale,
                )
            ]
        if name == "search_players":
            for key, value in arguments.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{key} must be finite")
            for lower, upper in (
                ("age_min", "age_max"),
                ("value_min_eur", "value_max_eur"),
                ("height_min_cm", "height_max_cm"),
            ):
                if (
                    arguments.get(lower) is not None
                    and arguments.get(upper) is not None
                    and arguments[lower] > arguments[upper]
                ):
                    raise ValueError(f"{lower} cannot exceed {upper}")
            arguments = {"limit": 500, **arguments}
            result = self.store.search(
                **arguments,
                include_unknown_value=self.include_unknown_value,
                currency_scale=self.scale,
            )
            filters = {
                key: value for key, value in arguments.items() if key not in ("limit", "offset")
            }
            search_id = f"search-{len(self.queries) + 1}"
            self.queries[search_id] = {
                "filters": filters,
                "matching_count": result["matching_count"],
                "preparation_id": self.store.metadata().get("preparation_id"),
                "complete": False,
            }
            result["search_id"] = search_id
            self._remember_matches(result)
            self.searched_ids.update(result["player_ids"])
            self.searches.append(
                {
                    "arguments": arguments,
                    "truncated": result["truncated"],
                    "matching_count": result["matching_count"],
                    "search_id": search_id,
                }
            )
            return result
        if name == BUILD_TASK_TOOL:
            return self._build_task(arguments["target"], arguments.get("threshold"))
        if name == TASK_PREDICTION_TOOL:
            return self._predict_search(
                arguments["search_id"],
                arguments.get("top_k", 25),
                arguments.get("rank_by", "expected"),
                task_id=arguments["task_id"],
            )
        if name == PREDICTION_TOOL:
            if ("search_id" in arguments) == ("player_ids" in arguments):
                raise ValueError("Provide exactly one of search_id or player_ids")
            if "search_id" in arguments:
                return self._predict_search(
                    arguments["search_id"],
                    arguments.get("top_k", 25),
                    arguments.get("rank_by", "expected"),
                )
        ids = arguments["player_ids"]
        players = self.store.get_players(ids, with_estimates=name == "get_player_details")
        if {row["player_id"] for row in players} != set(ids):
            raise ValueError("Every requested ID must belong to a player in the save")
        if name == "get_player_details":
            stats = self.store.season_stats(ids)
            return [
                {
                    key: value
                    for key, value in scale_money(row, self.scale).items()
                    if key != "split"
                }
                | {"season_stats": stats.get(row["player_id"])}
                for row in players
            ]
        assert self.predictor is not None
        return self._predict_ids(ids, players)

    def _predict_ids(self, ids: list[int], players: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert self.predictor is not None
        missing = [row for row in players if row["player_id"] not in self.scores]
        result = self.predictor.predict(missing) if missing else []
        if len(result) != len(missing) or {row["player_id"] for row in result} != {
            row["player_id"] for row in missing
        }:
            raise ValueError("Prediction result IDs do not match the requested players")
        for row in result:
            score = row[SCORE_FIELD]
            if (
                not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not SCORE_BOUNDS[0] <= score <= SCORE_BOUNDS[1]
            ):
                raise ValueError("Prediction result contains an invalid score")
        self.scores.update({row["player_id"]: row[SCORE_FIELD] for row in result})
        for row in result:
            low, high = row.get(LOW_FIELD), row.get(HIGH_FIELD)
            if low is not None and high is not None:
                if not (
                    math.isfinite(low)
                    and math.isfinite(high)
                    and SCORE_BOUNDS[0] <= low <= row[SCORE_FIELD] <= high <= SCORE_BOUNDS[1]
                ):
                    raise ValueError("Prediction result contains an invalid range")
                self.intervals[row["player_id"]] = (low, high)
            chance = row.get(CHANCE_FIELD)
            if chance is not None:
                if not (isinstance(chance, (int, float)) and 0.0 <= chance <= 1.0):
                    raise ValueError("Prediction result contains an invalid chance")
                self.chances[row["player_id"]] = float(chance)
        return sorted(
            [self.estimate(player_id) for player_id in ids],
            key=lambda row: (-row[SCORE_FIELD], row["player_id"]),
        )

    def _remember_matches(self, page: dict[str, Any]) -> None:
        for row in page.get("players", ()):
            if "profile_match" in row:
                self.profile_matches[row["player_id"]] = row["profile_match"]

    def estimate(self, player_id: int) -> dict[str, Any]:
        row = {"player_id": player_id, SCORE_FIELD: self.scores[player_id]}
        if player_id in self.intervals:
            row[LOW_FIELD], row[HIGH_FIELD] = self.intervals[player_id]
        if player_id in self.chances:
            row[CHANCE_FIELD] = self.chances[player_id]
        return row

    def rank_value(
        self, player_id: int, rank_by: str = "expected", task_id: str | None = None
    ) -> float:
        """The number a ranking mode sorts by (highest first).

        For tasks where low is good (injury proneness, ...) the numbers are negated, so the best
        player still comes first, and the best case is the bottom of the range.
        """
        if task_id is not None:
            task = self.tasks[task_id]
            if rank_by == "chance":
                return task.chances[player_id]
            sign = -1.0 if task.spec.info.better == "low" else 1.0
            low, high = task.intervals[player_id]
            best, worst = (high, low) if sign > 0 else (low, high)
            value = {"ceiling": best, "safe": worst}.get(rank_by, task.scores[player_id])
            return sign * value
        if rank_by == "chance" and player_id in self.chances:
            return self.chances[player_id]
        if player_id in self.intervals and rank_by == "ceiling":
            return self.intervals[player_id][1]
        if player_id in self.intervals and rank_by == "safe":
            return self.intervals[player_id][0]
        return self.scores[player_id]

    def rank_key(
        self, player_id: int, rank_by: str = "expected", task_id: str | None = None
    ) -> tuple[float, float, int]:
        """Sort key, best first. Chances often tie at the capped ends, so the estimate decides
        between equal chances, then the player_id."""
        return (
            -self.rank_value(player_id, rank_by, task_id),
            -self.rank_value(player_id, "expected", task_id),
            player_id,
        )

    def scored(self, player_id: int, task_id: str | None = None) -> bool:
        return player_id in (self.tasks[task_id].scores if task_id else self.scores)

    def check_ranking(self, rank_by: str, task_id: str | None = None) -> None:
        if task_id is not None and task_id not in self.tasks:
            raise ValueError("Unknown task_id; build the task first")
        if (
            rank_by == "chance"
            and task_id is not None
            and self.tasks[task_id].spec.threshold is None
        ):
            raise ValueError("rank_by chance needs a task built with a threshold")

    def _build_task(self, target: str, threshold: float | None) -> dict[str, Any]:
        if self.lab is None:
            raise ValueError("Custom prediction tasks are not available for this save")
        if threshold is not None and not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        spec = TaskSpec(target, threshold)
        self.lab.progress = self.progress  # says so when a model is actually fitted
        try:
            report = self.lab.build(spec)
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Building the prediction task failed; check TabPFN access and retry."
            ) from exc
        self.tasks.setdefault(spec.task_id, TaskScores(spec, report))
        info = spec.info
        return {
            "task_id": spec.task_id,
            "target": target,
            "meaning": info.label,
            "scale": list(info.scale),
            "better": info.better,
            "threshold": threshold,
            "goal": spec.describe_goal(),
            "quality": report,
            "next": f"Search with the user's filters, then pass search_id and task_id to {TASK_PREDICTION_TOOL}.",
        }

    def _predict_task_ids(self, task_id: str, players: list[dict[str, Any]]) -> None:
        assert self.lab is not None
        task = self.tasks[task_id]
        missing = [row for row in players if row["player_id"] not in task.scores]
        result = self.lab.predict(task.spec, missing) if missing else []
        if {row["player_id"] for row in result} != {row["player_id"] for row in missing}:
            raise ValueError("Prediction result IDs do not match the requested players")
        low_bound, high_bound = task.spec.info.scale
        for row in result:
            values = (row[LOW], row[ESTIMATE], row[HIGH])
            if not (
                all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
                and low_bound <= values[0] <= values[1] <= values[2] <= high_bound
            ):
                raise ValueError("Prediction result contains an invalid estimate")
            chance = row.get(CHANCE)
            if chance is not None and not 0.0 <= chance <= 1.0:
                raise ValueError("Prediction result contains an invalid chance")
        for row in result:
            player_id = row["player_id"]
            task.scores[player_id] = row[ESTIMATE]
            task.intervals[player_id] = (row[LOW], row[HIGH])
            if row.get(CHANCE) is not None:
                task.chances[player_id] = float(row[CHANCE])

    def task_estimate(self, player_id: int, task_id: str) -> dict[str, Any]:
        task = self.tasks[task_id]
        row = {
            "player_id": player_id,
            ESTIMATE: round(task.scores[player_id], 1),
            LOW: round(task.intervals[player_id][0], 1),
            HIGH: round(task.intervals[player_id][1], 1),
        }
        if player_id in task.chances:
            row[CHANCE] = task.chances[player_id]
        return row

    def _predict_search(
        self, search_id: str, top_k: int, rank_by: str = "expected", task_id: str | None = None
    ) -> dict[str, Any]:
        self.check_ranking(rank_by, task_id)
        if search_id not in self.queries:
            raise ValueError("Unknown search_id; use a search handle from this request")
        query = self.queries[search_id]
        if query["preparation_id"] != self.store.metadata().get("preparation_id"):
            raise ValueError("Dataset changed after search; start a new request")
        offset, seen, all_ids = 0, set(), []
        query["complete"] = False
        while True:
            page = self.store.search(
                **query["filters"],
                include_unknown_value=self.include_unknown_value,
                currency_scale=self.scale,
                limit=500,
                offset=offset,
            )
            ids = page["player_ids"]
            if page["matching_count"] != query["matching_count"] or seen.intersection(ids):
                raise ValueError("Search population changed during pagination; start a new request")
            self._remember_matches(page)
            if ids:
                self.searched_ids.update(ids)
                seen.update(ids)
                all_ids.extend(ids)
            if not page["has_more"]:
                break
            if not ids or page["next_offset"] is None:
                raise ValueError("Pagination made no progress; cannot claim complete scoring")
            offset = page["next_offset"]
        if len(seen) != query["matching_count"]:
            raise ValueError("Incomplete search scoring; no complete-pool shortlist is available")
        players = self.store.get_players(all_ids)
        if {row["player_id"] for row in players} != seen:
            raise ValueError("A matching player is missing from the database")
        if self.progress:
            label = f" for {self.tasks[task_id].spec.info.label}" if task_id else ""
            self.progress(f"scoring {len(seen):,} players{label}")
        try:
            if task_id is not None:
                self._predict_task_ids(task_id, players)
                scores = [self.task_estimate(player_id, task_id) for player_id in all_ids]
            else:
                scores = self._predict_ids(all_ids, players)
        except Exception as exc:
            # Never expose provider response bodies or claim completeness on failure.
            raise RuntimeError(
                "Complete-pool prediction failed; no partial shortlist returned. Check service access, quota and model limits, then retry without refitting."
            ) from exc
        self.prediction_operations.append(
            {"search_id": search_id, "player_ids": all_ids, "predictions": scores}
            | ({"task_id": task_id} if task_id else {})
        )
        query["complete"] = True
        if rank_by == "chance" and task_id is None and len(self.chances) < len(self.scores):
            raise ValueError("rank_by chance is not available for these estimates")
        ranked = sorted(seen, key=lambda player_id: self.rank_key(player_id, rank_by, task_id))[
            :top_k
        ]
        return {
            "search_id": search_id,
            "matching_count": len(seen),
            "scored_count": len(seen),
            "complete": True,
            "prediction_mode": "whole_pool_cached",
            "rank_by": rank_by,
            "player_ids": ranked,
            "ranked_players": [
                self.task_estimate(player_id, task_id) if task_id else self.estimate(player_id)
                for player_id in ranked
            ],
        } | ({"task_id": task_id} if task_id else {})

    def message_output(self, name: str, output: Any) -> Any:
        """Keep complete audit traces, but avoid sending thousands of IDs/records to the LLM."""
        if (
            name == "search_players"
            and self.predictor is not None
            and isinstance(output, dict)
            and "search_id" in output
        ):
            return {
                key: output[key]
                for key in (
                    "search_id",
                    "matching_count",
                    "unknown_value_count",
                    "estimated_value_count",
                    "returned_count",
                    "offset",
                    "has_more",
                    "next_offset",
                    "truncated",
                )
            } | {
                "prediction_instruction": f"Pass search_id to {PREDICTION_TOOL} (or, with a task_id, to {TASK_PREDICTION_TOOL}) to score ALL matches; then inspect the returned leaders."
            }
        return output
