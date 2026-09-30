from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .config import Currency
from .prediction import SCORE_BOUNDS, SCORE_FIELD, Predictor
from .visible_db import VisibleStore, scale_money

PREDICTION_TOOL = "predict_player_potential"

SEARCH_PROPERTIES = {
    "age_min": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
    "age_max": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
    "value_min_eur": {"type": ["number", "null"], "minimum": 0},
    "value_max_eur": {"type": ["number", "null"], "minimum": 0},
    "position": {
        "type": ["string", "null"],
        "description": "Canonical FM code such as MC, STC, DC, AML or GK; one position per call. Matches BOTH natural and accomplished labels, never natural-only.",
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
    predictions_enabled: bool = True, include_unknown_value: bool = True
) -> list[dict[str, Any]]:
    unknown_value_policy = (
        "Players whose market value the save does not store (value_known=false) are INCLUDED in "
        "budget filters and flagged; their budget fit cannot be confirmed."
        if include_unknown_value
        else "Players whose market value the save does not store fail budget filters."
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
                "Estimate each player's potential ability (1-200) with the saved TabPFN model; never refits. Prefer search_id: score EVERY matching player in one operation and return the global leaders plus coverage counts. Alternatively player_ids scores only those explicit IDs (at most 500). Results are estimates of hidden potential, not facts. Cached scores are reused. Rank highest predicted_potential first.",
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
                },
            )
        )
        tools[-1]["function"]["parameters"]["anyOf"] = [
            {"required": ["player_ids"]},
            {"required": ["search_id"]},
        ]
    return tools


class ScoutingTools:
    def __init__(
        self,
        store: VisibleStore,
        predictor: Predictor | None = None,
        *,
        include_unknown_value: bool = True,
        currency: Currency | None = None,
    ):
        self.store = store
        self.currency = currency or Currency()
        self.scale = self.currency.eur_per_internal_unit
        self.predictor = predictor
        self.include_unknown_value = include_unknown_value
        self.searched_ids: set[int] = set()
        self.scores: dict[int, float] = {}
        self.searches: list[dict[str, Any]] = []
        self.queries: dict[str, dict[str, Any]] = {}
        self.prediction_operations: list[dict[str, Any]] = []
        self.progress: Callable[[str], None] | None = None

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return tool_schemas(self.predictor is not None, self.include_unknown_value)

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
        if name == "search_players":
            for key, value in arguments.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{key} must be finite")
            for lower, upper in (("age_min", "age_max"), ("value_min_eur", "value_max_eur")):
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
        if name == PREDICTION_TOOL:
            if ("search_id" in arguments) == ("player_ids" in arguments):
                raise ValueError("Provide exactly one of search_id or player_ids")
            if "search_id" in arguments:
                return self._predict_search(arguments["search_id"], arguments.get("top_k", 25))
        ids = arguments["player_ids"]
        players = self.store.get_players(ids)
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
        return sorted(
            [{"player_id": player_id, SCORE_FIELD: self.scores[player_id]} for player_id in ids],
            key=lambda row: (-row[SCORE_FIELD], row["player_id"]),
        )

    def _predict_search(self, search_id: str, top_k: int) -> dict[str, Any]:
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
            self.progress(f"scoring {len(seen):,} players")
        try:
            scores = self._predict_ids(all_ids, players)
        except Exception as exc:
            # Never expose provider response bodies or claim completeness on failure.
            raise RuntimeError(
                "Complete-pool prediction failed; no partial shortlist returned. Check service access, quota and model limits, then retry without refitting."
            ) from exc
        self.prediction_operations.append(
            {"search_id": search_id, "player_ids": all_ids, "predictions": scores}
        )
        query["complete"] = True
        ranked = sorted(seen, key=lambda player_id: (-self.scores[player_id], player_id))[:top_k]
        return {
            "search_id": search_id,
            "matching_count": len(seen),
            "scored_count": len(seen),
            "complete": True,
            "prediction_mode": "whole_pool_cached",
            "player_ids": ranked,
            "ranked_players": [
                {"player_id": player_id, SCORE_FIELD: self.scores[player_id]}
                for player_id in ranked
            ],
        }

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
                    "returned_count",
                    "offset",
                    "has_more",
                    "next_offset",
                    "truncated",
                )
            } | {
                "prediction_instruction": f"Pass search_id to {PREDICTION_TOOL} to score ALL matches; then inspect the returned leaders."
            }
        return output
