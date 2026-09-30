from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .prediction import Predictor
from .visible_db import VisibleStore

SEARCH_PROPERTIES = {
    "age_min": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
    "age_max": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
    "value_min_eur": {"type": ["number", "null"], "minimum": 0},
    "value_max_eur": {"type": ["number", "null"], "minimum": 0},
    "position": {
        "type": ["string", "null"],
        "description": "Canonical FM code such as MC, STC, DC, AML or GK; one position per call. Matches BOTH natural and accomplished labels, never natural-only.",
    },
    "club": {"type": ["string", "null"]},
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
    heldout_only: bool = True,
    regression: bool = False,
    include_unknown_value: bool = True,
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
            "Inspect save date, held-out candidate counts, available filters and field coverage.",
            {},
        ),
        _function(
            "search_players",
            "Find held-out players using inclusive bounds. Returns a query-scoped search_id and one page of at most 500 records, with next_offset, total matching_count and unknown_value_count. "
            + unknown_value_policy
            + " For predictive rankings pass search_id to predict_wonderkid_probability: it scores EVERY match, not just this page.",
            SEARCH_PROPERTIES,
        ),
        _function(
            "get_player_details",
            "Inspect observable attributes of up to 25 held-out players. No hidden variables are available.",
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
                "predict_wonderkid_probability",
                "Estimate future potential with the saved TabPFN model, never refit. Prefer search_id: score EVERY matching player in one prediction operation and return global leading probabilities plus coverage counts. Alternatively player_ids scores only those explicit IDs (at most 500). Results are estimates, not hidden facts. Cached scores are reused.",
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
    if not heldout_only:
        for tool in tools:
            tool["function"]["description"] = tool["function"]["description"].replace(
                "held-out", "full-save demo"
            )
    if regression:
        for tool in tools:
            function = tool["function"]
            function["description"] = (
                function["description"]
                .replace("predict_wonderkid_probability", "predict_player_potential")
                .replace("probabilities", "estimated potential scores")
            )
            if function["name"] == "predict_wonderkid_probability":
                function["name"] = "predict_player_potential"
                function["description"] += (
                    " Regression returns predicted_potential on a 1–200 scale, not a probability or true hidden ability. Rank highest predicted_potential first."
                )
    return tools


class ScoutingTools:
    def __init__(
        self,
        store: VisibleStore,
        predictor: Predictor | None = None,
        *,
        heldout_only: bool = True,
        include_unknown_value: bool = True,
    ):
        self.store = store
        self.predictor = predictor
        self.heldout_only = heldout_only
        self.include_unknown_value = include_unknown_value
        self.score_field = getattr(predictor, "score_field", "wonderkid_probability")
        self.score_bounds = getattr(predictor, "score_bounds", (0.0, 1.0))
        self.regression = self.score_field == "predicted_potential"
        self.prediction_tool = (
            "predict_player_potential" if self.regression else "predict_wonderkid_probability"
        )
        self.searched_ids: set[int] = set()
        self.probabilities: dict[int, float] = {}
        self.searches: list[dict[str, Any]] = []
        self.queries: dict[str, dict[str, Any]] = {}
        self.prediction_operations: list[dict[str, Any]] = []
        self.progress: Callable[[str], None] | None = None

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return tool_schemas(
            self.predictor is not None,
            self.heldout_only,
            self.regression,
            self.include_unknown_value,
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
            result["candidate_scope"] = "held_out" if self.heldout_only else "full_save_demo"
            result["unknown_value_policy"] = (
                "included_and_flagged" if self.include_unknown_value else "excluded"
            )
            result["prediction_task"] = (
                "pa_regression" if self.regression else "binary_classification"
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
                heldout_only=self.heldout_only,
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
        if name == self.prediction_tool:
            if ("search_id" in arguments) == ("player_ids" in arguments):
                raise ValueError("Provide exactly one of search_id or player_ids")
            if "search_id" in arguments:
                return self._predict_search(arguments["search_id"], arguments.get("top_k", 25))
        ids = arguments["player_ids"]
        players = self.store.get_players(ids, require_test=self.heldout_only)
        found = {row["player_id"] for row in players}
        if found != set(ids):
            raise ValueError("Every requested ID must belong to the authorized candidate pool")
        if name == "get_player_details":
            return [{key: value for key, value in row.items() if key != "split"} for row in players]
        assert self.predictor is not None
        return self._predict_ids(ids, players)

    def _predict_ids(self, ids: list[int], players: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert self.predictor is not None
        missing = [row for row in players if row["player_id"] not in self.probabilities]
        result = self.predictor.predict(missing) if missing else []
        if len(result) != len(missing) or {row["player_id"] for row in result} != {
            row["player_id"] for row in missing
        }:
            raise ValueError("Prediction result IDs do not match the requested players")
        for row in result:
            probability = row[self.score_field]
            if (
                not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not self.score_bounds[0] <= probability <= self.score_bounds[1]
            ):
                raise ValueError("Prediction result contains an invalid score")
        self.probabilities.update({row["player_id"]: row[self.score_field] for row in result})
        return sorted(
            [
                {"player_id": player_id, self.score_field: self.probabilities[player_id]}
                for player_id in ids
            ],
            key=lambda row: (-row[self.score_field], row["player_id"]),
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
                limit=500,
                offset=offset,
                heldout_only=self.heldout_only,
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
        players = self.store.get_players(all_ids, require_test=self.heldout_only)
        if {row["player_id"] for row in players} != seen:
            raise ValueError("A matching player is outside the authorized candidate pool")
        if self.progress:
            self.progress(
                f"TabPFN: scoring all {len(seen):,} matches in one operation (saved fit and cached scores reused)"
            )
        try:
            scores = self._predict_ids(all_ids, players)
        except Exception as exc:
            # Never expose provider response bodies or claim completeness on failure.
            raise RuntimeError(
                "Complete-pool prediction failed; no partial shortlist returned. Check service access, quota and model limits, then retry without refitting."
            ) from exc
        self.prediction_operations.append(
            {
                "search_id": search_id,
                "player_ids": all_ids,
                "predictions" if self.regression else "probabilities": scores,
            }
        )
        query["complete"] = True
        ranked = sorted(seen, key=lambda player_id: (-self.probabilities[player_id], player_id))[
            :top_k
        ]
        return {
            "search_id": search_id,
            "matching_count": len(seen),
            "scored_count": len(seen),
            "complete": True,
            "prediction_mode": "whole_pool_cached",
            "player_ids": ranked,
            "ranked_players": [
                {"player_id": player_id, self.score_field: self.probabilities[player_id]}
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
                "prediction_instruction": f"Pass search_id to {self.prediction_tool} to score ALL matches; then inspect the returned leaders."
            }
        return output
