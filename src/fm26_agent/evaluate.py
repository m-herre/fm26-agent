from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .agent import ScoutingAgent
from .backend import ChatBackend
from .config import Settings
from .metrics import ranking_metrics
from .prediction import Predictor
from .private_db import PrivateStore
from .runtime import write_report
from .tools import ScoutingTools
from .visible_db import VisibleStore, value_in_range

BENCHMARKS = (
    ("central_midfielder", "central midfielders", "MC"),
    ("striker", "strikers", "STC"),
    ("central_defender", "central defenders", "DC"),
    ("winger", "left wingers", "AML"),
    ("goalkeeper", "goalkeepers", "GK"),
)
BENCHMARK_AGE_MAX = 19
BENCHMARK_VALUE_MAX_EUR = 8_000_000


def benchmark_constraints(position: str) -> dict[str, Any]:
    return {
        "age_max": BENCHMARK_AGE_MAX,
        "value_max_eur": BENCHMARK_VALUE_MAX_EUR,
        "position": position,
    }


def benchmark_pool(
    players: list[dict[str, Any]], position: str, currency_scale: float = 1.0
) -> list[dict[str, Any]]:
    """Players the agent can see for a benchmark query. Unknown values pass the budget filter.

    Rows stay in internal units (they are model inputs); only the budget test uses euros."""
    return [
        row
        for row in players
        if row["age"] is not None
        and row["age"] <= BENCHMARK_AGE_MAX
        and value_in_range(
            None if row["value_eur"] is None else row["value_eur"] * currency_scale,
            None,
            BENCHMARK_VALUE_MAX_EUR,
        )
        and position in row["natural_positions"] + row["accomplished_positions"]
    ]


def check_constraints(
    result: dict[str, Any], expected: dict[str, Any], eligible_ids: set[int]
) -> str | None:
    if result.get("error"):
        return result["error"]
    constraints = {key: value for key, value in result["constraints"].items() if value is not None}
    if constraints != expected or result["requested_count"] != 5:
        return "Constraint parsing failed: normalized constraints differ from the fixed benchmark"
    ids = [row["player_id"] for row in result["recommendations"]]
    if not set(ids).issubset(eligible_ids):
        return "Recommended IDs violate benchmark eligibility"
    return None


def evaluate(
    settings: Settings,
    backend: ChatBackend,
    store: VisibleStore,
    predictor: Predictor,
    *,
    emit: Callable[[str], None] = print,
) -> dict[str, Any]:
    private = PrivateStore(settings.data.private_database)
    if private.preparation_id() != store.metadata().get("preparation_id"):
        raise ValueError("Visible database and evaluation labels belong to different preparations")
    truth = {row["player_id"]: row for row in private.rows("test")}
    all_test = store.get_players(store.test_ids(), require_test=True)
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "llm_model": settings.llm.model,
        "base_url": settings.llm.base_url,
        "temperature": settings.llm.temperature,
        "benchmark_version": "fm26-v5",
        "unknown_value_policy": "included_and_flagged",
        "prediction_task": getattr(predictor, "task", "binary_classification"),
        "llm_settings": {
            "max_tool_steps": settings.llm.max_tool_steps,
            "final_retries": settings.llm.final_retries,
            "max_output_tokens": settings.llm.max_output_tokens,
            "thinking": settings.llm.thinking,
        },
        "preparation_id": store.metadata().get("preparation_id"),
        "benchmarks": [],
    }
    for benchmark_id, label, position in BENCHMARKS:
        query = (
            f"Find me the five best {label} wonderkids under 20 with a market value of at most €8M."
        )
        expected = benchmark_constraints(position)
        # Search cannot return more than 500, so enumerate the complete pool locally for ground-truth scoring.
        eligible = benchmark_pool(all_test, position, settings.eur_per_internal_unit)
        eligible_ids = {row["player_id"] for row in eligible}
        entry: dict[str, Any] = {
            "id": benchmark_id,
            "query": query,
            "expected_constraints": expected,
            "eligible_count": len(eligible),
            "eligible_unknown_value_count": sum(row["value_eur"] is None for row in eligible),
            "eligible_wonderkid_count": sum(
                truth[player_id]["wonderkid"] for player_id in eligible_ids
            ),
            "runs": {},
        }
        emit(f"Benchmark {benchmark_id}: {len(eligible):,} eligible players")
        if not entry["eligible_wonderkid_count"]:
            emit(
                "  No positive labels in this filtered evaluation pool; this cannot measure wonderkid retrieval improvement."
            )
        for mode, active_predictor in (("agent_only", None), ("agent_tabpfn", predictor)):
            result = (
                ScoutingAgent(
                    backend,
                    ScoutingTools(store, active_predictor, currency=settings.currency),
                    settings.llm.max_tool_steps,
                    final_retries=settings.llm.final_retries,
                    trace=lambda message, mode=mode: emit(f"  → {mode}: {message}"),
                )
                .run(query)
                .to_dict()
            )
            failure = check_constraints(result, expected, eligible_ids)
            result["benchmark_failure"] = failure
            if failure is None:
                ids = [row["player_id"] for row in result["recommendations"]]
                result["metrics"] = ranking_metrics(ids, truth, sorted(eligible_ids), ks=(5,))
            entry["runs"][mode] = result
            emit(f"  {mode}: {failure or 'completed'}")
        capped = eligible[: settings.training.max_evaluation_rows]
        try:
            predictions = predictor.predict(capped)
            score_field = getattr(predictor, "score_field", "wonderkid_probability")
            predictions.sort(key=lambda row: (-row[score_field], row["player_id"]))
            entry["tabpfn_only"] = {
                "scored_count": len(capped),
                "truncated": len(capped) < len(eligible),
                "ranked_ids": [row["player_id"] for row in predictions[:10]],
                "metrics": ranking_metrics(
                    [row["player_id"] for row in predictions], truth, sorted(eligible_ids)
                ),
            }
        except Exception as exc:
            entry["tabpfn_only"] = {"error": f"{type(exc).__name__}: hosted prediction failed"}
        report["benchmarks"].append(entry)
    report["successful_runs"] = {
        mode: sum(
            entry["runs"][mode]["benchmark_failure"] is None for entry in report["benchmarks"]
        )
        for mode in ("agent_only", "agent_tabpfn")
    }
    output = write_report(settings, "benchmark", report)
    emit(f"Evaluation report: {output}")
    return report
