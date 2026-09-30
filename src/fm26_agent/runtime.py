"""Wiring shared by the command line and any other front end, such as a web app.

Nothing here prints or reads input: progress goes through an optional callback and results are
returned as data.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent import AgentResult, ScoutingAgent
from .backend import ChatBackend, OpenAICompatibleBackend
from .config import Settings
from .prediction import HostedPredictor, HostedRegressionPredictor, Predictor
from .prediction_cache import CachedPredictor
from .tools import ScoutingTools
from .visible_db import VisibleStore


def write_report(settings: Settings, prefix: str, payload: dict[str, Any]) -> Path:
    """Save a timestamped JSON report in the runs directory and return its path."""
    settings.data.runs_directory.mkdir(parents=True, exist_ok=True)
    path = settings.data.runs_directory / (
        f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return path


def load_predictor(settings: Settings, store: VisibleStore, regression: bool = False) -> Predictor:
    """Load the saved hosted fit (classifier or regressor), wrapped in the local score cache."""
    if not settings.tabpfn_token:
        raise ValueError("Set TABPFN_TOKEN before using hosted prediction")
    metadata = store.metadata()
    if not metadata.get("model_ready"):
        raise ValueError("The dataset/model is not ready; run prepare without --extract-only")
    if regression:
        path = settings.data.regression_reference or settings.data.model_reference.with_name(
            "regression-model.json"
        )
    else:
        path = settings.data.model_reference
    if not path.exists():
        raise ValueError(
            "Regression reference is missing; run fit-regression"
            if regression
            else "Classifier reference is missing; run prepare"
        )
    cls = HostedRegressionPredictor if regression else HostedPredictor
    predictor = cls.load(path, settings.data.feature_schema, metadata["preparation_id"])
    if settings.data.prediction_cache is not None:
        namespace = CachedPredictor.namespace_for(
            metadata["preparation_id"], path, predictor.schema.fingerprint
        )
        predictor = CachedPredictor(predictor, settings.data.prediction_cache, namespace)
    return predictor


def open_runtime(
    settings: Settings, use_prediction: bool, regression: bool = False
) -> tuple[ChatBackend, VisibleStore, Predictor | None]:
    """Open the LLM backend, the player database and (optionally) the saved prediction model."""
    if not settings.deepseek_api_key:
        raise ValueError("Set DEEPSEEK_API_KEY or LLM_API_KEY before using the agent")
    if not settings.data.visible_database.exists():
        raise ValueError("Player database is missing; run prepare first")
    store = VisibleStore(settings.data.visible_database)
    if store.metadata().get("eur_per_internal_unit") != settings.eur_per_internal_unit:
        raise ValueError("Currency conversion changed; run prepare again to rebuild euro values")
    predictor = load_predictor(settings, store, regression) if use_prediction else None
    return OpenAICompatibleBackend(settings.llm, settings.deepseek_api_key), store, predictor


def scout(
    settings: Settings,
    backend: ChatBackend,
    store: VisibleStore,
    predictor: Predictor | None,
    query: str,
    *,
    heldout_only: bool = False,
    include_unknown_value: bool = True,
    trace: Callable[[str], None] | None = None,
) -> tuple[AgentResult, Path]:
    """Run one independent scouting request and save its audit report."""
    agent = ScoutingAgent(
        backend,
        ScoutingTools(
            store,
            predictor,
            heldout_only=heldout_only,
            include_unknown_value=include_unknown_value,
        ),
        settings.llm.max_tool_steps,
        trace=trace,
        final_retries=settings.llm.final_retries,
    )
    result = agent.run(query)
    report = {
        "model": settings.llm.model,
        "temperature": settings.llm.temperature,
        "llm_settings": {
            "thinking": settings.llm.thinking,
            "max_output_tokens": settings.llm.max_output_tokens,
            "max_tool_steps": settings.llm.max_tool_steps,
            "final_retries": settings.llm.final_retries,
        },
        "preparation_id": store.metadata()["preparation_id"],
        **result.to_dict(),
    }
    return result, write_report(settings, "chat", report)
