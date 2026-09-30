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
from .custom_tasks import TaskLab
from .features import FeatureSchema
from .planner import PLANNER_VERSION, PlanningSession
from .prediction import STAR_LEVEL, HostedPredictor, Predictor
from .prediction_cache import CachedPredictor
from .private_db import PrivateStore
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


def load_predictor(settings: Settings, store: VisibleStore) -> Predictor:
    """Load the saved hosted fit, wrapped in the local score cache."""
    if not settings.tabpfn_ready:
        raise ValueError("A TabPFN key is needed to estimate potential (or install local TabPFN)")
    metadata = store.metadata()
    if not metadata.get("model_ready"):
        raise ValueError("Setup has not finished for this save; run it again")
    path = settings.data.model_reference
    if not path.exists():
        raise ValueError("The potential model is missing; run setup again")
    predictor = HostedPredictor.load(path, settings.data.feature_schema, metadata["preparation_id"])
    if settings.data.prediction_cache is not None:
        namespace = CachedPredictor.namespace_for(
            metadata["preparation_id"], path, predictor.schema.fingerprint
        )
        predictor = CachedPredictor(predictor, settings.data.prediction_cache, namespace)
    return predictor


def open_lab(
    settings: Settings, store: VisibleStore, progress: Callable[[str], None] | None = None
) -> TaskLab | None:
    """The workshop for agent-built TabPFN tasks (potential and value targets work even when a
    setup predates the hidden targets), or None when TabPFN isn't available.

    Fitted tasks are kept in data/tasks next to the potential model and reused across sessions.
    """
    if not settings.tabpfn_ready or not settings.data.private_database.exists():
        return None
    private = PrivateStore(settings.data.private_database)
    return TaskLab(
        store,
        private,
        FeatureSchema.load(settings.data.feature_schema),
        settings.tabpfn_backend,
        settings.data.model_reference.parent / "tasks",
        seed=settings.training.random_seed,
        progress=progress,
    )


def open_runtime(
    settings: Settings, use_prediction: bool = True
) -> tuple[ChatBackend, VisibleStore, Predictor | None]:
    """Open the LLM backend, the player database and (optionally) the saved potential model."""
    if not settings.deepseek_api_key:
        raise ValueError("A DeepSeek key is needed to answer questions")
    if not settings.data.visible_database.exists():
        raise ValueError("No save has been set up yet")
    store = VisibleStore(settings.data.visible_database)
    predictor = load_predictor(settings, store) if use_prediction else None
    return OpenAICompatibleBackend(settings.llm, settings.deepseek_api_key), store, predictor


def scout(
    settings: Settings,
    backend: ChatBackend,
    store: VisibleStore,
    predictor: Predictor | None,
    query: str,
    *,
    include_unknown_value: bool = True,
    trace: Callable[[str], None] | None = None,
    lab: TaskLab | None = None,
) -> tuple[AgentResult, Path]:
    """Run one independent scouting request and save its audit report."""
    agent = ScoutingAgent(
        backend,
        ScoutingTools(
            store,
            predictor,
            include_unknown_value=include_unknown_value,
            currency=settings.currency,
            star_level=getattr(predictor, "star_level", STAR_LEVEL),
            lab=lab if predictor is not None else None,
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


def open_session(
    settings: Settings,
    backend: ChatBackend,
    store: VisibleStore,
    lab: TaskLab,
    *,
    auto: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PlanningSession:
    """A planning conversation: agree an objective with the user, then run it."""
    return PlanningSession(
        backend,
        store,
        lab,
        currency=settings.currency,
        auto=auto,
        progress=progress,
        max_steps=settings.llm.max_tool_steps,
    )


def write_session_report(settings: Settings, session: PlanningSession) -> Path:
    return write_report(
        settings,
        "plan",
        {
            "planner_version": PLANNER_VERSION,
            "model": settings.llm.model,
            "usage": session.usage,
            "log": session.log,
        },
    )
