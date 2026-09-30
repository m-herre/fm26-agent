"""Controlled Plus/Fast/Thinking experiments; no model routing or held-out tuning."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from .prediction import HostedPredictor, HostedRegressionPredictor
from .regression import _snapshot, compare_models


def variant_parameters(name, task, seed, effort="medium", timeout=180):
    if name not in ("plus", "fast", "thinking") or task not in ("classifier", "regression"):
        raise ValueError("Unknown model variant or task")
    if effort not in ("medium", "high") or not 1 <= timeout <= 2400:
        raise ValueError(
            "Thinking effort must be medium/high and timeout between 1 and 2400 seconds"
        )
    params = {
        "model_path": "v3.5-fast_default" if name == "fast" else "v3.5_default",
        "random_state": seed,
        "n_estimators": 8,
        "text_handling": "advanced",
        "fit_mode": "fit_preprocessors" if name == "thinking" else "fit_with_cache",
    }
    if name == "thinking":
        params.update(
            thinking_mode=True,
            thinking_effort=effort,
            thinking_timeout_s=timeout,
            thinking_metric="average_precision" if task == "classifier" else "rmse",
        )
    return params


def _quote(train, test, name, effort):
    from tabpfn_client import estimate_cost

    operations = ("thinking_fit", "thinking_predict") if name == "thinking" else ("cache_predict",)
    result = {}
    for operation in operations:
        kwargs = {
            "model_version": "v3.5-fast" if name == "fast" else "v3.5",
            "operation": operation,
            "n_estimators": 8,
        }
        if operation == "thinking_fit":
            kwargs["thinking_effort"] = effort
        quote = estimate_cost(train, None if operation == "thinking_fit" else test, **kwargs)
        result[operation] = (
            quote.model_dump(mode="json") if hasattr(quote, "model_dump") else vars(quote)
        )
    return result


def compare_variants(
    settings,
    *,
    task="classifier",
    thinking_effort="medium",
    thinking_seconds=180,
    refit=False,
    preview=False,
    emit=print,
):
    import tabpfn_client

    if task not in ("classifier", "regression", "both"):
        raise ValueError("task must be classifier, regression or both")
    store, private, schema, train, test = _snapshot(settings)
    train_players = store.get_players([r["player_id"] for r in train])
    test_players = store.get_players([r["player_id"] for r in test], require_test=True)
    x_train, x_test = schema.transform(train_players), schema.transform(test_players)
    tasks = ("classifier", "regression") if task == "both" else (task,)
    plans = {}
    directory = settings.data.model_reference.parent / "variants"
    for current_task in tasks:
        for name in ("plus", "fast", "thinking"):
            params = variant_parameters(
                name, current_task, settings.training.random_seed, thinking_effort, thinking_seconds
            )
            key = f"{current_task}_{name}"
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "params": params,
                        "schema": schema.fingerprint,
                        "preparation": private.preparation_id(),
                        "task": current_task,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:16]
            plans[key] = {
                "task": current_task,
                "variant": name,
                "parameters": params,
                "reference": str(directory / f"{key}-{digest}.json"),
                "reuse_requested": not refit,
                "quote_scope": "global held-out sample plus Thinking fit where applicable; filtered-pool prediction requests add usage",
            }
    emit(
        f"Variant comparison: same {len(train):,} reference players, {len(schema.columns)} features; same held-out pools. No agent-based model selection, no dataset-size routing."
    )
    if not preview and not settings.tabpfn_token:
        raise ValueError(
            "Set TABPFN_TOKEN to fit/score variants; --preview shows the local plan without fitting"
        )
    for plan in plans.values():
        if settings.tabpfn_token:
            try:
                plan["cost_estimates"] = _quote(x_train, x_test, plan["variant"], thinking_effort)
            except Exception as exc:
                raise ValueError(
                    f"Token estimate for {plan['variant']} failed; check credentials, client/model support and service access. No variant fits started."
                ) from exc
        emit(json.dumps(plan, default=str))
    if preview:
        return {
            "preview": True,
            "train_rows": len(train),
            "features": len(schema.columns),
            "models": plans,
        }
    predictors, hashes = {}, {}
    for key, plan in plans.items():
        path = Path(plan["reference"])
        cls = HostedPredictor if plan["task"] == "classifier" else HostedRegressionPredictor
        try:
            started = perf_counter()
            if path.exists() and not refit:
                payload = json.loads(path.read_text())
                if payload.get("variant_parameters") != plan["parameters"]:
                    raise ValueError("Saved variant parameters differ; no silent fallback")
                predictor = cls.load(path, settings.data.feature_schema, private.preparation_id())
                plan["fit_reused"] = True
                emit(f"{key}: saved fit reused")
            else:
                estimator = (
                    tabpfn_client.TabPFNClassifier
                    if plan["task"] == "classifier"
                    else tabpfn_client.TabPFNRegressor
                )
                y = np.asarray(
                    [
                        r["wonderkid"] if plan["task"] == "classifier" else r["potential_ability"]
                        for r in train
                    ]
                )
                model = estimator(**plan["parameters"])
                emit(
                    f"{key}: fitting fixed reference ({'binary target' if plan['task'] == 'classifier' else 'user-authorized exact PA target'}); no held-out labels used"
                )
                model.fit(x_train, y)
                predictor = cls(model, schema)
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "version": 2,
                    "task": predictor.task,
                    "model_version": "v3.5",
                    "variant": plan["variant"],
                    "variant_parameters": plan["parameters"],
                    "fit_mode": plan["parameters"]["fit_mode"],
                    "preparation_id": private.preparation_id(),
                    "feature_fingerprint": schema.fingerprint,
                    "target_upload": "binary_label"
                    if plan["task"] == "classifier"
                    else "exact_pa_user_authorized",
                    "model": model.save_model(),
                }
                path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
                plan["fit_reused"] = False
            plan["fit_or_load_wall_seconds"] = perf_counter() - started
            predictors[key] = (
                predictor  # Deliberately bypass local score cache for latency measurement.
            )
            hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception as exc:
            plan["error"] = f"{type(exc).__name__}: variant fit/load failed; no fallback model used"
            emit(f"{key}: {plan['error']}")
            predictors[key] = None
    return compare_models(
        settings,
        None,
        None,
        models=predictors,
        experiment="plus-fast-thinking-v1",
        reference_hashes=hashes,
        report_prefix="variant-comparison",
        emit=emit,
        report_extra={
            "variant_plans": plans,
            "tabpfn_client_version": version("tabpfn-client"),
            "latency_scope": "single-request wall time including network/queue; local score cache bypassed; model-dependent KV mode recorded; not a repeated speed benchmark",
            "automatic_model_selection": False,
        },
    )
