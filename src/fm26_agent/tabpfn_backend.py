"""Where TabPFN-3.5 runs: on this computer (`tabpfn`, GPU or Apple MPS) or at Prior Labs.

Both backends are the same TabPFN-3.5 regressor with the same scikit-learn interface, so the
rest of the application doesn't care which one it gets. "auto" picks local when the `tabpfn`
package is installed and a GPU (CUDA or Apple MPS) is available, otherwise hosted.

Local weights (about 1 GB, downloaded once) are kept in data/tabpfn-weights inside the project.
"""

from __future__ import annotations

import importlib.util
import os
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

BACKENDS = ("auto", "local", "hosted")
WEIGHTS_FOLDER = Path("data") / "tabpfn-weights"


def use_project_weights(project_root: Path) -> None:
    """Keep local TabPFN weights inside the project (must run before `tabpfn` is imported)."""
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(project_root / WEIGHTS_FOLDER))


def local_available() -> bool:
    return importlib.util.find_spec("tabpfn") is not None


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@cache
def resolve(requested: str) -> str:
    """The backend to use for `requested` (auto, local or hosted)."""
    if requested not in BACKENDS:
        raise ValueError(f"tabpfn.backend must be one of {', '.join(BACKENDS)}")
    if requested != "auto":
        if requested == "local" and not local_available():
            raise ValueError(
                'Local TabPFN is not installed. Install it with: pip install -e ".[local]"'
            )
        return requested
    return "local" if local_available() and device() != "cpu" else "hosted"


def new_regressor(backend: str, random_seed: int) -> Any:
    """An unfitted TabPFN-3.5 regressor that keeps its training context for fast predictions."""
    if backend == "local":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor.create_default_for_version(
            "v3.5", device=device(), fit_mode="fit_with_cache", random_state=random_seed
        )
    from tabpfn_client import TabPFNRegressor

    return TabPFNRegressor(
        model_path="v3.5_default",
        fit_mode="fit_with_cache",
        text_handling="advanced",
        random_state=random_seed,
    )


def fit(model: Any, backend: str, table: Any, target: Any) -> None:
    """Fit once. Local TabPFN's advisory warnings (dataset size, device) are not for players."""
    from .prediction import with_rate_limit_retry

    if backend != "local":
        with_rate_limit_retry(lambda: model.fit(table, target))
        return
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(table, target)
    free_memory()


LOCAL_CHUNK = 4_000  # rows per local prediction: keeps GPU memory in check; local calls are free


def predict_quantiles(model: Any, backend: str, table: Any, levels: list[float]) -> np.ndarray:
    """Quantiles (levels x rows). Hosted: one request for everything, since each request costs
    the same whatever its size. Local: in chunks, since memory is the limit and calls are free."""
    from .prediction import with_rate_limit_retry

    if backend != "local":
        return np.asarray(
            with_rate_limit_retry(
                lambda: model.predict(table, output_type="quantiles", quantiles=levels)
            ),
            dtype=float,
        )
    parts = [
        np.asarray(
            model.predict(
                table.iloc[start : start + LOCAL_CHUNK], output_type="quantiles", quantiles=levels
            ),
            dtype=float,
        )
        for start in range(0, len(table), LOCAL_CHUNK)
    ]
    free_memory()
    return np.concatenate(parts, axis=1) if parts else np.empty((len(levels), 0))


def free_memory() -> None:
    """Hand cached GPU memory back after a local fit or prediction."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def save_fitted(model: Any, backend: str, reference: Path) -> str:
    """Persist a fitted model; returns what the model.json stores to find it again."""
    if backend == "local":
        from tabpfn.model_loading import save_fitted_tabpfn_model

        archive = reference.with_suffix(".tabpfn_fit")
        save_fitted_tabpfn_model(model, archive)  # the fitted state, not the 1 GB weights
        return archive.name
    return model.save_model()  # a handle to the fit kept at Prior Labs


def load_fitted(stored: str, backend: str, reference: Path) -> Any:
    if backend == "local":
        from tabpfn.model_loading import load_fitted_tabpfn_model

        return load_fitted_tabpfn_model(reference.parent / stored, device=device())
    from tabpfn_client import TabPFNRegressor

    return TabPFNRegressor.load_model(stored)
