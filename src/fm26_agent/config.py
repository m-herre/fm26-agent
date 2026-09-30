from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class DataSettings:
    visible_database: Path
    private_database: Path
    model_reference: Path
    feature_schema: Path
    runs_directory: Path
    prediction_cache: Path | None = None


@dataclass(frozen=True)
class TrainingSettings:
    wonderkid_threshold: int = 160  # only used to balance the training sample
    random_seed: int = 42
    max_train_rows: int = 10_000


@dataclass(frozen=True)
class LLMSettings:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    temperature: float = 0.0
    max_tool_steps: int = 8
    final_retries: int = 2
    max_output_tokens: int = 4096
    thinking: bool = False


@dataclass(frozen=True)
class Currency:
    """How internal save units become euros. Applied only when results are filtered or shown;
    the model always sees the raw internal units, so changing this never needs a refit."""

    eur_per_internal_unit: float = 1.0
    calibrated: bool = False


@dataclass(frozen=True)
class Settings:
    data: DataSettings
    training: TrainingSettings
    llm: LLMSettings
    eur_per_internal_unit: float
    config_path: Path
    currency_calibrated: bool = False
    tabpfn_backend_setting: str = "auto"  # auto | local | hosted

    @property
    def currency(self) -> Currency:
        return Currency(self.eur_per_internal_unit, self.currency_calibrated)

    @property
    def project_root(self) -> Path:
        """Everything the application reads or writes must stay inside this directory."""
        return self.config_path.parent

    @property
    def deepseek_api_key(self) -> str | None:
        return os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")

    @property
    def tabpfn_token(self) -> str | None:
        return os.getenv("TABPFN_TOKEN")

    @property
    def tabpfn_backend(self) -> str:
        """local or hosted. FM26_TABPFN_BACKEND overrides the config's [tabpfn] backend."""
        from .tabpfn_backend import resolve

        return resolve(os.getenv("FM26_TABPFN_BACKEND") or self.tabpfn_backend_setting)

    @property
    def tabpfn_ready(self) -> bool:
        """Whether TabPFN can run: locally needs nothing, hosted needs the key."""
        return self.tabpfn_backend == "local" or bool(self.tabpfn_token)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def demo_settings(settings: Settings) -> Settings:
    """The same settings with the databases kept in data/demo, so a demo never replaces a real setup."""
    folder = settings.project_root / "data" / "demo"
    data = settings.data
    return replace(
        settings,
        data=replace(
            data,
            visible_database=folder / "players.sqlite3",
            private_database=folder / "private" / "labels.sqlite3",
            model_reference=folder / "model.json",
            feature_schema=folder / "feature_schema.json",
            prediction_cache=folder / "predictions.sqlite3" if data.prediction_cache else None,
        ),
    )


def ensure_inside(root: Path, path: str | Path, what: str) -> Path:
    """Resolve path (following symlinks) and refuse anything outside the project directory."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{what} must stay inside the project directory {root}, not {resolved}")
    return resolved


def load_settings(path: str | Path = "config.toml") -> Settings:
    """Load settings. The config file is optional: without it everything uses its defaults and
    the project directory is the folder the file would be in."""
    config_path = Path(path).expanduser().resolve()
    raw: dict = {}
    if config_path.exists():
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    base = config_path.parent
    data = raw.get("data", {})
    money = raw.get("money", {})
    training = raw.get("training", {})
    llm = raw.get("llm", {})
    rate = float(money.get("eur_per_internal_unit", 1.0))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("money.eur_per_internal_unit must be greater than zero")
    settings = Settings(
        data=DataSettings(
            visible_database=_resolve(base, data.get("visible_database", "data/players.sqlite3")),
            private_database=_resolve(
                base, data.get("private_database", "data/private/labels.sqlite3")
            ),
            model_reference=_resolve(base, data.get("model_reference", "data/model.json")),
            feature_schema=_resolve(base, data.get("feature_schema", "data/feature_schema.json")),
            runs_directory=_resolve(base, data.get("runs_directory", "runs")),
            prediction_cache=_resolve(
                base, data.get("prediction_cache", "data/predictions.sqlite3")
            ),
        ),
        training=TrainingSettings(
            wonderkid_threshold=int(training.get("wonderkid_threshold", 160)),
            random_seed=int(training.get("random_seed", 42)),
            max_train_rows=int(training.get("max_train_rows", 10_000)),
        ),
        llm=LLMSettings(
            base_url=os.getenv("LLM_BASE_URL", llm.get("base_url", "https://api.deepseek.com")),
            model=os.getenv("LLM_MODEL", llm.get("model", "deepseek-flash")),
            temperature=float(llm.get("temperature", 0.0)),
            max_tool_steps=int(llm.get("max_tool_steps", 8)),
            final_retries=int(llm.get("final_retries", 2)),
            max_output_tokens=int(llm.get("max_output_tokens", 4096)),
            thinking=llm.get("thinking", False),
        ),
        eur_per_internal_unit=rate,
        config_path=config_path,
        currency_calibrated=money.get("calibrated", False),
        tabpfn_backend_setting=str(raw.get("tabpfn", {}).get("backend", "auto")),
    )
    if not 1 <= settings.training.wonderkid_threshold <= 200:
        raise ValueError("training.wonderkid_threshold must be between 1 and 200")
    if settings.training.max_train_rows < 2:
        raise ValueError("training.max_train_rows must allow at least two players")
    if not 1 <= settings.llm.max_tool_steps <= 20:
        raise ValueError("llm.max_tool_steps must be between 1 and 20")
    if not 0 <= settings.llm.final_retries <= 3:
        raise ValueError("llm.final_retries must be between 0 and 3")
    if not 512 <= settings.llm.max_output_tokens <= 16384:
        raise ValueError("llm.max_output_tokens must be between 512 and 16384")
    if not isinstance(settings.llm.thinking, bool):
        raise ValueError("llm.thinking must be a TOML boolean")
    if not isinstance(settings.currency_calibrated, bool):
        raise ValueError("money.calibrated must be a TOML boolean")
    outputs = [
        settings.data.visible_database,
        settings.data.private_database,
        settings.data.model_reference,
        settings.data.feature_schema,
    ]
    if settings.data.prediction_cache is not None:
        outputs.append(settings.data.prediction_cache)
    for name, configured in vars(settings.data).items():
        if configured is not None:
            ensure_inside(settings.project_root, configured, f"data.{name}")
    if len(set(outputs)) != len(outputs):
        raise ValueError("Database, model, and schema paths must be distinct")
    return settings
