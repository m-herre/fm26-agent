"""Local storage for the two API keys, so a non-technical user enters them only once.

Keys live in a `.env` file inside the project directory (gitignored, readable only by the
owner). Real environment variables always take precedence.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = ".env"
KEY_NAMES = ("DEEPSEEK_API_KEY", "TABPFN_TOKEN")


def load_env_file(root: Path) -> None:
    """Load NAME=value lines from <root>/.env into the environment, without overriding it."""
    path = root / ENV_FILE
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip("'\"")
        if name and value:
            os.environ.setdefault(name, value)


def save_key(root: Path, name: str, value: str) -> None:
    """Store one key in <root>/.env (replacing any earlier value) and use it in this session."""
    if name not in KEY_NAMES:
        raise ValueError(f"Unknown key name {name!r}")
    value = value.strip()
    if not value or any(ch.isspace() for ch in value):
        raise ValueError("A key must be one piece of text without spaces")
    path = root / ENV_FILE
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [line for line in lines if not line.strip().startswith(f"{name}=")]
    lines.append(f"{name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    os.environ[name] = value
