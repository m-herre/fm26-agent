from __future__ import annotations

import pytest

from fm26_agent.config import ensure_inside, load_settings
from fm26_agent.prepare import prepare


def write_config(directory, **data):
    lines = "\n".join(f'{key} = "{value}"' for key, value in data.items())
    path = directory / "config.toml"
    path.write_text(f"[data]\n{lines}\n\n[money]\neur_per_internal_unit = 1.0\n", encoding="utf-8")
    return path


def test_default_paths_resolve_inside_the_project(tmp_path):
    settings = load_settings(write_config(tmp_path))
    assert settings.project_root == tmp_path.resolve()
    for configured in vars(settings.data).values():
        assert configured.is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("visible_database", "../elsewhere/players.sqlite3"),
        ("runs_directory", "/tmp/fm26-runs"),
        ("prediction_cache", "~/fm26-cache.sqlite3"),
    ],
)
def test_configured_paths_cannot_leave_the_project(tmp_path, key, value):
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="must stay inside the project directory"):
        load_settings(write_config(project, **{key: value}))


def test_symlinks_cannot_be_used_to_escape(tmp_path):
    project, outside = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="must stay inside"):
        load_settings(write_config(project))


def test_save_file_must_be_inside_the_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    settings = load_settings(write_config(project))
    save = tmp_path / "career.fm"
    save.write_bytes(b"not a real save")
    with pytest.raises(ValueError, match="--save must stay inside"):
        prepare(settings, save)
    inside = project / "career.fm"
    inside.write_bytes(b"")
    assert ensure_inside(settings.project_root, inside, "--save") == inside.resolve()
