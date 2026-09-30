"""Visible-attribute, height and multi-position filters; agent-defined formula targets."""

from __future__ import annotations

import pytest
from test_agent import FakeBackend, call
from test_custom_tasks import FakePotential, _lab, fits, world  # noqa: F401 - fixtures
from test_objective import FakeLab

from fm26_agent.agent import _matches
from fm26_agent.objective import Objective, execute
from fm26_agent.planner import PlanningSession
from fm26_agent.targets import (
    TARGETS,
    combine_values,
    define_formula_target,
    forget_formula_targets,
)


def test_attribute_height_and_position_list_filters(store):
    everyone = store.search(limit=500)["matching_count"]
    fast = store.search(min_attributes={"pace": 17}, limit=500)
    assert 0 < fast["matching_count"] < everyone
    rows = store.get_players(fast["player_ids"])
    assert all(row["pace"] >= 17 for row in rows)
    assert all(_matches(row, {"min_attributes": {"pace": 17}}) for row in rows)
    assert store.search(height_min_cm=176, limit=500)["matching_count"] == 0  # fixture is 175 cm
    assert store.search(position=["DM", "STC"], limit=500)["matching_count"] == everyone  # DM
    assert store.search(position=["STC", "GK"], limit=500)["matching_count"] == 0
    with pytest.raises(ValueError, match="Unknown attribute"):
        store.search(min_attributes={"shoe_size": 3})


def test_objective_filters_accept_the_new_keys():
    goal = Objective.from_dict(
        {
            "filters": {
                "position": ["aml", "AMR"],
                "min_attributes": {"pace": 16},
                "height_min_cm": 180,
            },
            "conditions": [],
            "rank_by": {"target": "potential_ability"},
            "count": 3,
        },
        FakeLab().available(),
    )
    assert goal.filters["position"] == ["AML", "AMR"]


def test_formula_targets_are_validated_and_flip_low_is_good():
    target = define_formula_target(
        "mentality", "mentality", {"professionalism": 2, "controversy": 1}
    )
    assert TARGETS["mentality"] is target and target.better == "high" and not target.whole
    values = combine_values(
        target, {"professionalism": {1: 20.0}, "controversy": {1: 20.0, 2: 5.0}}
    )
    assert values == {1: pytest.approx((2 * 20 + 1) / 3)}  # controversy 20 counts as 1
    for bad, match in (
        (("pace_x", {"professionalism": 1}), "2 to 6"),
        (("current_ability", {"ambition": 1, "loyalty": 1}), "must be new"),
        (("pace", {"ambition": 1, "loyalty": 1}), "must be new"),
        (("mix", {"current_ability": 1, "loyalty": 1}), "can't be combined"),
        (("mix", {"ambition": 9, "loyalty": 1}), "between 0 and 5"),
    ):
        with pytest.raises(ValueError, match=match):
            define_formula_target(bad[0], "x", bad[1])


def test_the_lab_learns_a_formula_target_and_objectives_replay_it(world, fits, tmp_path):  # noqa: F811
    define_formula_target("mentality", "mentality", {"professionalism": 1, "ambition": 1})
    lab = _lab(world, tmp_path)
    assert "mentality" in lab.available()
    report = lab.build_quality("mentality")
    assert report["trained_on"] == 80 and report["checked_on"] == 40
    goal = Objective.from_dict(
        {
            "filters": {},
            "conditions": [{"target": "mentality", "at_least": 12, "min_chance": 0.5}],
            "rank_by": {"target": "mentality"},
            "count": 3,
        },
        lab.available(),
    )
    saved = goal.to_dict()
    assert saved["custom_targets"] == [
        {
            "name": "mentality",
            "label": "mentality",
            "combine": {"professionalism": 1.0, "ambition": 1.0},
        }
    ]
    forget_formula_targets()  # a later replay starts without the definition
    again = Objective.from_dict(saved, _lab(world, tmp_path).available())
    assert again == goal
    assert len(execute(again, world.visible, _lab(world, tmp_path)).shortlist) == 3


def test_planner_can_define_a_target_and_use_it(world, fits, tmp_path):  # noqa: F811
    lab = _lab(world, tmp_path)
    backend = FakeBackend(
        [
            call(
                "define_target",
                {"name": "leader", "label": "leadership mentality",
                 "combine": {"professionalism": 1, "pressure": 1}},
            ),
            call(
                "propose_objective",
                {"objective": {"filters": {}, "conditions": [],
                               "rank_by": {"target": "leader"}, "count": 2}},
                index=2,
            ),
        ]
    )  # fmt: skip
    planner = PlanningSession(backend, world.visible, lab)
    reply = planner.send("a natural leader")
    assert reply.kind == "objective" and "leadership mentality" in reply.text
    tool = [m for m in backend.messages[-1] if m.get("role") == "tool"][0]["content"]
    assert '"checked_on":40' in tool  # the quality report went back to the planner


def test_planner_prompt_keeps_constraints_and_adds_nothing(store):
    planner = PlanningSession(FakeBackend([]), store, FakeLab())
    prompt = planner.messages[0]["content"]
    assert "never widen" in prompt and "Add nothing the user didn't ask for" in prompt
    assert "never claim a filter" in prompt and "min_attributes" in prompt


def test_only_the_most_recent_local_fits_stay_on_disk(world, tmp_path):  # noqa: F811
    import os
    import time

    lab = _lab(world, tmp_path)
    lab.folder.mkdir(parents=True)
    for index in range(6):
        fit = lab.folder / f"t{index}.tabpfn_fit"
        fit.write_bytes(b"x")
        fit.with_suffix(".json").write_text("{}")
        os.utime(fit, (time.time() + index, time.time() + index))
    lab._prune()
    assert sorted(p.name for p in lab.folder.glob("*.tabpfn_fit")) == [
        "t2.tabpfn_fit", "t3.tabpfn_fit", "t4.tabpfn_fit", "t5.tabpfn_fit"
    ]  # fmt: skip
    assert not (lab.folder / "t0.json").exists()
