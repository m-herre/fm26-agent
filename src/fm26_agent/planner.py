"""Planning mode: the agent and the user agree an objective, then code carries it out.

Like a coding agent in plan mode. The planner LLM reads the request, asks short questions only
when a word could mean several things (always recommending an answer), and proposes an objective:
filters, conditions on hidden targets with a minimum chance, one ranking. The user approves it
("go") or edits it in plain words. Only then does anything run, deterministically, in
objective.execute. The LLM never picks players; afterwards it only writes short explanations for
the players code chose.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .backend import ChatBackend
from .config import Currency
from .custom_tasks import TaskLab
from .finder import describe
from .objective import FILTER_PROPERTIES, OBJECTIVE_SCHEMA, Objective, ObjectiveResult, execute
from .present import describe_filters, render_objective, render_result
from .targets import glossary
from .tools import ScoutingTools
from .visible_db import VisibleStore, profile_attributes

PLANNER_VERSION = "fm26-planner-v1"
GO_WORDS = {
    "go", "yes", "y", "ok", "okay", "run", "run it", "do it", "sure", "sounds good", "go ahead",
    "yes please", "perfect", "great", "looks good", "lgtm", "start", "proceed", "yep", "ja",
}  # fmt: skip
RESET_WORDS = {"new search", "start over", "reset", "new"}

PLANNER_PROMPT = """You are a Football Manager 2026 scouting assistant working in PLANNING MODE with a game
player, not a data scientist. Your job is to agree an OBJECTIVE with the user; the application then
runs it exactly (TabPFN predicts hidden values for every matching player) and shows the result. You
never pick players yourself.

An objective has:
- filters: what the save can filter by: age, price, position (one code or a list, e.g.
  ["AML", "AMR"] for wingers), club, preferred_foot, contract_ends_within_days, similar_to,
  height_min_cm / height_max_cm, and min_attributes for VISIBLE attributes (1-20, e.g.
  {{"pace": 16}}; "tall" ≈ height_min_cm 188, "good in the air" ≈ heading and jumping_reach 14+).
  Bounds are inclusive ("under 20" = age_max 19; "max €8M" = value_max_eur 8000000). Positions: STC
  striker, MC central midfielder, DC centre-back, AML/AMR wingers, GK goalkeeper, DL/DR full-backs,
  DM, AMC. A position matches natural AND accomplished.
- Keep every explicit constraint exactly as the user gave it: never widen, narrow or shift an
  age, price or other limit ("25 years old" is age 25, not 23-27). Only change one if the user
  agrees (ask) or asks for it.
- Add nothing the user didn't ask for: no extra conditions, filters or caps "to help". If
  something would clearly help, offer it as an option in ask_user (or mention it in a reading as
  a suggestion) instead of adding it.
- conditions (0-4): a hidden target, at_least or at_most a level on its scale, and min_chance, the
  chance TabPFN must give that the player meets it (0.25 "could", 0.5 "likely", 0.7 "very likely").
- rank_by: ONE target and a mode: expected (best estimate), ceiling (best case, for upside),
  safe (worst case, for low risk) or chance (most likely to meet a level; give at_least/at_most).
- count (default 5) and readings: how you interpreted each vague phrase, in plain words. Every
  wish the user expressed must be covered by a filter, a condition, the ranking or a reading. If
  something can't be done (nationality, league, wage, anything not in the filters or targets),
  add a reading saying so, e.g. {{"phrase": "from Brazil", "meaning": "can't be filtered, so it is
  ignored"}}. Never drop a wish silently, and never claim a filter or condition that isn't in the
  objective: readings describe only what the objective really contains or what is ignored.
- A wish that blends several hidden qualities ("strong mentality", "a leader", "big-game
  mentality") can get its OWN target: call define_target (a weighted average of hidden
  attributes / personality), read its quality report, then use it like any other target.

Hidden targets this save can predict (name, scale, which end is good, meaning):
{glossary}

Recommended readings (use them as defaults; ask only if the choice changes the answer a lot):
- "world class", "star", "could become great": potential_ability at_least 160 (min_chance 0.25-0.5).
  With no age given for a prospect, add age_max 21. Rank by chance of potential 160+ when it's the
  main wish.
- "in his prime", "ready now", "proven", "best right now": current_ability (e.g. at_least 140 or
  rank by it); "prime" with no age means age 24-29.
- "undervalued", "bargain", "cheap for what he is": price_vs_fair_value at_most 0.8 (he costs at
  most 80% of what his profile is worth), or price_vs_peers at_most 0.8 (cheaper than players of
  similar ability at his position). Recommend price_vs_fair_value; offer price_vs_peers.
- "reliable", "consistent": consistency at_least 14. "big-game player": important_matches.
- "stays fit": injury_proneness at_most 8 - but it is not predictable from visible data; say so and
  suggest dropping it.
- "still improving": growth_room at_least 15. "model professional": professionalism at_least 15.

How to work:
1. Read the request. Use count_matches to see how many players the filters leave, and check_target
   for any hidden target other than potential and current ability, to know how well it can be
   predicted. If a target's verdict is "weak" or "not predictable", tell the user and recommend
   dropping it or treating it loosely.
2. ASK before proposing when (a) a word has more than one reading above and the user hasn't said
   which (e.g. "undervalued": fair value or similar players; "world class" for a player over 23:
   how likely must it be, and is potential 160 or current ability the point), (b) the request is
   too vague to run, or (c) wishes conflict. Call ask_user with 1-3 short questions, 2-4 options
   each, the recommended option first and marked "(recommended)"; mention in an option what the
   data says when it matters (e.g. "few 25-year-olds reach 160"). Don't ask about things with an
   obvious default (position codes, inclusive bounds). Don't ask twice about the same thing.
3. Otherwise, or once answered, call propose_objective. The user sees it as a card and replies "go"
   or asks for changes; then propose the revised objective.
4. After a result is shown, a follow-up ("cheaper", "younger", "relax it") means: propose the
   adjusted objective. A new, unrelated request starts a new objective.
Every turn ends with exactly one call to ask_user or propose_objective. Only if the message isn't
about finding players (small talk, a question about the game) reply in one or two plain sentences
without a tool.
"""

AUTO_PROMPT = """
The user cannot answer questions right now: never call ask_user. Use the recommended readings, list
them in readings, and call propose_objective; it runs straight away.
"""

EXPLAIN_PROMPT = """You write short scouting notes for a Football Manager player. For each player give one or
two plain sentences on what stands out in the visible profile (age, position, the attributes and
traits listed) that fits what the user asked for. No numbers about potential, ability, chances or
prices: the application prints those itself. No jargon, no talk of models. Also give an optional
note of at most two sentences on the result as a whole (for example if few players qualified),
again without numbers. Reply with JSON only:
{"explanations": [{"player_id": 1, "explanation": "..."}], "note": "..."}"""


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def planner_tools(targets: list[str], auto: bool) -> list[dict[str, Any]]:
    tools = [
        _function(
            "get_database_summary", "Save date, player counts and available positions.", {}, []
        ),
        _function(
            "find_player",
            "Look a player up by name to get his player_id (for similar_to).",
            {"name": {"type": "string", "minLength": 2, "maxLength": 80}},
            ["name"],
        ),
        _function(
            "count_matches",
            "How many players these filters leave (visible data only; nothing is predicted).",
            {
                "filters": {
                    "type": "object",
                    "properties": FILTER_PROPERTIES,
                    "additionalProperties": False,
                }
            },
            ["filters"],
        ),
        _function(
            "check_target",
            "How well TabPFN can predict a hidden target from visible data: builds (or reuses) the "
            "model and returns its self-check on players it never saw, with a verdict (useful, weak, "
            "not predictable).",
            {"target": {"type": "string", "enum": targets}},
            ["target"],
        ),
        _function(
            "define_target",
            "Design a NEW hidden target when the wish is a blend of several hidden qualities that "
            "no single target covers (e.g. 'mentality', 'leader', 'dressing-room influence', "
            "'big-game mentality'): a weighted average of 2-6 hidden attributes or personality "
            "values (1-20; values where low is good are reversed automatically, so the result is "
            "always high-is-good on 1-20). TabPFN learns it on the spot and checks itself on "
            "players it never saw; you get the quality report back. Then use its name in the "
            "objective like any other target.",
            {
                "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{2,30}$"},
                "label": {"type": "string", "maxLength": 40},
                "description": {"type": "string", "maxLength": 200},
                "combine": {
                    "type": "object",
                    "additionalProperties": {"type": "number", "exclusiveMinimum": 0, "maximum": 5},
                    "minProperties": 2,
                    "maxProperties": 6,
                },
            },
            ["name", "label", "combine"],
        ),
        _function(
            "propose_objective",
            "Show the user the objective as a card to approve or change."
            + (" In this session it runs immediately." if auto else ""),
            {"objective": OBJECTIVE_SCHEMA},
            ["objective"],
        ),
    ]
    if not auto:
        tools.append(
            _function(
                "ask_user",
                "Ask the user 1-3 short questions before proposing, each with 2-4 options; put the "
                "recommended option first and end its label with (recommended).",
                {
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["question", "options"],
                            "properties": {
                                "question": {"type": "string", "maxLength": 300},
                                "options": {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 4,
                                    "items": {"type": "string", "maxLength": 200},
                                },
                            },
                        },
                    }
                },
                ["questions"],
            )
        )
    return tools


@dataclass
class Reply:
    kind: str  # questions | objective | shortlist | chat | error
    text: str
    objective: Objective | None = None
    result: ObjectiveResult | None = None
    questions: list[dict[str, Any]] = field(default_factory=list)


def is_go(text: str) -> bool:
    return re.sub(r"[^a-z ]", "", text.lower()).strip() in GO_WORDS


def render_questions(questions: list[dict[str, Any]]) -> str:
    lines = [
        "A couple of things to pin down first:" if len(questions) > 1 else "One question first:"
    ]
    for number, item in enumerate(questions, 1):
        lines.append(f"{number}. {item['question']}")
        for letter, option in zip("abcd", item["options"], strict=False):
            lines.append(f"   {letter}) {option}")
    lines.append('Answer in your own words or like "1a 2b"; "go" takes the recommended options.')
    return "\n".join(lines)


class PlanningSession:
    """One conversation. send() takes whatever the user typed and returns what to show."""

    def __init__(
        self,
        backend: ChatBackend,
        store: VisibleStore,
        lab: TaskLab,
        *,
        currency: Currency | None = None,
        include_unknown_value: bool = True,
        auto: bool = False,
        progress: Callable[[str], None] | None = None,
        max_steps: int = 8,
    ):
        self.backend, self.store, self.lab = backend, store, lab
        self.currency = currency or Currency()
        self.include_unknown_value = include_unknown_value
        self.auto, self.progress, self.max_steps = auto, progress, max_steps
        self.lab.progress = progress
        self.lookup = ScoutingTools(store, None, currency=self.currency)
        self.objective: Objective | None = None
        self.state = "idle"  # idle | asking | proposed
        self.usage: dict[str, int] = {}
        self.log: list[dict[str, Any]] = []  # everything shown and run, for the session report
        self.reset()

    # -- conversation ---------------------------------------------------------------------

    def reset(self) -> None:
        targets = self.lab.available()
        prompt = PLANNER_PROMPT.format(glossary=glossary(targets)) + (
            AUTO_PROMPT if self.auto else ""
        )
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
        self.tools = planner_tools(targets, self.auto)
        self.objective, self.state = None, "idle"

    def send(self, text: str) -> Reply:
        text = text.strip()
        if text.lower() in RESET_WORDS:
            self.reset()
            return Reply("chat", "Starting fresh. What are you looking for?")
        if self.state == "proposed" and self.objective is not None and is_go(text):
            return self.run(self.objective)
        if self.state == "asking" and is_go(text):
            text = "Take the recommended options."
        self.messages.append({"role": "user", "content": text})
        return self._plan()

    def _plan(self) -> Reply:
        for _ in range(self.max_steps):
            reply = self.backend.complete(self.messages, self.tools)
            for key, value in reply.usage.items():
                self.usage[key] = self.usage.get(key, 0) + value
            message = dict(reply.message)
            if message.get("content") is None:
                message["content"] = ""
            self.messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                self.state = "idle"
                return Reply("chat", (message.get("content") or "").strip())
            final: Reply | None = None
            for call in calls:
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                    output, shown = self._tool(name, arguments)
                except (ValueError, TypeError, KeyError) as exc:
                    output, shown = {"error": str(exc)}, None
                self.log.append({"tool": name, "result": output if shown is None else "shown"})
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(output, default=str, separators=(",", ":")),
                    }
                )
                final = shown or final
            if final is not None:
                if final.kind == "objective" and self.auto:
                    return self.run(final.objective)
                return final
        return Reply("error", "Sorry, I couldn't pin that down. Could you say it another way?")

    def _tool(self, name: str, arguments: dict[str, Any]) -> tuple[Any, Reply | None]:
        if name == "get_database_summary":
            return self.lookup.call("get_database_summary", {}), None
        if name == "find_player":
            return self.lookup.call("find_player", arguments), None
        if name == "count_matches":
            filters = Objective.from_dict(
                {
                    "filters": arguments["filters"],
                    "conditions": [],
                    "rank_by": {"target": "potential_ability"},
                    "count": 1,
                }
            ).filters
            page = self.store.search(
                **filters,
                include_unknown_value=self.include_unknown_value,
                currency_scale=self.currency.eur_per_internal_unit,
                limit=1,
            )
            return {
                "matching_count": page["matching_count"],
                "without_stored_price": page["unknown_value_count"] + page["estimated_value_count"],
            }, None
        if name == "check_target":
            report = self.lab.build_quality(arguments["target"])
            return {"target": arguments["target"], "quality": report}, None
        if name == "define_target":
            from .targets import define_formula_target

            target = define_formula_target(
                arguments["name"],
                arguments["label"],
                arguments["combine"],
                arguments.get("description", ""),
            )
            self.lab.forget(target.name)
            report = self.lab.build_quality(target.name)
            return {
                "target": target.name,
                "meaning": target.description,
                "scale": list(target.scale),
                "better": target.better,
                "quality": report,
            }, None
        if name == "ask_user":
            questions = arguments["questions"]
            self.state = "asking"
            return {"shown": True, "waiting_for": "the user's answers"}, Reply(
                "questions", render_questions(questions), questions=questions
            )
        if name == "propose_objective":
            objective = Objective.from_dict(arguments["objective"], self.lab.available())
            quality = {
                target: self.lab.build_quality(target)
                for target in objective.targets
                if target not in ("potential_ability",)
            }
            pool = self.store.search(
                **objective.filters,
                include_unknown_value=self.include_unknown_value,
                currency_scale=self.currency.eur_per_internal_unit,
                limit=1,
            )["matching_count"]
            self.objective, self.state = objective, "proposed"
            card = render_objective(objective, quality, pool)
            self.log.append({"objective": objective.to_dict()})
            return {"shown": True, "waiting_for": "go or changes"}, Reply(
                "objective", card + ("" if self.auto else "\nGo, or change something?"), objective
            )
        raise ValueError(f"Unknown tool {name}")

    # -- running ---------------------------------------------------------------------------

    def run(self, objective: Objective) -> Reply:
        if self.progress:
            self.progress("running objective")
        result = execute(
            objective,
            self.store,
            self.lab,
            include_unknown_value=self.include_unknown_value,
            currency_scale=self.currency.eur_per_internal_unit,
        )
        explanations, note = self.explain(objective, result)
        fair = (
            self.lab.values.fair_values(
                self.store.get_players([row["player_id"] for row in result.shortlist])
            )
            if "price_vs_fair_value" in objective.targets
            else {}
        )
        text = render_result(
            result,
            self.store,
            explanations=explanations,
            note=note,
            fair_values=fair,
            scale=self.currency.eur_per_internal_unit,
        )
        if self.auto:
            text = render_objective(objective, None, result.pool_size) + "\n\n" + text
        names = {
            row["player_id"]: row["name"]
            for row in self.store.get_players([row["player_id"] for row in result.shortlist])
        }
        self.messages.append(
            {
                "role": "user",
                "content": "[application] The objective ran. "
                + " → ".join(f"{stage.remaining} {stage.label}" for stage in result.funnel)
                + ". Shortlist: "
                + (", ".join(names[row["player_id"]] for row in result.shortlist) or "nobody")
                + ". Suggestions: "
                + ("; ".join(result.suggestions) or "none")
                + ". If the user follows up, propose an adjusted objective.",
            }
        )
        self.state = "idle"
        self.log.append({"result": result.to_dict(), "explanations": explanations, "note": note})
        return Reply("shortlist", text, objective, result)

    def explain(self, objective: Objective, result: ObjectiveResult) -> tuple[dict[int, str], str]:
        """Short notes on the players code chose. Falls back to plain facts if the LLM fails."""
        ids = [row["player_id"] for row in result.shortlist]
        players = self.store.get_players(ids)
        fallback = {row["player_id"]: describe(row) for row in players}
        if not ids:
            return {}, ""
        profiles = []
        for row in players:
            names = profile_attributes("GK" in row["natural_positions"])
            best = sorted((n for n in names if row.get(n) is not None), key=lambda n: -row[n])[:6]
            profiles.append(
                {
                    "player_id": row["player_id"],
                    "age": row["age"],
                    "positions": row["natural_positions"] + row["accomplished_positions"],
                    "preferred_foot": row["preferred_foot"],
                    "best_attributes": {n: row[n] for n in best},
                    "traits": row["traits"][:5],
                }
            )
        request = {
            "what_the_user_wants": describe_filters(objective.filters)
            + "; "
            + "; ".join(c.describe() for c in objective.conditions)
            + "; ranked by "
            + objective.rank_by.describe(),
            "readings": [meaning for _, meaning in objective.readings],
            "players": profiles,
        }
        try:
            reply = self.backend.complete(
                [
                    {"role": "system", "content": EXPLAIN_PROMPT},
                    {"role": "user", "content": json.dumps(request)},
                ],
                [],
            )
            for key, value in reply.usage.items():
                self.usage[key] = self.usage.get(key, 0) + value
            content = (reply.message.get("content") or "").strip().removeprefix("```json")
            data = json.loads(content.removeprefix("```").removesuffix("```"))
            written = {
                int(item["player_id"]): str(item["explanation"])[:400]
                for item in data.get("explanations", [])
            }
            if set(written) != set(ids):  # must explain exactly the players code chose
                return fallback, ""
            return written, str(data.get("note") or "")[:400]
        except Exception:
            return fallback, ""
