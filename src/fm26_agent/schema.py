from __future__ import annotations

VISIBLE_ATTRIBUTES = (
    "crossing",
    "dribbling",
    "finishing",
    "heading",
    "long_shots",
    "marking",
    "off_the_ball",
    "passing",
    "penalty_taking",
    "tackling",
    "vision",
    "handling",
    "aerial_reach",
    "command_of_area",
    "communication",
    "kicking",
    "throwing",
    "anticipation",
    "decisions",
    "one_on_ones",
    "positioning",
    "reflexes",
    "first_touch",
    "technique",
    "flair",
    "corners",
    "teamwork",
    "work_rate",
    "long_throws",
    "eccentricity",
    "rushing_out",
    "punching",
    "acceleration",
    "free_kick_taking",
    "strength",
    "stamina",
    "pace",
    "jumping_reach",
    "leadership",
    "balance",
    "bravery",
    "aggression",
    "agility",
    "natural_fitness",
    "determination",
    "composure",
    "concentration",
)

HIDDEN_ATTRIBUTES = {
    "dirtiness",
    "consistency",
    "important_matches",
    "injury_proneness",
    "versatility",
}

FORBIDDEN_FEATURE_FRAGMENTS = {
    "ability_current",
    "ability_potential",
    "potential_range",
    "current_ability",
    "potential_ability",
    "raw_",
    "reputation",
    "personality",
    "player_id",
    "name",
    *HIDDEN_ATTRIBUTES,
}

POSITION_CODES = (
    "GK",
    "SW",
    "DL",
    "DC",
    "DR",
    "DM",
    "ML",
    "MC",
    "MR",
    "AML",
    "AMC",
    "AMR",
    "STC",
    "WBL",
    "WBR",
)

POSITION_ALIASES = {
    "GOALKEEPER": "GK",
    "GK": "GK",
    "SWEEPER": "SW",
    "SW": "SW",
    "LEFT BACK": "DL",
    "DL": "DL",
    "LB": "DL",
    "CENTRAL DEFENDER": "DC",
    "CENTRE BACK": "DC",
    "CENTER BACK": "DC",
    "CB": "DC",
    "DC": "DC",
    "RIGHT BACK": "DR",
    "DR": "DR",
    "RB": "DR",
    "DEFENSIVE MIDFIELDER": "DM",
    "DM": "DM",
    "LEFT MIDFIELDER": "ML",
    "ML": "ML",
    "LM": "ML",
    "CENTRAL MIDFIELDER": "MC",
    "CENTRAL MIDFIELD": "MC",
    "CM": "MC",
    "MC": "MC",
    "RIGHT MIDFIELDER": "MR",
    "MR": "MR",
    "RM": "MR",
    "LEFT WINGER": "AML",
    "AML": "AML",
    "LW": "AML",
    "ATTACKING MIDFIELDER": "AMC",
    "AMC": "AMC",
    "CAM": "AMC",
    "RIGHT WINGER": "AMR",
    "AMR": "AMR",
    "RW": "AMR",
    "STRIKER": "STC",
    "FORWARD": "STC",
    "ST": "STC",
    "STC": "STC",
    "LEFT WING BACK": "WBL",
    "WBL": "WBL",
    "LWB": "WBL",
    "RIGHT WING BACK": "WBR",
    "WBR": "WBR",
    "RWB": "WBR",
}


def normalize_position(value: str | None) -> str | None:
    if value is None:
        return None
    key = " ".join(value.replace("-", " ").strip().upper().split())
    if key not in POSITION_ALIASES:
        raise ValueError(f"Unsupported position {value!r}; use one of {', '.join(POSITION_CODES)}")
    return POSITION_ALIASES[key]


def assert_safe_features(columns: list[str] | tuple[str, ...]) -> None:
    bad = [
        column
        for column in columns
        if column.lower()
        in {
            "ca",
            "pa",
            "ambition",
            "professionalism",
            "adaptability",
            "loyalty",
            "pressure",
            "sportsmanship",
            "temperament",
            "controversy",
        }
        or any(fragment in column.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
    ]
    if bad:
        raise ValueError(f"Forbidden model features: {', '.join(sorted(bad))}")
