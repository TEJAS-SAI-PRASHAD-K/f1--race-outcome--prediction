"""Turn raw results into one modelling row per driver per race.

THE ONE RULE: a feature may only use information that existed before lights
out. Every rolling statistic is therefore built as `shift(1)` first, rolling
window second. Shifting first means the current race can never contribute to
its own features.

Two subtleties that are easy to get wrong, and that this module handles
explicitly:

1. Anything computed at *team* or *circuit* level must be aggregated to one
   row per race BEFORE shifting. Shifting a 20-row-per-race frame by one row
   just leaks the driver sitting next to you in the same race.

2. Missing values must be filled deliberately. Ridge cannot accept NaN, and
   trees would happily treat "rookie with no history" as a signal of its own.
   The fallback chain is driver -> team -> field median for that race.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pred.config import FEATURE_SET_VERSION
from f1pred.db import read_sql, upsert_df

# Every feature column this module produces. config.yaml decides which of
# these a given model is allowed to see; they are always all computed.
FEATURE_COLUMNS = [
    "driver_avg_finish_3", "driver_avg_finish_5", "driver_points_5",
    "driver_dnf_rate_10", "driver_experience_races",
    "driver_vs_teammate_quali_delta_5",
    "team_avg_finish_3", "team_avg_finish_5", "team_avg_quali_pos_3",
    "team_dnf_rate_10",
    "driver_circuit_avg_finish_3", "driver_circuit_appearances",
    "circuit_overtaking_index", "circuit_dnf_rate",
    "circuit_laps", "circuit_is_street",
    "season_round", "is_sprint_weekend",
    "grid_position", "quali_gap_to_pole_pct", "quali_gap_to_teammate_pct",
]

# Stored alongside the features but excluded from the model inputs: these are
# what the naive baselines rank on.
BASELINE_COLUMNS = ["prev_finish_position", "season_points_to_date"]

_LOAD_SQL = """
SELECT r.event_id, r.driver_code, r.team_key, r.grid, r.finish_position,
       r.classified, r.dnf, r.points, r.laps_completed,
       e.season, e.round, e.event_date, e.circuit_key, e.event_format,
       c.is_street, c.lap_count,
       q.best_ms, q.quali_position
FROM raw.race_results r
JOIN raw.events    e ON e.event_id = r.event_id
JOIN raw.circuits  c ON c.circuit_key = e.circuit_key
LEFT JOIN raw.qualifying q
       ON q.event_id = r.event_id AND q.driver_code = r.driver_code
ORDER BY e.event_date, r.event_id, r.finish_position
"""


_UPCOMING_SQL = """
SELECT e.event_id, e.season, e.round, e.event_name, e.event_date,
       e.circuit_key, e.event_format, c.is_street, c.lap_count
FROM raw.events e
JOIN raw.circuits c ON c.circuit_key = e.circuit_key
WHERE NOT EXISTS (SELECT 1 FROM raw.race_results r WHERE r.event_id = e.event_id)
  AND e.event_date >= CURRENT_DATE - INTERVAL '2 days'
ORDER BY e.event_date
LIMIT 1
"""


def next_event() -> pd.Series | None:
    """The soonest race with no result yet, or None at season's end."""
    df = read_sql(_UPCOMING_SQL)
    if df.empty:
        return None
    row = df.iloc[0].copy()
    row["event_date"] = pd.to_datetime(row["event_date"])
    return row


def _upcoming_entries(history: pd.DataFrame) -> pd.DataFrame:
    """Placeholder rows for the next race: who is expected on the grid.

    Entry list comes from qualifying if it has run, otherwise from the most
    recent race. Every target column is left null, so these rows can never
    contribute to anyone's rolling features - and because they sort last by
    date, the shift(1) windows of earlier races never reach them either.
    """
    event = next_event()
    if event is None or history.empty:
        return pd.DataFrame()

    qual = read_sql(
        "SELECT driver_code, best_ms, quali_position FROM raw.qualifying "
        "WHERE event_id = :eid", {"eid": int(event["event_id"])})

    # Latest known team for each driver, so a mid-season switch is respected.
    latest_team = (history.sort_values("event_date")
                          .groupby("driver_code")["team_key"].last())

    if not qual.empty:
        entries = qual.copy()
        # Grid equals qualifying order here; penalties are not published in
        # this data, so a grid drop will not be reflected.
        entries["grid"] = entries["quali_position"].astype(float)
    else:
        last_event = history.sort_values("event_date")["event_id"].iloc[-1]
        entries = history.loc[history["event_id"] == last_event,
                              ["driver_code"]].copy()
        entries["best_ms"] = np.nan
        entries["quali_position"] = np.nan
        entries["grid"] = np.nan

    entries["team_key"] = entries["driver_code"].map(latest_team)
    entries = entries.dropna(subset=["team_key"])

    for col, value in [
        ("event_id", int(event["event_id"])), ("season", int(event["season"])),
        ("round", int(event["round"])), ("event_date", event["event_date"]),
        ("circuit_key", event["circuit_key"]),
        ("event_format", event["event_format"]),
        ("is_street", bool(event["is_street"])),
        ("lap_count", event["lap_count"]),
        ("finish_position", np.nan), ("classified", np.nan),
        ("dnf", np.nan), ("points", np.nan), ("laps_completed", np.nan),
    ]:
        entries[col] = value
    return entries


def load_entries(include_upcoming: bool = False) -> pd.DataFrame:
    df = read_sql(_LOAD_SQL)
    df["event_date"] = pd.to_datetime(df["event_date"])
    if include_upcoming:
        upcoming = _upcoming_entries(df)
        if not upcoming.empty:
            df = pd.concat([df, upcoming[df.columns.intersection(
                upcoming.columns)]], ignore_index=True)
    return df.sort_values(["event_date", "event_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# rolling helpers - all of them shift before they roll
# --------------------------------------------------------------------------

def _roll_mean(df: pd.DataFrame, by: str | list[str], col: str,
               window: int) -> pd.Series:
    """Mean of the previous `window` values, excluding the current row."""
    return df.groupby(by, sort=False)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )


def _roll_sum(df: pd.DataFrame, by: str | list[str], col: str,
              window: int) -> pd.Series:
    return df.groupby(by, sort=False)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )


def _expanding_mean(df: pd.DataFrame, by: str, col: str) -> pd.Series:
    """Mean over everything seen before this row, excluding it."""
    return df.groupby(by, sort=False)[col].transform(
        lambda s: s.shift(1).expanding().mean()
    )


# --------------------------------------------------------------------------
# feature blocks
# --------------------------------------------------------------------------

def _driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-driver form. finish_position already ranks retirees at the back,
    so a DNF naturally drags recent form down without special-casing."""
    out = pd.DataFrame(index=df.index)
    out["driver_avg_finish_3"] = _roll_mean(df, "driver_code", "finish_position", 3)
    out["driver_avg_finish_5"] = _roll_mean(df, "driver_code", "finish_position", 5)
    out["driver_points_5"] = _roll_sum(df, "driver_code", "points", 5)
    out["driver_dnf_rate_10"] = _roll_mean(df, "driver_code", "dnf_num", 10)
    # Races started before this one: 0 for a debutant.
    out["driver_experience_races"] = df.groupby("driver_code", sort=False).cumcount()

    # Baseline inputs, not model features by default.
    out["prev_finish_position"] = df.groupby("driver_code", sort=False)[
        "finish_position"].shift(1)
    # Championship points banked BEFORE this race, hence shift then cumsum.
    out["season_points_to_date"] = df.groupby(
        ["season", "driver_code"], sort=False)["points"].transform(
            lambda s: s.shift(1).cumsum())
    return out


def _teammate_delta(df: pd.DataFrame) -> pd.Series:
    """Rolling qualifying gap to the driver in the same car, in percent.

    The strongest available signal of driver skill independent of machinery:
    it compares two drivers who had identical equipment on the same day.
    """
    # Mean qualifying time of the other car(s) in the same team, same race.
    grp = df.groupby(["event_id", "team_key"])["best_ms"]
    team_sum = grp.transform("sum")
    team_cnt = grp.transform("count")
    others = (team_sum - df["best_ms"].fillna(0)) / (team_cnt - 1).replace(0, np.nan)

    gap_pct = (df["best_ms"] - others) / others * 100.0
    tmp = df[["driver_code"]].copy()
    tmp["gap"] = gap_pct
    # Rolling over the driver's last 5 races, shifted so today is excluded.
    return _roll_mean(tmp, "driver_code", "gap", 5)


def _team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Car performance. Aggregated to one row per team per race BEFORE
    shifting, otherwise the sister car in the same race leaks straight in."""
    per_race = (
        df.groupby(["team_key", "event_id", "event_date"], as_index=False)
          .agg(team_finish=("finish_position", "mean"),
               team_quali=("quali_position", "mean"),
               team_dnf=("dnf_num", "mean"))
          .sort_values(["event_date", "event_id"])
          .reset_index(drop=True)
    )
    per_race["team_avg_finish_3"] = _roll_mean(per_race, "team_key", "team_finish", 3)
    per_race["team_avg_finish_5"] = _roll_mean(per_race, "team_key", "team_finish", 5)
    per_race["team_avg_quali_pos_3"] = _roll_mean(per_race, "team_key", "team_quali", 3)
    per_race["team_dnf_rate_10"] = _roll_mean(per_race, "team_key", "team_dnf", 10)

    cols = ["team_key", "event_id", "team_avg_finish_3", "team_avg_finish_5",
            "team_avg_quali_pos_3", "team_dnf_rate_10"]
    return df[["team_key", "event_id"]].merge(
        per_race[cols], on=["team_key", "event_id"], how="left"
    ).drop(columns=["team_key", "event_id"]).set_index(df.index)


def _circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """How this track behaves, learned only from its own past runnings.

    Aggregated per race first for the same reason as team features.
    """
    per_race = (
        df.assign(pos_change=(df["finish_position"] - df["grid"]).abs())
          .groupby(["circuit_key", "event_id", "event_date"], as_index=False)
          .agg(mean_pos_change=("pos_change", "mean"),
               mean_dnf=("dnf_num", "mean"))
          .sort_values(["event_date", "event_id"])
          .reset_index(drop=True)
    )
    # How much the order shuffles here: high = overtaking track, low = procession.
    per_race["circuit_overtaking_index"] = _expanding_mean(
        per_race, "circuit_key", "mean_pos_change")
    # Attrition proxy. Stands in for the safety-car rate, which needs lap
    # timing data that FastF1 now puts behind an F1 TV subscription.
    per_race["circuit_dnf_rate"] = _expanding_mean(
        per_race, "circuit_key", "mean_dnf")

    cols = ["circuit_key", "event_id", "circuit_overtaking_index", "circuit_dnf_rate"]
    merged = df[["circuit_key", "event_id"]].merge(
        per_race[cols], on=["circuit_key", "event_id"], how="left"
    ).drop(columns=["circuit_key", "event_id"]).set_index(df.index)

    merged["circuit_laps"] = df["lap_count"].astype(float)
    merged["circuit_is_street"] = df["is_street"].astype(float)
    return merged


def _driver_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["driver_circuit_avg_finish_3"] = _roll_mean(
        df, ["driver_code", "circuit_key"], "finish_position", 3)
    out["driver_circuit_appearances"] = df.groupby(
        ["driver_code", "circuit_key"], sort=False).cumcount()
    return out


def _quali_features(df: pd.DataFrame) -> pd.DataFrame:
    """Known on Saturday evening, so only used in post_quali mode."""
    out = pd.DataFrame(index=df.index)
    out["grid_position"] = df["grid"].astype(float)

    pole = df.groupby("event_id")["best_ms"].transform("min")
    # Percent, not seconds: a 0.5s gap means very different things at Monaco
    # and at Spa, but a 0.5% gap is comparable across both.
    out["quali_gap_to_pole_pct"] = (df["best_ms"] - pole) / pole * 100.0

    grp = df.groupby(["event_id", "team_key"])["best_ms"]
    others = ((grp.transform("sum") - df["best_ms"].fillna(0))
              / (grp.transform("count") - 1).replace(0, np.nan))
    out["quali_gap_to_teammate_pct"] = (df["best_ms"] - others) / others * 100.0
    return out


# --------------------------------------------------------------------------
# leakage guard
# --------------------------------------------------------------------------

def check_shift_discipline(df: pd.DataFrame) -> None:
    """Fail loudly if a rolling feature saw the current race.

    The test: a driver's very first race cannot have any prior-form value. If
    one is populated there, the shift was lost somewhere.
    """
    debut = df["driver_experience_races"] == 0
    if not debut.any():
        return
    for col in ("driver_avg_finish_3", "driver_avg_finish_5",
                "driver_points_5", "driver_dnf_rate_10"):
        leaked = df.loc[debut, col].notna().sum()
        if leaked:
            raise AssertionError(
                f"leakage in {col}: {leaked} debut rows have prior-form values"
            )


# --------------------------------------------------------------------------
# imputation
# --------------------------------------------------------------------------

def impute(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill gaps deliberately: driver -> team -> field median -> global median.

    A rookie inherits their team's number rather than a NaN or a zero, which
    is both the honest guess and the one that keeps Ridge usable.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        s = out[col]
        # 1. the sister car's value in this same race
        team_med = out.groupby(["event_id", "team_key"])[col].transform("median")
        s = s.fillna(team_med)
        # 2. the field median for this race
        race_med = out.groupby("event_id")[col].transform("median")
        s = s.fillna(race_med)
        # 3. a global fallback, for a column that is empty everywhere early on
        s = s.fillna(s.median())
        out[col] = s.fillna(0.0)
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build_features(include_upcoming: bool = False) -> pd.DataFrame:
    df = load_entries(include_upcoming=include_upcoming)
    if df.empty:
        raise RuntimeError("raw.race_results is empty - run scripts/ingest.py first")

    df["dnf_num"] = pd.to_numeric(df["dnf"], errors="coerce").astype(float)
    df["points"] = df["points"].fillna(0.0)

    blocks = [
        _driver_features(df),
        _team_features(df),
        _circuit_features(df),
        _driver_circuit_features(df),
        _quali_features(df),
    ]
    feats = pd.concat(blocks, axis=1)
    feats["driver_vs_teammate_quali_delta_5"] = _teammate_delta(df)
    feats["season_round"] = df["round"].astype(float)
    feats["is_sprint_weekend"] = (
        df["event_format"].fillna("conventional") != "conventional"
    ).astype(float)

    out = pd.concat(
        [df[["event_id", "driver_code", "season", "round", "event_date",
             "circuit_key", "team_key", "finish_position", "dnf",
             "classified", "grid"]],
         feats],
        axis=1,
    )

    check_shift_discipline(out)
    out = impute(out, FEATURE_COLUMNS + BASELINE_COLUMNS)
    out["feature_set_version"] = FEATURE_SET_VERSION
    return out


def persist(df: pd.DataFrame) -> int:
    cols = (["event_id", "driver_code", "season", "round", "event_date",
             "circuit_key", "team_key"]
            + FEATURE_COLUMNS + BASELINE_COLUMNS
            + ["finish_position", "dnf", "classified", "feature_set_version"])
    frame = df[cols].copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"]).dt.date
    return upsert_df(frame, "core.entry_features", ["event_id", "driver_code"])


def load_features() -> pd.DataFrame:
    """Read back the persisted modelling table, newest last."""
    df = read_sql("SELECT * FROM core.entry_features ORDER BY event_date, event_id")
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df
