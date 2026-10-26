"""Pull race results and qualifying times out of FastF1 / Jolpica.

IMPORTANT, and the reason this module is smaller than you might expect:
FastF1 3.9 moved lap-by-lap timing behind an F1 TV subscription. Lap data,
weather and track-status flags all fail to load without one. What still comes
through free, via the Jolpica backend, is exactly what this model needs most:

    - race results   (grid, finishing position, status, points, laps)
    - qualifying     (Q1/Q2/Q3 times and positions)

So practice long-run pace and safety-car rates are not available here. See
openf1_source.py, which supplies lap-level detail for 2023+ without auth.

Everything is fetched a SEASON at a time, not a race at a time. Loading each
race session individually costs roughly seven API calls and blows through
Jolpica's 500-calls-per-hour budget before 2020. The bulk endpoints bring a
full nine-season backfill down to about a hundred calls.
"""

from __future__ import annotations

import re
import warnings

import fastf1
import pandas as pd
from fastf1.ergast import Ergast

from f1pred.config import CACHE_DIR

# Jolpica caps a page at 100 rows, i.e. five races.
PAGE_LIMIT = 100

# Circuits where a mistake ends in a wall rather than a gravel trap. Hand
# maintained because no free source exposes it as a flag.
STREET_CIRCUITS = {
    "monaco", "monte_carlo", "baku", "singapore", "marina_bay", "jeddah",
    "miami", "las_vegas", "melbourne", "montreal", "valencia", "sochi",
}

# Teams rebrand constantly. Without this, a rolling "team form" feature resets
# to nothing every time a name changes, throwing away the car's real history.
# Keys are Jolpica constructorIds, which are stable within a brand era.
TEAM_ALIASES = {
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
    "renault": "alpine",
    "lotus_f1": "alpine",
    "toro_rosso": "rb",
    "alphatauri": "rb",
    "rb": "rb",
    "sauber": "audi",
    "alfa": "audi",
    "alfa_romeo": "audi",
}


def _slug(value: str) -> str:
    """Lowercase underscore slug, so keys survive punctuation and accents."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def canonical_team(constructor_id: str) -> str:
    key = _slug(constructor_id)
    return TEAM_ALIASES.get(key, key)


def enable_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(CACHE_DIR))
    fastf1.set_log_level("ERROR")
    warnings.filterwarnings("ignore")


def event_id(season: int, round_no: int) -> int:
    """Deterministic key: 2026 round 19 -> 202619."""
    return int(season) * 100 + int(round_no)


def get_schedule(season: int) -> pd.DataFrame:
    """Calendar for a season, one row per round.

    Taken from FastF1's own schedule rather than Jolpica because it lists
    races that have not happened yet, which is what the next-race prediction
    needs, and because it carries the sprint/conventional event format.
    """
    sched = fastf1.get_event_schedule(season, include_testing=False)
    sched = sched[sched["RoundNumber"] > 0].copy()
    return pd.DataFrame({
        "event_id": [event_id(season, r) for r in sched["RoundNumber"]],
        "season": season,
        "round": sched["RoundNumber"].astype(int),
        "event_name": sched["EventName"],
        "event_date": pd.to_datetime(sched["EventDate"]).dt.date,
        "circuit_key": sched["Location"].map(_slug),
        "country": sched["Country"],
        "event_format": sched["EventFormat"],
    }).reset_index(drop=True)


def _all_pages(fetch, season: int) -> list[tuple[pd.Series, pd.DataFrame]]:
    """Walk Jolpica's pagination and return (race description, rows) pairs."""
    pairs: list[tuple[pd.Series, pd.DataFrame]] = []
    fetched = 0
    response = fetch(season=season)
    while True:
        for i, frame in enumerate(response.content):
            pairs.append((response.description.iloc[i], frame))
            fetched += len(frame)
        # Count rows actually returned. Assuming a fixed field size silently
        # truncates seasons where entry lists varied.
        if fetched >= (response.total_results or 0):
            break
        try:
            response = response.get_next_result_page()
        except Exception:
            break
        if response is None or not len(response.content):
            break
    return pairs


def _timedelta_to_ms(series: pd.Series) -> pd.Series:
    return (pd.to_timedelta(series).dt.total_seconds() * 1000).round()


# Two vocabularies for "finished but a lap or more down" coexist in the data:
# the older "+1 Lap" form and the newer bare "Lapped". Missing the second one
# marks a third of the modern field as retirements and wrecks both the DNF
# model and every rolling reliability feature.
FINISHED_STATUSES = {"Finished", "Lapped"}


def _is_dnf(status: str) -> bool:
    """True when the car did not reach the flag.

    Everything that is not a finish - accident, engine, gearbox, disqualified,
    withdrawn, did not start - counts. This is the conventional definition and
    it keeps the position regressor's target clean.
    """
    s = str(status)
    return not (s in FINISHED_STATUSES
                or re.fullmatch(r"\+\d+ Laps?", s) is not None)


def get_season_results(season: int) -> pd.DataFrame:
    """Every race result for one season, in a handful of API calls."""
    ergast = Ergast(result_type="pandas", auto_cast=True, limit=PAGE_LIMIT)
    rows = []
    for desc, res in _all_pages(ergast.get_race_results, season):
        if res is None or res.empty:
            continue
        rnd = int(desc["round"])
        grid = pd.to_numeric(res["grid"], errors="coerce")
        # A pit-lane start is recorded as grid 0; treat it as the back row.
        grid = grid.replace(0, float(len(res)))

        rows.append(pd.DataFrame({
            "event_id": event_id(season, rnd),
            "driver_code": res["driverCode"].astype(str),
            "full_name": (res["givenName"].astype(str) + " "
                          + res["familyName"].astype(str)),
            "team_name": res["constructorName"].astype(str),
            "team_key": res["constructorId"].map(canonical_team),
            "grid": grid,
            "finish_position": pd.to_numeric(res["position"], errors="coerce"),
            "status": res["status"].astype(str),
            # positionText is a number when classified, else R / D / W / N / F.
            "classified": pd.to_numeric(res["positionText"],
                                        errors="coerce").notna(),
            "dnf": res["status"].map(_is_dnf),
            "points": pd.to_numeric(res["points"], errors="coerce"),
            "laps_completed": pd.to_numeric(res["laps"], errors="coerce"),
        }))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def get_season_qualifying(season: int) -> pd.DataFrame:
    """Every qualifying result for one season."""
    ergast = Ergast(result_type="pandas", auto_cast=True, limit=PAGE_LIMIT)
    rows = []
    for desc, res in _all_pages(ergast.get_qualifying_results, season):
        if res is None or res.empty:
            continue
        rnd = int(desc["round"])
        q1, q2, q3 = (_timedelta_to_ms(res[c]) if c in res.columns
                      else pd.Series(index=res.index, dtype=float)
                      for c in ("Q1", "Q2", "Q3"))
        rows.append(pd.DataFrame({
            "event_id": event_id(season, rnd),
            "driver_code": res["driverCode"].astype(str),
            "q1_ms": q1, "q2_ms": q2, "q3_ms": q3,
            # Best of the three: the lap that actually set their grid slot.
            "best_ms": pd.concat([q1, q2, q3], axis=1).min(axis=1),
            "quali_position": pd.to_numeric(res["position"], errors="coerce"),
        }))

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["best_ms"]).reset_index(drop=True)
