"""OpenF1: free lap-level data for 2023 onwards, no authentication.

This module matters more than it was meant to. FastF1 3.9 put lap timing
behind an F1 TV subscription, and practice long-run pace - the best available
read on race trim before lights out - is built from lap times. OpenF1 serves
the same laps for free from 2023, so the feature is recoverable for the
seasons that actually resemble the current car.

What "long-run pace" means here: the median of a driver's clean practice
laps, not their fastest. The headline FP2 time is a low-fuel qualifying
simulation and says little about Sunday; the median of the laps they ran on
race fuel says a lot.

Free tier: 3 requests/second, 30/minute, personal and non-commercial use.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"
FIRST_SEASON = 2023          # OpenF1 has nothing earlier

# Free tier allowance, with a little headroom.
MAX_PER_SECOND = 3
MAX_PER_MINUTE = 30

# FP2 is the long-run session on a conventional weekend. FP3 is mostly
# qualifying preparation, and FP1 is often run by reserve drivers, so they
# are only fallbacks.
PRACTICE_PREFERENCE = ("Practice 2", "Practice 3", "Practice 1")


class _RateLimiter:
    """Keeps the client inside OpenF1's published free-tier limits."""

    def __init__(self) -> None:
        self._calls: list[float] = []

    def wait(self) -> None:
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < 60]
        if len(self._calls) >= MAX_PER_MINUTE:
            time.sleep(60 - (now - self._calls[0]) + 0.1)
        recent = [t for t in self._calls if time.monotonic() - t < 1]
        if len(recent) >= MAX_PER_SECOND:
            time.sleep(1.0)
        self._calls.append(time.monotonic())


_limiter = _RateLimiter()


MAX_RETRIES = 3


def _get(path: str, **params) -> list[dict]:
    """GET with a short retry. A whole season's backfill is ~45 requests, so
    one dropped connection should not cost the lot."""
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        _limiter.wait()
        try:
            response = requests.get(f"{BASE_URL}/{path}", params=params,
                                    timeout=120)
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 404:
            return []
        if response.status_code == 429:          # throttled: back off and retry
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"openf1 {path} failed after {MAX_RETRIES} tries: {last}")


def list_practice_sessions(season: int) -> pd.DataFrame:
    """Every practice session of a season, best session per weekend first."""
    rows = _get("sessions", year=season, session_type="Practice")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[~df.get("is_cancelled", False).fillna(False)]
    df["date_start"] = pd.to_datetime(df["date_start"], utc=True)
    # Rank the sessions so the preferred one per weekend is picked first.
    order = {name: i for i, name in enumerate(PRACTICE_PREFERENCE)}
    df["preference"] = df["session_name"].map(order)
    return df.dropna(subset=["preference"]).sort_values(
        ["meeting_key", "preference"]).reset_index(drop=True)


def long_run_pace(session_key: int) -> pd.DataFrame:
    """Median clean lap per driver for one session.

    A lap is dropped when it has no recorded duration, when it is an out-lap,
    or when it is slower than 107% of the session's best. That last filter
    removes in-laps, cool-downs and traffic, which otherwise drag the median
    around far more than any real pace difference.
    """
    laps = _get("laps", session_key=int(session_key))
    if not laps:
        return pd.DataFrame()

    df = pd.DataFrame(laps)
    if "lap_duration" not in df.columns:
        return pd.DataFrame()

    df = df[df["lap_duration"].notna()]
    if "is_pit_out_lap" in df.columns:
        df = df[~df["is_pit_out_lap"].fillna(False).astype(bool)]
    if df.empty:
        return pd.DataFrame()

    cutoff = df["lap_duration"].min() * 1.07
    df = df[df["lap_duration"] <= cutoff]
    if df.empty:
        return pd.DataFrame()

    drivers = _get("drivers", session_key=int(session_key))
    if not drivers:
        return pd.DataFrame()
    code_by_number = {d["driver_number"]: d.get("name_acronym")
                      for d in drivers}

    pace = (df.groupby("driver_number")["lap_duration"]
              .agg(longrun_median_ms=lambda s: round(s.median() * 1000),
                   clean_laps="count")
              .reset_index())
    pace["driver_code"] = pace["driver_number"].map(code_by_number)
    return pace.dropna(subset=["driver_code"]).drop(columns=["driver_number"])


def practice_pace_for_season(season: int, events: pd.DataFrame) -> pd.DataFrame:
    """Long-run pace for every weekend of a season, keyed to our event_ids.

    `events` must have event_id and event_date. Sessions are matched to
    events by date rather than by circuit name, because the two APIs spell
    venues differently and a date is unambiguous.
    """
    sessions = list_practice_sessions(season)
    if sessions.empty or events.empty:
        return pd.DataFrame()

    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], utc=True)

    out = []
    for meeting_key, group in sessions.groupby("meeting_key"):
        session = group.iloc[0]          # already the preferred session
        # Practice runs on the Friday or Thursday before a Sunday race.
        delta = (events["event_date"] - session["date_start"]).dt.days
        candidates = events[(delta >= 0) & (delta <= 4)]
        if candidates.empty:
            continue

        pace = long_run_pace(session["session_key"])
        if pace.empty:
            continue
        pace["event_id"] = int(candidates.iloc[0]["event_id"])
        pace["session_name"] = session["session_name"]
        out.append(pace)

    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)[
        ["event_id", "driver_code", "session_name",
         "longrun_median_ms", "clean_laps"]]
