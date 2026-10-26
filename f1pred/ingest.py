"""Walk the calendar and load each season into the raw.* tables.

Season at a time, not race at a time: Jolpica allows 500 API calls an hour,
and loading each race session individually burns through that before 2020.

Incremental by design. raw.ingest_log records each season already fetched, so
a weekly refresh only re-reads the current season. Pass force=True to redo
everything.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from f1pred.db import get_engine, read_sql, upsert_df
from f1pred.sources import fastf1_source as ff1


class RateLimited(RuntimeError):
    """Jolpica's hourly budget is spent; progress so far is already saved."""


def _log(season: int, session: str, status: str, rows: int,
         message: str | None = None) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO raw.ingest_log "
                "  (source, season, round, session_name, status, row_count, message) "
                "VALUES ('fastf1', :season, 0, :sess, :status, :rows, :msg) "
                "ON CONFLICT (source, season, round, session_name) DO UPDATE SET "
                "  status = EXCLUDED.status, row_count = EXCLUDED.row_count, "
                "  message = EXCLUDED.message, fetched_at = now()"
            ),
            {"season": season, "sess": session, "status": status,
             "rows": rows, "msg": message},
        )


def _completed_seasons() -> set[int]:
    df = read_sql(
        "SELECT DISTINCT season FROM raw.ingest_log "
        "WHERE source = 'fastf1' AND session_name = 'R' AND status = 'ok'"
    )
    return {int(s) for s in df["season"]}


def _upsert_calendar(season: int) -> pd.DataFrame:
    """Circuits and events must exist before results can reference them."""
    schedule = ff1.get_schedule(season)

    circuits = (schedule[["circuit_key", "event_name", "country"]]
                .drop_duplicates("circuit_key")
                .rename(columns={"event_name": "name"})
                .assign(is_street=lambda d:
                        d["circuit_key"].isin(ff1.STREET_CIRCUITS)))
    upsert_df(circuits, "raw.circuits", ["circuit_key"],
              update_cols=["name", "country", "is_street"])

    upsert_df(schedule.drop(columns=["country"]), "raw.events", ["event_id"])
    return schedule


def _upsert_dimensions(results: pd.DataFrame) -> None:
    upsert_df(results[["driver_code", "full_name"]].drop_duplicates("driver_code"),
              "raw.drivers", ["driver_code"])
    upsert_df(results[["team_key", "team_name"]].drop_duplicates("team_key"),
              "raw.teams", ["team_key"])


def _update_lap_counts(results: pd.DataFrame) -> None:
    """Typical race distance per circuit, used as a feature."""
    laps = (results.groupby("event_id")["laps_completed"].max()
            .dropna().astype(int).reset_index())
    if laps.empty:
        return
    with get_engine().begin() as conn:
        conn.execute(
            text("UPDATE raw.circuits c SET lap_count = :laps "
                 "FROM raw.events e "
                 "WHERE e.circuit_key = c.circuit_key AND e.event_id = :eid"),
            [{"laps": int(r.laps_completed), "eid": int(r.event_id)}
             for r in laps.itertuples()],
        )


def ingest_season(season: int) -> dict[str, int]:
    counts = {"events": 0, "results": 0, "quali": 0}

    schedule = _upsert_calendar(season)
    counts["events"] = len(schedule)
    known_events = set(schedule["event_id"])

    try:
        results = ff1.get_season_results(season)
    except Exception as exc:
        if "rate limit" in str(exc).lower() or "calls/h" in str(exc):
            raise RateLimited(str(exc)) from exc
        _log(season, "R", "error", 0, str(exc)[:300])
        return counts

    if results.empty:
        _log(season, "R", "empty", 0)
        return counts

    # A round in the results but not the schedule would break the foreign key.
    results = results[results["event_id"].isin(known_events)]
    _upsert_dimensions(results)

    race_cols = ["event_id", "driver_code", "team_key", "grid",
                 "finish_position", "status", "classified", "dnf",
                 "points", "laps_completed"]
    counts["results"] = upsert_df(results[race_cols], "raw.race_results",
                                  ["event_id", "driver_code"])
    _log(season, "R", "ok", counts["results"])
    _update_lap_counts(results)

    # --- qualifying: also present for an upcoming race the Sunday before ---
    try:
        qual = ff1.get_season_qualifying(season)
    except Exception as exc:
        if "rate limit" in str(exc).lower() or "calls/h" in str(exc):
            raise RateLimited(str(exc)) from exc
        _log(season, "Q", "error", 0, str(exc)[:300])
        return counts

    if qual.empty:
        _log(season, "Q", "empty", 0)
        return counts

    # raw.qualifying references raw.drivers, so drop anyone not yet known.
    known_drivers = set(read_sql("SELECT driver_code FROM raw.drivers")["driver_code"])
    qual = qual[qual["event_id"].isin(known_events)
                & qual["driver_code"].isin(known_drivers)]
    counts["quali"] = upsert_df(qual, "raw.qualifying",
                                ["event_id", "driver_code"])
    _log(season, "Q", "ok", counts["quali"])
    return counts


def ingest_range(start: int, end: int, force: bool = False) -> None:
    ff1.enable_cache()
    done = set() if force else _completed_seasons()

    for season in range(start, end + 1):
        # The newest season is always re-read: it gains a race every weekend.
        if season in done and season != end:
            print(f"  {season}: already ingested, skipping")
            continue
        try:
            c = ingest_season(season)
        except RateLimited as exc:
            print(f"\n  stopped at {season}: {exc}")
            print("  Jolpica allows 500 calls/hour. Everything fetched so far "
                  "is saved —\n  re-run the same command in an hour to "
                  "continue where it left off.")
            return
        print(f"  {season}: {c['events']:>2} events, {c['results']:>4} results, "
              f"{c['quali']:>4} quali rows")
