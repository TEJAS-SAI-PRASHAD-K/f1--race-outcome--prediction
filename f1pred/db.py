"""PostgreSQL access: an engine, an upsert, and a read helper.

Deliberately SQLAlchemy Core with hand-written SQL rather than an ORM. The
point of this project is to understand what happens, and visible SQL beats a
mapping layer for that.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from f1pred.config import PROJECT_ROOT, load_config

SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"

_engine: Engine | None = None


def get_engine() -> Engine:
    """Process-wide engine. Pooled, so repeated calls are free."""
    global _engine
    if _engine is None:
        _engine = create_engine(load_config().database_url, future=True)
    return _engine


def apply_schema() -> None:
    """Run db/schema.sql. It is idempotent, so this is safe to repeat."""
    sql = SCHEMA_PATH.read_text()
    with get_engine().begin() as conn:
        conn.exec_driver_sql(sql)


def read_sql(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params=params or {})


def upsert_df(
    df: pd.DataFrame,
    table: str,
    conflict_cols: Iterable[str],
    update_cols: Iterable[str] | None = None,
) -> int:
    """INSERT ... ON CONFLICT DO UPDATE for a whole DataFrame.

    Re-running ingestion then refreshes rows instead of failing on duplicates,
    which is what makes the weekly update loop painless.

    `table` is schema-qualified, e.g. "raw.race_results".
    """
    if df.empty:
        return 0

    cols = list(df.columns)
    conflict_cols = list(conflict_cols)
    if update_cols is None:
        update_cols = [c for c in cols if c not in conflict_cols]
    update_cols = list(update_cols)

    col_list = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    conflict_list = ", ".join(f'"{c}"' for c in conflict_cols)

    if update_cols:
        assignments = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
        action = f"DO UPDATE SET {assignments}"
    else:
        action = "DO NOTHING"

    stmt = text(
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_list}) {action}"
    )

    # NaN/NaT are not valid SQL parameters; None is.
    records = df.astype(object).where(pd.notna(df), None).to_dict("records")
    with get_engine().begin() as conn:
        conn.execute(stmt, records)
    return len(records)


def git_sha() -> str | None:
    """Short commit hash, recorded with every run so results stay traceable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def start_run(kind: str, mode: str, config_json: dict, notes: str | None = None) -> int:
    """Open a row in ml.runs and return its id.

    Every backtest, tune and prediction is recorded with the exact config it
    used, so two experiments can be compared with SQL rather than memory.
    """
    import json

    with get_engine().begin() as conn:
        row = conn.execute(
            text(
                "INSERT INTO ml.runs (kind, mode, config_json, git_sha, notes) "
                "VALUES (:kind, :mode, CAST(:cfg AS JSONB), :sha, :notes) "
                "RETURNING run_id"
            ),
            {
                "kind": kind,
                "mode": mode,
                "cfg": json.dumps(config_json),
                "sha": git_sha(),
                "notes": notes,
            },
        ).one()
    return int(row[0])
