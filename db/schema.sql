-- ===========================================================================
-- f1_outcome_pred schema. Idempotent: safe to re-run.
--
-- Three schemas mirror the three pipeline stages, so it is always obvious
-- which stage a table belongs to:
--   raw   - exactly what the APIs returned, lightly typed
--   core  - one modelling row per driver-race, features already computed
--   ml    - experiment tracking: every backtest and tuning run is kept
--
-- event_id is deterministic: season * 100 + round (2026 round 20 -> 202620).
-- No sequences, no surrogate-key lookups, and a human can read it.
-- ===========================================================================

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS ml;

-- --------------------------------------------------------------------- raw

CREATE TABLE IF NOT EXISTS raw.circuits (
    circuit_key   TEXT PRIMARY KEY,          -- slug of the location, stable across years
    name          TEXT NOT NULL,
    country       TEXT,
    is_street     BOOLEAN NOT NULL DEFAULT FALSE,
    lap_count     SMALLINT                   -- typical race distance in laps
);

-- NOTE: keyed on the 3-letter abbreviation (VER, HAM, ...). Unique across
-- 2018-2026, readable in query output, but it would silently merge two
-- drivers if the FIA ever reissued a code. Acceptable for this window.
CREATE TABLE IF NOT EXISTS raw.drivers (
    driver_code   TEXT PRIMARY KEY,
    full_name     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw.teams (
    team_key      TEXT PRIMARY KEY,
    team_name     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw.events (
    event_id      INTEGER PRIMARY KEY,       -- season * 100 + round
    season        SMALLINT NOT NULL,
    round         SMALLINT NOT NULL,
    event_name    TEXT NOT NULL,
    event_date    DATE NOT NULL,
    circuit_key   TEXT NOT NULL REFERENCES raw.circuits(circuit_key),
    event_format  TEXT,                      -- conventional / sprint variants
    UNIQUE (season, round)
);

CREATE TABLE IF NOT EXISTS raw.race_results (
    event_id        INTEGER NOT NULL REFERENCES raw.events(event_id) ON DELETE CASCADE,
    driver_code     TEXT NOT NULL REFERENCES raw.drivers(driver_code),
    team_key        TEXT NOT NULL REFERENCES raw.teams(team_key),
    grid            SMALLINT,
    finish_position SMALLINT,
    status          TEXT,                    -- "Finished", "+1 Lap", "Accident", ...
    classified      BOOLEAN NOT NULL,        -- officially classified at the end
    dnf             BOOLEAN NOT NULL,
    points          REAL,
    laps_completed  SMALLINT,
    PRIMARY KEY (event_id, driver_code)
);

CREATE TABLE IF NOT EXISTS raw.qualifying (
    event_id        INTEGER NOT NULL REFERENCES raw.events(event_id) ON DELETE CASCADE,
    driver_code     TEXT NOT NULL REFERENCES raw.drivers(driver_code),
    q1_ms           INTEGER,
    q2_ms           INTEGER,
    q3_ms           INTEGER,
    best_ms         INTEGER,                 -- fastest of the three
    quali_position  SMALLINT,
    PRIMARY KEY (event_id, driver_code)
);

-- Median of clean long-run laps, which is a far better race-pace signal than
-- the headline fastest practice lap (that one is a low-fuel qualifying sim).
CREATE TABLE IF NOT EXISTS raw.practice_pace (
    event_id          INTEGER NOT NULL REFERENCES raw.events(event_id) ON DELETE CASCADE,
    driver_code       TEXT NOT NULL REFERENCES raw.drivers(driver_code),
    session_name      TEXT NOT NULL,         -- FP1 / FP2 / FP3
    longrun_median_ms INTEGER,
    clean_laps        SMALLINT,
    PRIMARY KEY (event_id, driver_code, session_name)
);

CREATE TABLE IF NOT EXISTS raw.weather (
    event_id      INTEGER NOT NULL REFERENCES raw.events(event_id) ON DELETE CASCADE,
    session_name  TEXT NOT NULL,
    air_temp_c    REAL,
    track_temp_c  REAL,
    humidity      REAL,
    rainfall      BOOLEAN,
    PRIMARY KEY (event_id, session_name)
);

-- Safety car / VSC exposure, read off race lap track-status flags.
CREATE TABLE IF NOT EXISTS raw.race_flags (
    event_id      INTEGER PRIMARY KEY REFERENCES raw.events(event_id) ON DELETE CASCADE,
    had_sc        BOOLEAN,
    had_vsc       BOOLEAN,
    sc_laps       SMALLINT
);

-- OpenF1 covers 2023+ only; strictly additive, never required.
CREATE TABLE IF NOT EXISTS raw.openf1_enrich (
    event_id        INTEGER NOT NULL REFERENCES raw.events(event_id) ON DELETE CASCADE,
    driver_code     TEXT NOT NULL REFERENCES raw.drivers(driver_code),
    pit_stop_count  SMALLINT,
    stint_count     SMALLINT,
    compounds       TEXT[],
    PRIMARY KEY (event_id, driver_code)
);

-- Lets an ingest run skip work it has already done, so the weekly refresh
-- only fetches the new weekend instead of re-walking nine seasons.
CREATE TABLE IF NOT EXISTS raw.ingest_log (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT NOT NULL,             -- fastf1 / openf1
    season        SMALLINT NOT NULL,
    round         SMALLINT NOT NULL,
    session_name  TEXT NOT NULL,
    status        TEXT NOT NULL,             -- ok / empty / error
    row_count     INTEGER,
    message       TEXT,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, season, round, session_name)
);

CREATE INDEX IF NOT EXISTS idx_events_season_round ON raw.events (season, round);
CREATE INDEX IF NOT EXISTS idx_race_results_event  ON raw.race_results (event_id);

-- -------------------------------------------------------------------- core

-- One row per driver per race: every feature, plus the targets. Features are
-- always computed in full; config.yaml decides which ones a given model is
-- allowed to see, so an ablation never needs a schema change.
CREATE TABLE IF NOT EXISTS core.entry_features (
    event_id                         INTEGER NOT NULL REFERENCES raw.events(event_id) ON DELETE CASCADE,
    driver_code                      TEXT NOT NULL REFERENCES raw.drivers(driver_code),
    season                           SMALLINT NOT NULL,
    round                            SMALLINT NOT NULL,
    event_date                       DATE NOT NULL,
    circuit_key                      TEXT NOT NULL,
    team_key                         TEXT NOT NULL,

    -- known before the weekend starts
    driver_avg_finish_3              REAL,
    driver_avg_finish_5              REAL,
    driver_points_5                  REAL,
    driver_dnf_rate_10               REAL,
    driver_experience_races          REAL,
    driver_vs_teammate_quali_delta_5 REAL,
    team_avg_finish_3                REAL,
    team_avg_finish_5                REAL,
    team_avg_quali_pos_3             REAL,
    team_dnf_rate_10                 REAL,
    driver_circuit_avg_finish_3      REAL,
    driver_circuit_appearances       REAL,
    circuit_overtaking_index         REAL,
    circuit_sc_rate                  REAL,
    circuit_laps                     REAL,
    circuit_is_street                REAL,
    season_round                     REAL,
    is_sprint_weekend                REAL,

    -- known only once qualifying has run
    grid_position                    REAL,
    quali_gap_to_pole_pct            REAL,
    quali_gap_to_teammate_pct        REAL,
    fp_longrun_gap_pct               REAL,
    fp_longrun_rank                  REAL,

    -- targets
    finish_position                  SMALLINT,
    dnf                              BOOLEAN,
    classified                       BOOLEAN,

    feature_set_version              SMALLINT NOT NULL,
    PRIMARY KEY (event_id, driver_code)
);

CREATE INDEX IF NOT EXISTS idx_entry_features_date ON core.entry_features (event_date);

-- ---------------------------------------------------------------------- ml

CREATE TABLE IF NOT EXISTS ml.runs (
    run_id       BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind         TEXT NOT NULL,              -- backtest / tune / predict
    mode         TEXT NOT NULL,              -- pre_weekend / post_quali
    config_json  JSONB NOT NULL,             -- the knobs this run actually used
    git_sha      TEXT,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS ml.backtest_metrics (
    run_id          BIGINT NOT NULL REFERENCES ml.runs(run_id) ON DELETE CASCADE,
    model_name      TEXT NOT NULL,           -- ridge / rf / xgb / baseline_*
    mae             REAL,
    spearman        REAL,
    podium_overlap  REAL,                    -- of 3
    winner_acc      REAL,
    top10_acc       REAL,                    -- of 10
    n_races         INTEGER,
    n_rows          INTEGER,
    PRIMARY KEY (run_id, model_name)
);

CREATE TABLE IF NOT EXISTS ml.predictions (
    run_id            BIGINT NOT NULL REFERENCES ml.runs(run_id) ON DELETE CASCADE,
    event_id          INTEGER NOT NULL,
    driver_code       TEXT NOT NULL,
    model_name        TEXT NOT NULL,
    predicted_value   REAL,                  -- raw regressor output
    predicted_rank    SMALLINT,              -- after ranking within the race
    p_dnf             REAL,
    actual_position   SMALLINT,              -- null when predicting a future race
    PRIMARY KEY (run_id, event_id, driver_code, model_name)
);

CREATE TABLE IF NOT EXISTS ml.tuned_params (
    model_name   TEXT NOT NULL,
    mode         TEXT NOT NULL,
    params       JSONB NOT NULL,
    mae          REAL,
    run_id       BIGINT REFERENCES ml.runs(run_id) ON DELETE SET NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_name, mode)
);
