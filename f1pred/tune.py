"""Search for the settings that actually score best, chronologically.

Two stages, on purpose:

1. Find the recency half-life (and the 2026 weight multiplier). With new
   technical regulations this year, how hard old races are discounted moves
   the score far more than any model hyperparameter. Stage one uses Ridge
   alone because it fits in milliseconds.

2. Grid-search each model's own hyperparameters at that recency setting.

A full joint grid would be hours of refitting for a worse answer, since the
two stages barely interact.

Every trial is written to ml.backtest_metrics under one ml.runs row, so the
whole search is inspectable in SQL afterwards rather than lost to scrollback.
"""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import replace

import pandas as pd
from sqlalchemy import text

from f1pred.backtest import run as backtest_run
from f1pred.config import Config
from f1pred.db import get_engine, start_run, upsert_df
from f1pred.models import MODEL_NAMES


def _with_recency(cfg: Config, half_life: float, multiplier: float) -> Config:
    raw = copy.deepcopy(cfg.raw)
    raw["training"]["recency_half_life_races"] = half_life
    raw["training"]["season_weight_multiplier"] = {2026: multiplier}
    return replace(cfg, raw=raw)


def _score(cfg: Config, mode: str, n_races: int, model: str,
           skip_last: int, overrides: dict | None = None) -> dict[str, float]:
    results, _ = backtest_run(cfg, mode, n_test_races=n_races,
                              model_overrides=overrides,
                              only_models=[model], include_baselines=False,
                              skip_last=skip_last,
                              persist=False, verbose=False)
    return results[model]


def tune(cfg: Config, mode: str, verbose: bool = True) -> dict:
    n_races = int(cfg.raw["tuning"]["n_test_races"])
    half_lives = cfg.raw["tuning"]["half_life_grid"]
    multipliers = cfg.raw["tuning"]["season_multiplier_grid"]
    # Select on the block of races immediately BEFORE the backtest window, so
    # scripts/backtest.py still reports on races this search never saw.
    skip_last = cfg.n_test_races

    trials: list[dict] = []

    # ---- stage 1: recency, scored on Ridge ----------------------------
    if verbose:
        print(f"stage 1: recency over {len(half_lives)*len(multipliers)} "
              f"combinations, scored on ridge")
        print(f"         validating on {n_races} races ending {skip_last} "
              f"races before the present, so the backtest window stays clean")
    best_recency, best_mae = (cfg.recency_half_life, 1.0), float("inf")
    for hl, mult in itertools.product(half_lives, multipliers):
        res = _score(_with_recency(cfg, hl, mult), mode, n_races, "ridge",
                     skip_last)
        mae = res["mae"]
        trials.append({"model_name": f"ridge[hl={hl},x2026={mult}]", **res})
        if verbose:
            print(f"  half_life={hl:<5} 2026x={mult:<4} ridge MAE {mae:.3f}")
        if mae < best_mae:
            best_mae, best_recency = mae, (hl, mult)

    hl, mult = best_recency
    tuned_cfg = _with_recency(cfg, hl, mult)
    if verbose:
        print(f"  -> best recency: half_life={hl}, 2026 multiplier={mult} "
              f"(MAE {best_mae:.3f})\n")

    # ---- stage 2: per-model hyperparameters ---------------------------
    best_params: dict[str, dict] = {}
    best_scores: dict[str, float] = {}
    for name in MODEL_NAMES:
        grid = cfg.model_grid(name)
        if not grid:
            continue
        keys = list(grid)
        combos = list(itertools.product(*(grid[k] for k in keys)))
        if verbose:
            print(f"stage 2: {name} over {len(combos)} combinations")

        for combo in combos:
            params = dict(zip(keys, combo))
            res = _score(tuned_cfg, mode, n_races, name, skip_last,
                         overrides={name: params})
            mae = res["mae"]
            label = ",".join(f"{k}={v}" for k, v in params.items())
            trials.append({"model_name": f"{name}[{label}]", **res})
            if verbose:
                print(f"  {label:<48} MAE {mae:.3f}")
            if mae < best_scores.get(name, float("inf")):
                best_scores[name] = mae
                best_params[name] = params
        if verbose:
            print(f"  -> best {name}: {best_params.get(name)} "
                  f"(MAE {best_scores.get(name, float('nan')):.3f})\n")

    run_id = start_run(
        kind="tune", mode=mode,
        config_json={"half_life_grid": half_lives,
                     "season_multiplier_grid": multipliers,
                     "n_test_races": n_races, "skip_last": skip_last},
        notes=f"two-stage search, {len(trials)} trials",
    )
    frame = pd.DataFrame(trials)
    frame.insert(0, "run_id", run_id)
    upsert_df(frame, "ml.backtest_metrics", ["run_id", "model_name"])

    _persist_best(run_id, mode, hl, mult, best_params, best_scores)

    return {"half_life": hl, "season_multiplier": mult,
            "params": best_params, "mae": best_scores, "run_id": run_id}


def _persist_best(run_id: int, mode: str, half_life: float, multiplier: float,
                  params: dict[str, dict], scores: dict[str, float]) -> None:
    """Winners go to ml.tuned_params, which predict.py reads automatically."""
    rows = [{"model_name": name, "mode": mode,
             "params": json.dumps(p), "mae": scores.get(name),
             "run_id": run_id}
            for name, p in params.items()]
    rows.append({"model_name": "_recency", "mode": mode,
                 "params": json.dumps({"recency_half_life_races": half_life,
                                       "season_weight_multiplier": {"2026": multiplier}}),
                 "mae": None, "run_id": run_id})
    with get_engine().begin() as conn:
        for row in rows:
            conn.execute(
                text("INSERT INTO ml.tuned_params "
                     "  (model_name, mode, params, mae, run_id) "
                     "VALUES (:model_name, :mode, CAST(:params AS JSONB), "
                     "        :mae, :run_id) "
                     "ON CONFLICT (model_name, mode) DO UPDATE SET "
                     "  params = EXCLUDED.params, mae = EXCLUDED.mae, "
                     "  run_id = EXCLUDED.run_id, created_at = now()"),
                row,
            )
