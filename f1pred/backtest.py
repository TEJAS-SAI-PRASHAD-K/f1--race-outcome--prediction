"""Walk-forward backtest: train on the past, score the next race, repeat.

Why not train_test_split or KFold? Because a shuffled split lets the model
train on races that happened AFTER the one it is scoring. Rolling form
features make that catastrophic - the model effectively reads next week's
newspaper, and a broken pipeline will look brilliant. Chronological
evaluation is the only honest option here, so there is no shuffle switch.

For each held-out race: fit on every race strictly before it, predict, move
on. Metrics are averaged over the held-out races.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pred import baselines, models
from f1pred.config import Config
from f1pred.db import get_engine, start_run, upsert_df
from f1pred.features import load_features
from f1pred.metrics import evaluate, rank_within_race


def _race_order(df: pd.DataFrame) -> pd.DataFrame:
    return (df[["event_id", "event_date"]]
            .drop_duplicates()
            .sort_values(["event_date", "event_id"])
            .reset_index(drop=True))


def training_rows(train: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Rows the position regressor learns from.

    Retirements are dropped by default: "where would this car have finished"
    is a different question from "did its gearbox survive", and mixing them
    makes the target noisy. DNF risk gets its own model in predict.py.
    """
    if cfg.exclude_dnf_from_regression:
        train = train[~train["dnf"].astype(bool)]
    return train.dropna(subset=["finish_position"])


def run(cfg: Config, mode: str, n_test_races: int | None = None,
        model_overrides: dict[str, dict] | None = None,
        shuffle_target: bool = False, only_models: list[str] | None = None,
        include_baselines: bool = True, skip_last: int = 0,
        persist: bool = True, verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """Returns (metrics per model, all out-of-sample predictions).

    `only_models` restricts which algorithms are fitted, which is what makes
    hyperparameter search affordable: a tuning trial for Ridge has no reason
    to refit a random forest 30 times.

    `skip_last` ends the scoring window that many races before the present.
    Tuning uses it to select on an EARLIER block than the one the final
    backtest reports on; otherwise hyperparameters are chosen on the same
    races they are later judged by, and the headline number flatters itself.
    """
    features = load_features()
    feature_names = cfg.feature_names(mode)
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise KeyError(f"features missing from core.entry_features: {missing}")

    # Only races that have actually been run can be scored.
    scored = features.dropna(subset=["finish_position"])
    order = _race_order(scored)
    n_test = n_test_races or cfg.n_test_races
    end = len(order) - skip_last
    test_events = order.iloc[max(0, end - n_test):end]["event_id"].tolist()

    collected: list[pd.DataFrame] = []

    for event_id in test_events:
        event_date = scored.loc[scored["event_id"] == event_id,
                                "event_date"].iloc[0]
        train = scored[scored["event_date"] < event_date]
        if train["event_id"].nunique() < cfg.min_train_races:
            continue

        test = features[features["event_id"] == event_id].copy()
        fitting = training_rows(train, cfg)
        if fitting.empty or test.empty:
            continue

        X_train = fitting[feature_names]
        y_train = fitting["finish_position"].astype(float)
        if shuffle_target:
            # Leakage probe: destroy the signal inside each race. Any model
            # still scoring well is reading something it should not.
            # The generator is advanced per group rather than reseeded, so
            # every race gets a DIFFERENT permutation. Reusing one seed would
            # apply the same mapping everywhere, which a model can simply
            # learn - making the probe look clean when it is not.
            rng = np.random.default_rng(0)
            y_train = (y_train.groupby(fitting["event_id"])
                       .transform(lambda s: rng.permutation(s.to_numpy())))
        weights = models.sample_weights(fitting, cfg)

        chosen = models.make_models(cfg, model_overrides)
        if only_models is not None:
            chosen = {k: v for k, v in chosen.items() if k in only_models}
        for name, model in chosen.items():
            models.fit(model, X_train, y_train, weights)
            preds = test.copy()
            preds["model_name"] = name
            preds["predicted_value"] = model.predict(test[feature_names])
            collected.append(preds)

        for name in (baselines.applicable(mode) if include_baselines else []):
            preds = test.copy()
            preds["model_name"] = name
            preds["predicted_value"] = baselines.predict(test, name)
            collected.append(preds)

        if verbose:
            print(".", end="", flush=True)

    if verbose:
        print()
    if not collected:
        raise RuntimeError(
            "no races scored - is there enough history for min_train_races?")

    all_preds = pd.concat(collected, ignore_index=True)
    all_preds["predicted_rank"] = (
        all_preds.groupby(["model_name", "event_id"])["predicted_value"]
        .rank(method="first", ascending=True).astype(int)
    )

    results = {
        name: evaluate(group)
        for name, group in all_preds.groupby("model_name")
    }

    if persist:
        run_id = start_run(
            kind="backtest", mode=mode,
            config_json={"training": cfg.raw["training"],
                         "backtest": {"n_test_races": n_test},
                         "features": feature_names,
                         "shuffle_target": shuffle_target},
            notes=f"walk-forward over {len(test_events)} races",
        )
        _persist(run_id, results, all_preds)

    return results, all_preds


def _persist(run_id: int, results: dict, preds: pd.DataFrame) -> None:
    metrics = pd.DataFrame([
        {"run_id": run_id, "model_name": name, **m} for name, m in results.items()
    ])
    upsert_df(metrics, "ml.backtest_metrics", ["run_id", "model_name"])

    rows = preds[["event_id", "driver_code", "model_name", "predicted_value",
                  "predicted_rank", "finish_position"]].copy()
    rows = rows.rename(columns={"finish_position": "actual_position"})
    rows.insert(0, "run_id", run_id)
    rows["p_dnf"] = np.nan
    upsert_df(rows, "ml.predictions",
              ["run_id", "event_id", "driver_code", "model_name"])
