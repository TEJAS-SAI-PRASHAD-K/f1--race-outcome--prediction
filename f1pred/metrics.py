"""How good is a predicted finishing order?

MAE alone is a poor summary - it says nothing about whether the top of the
order is right, which is the part anyone cares about. So every evaluation
reports five numbers side by side, and always next to a naive baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_within_race(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Turn raw regressor output into positions 1..N inside each race.

    Ranking rather than rounding guarantees a valid finishing order: no two
    drivers can both be predicted P4, and nobody is predicted P21.
    """
    return (df.groupby("event_id")[value_col]
              .rank(method="first", ascending=True)
              .astype(int))


def _one_race(actual: pd.Series, predicted_rank: pd.Series) -> dict[str, float]:
    n = len(actual)
    out = {"mae": float(np.abs(predicted_rank - actual).mean())}

    # Spearman is undefined when everything ties or there is a single entry.
    if n > 2 and actual.nunique() > 1 and predicted_rank.nunique() > 1:
        rho = spearmanr(actual, predicted_rank).statistic
        out["spearman"] = float(rho) if np.isfinite(rho) else np.nan
    else:
        out["spearman"] = np.nan

    order = pd.DataFrame({"actual": actual, "pred": predicted_rank})
    pred_top3 = set(order.nsmallest(3, "pred").index)
    act_top3 = set(order.nsmallest(3, "actual").index)
    out["podium_overlap"] = float(len(pred_top3 & act_top3))

    pred_top10 = set(order.nsmallest(min(10, n), "pred").index)
    act_top10 = set(order.nsmallest(min(10, n), "actual").index)
    out["top10_acc"] = float(len(pred_top10 & act_top10))

    pred_winner = order["pred"].idxmin()
    act_winner = order["actual"].idxmin()
    out["winner_acc"] = float(pred_winner == act_winner)
    return out


def evaluate(df: pd.DataFrame, value_col: str = "predicted_value",
             actual_col: str = "finish_position") -> dict[str, float]:
    """Score a frame of predictions, averaging the per-race metrics.

    Averaging per race rather than per row stops a 20-car grand prix from
    outweighing a shortened one, and keeps the podium/winner numbers
    interpretable as "per race".
    """
    df = df.dropna(subset=[actual_col]).copy()
    if df.empty:
        return {k: np.nan for k in
                ("mae", "spearman", "podium_overlap", "winner_acc",
                 "top10_acc", "n_races", "n_rows")}

    if "predicted_rank" not in df.columns:
        df["predicted_rank"] = rank_within_race(df, value_col)

    rows = [
        _one_race(g[actual_col].astype(float), g["predicted_rank"].astype(float))
        for _, g in df.groupby("event_id")
    ]
    per_race = pd.DataFrame(rows)

    return {
        "mae": float(per_race["mae"].mean()),
        "spearman": float(per_race["spearman"].mean(skipna=True)),
        "podium_overlap": float(per_race["podium_overlap"].mean()),
        "winner_acc": float(per_race["winner_acc"].mean()),
        "top10_acc": float(per_race["top10_acc"].mean()),
        "n_races": int(len(per_race)),
        "n_rows": int(len(df)),
    }


def format_table(results: dict[str, dict[str, float]]) -> str:
    """Render the comparison table that every evaluation prints."""
    header = (f"{'model':<24}{'MAE':>7}{'Spearman':>10}"
              f"{'Podium':>9}{'Winner':>9}{'Top10':>8}{'Races':>7}")
    lines = [header, "-" * len(header)]
    for name, m in sorted(results.items(), key=lambda kv: kv[1].get("mae", 9e9)):
        lines.append(
            f"{name:<24}{m['mae']:>7.2f}{m['spearman']:>10.3f}"
            f"{m['podium_overlap']:>8.2f}/3{m['winner_acc']:>8.0%}"
            f"{m['top10_acc']:>7.2f}{m['n_races']:>7}"
        )
    return "\n".join(lines)
