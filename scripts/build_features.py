"""Compute core.entry_features from the raw tables.

    python scripts/build_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1pred.features import FEATURE_COLUMNS, build_features, persist  # noqa: E402


def main() -> None:
    df = build_features()
    n = persist(df)
    print(f"built {n} feature rows over {df['event_id'].nunique()} races "
          f"({df['season'].min()}-{df['season'].max()})")
    print(f"{len(FEATURE_COLUMNS)} feature columns, no NaNs: "
          f"{not df[FEATURE_COLUMNS].isna().any().any()}")


if __name__ == "__main__":
    main()
