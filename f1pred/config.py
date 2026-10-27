"""Load config.yaml and .env into a plain object.

Deliberately thin: one YAML file holds every knob, and this module just reads
it. Nothing else in the project is allowed to hardcode a tunable value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Bumped whenever features.py changes shape, so stale rows in
# core.entry_features are detectable instead of silently mixed in.
FEATURE_SET_VERSION = 1

# The two ways the model can be run. pre_weekend knows nothing about the
# running order; post_quali knows the grid, which is a far stronger signal.
MODES = ("pre_weekend", "post_quali")


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(repr=False)
    database_url: str

    # --- convenience accessors, so callers never dig through nested dicts ---
    @property
    def start_season(self) -> int:
        return int(self.raw["data"]["start_season"])

    @property
    def end_season(self) -> int:
        return int(self.raw["data"]["end_season"])

    @property
    def load_practice_pace(self) -> bool:
        return bool(self.raw["data"]["load_practice_pace"])

    @property
    def use_openf1_enrichment(self) -> bool:
        return bool(self.raw["data"]["use_openf1_enrichment"])

    @property
    def recency_half_life(self) -> float:
        return float(self.raw["training"]["recency_half_life_races"])

    @property
    def season_weight_multiplier(self) -> dict[int, float]:
        raw = self.raw["training"].get("season_weight_multiplier") or {}
        return {int(k): float(v) for k, v in raw.items()}

    @property
    def exclude_dnf_from_regression(self) -> bool:
        return bool(self.raw["training"]["exclude_dnf_from_regression"])

    @property
    def n_test_races(self) -> int:
        return int(self.raw["backtest"]["n_test_races"])

    @property
    def min_train_races(self) -> int:
        return int(self.raw["backtest"]["min_train_races"])

    def feature_names(self, mode: str) -> list[str]:
        """Feature columns for a mode.

        pre_weekend gets the base set only. post_quali additionally gets
        everything that is unknown until Saturday evening.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        names = list(self.raw["features"]["base"])
        if mode == "post_quali":
            names += list(self.raw["features"]["post_quali"])
        return names

    @property
    def enabled_models(self) -> list[str]:
        """Algorithms to fit, in config order. Defaults to all configured."""
        names = self.raw["models"].get("enabled")
        if not names:
            names = [k for k in self.raw["models"] if k != "enabled"]
        return list(names)

    def model_defaults(self, name: str) -> dict[str, Any]:
        return dict(self.raw["models"][name]["defaults"])

    def model_grid(self, name: str) -> dict[str, list[Any]]:
        return dict(self.raw["models"][name].get("grid") or {})


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> Config:
    """Read config.yaml + .env. Cached, so it is cheap to call anywhere."""
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = path or DEFAULT_CONFIG_PATH
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and adjust it."
        )
    return Config(raw=raw, database_url=database_url)
