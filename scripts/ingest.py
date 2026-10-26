"""Load FastF1 race results and qualifying into the raw schema.

    python scripts/ingest.py                 # seasons from config.yaml
    python scripts/ingest.py --seasons 2026  # just this year
    python scripts/ingest.py --force         # re-fetch even if already logged
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1pred.config import load_config  # noqa: E402
from f1pred.ingest import ingest_openf1, ingest_range  # noqa: E402


def parse_seasons(spec: str | None, cfg) -> tuple[int, int]:
    if not spec:
        return cfg.start_season, cfg.end_season
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    return int(spec), int(spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", help="e.g. 2026 or 2018-2026")
    ap.add_argument("--force", action="store_true",
                    help="ignore raw.ingest_log and re-fetch")
    ap.add_argument("--openf1-only", action="store_true",
                    help="skip FastF1 and refresh only the OpenF1 pace data")
    args = ap.parse_args()

    cfg = load_config()
    start, end = parse_seasons(args.seasons, cfg)

    if not args.openf1_only:
        print(f"ingesting {start}-{end} from FastF1/Jolpica")
        ingest_range(start, end, force=args.force)

    if cfg.use_openf1_enrichment:
        print(f"ingesting {start}-{end} practice pace from OpenF1")
        ingest_openf1(start, end)


if __name__ == "__main__":
    main()
