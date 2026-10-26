"""Create the schemas and tables. Idempotent: re-running changes nothing.

    python scripts/init_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1pred.db import apply_schema, read_sql  # noqa: E402


def main() -> None:
    apply_schema()
    tables = read_sql(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN ('raw', 'core', 'ml')
        ORDER BY table_schema, table_name
        """
    )
    print(f"schema applied — {len(tables)} tables")
    for schema, group in tables.groupby("table_schema"):
        print(f"  {schema}: {', '.join(group['table_name'])}")


if __name__ == "__main__":
    main()
