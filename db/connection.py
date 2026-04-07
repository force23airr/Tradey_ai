"""DuckDB connection — single shared connection for the whole project."""

import duckdb
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tradey.duckdb"


def get_conn() -> duckdb.DuckDBPyConnection:
    """Return a persistent DuckDB connection to tradey.duckdb."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))
