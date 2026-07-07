"""
Data ingestion & normalization layer.

Design choice: we load the CSV once into a pandas DataFrame at startup and keep
it in memory (500 rows is trivial). This avoids the operational overhead of a
database for a dataset this size, while still exposing a clean, typed
DataFrame that both the analytics tools and the anomaly engine can share.

If this needed to scale past a few hundred thousand rows, the natural next
step would be to load into SQLite/DuckDB and let the LLM-selected tools issue
SQL instead of pandas ops (see README "Scaling" section).
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
]


def load_tickets(csv_path: str | Path) -> pd.DataFrame:
    """Load and normalize the support ticket CSV.

    Handles:
      - missing/aliased column names (resp_time_hrs vs response_time_hrs, etc.)
      - mixed date formats (day-first and ISO)
      - blank strings vs NaN for unresolved tickets
    """
    df = pd.read_csv(csv_path)

    # normalize column name aliases seen across sample data / spec
    rename_map = {
        "resp_time_hrs": "response_time_hrs",
        "resol_time_hrs": "resolution_time_hrs",
        "cust_rating": "customer_rating",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # dayfirst=True handles DD-MM-YYYY; pandas falls back gracefully for ISO too
    df["created_at"] = pd.to_datetime(df["created_at"], dayfirst=True, errors="coerce")

    numeric_cols = ["response_time_hrs", "resolution_time_hrs", "customer_rating"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["category", "priority", "status", "agent_id"]:
        df[col] = df[col].astype(str).str.strip()

    df["issue_summary"] = df["issue_summary"].astype(str).str.strip()

    # sanity: unresolved tickets should have null resolution_time_hrs / customer_rating
    unresolved_mask = df["status"].isin(["Open", "Escalated"])
    df.loc[unresolved_mask & df["resolution_time_hrs"].isna(), "resolution_time_hrs"] = pd.NA

    return df.reset_index(drop=True)


_CACHE: dict[str, pd.DataFrame] = {}


def get_tickets_df(csv_path: str | Path = "data/support_tickets.csv") -> pd.DataFrame:
    """Cached accessor so we don't re-read the CSV on every request."""
    key = str(csv_path)
    if key not in _CACHE:
        _CACHE[key] = load_tickets(csv_path)
    return _CACHE[key]


def reload_tickets(csv_path: str | Path = "data/support_tickets.csv") -> pd.DataFrame:
    """Force a reload (e.g. if the underlying CSV changed)."""
    key = str(csv_path)
    _CACHE[key] = load_tickets(csv_path)
    return _CACHE[key]
