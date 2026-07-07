"""
Deterministic analytics functions ("tools") that the LLM can invoke.

Architecture note: we deliberately do NOT let the LLM generate and execute
raw pandas/SQL code against the dataset. Instead the LLM sees a small fixed
set of typed functions (via tool-calling) and picks one + arguments. This:
  - removes code-injection / arbitrary-execution risk entirely
  - keeps every numeric answer 100% deterministic (LLM only phrases the
    final sentence, it never computes the number itself)
  - is easy to unit test in isolation from the LLM

This trades a bit of flexibility (an odd one-off question might not map to
any tool) for reliability and safety, which is the right trade-off for a
support-ticket analytics system where numbers need to be trustworthy.
"""

from __future__ import annotations

import pandas as pd
from typing import Optional


def _filter(df: pd.DataFrame, *, status=None, priority=None, category=None, agent_id=None) -> pd.DataFrame:
    out = df
    if status:
        out = out[out["status"].str.lower() == status.lower()]
    if priority:
        out = out[out["priority"].str.lower() == priority.lower()]
    if category:
        out = out[out["category"].str.lower() == category.lower()]
    if agent_id:
        out = out[out["agent_id"].str.lower() == agent_id.lower()]
    return out


def count_tickets(df: pd.DataFrame, status: Optional[str] = None, priority: Optional[str] = None,
                   category: Optional[str] = None, agent_id: Optional[str] = None) -> dict:
    """Count tickets matching optional filters."""
    filtered = _filter(df, status=status, priority=priority, category=category, agent_id=agent_id)
    return {"count": int(len(filtered)), "filters": {"status": status, "priority": priority,
                                                       "category": category, "agent_id": agent_id}}


def average_metric(df: pd.DataFrame, metric: str, category: Optional[str] = None,
                    priority: Optional[str] = None, agent_id: Optional[str] = None,
                    status: Optional[str] = None) -> dict:
    """Average of customer_rating, response_time_hrs, or resolution_time_hrs, with optional filters."""
    if metric not in {"customer_rating", "response_time_hrs", "resolution_time_hrs"}:
        return {"error": f"unsupported metric '{metric}'"}
    filtered = _filter(df, status=status, priority=priority, category=category, agent_id=agent_id)
    series = filtered[metric].dropna()
    if series.empty:
        return {"average": None, "sample_size": 0, "metric": metric}
    return {"average": round(float(series.mean()), 2), "sample_size": int(series.count()), "metric": metric}


def agent_leaderboard(df: pd.DataFrame, rank_by: str = "resolved_count", top_n: int = 5) -> dict:
    """Rank agents by resolved_count, avg_rating, or avg_resolution_time (ascending = best for time)."""
    resolved = df[df["status"] == "Resolved"]
    grouped = resolved.groupby("agent_id")
    if rank_by == "resolved_count":
        result = grouped.size().sort_values(ascending=False)
        rows = [{"agent_id": a, "resolved_count": int(c)} for a, c in result.head(top_n).items()]
    elif rank_by == "avg_rating":
        result = grouped["customer_rating"].mean().sort_values(ascending=True)  # ascending -> worst first, caller can reverse
        rows = [{"agent_id": a, "avg_rating": round(float(v), 2)} for a, v in result.items()]
        rows_sorted_desc = sorted(rows, key=lambda r: r["avg_rating"], reverse=True)
        rows_sorted_asc = sorted(rows, key=lambda r: r["avg_rating"])
        return {"rank_by": rank_by, "best": rows_sorted_desc[:top_n], "worst": rows_sorted_asc[:top_n]}
    elif rank_by == "avg_resolution_time":
        result = grouped["resolution_time_hrs"].mean().sort_values(ascending=True)
        rows = [{"agent_id": a, "avg_resolution_time_hrs": round(float(v), 2)} for a, v in result.head(top_n).items()]
    else:
        return {"error": f"unsupported rank_by '{rank_by}'"}
    return {"rank_by": rank_by, "top": rows}


def filter_tickets(df: pd.DataFrame, status: Optional[str] = None, priority: Optional[str] = None,
                    category: Optional[str] = None, min_resolution_hrs: Optional[float] = None,
                    max_age_hrs: Optional[float] = None, limit: int = 20) -> dict:
    """Return a (capped) list of tickets matching filters. max_age_hrs filters on hours since created_at (relative to the latest timestamp in the dataset, used as a stand-in for 'now')."""
    filtered = _filter(df, status=status, priority=priority, category=category)
    if min_resolution_hrs is not None:
        filtered = filtered[filtered["resolution_time_hrs"] >= min_resolution_hrs]
    if max_age_hrs is not None:
        reference_now = df["created_at"].max()
        age_hrs = (reference_now - filtered["created_at"]).dt.total_seconds() / 3600
        filtered = filtered[age_hrs >= max_age_hrs]
    cols = ["ticket_id", "created_at", "category", "priority", "status",
            "resolution_time_hrs", "agent_id", "issue_summary"]
    rows = filtered[cols].head(limit).copy()
    rows["created_at"] = rows["created_at"].astype(str)
    return {"total_matches": int(len(filtered)), "returned": rows.to_dict(orient="records")}
