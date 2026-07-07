"""
Anomaly detection over the ticket dataset.

Two anomaly types, deliberately kept rule-based / statistical rather than
ML-based: at 500 rows there isn't enough data to train anything meaningful,
and simple, explainable rules are exactly what a support-ops team would
trust and act on.

1. long_resolution: resolution_time_hrs beyond a per-category IQR fence
   (Q3 + 1.5*IQR). Using per-category fences instead of one global cutoff
   avoids flagging every "Technical" ticket just because Technical tickets
   are inherently slower than "General" ones.
2. stale_unresolved: unresolved tickets (Open/Escalated) whose age exceeds
   a priority-based SLA threshold (default: Critical=24h, High=48h,
   Medium=72h, Low=120h). Thresholds are configurable.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_SLA_HOURS = {
    "Critical": 24,
    "High": 48,
    "Medium": 72,
    "Low": 120,
}


def detect_long_resolutions(df: pd.DataFrame) -> list[dict]:
    resolved = df[df["resolution_time_hrs"].notna()].copy()
    anomalies = []
    for category, group in resolved.groupby("category"):
        q1, q3 = group["resolution_time_hrs"].quantile([0.25, 0.75])
        iqr = q3 - q1
        fence = q3 + 1.5 * iqr
        flagged = group[group["resolution_time_hrs"] > fence]
        for _, row in flagged.iterrows():
            anomalies.append({
                "ticket_id": row["ticket_id"],
                "category": category,
                "resolution_time_hrs": float(row["resolution_time_hrs"]),
                "category_fence_hrs": round(float(fence), 1),
                "reason": f"Resolution time {row['resolution_time_hrs']}h exceeds the "
                          f"{category} category's normal upper bound (~{round(fence,1)}h)",
            })
    return anomalies


def detect_stale_unresolved(df: pd.DataFrame, sla_hours: dict | None = None) -> list[dict]:
    sla = sla_hours or DEFAULT_SLA_HOURS
    reference_now = df["created_at"].max()  # dataset has no true "now"; use latest timestamp seen
    unresolved = df[df["status"].isin(["Open", "Escalated"])].copy()
    unresolved["age_hrs"] = (reference_now - unresolved["created_at"]).dt.total_seconds() / 3600

    anomalies = []
    for _, row in unresolved.iterrows():
        threshold = sla.get(row["priority"], 72)
        if row["age_hrs"] >= threshold:
            anomalies.append({
                "ticket_id": row["ticket_id"],
                "priority": row["priority"],
                "status": row["status"],
                "age_hrs": round(float(row["age_hrs"]), 1),
                "sla_threshold_hrs": threshold,
                "reason": f"{row['priority']} priority ticket has been {row['status'].lower()} for "
                          f"{round(row['age_hrs'],1)}h, past the {threshold}h SLA",
            })
    return anomalies


def detect_all_anomalies(df: pd.DataFrame) -> dict:
    long_res = detect_long_resolutions(df)
    stale = detect_stale_unresolved(df)
    return {
        "long_resolution_anomalies": long_res,
        "stale_unresolved_anomalies": stale,
        "total_anomalies": len(long_res) + len(stale),
    }
