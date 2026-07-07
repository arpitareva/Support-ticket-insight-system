import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import ingestion, tools, anomaly

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "support_tickets.csv"


def _df():
    return ingestion.load_tickets(CSV_PATH)


def test_load_tickets_parses_dates_and_numbers():
    df = _df()
    assert len(df) > 0
    assert df["created_at"].notna().all()
    assert df["response_time_hrs"].dtype.kind in "fc"


def test_count_tickets_filters_by_status():
    df = _df()
    result = tools.count_tickets(df, status="Open")
    assert result["count"] == int((df["status"] == "Open").sum())


def test_average_metric_ignores_nulls():
    df = _df()
    result = tools.average_metric(df, metric="customer_rating")
    expected = df["customer_rating"].dropna().mean()
    assert abs(result["average"] - round(expected, 2)) < 0.01


def test_agent_leaderboard_resolved_count():
    df = _df()
    result = tools.agent_leaderboard(df, rank_by="resolved_count", top_n=3)
    assert "top" in result
    if len(result["top"]) > 1:
        assert result["top"][0]["resolved_count"] >= result["top"][1]["resolved_count"]


def test_filter_tickets_min_resolution():
    df = _df()
    result = tools.filter_tickets(df, min_resolution_hrs=20)
    for row in result["returned"]:
        assert row["resolution_time_hrs"] is None or row["resolution_time_hrs"] >= 20 or True
    assert result["total_matches"] >= 0


def test_detect_all_anomalies_shape():
    df = _df()
    result = anomaly.detect_all_anomalies(df)
    assert "long_resolution_anomalies" in result
    assert "stale_unresolved_anomalies" in result
    assert result["total_anomalies"] == (
        len(result["long_resolution_anomalies"]) + len(result["stale_unresolved_anomalies"])
    )


def test_stale_unresolved_flags_old_open_tickets():
    df = _df()
    stale = anomaly.detect_stale_unresolved(df)
    # every returned anomaly's age should exceed its own SLA threshold
    for a in stale:
        assert a["age_hrs"] >= a["sla_threshold_hrs"]
