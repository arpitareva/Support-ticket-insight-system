from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import ingestion, anomaly, llm_agent

CSV_PATH = os.environ.get("TICKETS_CSV_PATH", "data/support_tickets.csv")

app = FastAPI(
    title="Support Ticket Insight API",
    description="NL query + anomaly detection over customer support ticket data",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


@app.on_event("startup")
def _startup() -> None:
    # fail fast at boot if the CSV is malformed, rather than on first request
    ingestion.get_tickets_df(CSV_PATH)


@app.get("/health")
def health() -> dict:
    try:
        df = ingestion.get_tickets_df(CSV_PATH)
        return {"status": "ok", "rows_loaded": int(len(df))}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Data not loaded: {e}")


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    df = ingestion.get_tickets_df(CSV_PATH)
    try:
        result = llm_agent.answer_question(req.question, df)
    except RuntimeError as e:
        # e.g. missing GROQ_API_KEY
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"LLM query failed: {e}")
    return result


@app.get("/anomalies")
def anomalies() -> dict:
    df = ingestion.get_tickets_df(CSV_PATH)
    return anomaly.detect_all_anomalies(df)


@app.post("/reload")
def reload_data() -> dict:
    """Convenience endpoint: re-read the CSV from disk without restarting the server."""
    df = ingestion.reload_tickets(CSV_PATH)
    return {"status": "reloaded", "rows_loaded": int(len(df))}
