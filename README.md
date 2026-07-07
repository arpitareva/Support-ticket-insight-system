# Support Ticket Insight

An AI-powered system for querying and monitoring customer support ticket data.

**Ingests** a CSV of support tickets → **answers natural language questions**
about it via an LLM → **detects anomalies** (slow resolutions, SLA breaches)
→ exposes both a **REST API** and a **Streamlit UI**.

## Quick start

```bash
git clone <your-repo-url>
cd dotmappers-assessment
cp .env.example .env
# edit .env and paste in a free Groq API key from https://console.groq.com/keys
```

**Option A — Docker :**
```bash
docker-compose up
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

**Option B — local Python:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export $(cat .env | xargs)

uvicorn app.main:app --reload            # API on :8000
streamlit run ui/streamlit_app.py         # UI on :8501 (separate terminal)
```


## Architecture

```
CSV file
   │
   ▼
ingestion.py  ── loads + normalizes into a pandas DataFrame (in-memory, cached)
   │
   ├──────────────┬─────────────────────┐
   ▼              ▼                     ▼
tools.py      anomaly.py            (shared DataFrame)
(deterministic   (rule-based /
 aggregations)    statistical rules)
   ▲              ▲
   │              │
llm_agent.py ──────┘
(Groq tool-calling: picks a tool + args, we execute it, LLM phrases the answer)
   ▲
   │
main.py (FastAPI: /health, /query, /anomalies, /reload)
   ▲
   │
ui/streamlit_app.py (thin client over the same API)
```

### Key design decision: tool-calling instead of code generation

The LLM is **never** allowed to generate and execute arbitrary pandas/SQL
code against the data. Instead it sees a small fixed set of typed Python
functions (`count_tickets`, `average_metric`, `agent_leaderboard`,
`filter_tickets`, `detect_all_anomalies`) exposed as Groq tool schemas, picks
one + its arguments, and we run it deterministically. The LLM's only other
job is to phrase the final sentence using the *real* numbers we computed.

Trade-off: an unusual one-off question that doesn't map to any of the five
tools won't be answered numerically (the model will say so rather than
guess). In exchange, every number the system reports is reproducible,
testable in isolation from the LLM, and immune to code-injection or
hallucinated statistics — which matters more for a support-ops tool people
will actually trust.

### Anomaly detection logic

- **Long resolution times**: per-category IQR fence (`Q3 + 1.5×IQR`), so a
  Technical ticket isn't flagged just for taking longer than a General one —
  each category has its own "normal" baseline.
- **Stale unresolved tickets**: priority-based SLA thresholds (Critical
  24h / High 48h / Medium 72h / Low 120h, configurable in `anomaly.py`),
  measured against the latest timestamp seen in the dataset (used as a
  stand-in for "now" since the data is historical).

### Why in-memory pandas instead of a database

500 rows fits trivially in memory; a database would add operational
overhead with no benefit at this scale. See "Scaling" below for what
changes if the dataset grows.

## Model / tools used

- **LLM**: Groq free tier, `openai/gpt-oss-120b` (fast, free, supports
  OpenAI-compatible tool calling) — swap via `GROQ_MODEL` env var.
- **API**: FastAPI + Uvicorn
- **UI**: Streamlit
- **Data**: pandas

## API endpoints

| Method | Path         | Description                                  |
|--------|--------------|-----------------------------------------------|
| GET    | `/health`    | Confirms the CSV loaded and row count         |
| POST   | `/query`     | `{"question": "..."}` → NL answer + tool trace |
| GET    | `/anomalies` | Full anomaly report (long resolutions + stale tickets) |
| POST   | `/reload`    | Re-reads the CSV from disk without a restart   |

## Example queries & outputs

```bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many tickets are currently open?"}'
```
```json
{
  "answer": "There are 4 tickets currently marked as Open.",
  "tool_used": "count_tickets",
  "tool_args": {"status": "Open"},
  "tool_result": {"count": 4, "filters": {...}}
}
```

```bash
curl localhost:8000/anomalies
```
```json
{
  "long_resolution_anomalies": [
    {"ticket_id": "TKT-015", "category": "Billing", "resolution_time_hrs": 37.7,
     "category_fence_hrs": 26.3, "reason": "Resolution time 37.7h exceeds the Billing category's normal upper bound (~26.3h)"}
  ],
  "stale_unresolved_anomalies": [
    {"ticket_id": "TKT-007", "priority": "High", "status": "Open", "age_hrs": 1609.4,
     "sla_threshold_hrs": 48, "reason": "High priority ticket has been open for 1609.4h, past the 48h SLA"}
  ],
  "total_anomalies": 2
}
```

## Tests

Deterministic logic (ingestion, tools, anomaly rules) is unit-tested and
requires **no API key**:
```bash
pytest tests/ -v
```
The LLM query path is exercised via the `/query` endpoint or the Streamlit
UI once `GROQ_API_KEY` is set.

## Known limitations

- The tool-calling agent can only answer questions that map to one of its
  five tools; genuinely novel analytical questions ("correlate rating with
  agent tenure") would need a new tool added, not just a different query.
- SLA thresholds and IQR-fence sensitivity are hardcoded defaults — in
  production these would be configurable per team/organization.
- `filter_tickets` results are capped (`limit`, default 20) to keep LLM
  context small; there's no pagination in the current API.
- Single in-memory DataFrame — fine at 500 rows, not designed for
  concurrent writes or very large datasets (see Scaling below).
- No auth on the API — out of scope for this assessment but would be
  required before any real deployment.

## Scaling

At meaningfully larger volumes (tens of thousands+ rows, or multiple
concurrent writers), the two changes I'd make first:
1. Move ingestion from an in-memory DataFrame to SQLite/DuckDB, and have the
   tool functions issue parameterized SQL instead of pandas filtering —
   same tool-calling architecture, just a different execution backend.
2. Precompute the anomaly scan on a schedule (e.g. every few minutes) rather
   than on every `/anomalies` request, since the IQR fences don't need to be
   recalculated per-request.
