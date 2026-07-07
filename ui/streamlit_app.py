"""
Streamlit UI for the Support Ticket Insight system.

Talks to the FastAPI backend over HTTP so the UI stays a thin client — the
same API used here is what the evaluator can also hit directly with curl or
Swagger (/docs).
"""

import os
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Support Ticket Insight", page_icon="🎫", layout="wide")
st.title("🎫 Support Ticket Insight")
st.caption("Ask questions about your support ticket data, or check for anomalies.")

tab_query, tab_anomalies, tab_health = st.tabs(["Ask a question", "Anomalies", "System health"])

with tab_query:
    st.subheader("Natural language query")
    example_qs = [
        "How many tickets are currently open?",
        "Which agent resolved the most tickets?",
        "What is the average customer rating for Technical category tickets?",
        "Show me all Critical tickets not resolved within 12 hours.",
    ]
    st.write("Examples:", " · ".join(f"`{q}`" for q in example_qs))

    question = st.text_input("Your question", placeholder="e.g. How many critical tickets are unresolved?")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{API_BASE}/query", json={"question": question}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                st.success(data.get("answer", "(no answer returned)"))
                with st.expander("Show underlying tool call & raw result"):
                    st.json(data)
            except requests.RequestException as e:
                st.error(f"Request to API failed: {e}")

with tab_anomalies:
    st.subheader("Detected anomalies")
    if st.button("Run anomaly scan"):
        with st.spinner("Scanning..."):
            try:
                resp = requests.get(f"{API_BASE}/anomalies", timeout=30)
                resp.raise_for_status()
                data = resp.json()
                st.metric("Total anomalies", data.get("total_anomalies", 0))

                st.markdown("**Abnormally long resolution times**")
                long_res = data.get("long_resolution_anomalies", [])
                if long_res:
                    st.dataframe(long_res)
                else:
                    st.write("None found.")

                st.markdown("**Unresolved tickets past SLA**")
                stale = data.get("stale_unresolved_anomalies", [])
                if stale:
                    st.dataframe(stale)
                else:
                    st.write("None found.")
            except requests.RequestException as e:
                st.error(f"Request to API failed: {e}")

with tab_health:
    st.subheader("API health")
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=10)
        resp.raise_for_status()
        st.json(resp.json())
    except requests.RequestException as e:
        st.error(f"Could not reach API at {API_BASE}: {e}")
