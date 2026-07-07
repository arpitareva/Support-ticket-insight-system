"""
LLM integration layer.

Uses Groq's free tier (OpenAI-compatible tool-calling API) as the NLU engine.
The LLM never touches the data directly and never generates executable code;
it only:
  1. reads the user's question,
  2. picks ONE of the tool functions below + arguments (structured tool call),
  3. after we execute that tool deterministically in Python, restates the
     result in natural language.

This two-pass "reason -> call -> phrase" pattern is what keeps every number
in the final answer traceable back to a pandas computation instead of an
LLM hallucination.
"""

from __future__ import annotations

import json
import os
from groq import Groq

from . import tools, anomaly

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "count_tickets",
            "description": "Count tickets matching optional filters on status, priority, category, agent_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["Open", "Resolved", "Escalated"]},
                    "priority": {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
                    "category": {"type": "string", "enum": ["Billing", "Technical", "General"]},
                    "agent_id": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "average_metric",
            "description": "Average of customer_rating, response_time_hrs, or resolution_time_hrs, with optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["customer_rating", "response_time_hrs", "resolution_time_hrs"]},
                    "status": {"type": "string", "enum": ["Open", "Resolved", "Escalated"]},
                    "priority": {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
                    "category": {"type": "string", "enum": ["Billing", "Technical", "General"]},
                    "agent_id": {"type": "string"},
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_leaderboard",
            "description": "Rank agents by resolved_count, avg_rating, or avg_resolution_time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rank_by": {"type": "string", "enum": ["resolved_count", "avg_rating", "avg_resolution_time"]},
                    "top_n": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_tickets",
            "description": "List tickets matching filters: status, priority, category, min_resolution_hrs, "
                            "max_age_hrs (hours since creation, for finding stale tickets).",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["Open", "Resolved", "Escalated"]},
                    "priority": {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
                    "category": {"type": "string", "enum": ["Billing", "Technical", "General"]},
                    "min_resolution_hrs": {"type": "number"},
                    "max_age_hrs": {"type": "number"},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_all_anomalies",
            "description": "Detect anomalies: abnormally long resolution times and unresolved tickets past their SLA.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_TOOL_IMPL = {
    "count_tickets": tools.count_tickets,
    "average_metric": tools.average_metric,
    "agent_leaderboard": tools.agent_leaderboard,
    "filter_tickets": tools.filter_tickets,
    "detect_all_anomalies": lambda df: anomaly.detect_all_anomalies(df),
}

SYSTEM_PROMPT = (
    "You are an analytics assistant over a customer-support ticket dataset. "
    "For every question, you MUST call exactly one of the provided tools to get "
    "real numbers -- never invent statistics yourself. After you receive the tool "
    "result, answer the user's question in one or two clear sentences using those "
    "exact numbers. If no tool fits the question, say so honestly instead of guessing."
)


def _client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and set it as an environment variable."
        )
    return Groq(api_key=api_key)


def answer_question(question: str, df) -> dict:
    """Run the reason -> call -> phrase loop for one NL question. Returns dict with
    the final answer text plus the raw tool result for transparency/debugging."""
    client = _client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    first = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        temperature=0,
    )
    choice = first.choices[0].message

    if not choice.tool_calls:
        # LLM decided no tool applied -- surface its own text rather than force a call
        return {"answer": choice.content, "tool_used": None, "tool_result": None}

    call = choice.tool_calls[0]
    fn_name = call.function.name
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    impl = _TOOL_IMPL.get(fn_name)
    if impl is None:
        return {"answer": f"Model requested unknown tool '{fn_name}'.", "tool_used": fn_name, "tool_result": None}

    tool_result = impl(df, **args)

    messages.append({"role": "assistant", "content": None, "tool_calls": [call.model_dump()]})
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "name": fn_name,
        "content": json.dumps(tool_result, default=str),
    })

    second = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    final_text = second.choices[0].message.content

    return {"answer": final_text, "tool_used": fn_name, "tool_args": args, "tool_result": tool_result}
