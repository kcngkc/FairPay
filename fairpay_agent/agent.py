"""
FairPay AI — Production Multi-Agent Compensation Analysis Pipeline
Google Cloud Rapid Agent Hackathon 2026 — Fivetran Track

Architecture:
============
SequentialAgent (deterministic orchestration — no LLM controls flow)
├── 0. input_router        — Validates & classifies user intent
│                            (gemini-2.5-flash — fast classification)
├── 1. data_health_agent   — Fivetran MCP: pipeline health check
│                            (gemini-2.5-flash — fast tool calling)
├── 2. data_gap_agent      — BigQuery: coverage gap detection
│                            (gemini-3.5-flash — deterministic reasoning)
├── 3. benchmarking_agent  — BigQuery: market vs internal analysis
│                            (gemini-2.5-flash — deterministic math)
└── 4. narrative_agent     — Gemini: executive report + HITL gate
                             (gemini-2.5-pro — nuanced prose)

Data Flow (via session.state — output_key → {variable_name}):
=============================================================
input_router       → state["user_intent"]
data_health_agent  → state["health_report"]
data_gap_agent     → state["gap_report"]
benchmarking_agent → state["benchmark_results"]
narrative_agent    → state["final_report"]

HITL Pattern:
=============
escalate_market_misalignment is called ONLY when compa_ratio < 0.85
(Significantly Below Market). ADK shows the escalation result and the
user decides whether to act. This is an exception gate, not routine.

Model Strategy:
===============
- gemini-2.5-flash: tool-calling + classification (speed + cost)
- gemini-2.5-pro: reasoning + prose (accuracy + nuance)
- Each agent has tuned GenerateContentConfig for its role
"""

import os
import json
import time
from google.cloud import bigquery
from google.adk.agents import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool import MCPToolset
from google.genai import types
from mcp import StdioServerParameters


# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIVETRAN_PYTHON = os.path.join(PROJECT_ROOT, "fivetran-mcp", ".venv", "bin", "python3")
FIVETRAN_SERVER = os.path.join(PROJECT_ROOT, "fivetran-mcp", "server.py")

GCP_PROJECT = "fairpay-498216"
BQ_MARKET = f"`{GCP_PROJECT}.google_sheets.bls_oews_market_data`"
BQ_XREF = f"`{GCP_PROJECT}.google_sheets.xref_position_to_soc`"
BQ_HRIS = f"`{GCP_PROJECT}.internal.internal_hris_compensation`"


# ═══════════════════════════════════════════════════════════
# MODEL CONFIGS (tuned per agent role)
# ═══════════════════════════════════════════════════════════

TOOL_AGENT_CONFIG = types.GenerateContentConfig(
    temperature=0.1,
    top_p=0.95,
    max_output_tokens=4096,
)

BENCHMARK_CONFIG = types.GenerateContentConfig(
    temperature=0.2,
    top_p=0.95,
    max_output_tokens=4096,
)

NARRATIVE_CONFIG = types.GenerateContentConfig(
    temperature=0.5,
    top_p=0.95,
    max_output_tokens=8192,
)


# ═══════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════

# -- BigQuery Tool --

_bq_client = bigquery.Client(project=GCP_PROJECT)


def query_bigquery(sql: str) -> str:
    """Execute a SQL query against BigQuery and return JSON results.

    Use this tool to query compensation and market data tables.
    Always use fully-qualified table names with backticks.

    Args:
        sql: A BigQuery Standard SQL query string.

    Returns:
        JSON string containing query results as a list of row dicts,
        or a JSON object with an error message if the query fails.
    """
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            query_job = _bq_client.query(sql)
            rows = list(query_job.result(timeout=30))
            if not rows:
                return json.dumps({"status": "success", "rows": [], "row_count": 0})
            results = [dict(row) for row in rows]
            return json.dumps(
                {"status": "success", "rows": results, "row_count": len(results)},
                default=str,
            )
        except Exception as e:
            if attempt < max_retries and any(
                keyword in str(e).lower()
                for keyword in ["timeout", "rate", "503", "retry"]
            ):
                time.sleep(2 ** attempt)
                continue
            return json.dumps({
                "status": "error",
                "error": str(e),
                "sql": sql,
                "suggestion": "Check table name, column names, or permissions.",
            })


bigquery_tool = FunctionTool(query_bigquery)


# -- Human-in-the-Loop: Market Misalignment Escalation --
# EXCEPTION gate — called only when compa_ratio < 0.85.
# This flags critically underpaid positions for HR action.


def escalate_market_misalignment(finding_summary: str, severity: str) -> dict:
    """Escalate a market misalignment finding to HR leadership.

    WHEN TO CALL THIS TOOL:
    - Call with severity='critical' when compa_ratio < 0.75
    - Call with severity='high' when compa_ratio is 0.75 to 0.85

    WHEN NOT TO CALL THIS TOOL:
    - Do NOT call when compa_ratio >= 0.85
    - Do NOT call for routine reports where pay is at or above market

    Args:
        finding_summary: A 2-3 sentence factual summary of the finding.
            Example: "Software Developers in Cupertino have a compa-ratio
            of 0.78 (Below Market). Internal avg $142K vs market median
            $213K. 3 employees affected."
        severity: Must be 'high' (0.75-0.85) or 'critical' (<0.75).

    Returns:
        Dict with escalation status, summary, and recommended action.
    """
    return {
        "status": "escalated_to_hr",
        "finding_summary": finding_summary,
        "severity": severity,
        "recommended_action": (
            f"MARKET MISALIGNMENT ALERT (severity: {severity}): "
            "This finding has been escalated to HR leadership. "
            "The compensation committee should review market adjustment "
            "options within 30 days to reduce attrition risk."
        ),
    }

# -- Fivetran MCP Toolset --
# NOTE: StdioServerParameters.env REPLACES the subprocess environment.
# We must include PATH and HOME for the MCP server to function.

# Fivetran MCP — auto-detects local vs Cloud Run.
# Local: connects via StdioServerParameters (subprocess)
# Cloud Run: fivetran_toolset = Rest API (graceful skip)

_mcp_available = os.path.exists(FIVETRAN_PYTHON) and os.path.exists(FIVETRAN_SERVER)

if _mcp_available:
    fivetran_toolset = MCPToolset(
        connection_params=StdioServerParameters(
            command=FIVETRAN_PYTHON,
            args=[FIVETRAN_SERVER],
            env={
                "FIVETRAN_API_KEY": os.environ.get("FIVETRAN_API_KEY", ""),
                "FIVETRAN_API_SECRET": os.environ.get("FIVETRAN_API_SECRET", ""),
                "FIVETRAN_ALLOW_WRITES": "true",
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", ""),
            },
        ),
    )
else:
    fivetran_toolset = None


# ═══════════════════════════════════════════════════════════
# SESSION CACHE — Skip agents if results already exist
# ═══════════════════════════════════════════════════════════

def skip_if_cached(callback_context: CallbackContext) -> types.Content | None:
    """Skip an agent if its output_key already exists in session state.

    Used by data_health and data_gap agents so they only run once
    per session. Subsequent turns reuse the cached report.

    To force a refresh, include 'refresh' or 'recheck' in the message.
    """
    agent_name = callback_context.agent_name
    state = callback_context.state

    # Check if user wants to force a refresh
    user_message = state.get("user_message", "")
    if isinstance(user_message, str) and any(
        word in user_message.lower()
        for word in ["refresh", "recheck", "rerun", "check pipeline"]
    ):
        return None

    cache_keys = {
        "data_health": "health_report",
        "data_gap_detector": "gap_report",
    }

    output_key = cache_keys.get(agent_name)
    if output_key and output_key in state and state[output_key]:
        cached = state[output_key]
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=str(cached))],
        )

    return None


# ═══════════════════════════════════════════════════════════
DEFAULT_STATE = {
    "health_report": "",
    "gap_report": "",
    "user_intent": "",
    "benchmark_results": "",
    "final_report": "",
}


def initialize_state(callback_context: CallbackContext) -> None:
    """Set default state values if they don't exist yet."""
    for key, default in DEFAULT_STATE.items():
        if key not in callback_context.state:
            callback_context.state[key] = default


# ═══════════════════════════════════════════════════════════
# # AGENT 0: INPUT ROUTER
# ═══════════════════════════════════════════════════════════

input_router = LlmAgent(
    name="input_router",
    before_agent_callback=initialize_state,
    model="gemini-3.5-flash",
    description="Validates user input and classifies intent for the pipeline.",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0, top_p=0.95, max_output_tokens=512,
    ),
    output_key="user_intent",
    instruction="""You are the Input Router for FairPay AI.

YOUR JOB: Classify the user's message into one of these categories:

1. COMPENSATION_QUERY — User asks about pay, salary, market rate,
   benchmarking, compensation for a specific role or location.
   Extract: role, location, analysis_type.

2. PIPELINE_CHECK — User asks about data freshness, pipeline status,
   Fivetran connectors, or sync health.

3. REFRESH — User says "refresh", "recheck", "rerun", "reload connection schema" , "Sync connection" or similar.

4. OUT_OF_SCOPE — Greetings, unrelated questions, or anything not
   about compensation analysis or data pipelines.

══════════════════════════════════════
YOU HAVE NO TOOLS. Do not call any tools.
Just classify the message and output the result.
══════════════════════════════════════

OUTPUT FORMAT:
Intent: [COMPENSATION_QUERY | PIPELINE_CHECK | REFRESH | OUT_OF_SCOPE]
Role: [extracted role or "N/A"]
Location: [extracted location or "all"]
Analysis Type: [benchmark | comparison | overview]
Original Question: [user's exact message]

If OUT_OF_SCOPE, add:
Message: FairPay AI is a compensation benchmarking system. I can help
you benchmark salaries against market data or check data pipeline
health. What would you like to analyze?""",
)


# ═══════════════════════════════════════════════════════════
# AGENT 1: DATA HEALTH (Fivetran MCP)
# ═══════════════════════════════════════════════════════════

data_health_agent = LlmAgent(
    name="data_health",
    model="gemini-3.5-flash",
    description="Checks Fivetran pipeline health and passes user question downstream.",
    generate_content_config=TOOL_AGENT_CONFIG,
    tools=[fivetran_toolset] if fivetran_toolset else [],
    output_key="health_report",
    before_agent_callback=skip_if_cached,
    instruction="""You are the Data Health Agent, the FIRST step in the
FairPay AI compensation analysis pipeline.

YOU HAVE TWO JOBS — ALWAYS DO BOTH:

JOB 1: Check Fivetran pipeline health.
JOB 2: Pass through the user's question for downstream agents.

You are NOT the only agent. After you finish, three more agents will
run: Data Gap Detector, Benchmarking Agent, and Narrative Agent.
They need your health report AND the user's original question.

══════════════════════════════════════
YOUR TOOLS (use ONLY these):
══════════════════════════════════════
Fivetran MCP tools:
- list_connections → list all Fivetran connectors
- get_connection_details → get sync status, last sync time, setup_state
- get_connection_state → check current sync state
- get_connection_schema_config → verify table schema matches expectations
- reload_connection_schema_config → refresh if HRIS structure changed
- sync_connection → trigger a new sync (only when user explicitly asks)
- modify_connection_state → pause/resume connectors

DO NOT USE: query_bigquery, escalate_market_misalignment
Those tools belong to other agents.
══════════════════════════════════════

JOB 1: PIPELINE HEALTH CHECK
1. Call `list_connections` to discover all Fivetran connectors.
2. For each connection, call `get_connection_details`.
3. Report each connector's name, sync status, last sync, errors.

Ignore any connector named "fivetran_metadata_removable_shipping".

JOB 2: PASS THROUGH USER QUESTION
After the health check, ALWAYS include the user's original question
at the end of your output, exactly as they wrote it.

OUTPUT FORMAT (MANDATORY):
Pipeline Health Summary:
- [connector_name]: [status] | Last sync: [timestamp] | Errors: [none/desc]
Overall: [HEALTHY / DEGRADED / CRITICAL]


CRITICAL RULES:
- ALWAYS call list_connections first. This is mandatory.
- ALWAYS include "User Question:" at the end. This is mandatory.
- NEVER refuse to process a message. You are part of a pipeline.
- NEVER say "I cannot answer that" or "that's not my job."
- Execute immediately. Do NOT ask permission.""",
)


# ═══════════════════════════════════════════════════════════
# AGENT 2: DATA GAP DETECTION (BigQuery)
# ═══════════════════════════════════════════════════════════

data_gap_agent = LlmAgent(
    name="data_gap_detector",
    model="gemini-3.5-flash",
    description="Detects gaps between internal HRIS positions and available BLS market benchmarks.",
    generate_content_config=TOOL_AGENT_CONFIG,
    tools=[bigquery_tool],
    output_key="gap_report",
    before_agent_callback=skip_if_cached,
    instruction=f"""You are the Data Gap Detector for FairPay AI.

Read {{health_report}} for pipeline freshness context.

YOUR JOB: Detect gaps between internal HRIS data and available BLS
market benchmarks so downstream agents know what data is missing.

══════════════════════════════════════
YOUR TOOLS (use ONLY these):
══════════════════════════════════════
- query_bigquery → Execute SQL against BigQuery

DO NOT USE any Fivetran tools or escalate_market_misalignment.
Those tools belong to other agents.
══════════════════════════════════════

STEP 1: Find all distinct position-location combinations in HRIS:
```sql
SELECT DISTINCT soc_code, location_type
FROM {BQ_HRIS}
WHERE soc_code IS NOT NULL
ORDER BY soc_code, location_type
```

STEP 2: Check which SOC codes exist in market data:
```sql
SELECT DISTINCT OCC_CODE, location_type
FROM {BQ_MARKET}
ORDER BY OCC_CODE, location_type
```

STEP 3: Compare the two result sets. Flag any (soc_code, location_type)
pairs from HRIS that do NOT have a matching row in market data.

NOTE: If there is a new location with all HRIS hired date less than 14 days ago and its positions may not yet have SOC mappings in the xref table. List them as Location gaps.

OUTPUT FORMAT:
Data Gap Report:
- Matched pairs: [count] positions have market benchmarks
- Missing pairs: [list each soc_code + location_type with no market data]
- Location gaps: [location and any position missing from xref or market data]
- Action needed: none | expand_bls_data | verify_soc_codes | update_xref

Execute immediately. Do NOT ask the user for anything.""",
)


# ═══════════════════════════════════════════════════════════
# AGENT 3: BENCHMARKING (BigQuery)
# ═══════════════════════════════════════════════════════════

benchmarking_agent = LlmAgent(
    name="benchmarking",
    model="gemini-3.5-flash",
    description="Performs compensation benchmarking with SAFE_CAST for dirty BLS data, computes compa-ratio and deterministic confidence score.",
    generate_content_config=BENCHMARK_CONFIG,
    tools=[bigquery_tool],
    output_key="benchmark_results",
    instruction=f"""You are the Benchmarking Agent for FairPay AI.

Read {{health_report}} and {{gap_report}} for context.

YOUR JOB: Perform compensation benchmarking using BigQuery data.

══════════════════════════════════════
YOUR TOOLS (use ONLY these):
══════════════════════════════════════
- query_bigquery → Execute SQL against BigQuery

DO NOT USE any Fivetran tools or escalate_market_misalignment.
Those tools belong to other agents.
══════════════════════════════════════

IMPORTANT — DATA QUALITY:
BLS OEWS data contains '#' and '*' for suppressed/unavailable values.
ALWAYS use SAFE_CAST with REPLACE to handle these characters.

STEP 1: Identify the role and location from the user question in
{{health_report}} (look for "User Question:" at the end).

STEP 2: Look up the SOC code:
```sql
SELECT company_position, soc_code, soc_title, location_type, level
FROM {BQ_XREF}
WHERE LOWER(company_position) LIKE LOWER('%[position]%')
```

STEP 3: Query market data for LOCAL area + NATIONAL.
Use SAFE_CAST on ALL wage columns:
```sql
SELECT OCC_CODE, OCC_TITLE, location_type, AREA, AREA_TITLE,
  SAFE_CAST(REPLACE(REPLACE(A_MEDIAN, '#', ''), '*', '') AS NUMERIC) AS A_MEDIAN,
  SAFE_CAST(REPLACE(REPLACE(A_MEAN, '#', ''), '*', '') AS NUMERIC) AS A_MEAN,
  SAFE_CAST(REPLACE(REPLACE(a_pct_10, '#', ''), '*', '') AS NUMERIC) AS a_pct_10,
  SAFE_CAST(REPLACE(REPLACE(a_pct_25, '#', ''), '*', '') AS NUMERIC) AS a_pct_25,
  SAFE_CAST(REPLACE(REPLACE(a_pct_75, '#', ''), '*', '') AS NUMERIC) AS a_pct_75,
  SAFE_CAST(REPLACE(REPLACE(a_pct_90, '#', ''), '*', '') AS NUMERIC) AS a_pct_90,
  SAFE_CAST(REPLACE(REPLACE(TOT_EMP, '#', ''), '*', '') AS NUMERIC) AS TOT_EMP
FROM {BQ_MARKET}
WHERE OCC_CODE = '[soc_code]'
  AND location_type IN ('[local_type]', 'national')
```

Area code reference:
- manufacturing (Fort Worth) = AREA 19100
- sales_office_SJ (Cupertino) = AREA 41940
- sales_office_LA (Brea) = AREA 31080
- sales_office_WA (Bellevue) = AREA 42660
- sales_office_AU (Austin) = AREA 12420
- national = AREA 99

STEP 4: Query internal compensation:
```sql
SELECT company_position, base_salary_annual, gender,
       location_type, level, hire_date
FROM {BQ_HRIS}
WHERE soc_code = '[soc_code]'
  AND location_type = '[location_type]'
```

The level column tells the percentile to compare against:
Junior / Individual → 25th–50th percentile
Senior / Supervisor → 50th–75th percentile
Manager / Director → 75th–90th percentile
Executive → 90th percentile

STEP 5: Compute metrics (do the math yourself from query results):
- compa_ratio = AVG(internal base_salary_annual) / market A_MEDIAN
- Tenure-pay correlation flag: if newest hires (< 1 year) earn more
  than 3+ year employees at the same level, flag as "pay compression detected"
- Competitiveness (5-level scale):
  compa_ratio < 0.85 = "Significantly Below Market" [RED]
  0.85-0.95 = "Below Market" [ORANGE]
  0.95-1.05 = "At Market" [GREEN]
  1.05-1.15 = "Above Market" [BLUE]
  > 1.15 = "Significantly Above Market" [PURPLE]

If any SAFE_CAST returns NULL for a wage column, note that value as
"suppressed" and exclude it from calculations. Do NOT treat NULL as 0.

STEP 6: Compute DETERMINISTIC confidence score (start at 5, deduct):
- Deduct 1 if any pipeline last_sync > 7 days ago (check {{health_report}})
- Deduct 1 if internal sample size < 3 employees
- Deduct 1 if market data contains suppressed values (NULL from SAFE_CAST)
- Deduct 1 if SOC code is a broad group (ends in 0, e.g., 13-1020)
- Minimum score = 1

OUTPUT FORMAT (structured text for the narrative agent):
Benchmark Results:
- Role: [company_position]
- SOC Code: [XX-XXXX] — [OCC_TITLE]
- Location: [city description]
- Market Median (Local): $[amount]
- Market Mean (Local): $[amount]
- Market 10th (Local): $[amount]
- Market 25th (Local): $[amount]
- Market 75th (Local): $[amount]
- Market 90th (Local): $[amount]
- Market Median (National): $[amount]
- Market Mean (National): $[amount]
- Internal Avg: $[amount]
- Internal Count: [N]
- Internal Min: $[amount]
- Internal Max: $[amount]
- Internal Levels: [list levels and count per level]
- Compa-Ratio: [X.XX]
- Competitiveness: [rating + color indicator]
- Tenure-Pay Flag: [none / pay compression detected]
- Suppressed Values: [list any NULL columns from SAFE_CAST]
- Confidence Score: [1-5]
- Confidence Factors: [list of deductions applied]

If the user asked about multiple positions or locations, repeat the
analysis for each one.

Execute ALL steps immediately. Do NOT ask for permission.""",
)


# ═══════════════════════════════════════════════════════════
# AGENT 4: NARRATIVE + HITL MARKET MISALIGNMENT GATE
# ═══════════════════════════════════════════════════════════

narrative_agent = LlmAgent(
    name="narrative",
    model="gemini-3.1-pro-preview",
    description="Generates executive compensation report. Escalates to HR when employees are significantly below market.",
    generate_content_config=NARRATIVE_CONFIG,
    tools=[],
    output_key="final_report",
    instruction="""You are the Narrative Agent for FairPay AI.

Read {benchmark_results}, {health_report}, and {gap_report}.

YOUR JOB: Generate the polished executive report using the EXACT
format below. Use ONLY the data from {benchmark_results}. Do NOT
recalculate any numbers. Do NOT invent data.

══════════════════════════════════════
DO NOT USE: any Fivetran tools.
Those tools belong to other agents.
══════════════════════════════════════

══════════════════════════════════════════════════════════════
CRITICAL BEHAVIOR RULES
══════════════════════════════════════════════════════════════

1. NEVER ask the user for permission to run queries or look up data.
   Deliver the complete report in a SINGLE response.

2. If data is missing, state it as a finding. Do not stop and ask.

═══════════════════════════════════════════════════════
REPORT FORMAT — SINGLE ROLE
═══════════════════════════════════════════════════════

## FairPay AI — Compensation Analysis Report

**Role:** [from benchmark_results]
**SOC Code:** [from benchmark_results]
**Location:** [from benchmark_results]
**Data Sources:** BLS OEWS May 2025 | Internal HRIS
**Pipeline Health:** [from health_report — HEALTHY/DEGRADED/CRITICAL]

---

### 1. Executive Summary
[2-3 sentences. Lead with key finding. Be direct and factual.]

### 2. Market Position

| Metric | Internal | Local Market | National |
|--------|----------|-------------|----------|
| Median | — | $XXX,XXX | $XXX,XXX |
| Mean   | $XXX,XXX (avg) | $XXX,XXX | $XXX,XXX |
| 10th Pct | — | $XXX,XXX | $XXX,XXX |
| 25th Pct | — | $XXX,XXX | $XXX,XXX |
| 75th Pct | — | $XXX,XXX | $XXX,XXX |
| 90th Pct | — | $XXX,XXX | $XXX,XXX |
| Min/Max | $XXX,XXX / $XXX,XXX | — | — |

**Compa-Ratio:** X.XX — [Competitiveness rating with color indicator]

### 3. Compensation Distribution

| Level | Count | Avg Salary | Relevant Percentile |
|-------|-------|------------|---------------------|
| [level] | X | $XXX,XXX | vs [Nth] market pct |

[If tenure-pay flag is "pay compression detected", add:]
**Tenure-Pay Alert:** Recent hires are earning more than tenured
employees at the same level. This may indicate pay compression.

### 4. Data Coverage
[Summarize gap_report — any missing market data for positions]
[Note any suppressed values from SAFE_CAST]

### 5. Confidence Score
**Rating:** [star emojis] ([X]/5)
**Factors:**
[List each factor from benchmark_results confidence_factors]

### 6. Recommended Actions
1. [Action] — **Owner:** [Team]
2. [Action] — **Owner:** [Team]
3. [Action] — **Owner:** [Team]

---

═══════════════════════════════════════════════════════
COMPARISON REPORT FORMAT (when multiple roles/locations)
═══════════════════════════════════════════════════════

## FairPay AI — Multi-Location Compensation Comparison

**SOC Code:** [XX-XXXX] — [Title]
**Locations Compared:** [list all]
**Data Sources:** BLS OEWS May 2025 | Internal HRIS

---

### Market Position Comparison

| Metric | [Location 1] | [Location 2] | [Location 3] | National |
|--------|-------------|-------------|-------------|----------|
| Market Median | $XXX | $XXX | $XXX | $XXX |
| Internal Avg | $XXX | $XXX | $XXX | — |
| Compa-Ratio | X.XX | X.XX | X.XX | — |
| Rating | [emoji] | [emoji] | [emoji] | — |
| Headcount | N | N | N | — |
| Tenure Flag | [none/flag] | [none/flag] | [none/flag] | — |

### Key Findings
1. [Highest paid location vs lowest — delta in $ and %]
2. [Any location significantly below market (compa < 0.85)]
3. [Any location with pay compression detected]

Use this format when {benchmark_results} contains data for 2+ locations.
Use the single-role format when only 1 location is analyzed.

**Advisory Notice:** This analysis is for decision support only.
FairPay AI does not make compensation decisions. All recommendations
require human review and approval.

---

═══════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════

1. Do NOT use hype words: revolutionize, game-changing, pivotal,
   cutting-edge, leverage, synergy, unlock, empower.

2. Do NOT recalculate the confidence score. Use EXACTLY what
   {benchmark_results} provides.

3. Do NOT invent numbers. If a metric is missing from
   {benchmark_results}, write "Data not available."

4. MARKET MISALIGNMENT ESCALATION — EXCEPTION ONLY:
4. MARKET MISALIGNMENT ESCALATION — EXCEPTION ONLY:
   - If compa_ratio < 0.75, add this block to the report:

     ### ⚠️ ESCALATION: CRITICAL MARKET MISALIGNMENT
     **Severity:** CRITICAL (compa-ratio < 0.75)
     **Status:** Escalated to HR Leadership
     **Finding:** [2-sentence factual summary with numbers]
     **Action Required:** Compensation committee must review market
     adjustment options within 30 days to reduce attrition risk.

   - If compa_ratio is 0.75-0.85, add this block:

     ### ⚠️ ESCALATION: HIGH MARKET MISALIGNMENT
     **Severity:** HIGH (compa-ratio 0.75–0.85)
     **Status:** Escalated to HR Leadership
     **Finding:** [2-sentence factual summary with numbers]
     **Action Required:** Compensation committee should review
     market adjustment options within 30 days.

   - If compa_ratio >= 0.85: do NOT add an escalation block.
     State: "Compensation is within acceptable market range."
5. Do NOT ask follow-up questions. Do NOT end with "Would you like
   me to..." or any continuation prompt. This is a terminal report.
   End with the report content and nothing else.
""",
)


# ═══════════════════════════════════════════════════════════
# ROOT AGENT: SEQUENTIAL ORCHESTRATOR (must be LAST)
# ═══════════════════════════════════════════════════════════

root_agent = SequentialAgent(
    name="fairpay_orchestrator",
    description=(
        "FairPay AI: deterministic 5-step compensation analysis pipeline. "
        "Step 0: Input router validates & classifies user request. "
        "Step 1: Fivetran MCP pipeline health check. "
        "Step 2: BigQuery data coverage gap detection. "
        "Step 3: Market vs internal benchmarking with SAFE_CAST and deterministic scoring. "
        "Step 4: Executive report with HITL market misalignment gate."
    ),
    sub_agents=[
        input_router,
        data_health_agent,
        data_gap_agent,
        benchmarking_agent,
        narrative_agent,
    ],
)
