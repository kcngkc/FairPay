# FairPay AI ⚖️💰

**Autonomous Compensation Benchmarking Agent — Built for the Real World**

*Google Cloud Rapid Agent Hackathon 2026 — Fivetran Track*

[![Google ADK](https://img.shields.io/badge/Google_ADK-2.1.0-4285F4?style=for-the-badge&logo=google-cloud)](https://adk.dev/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash_|_Pro-8E75B2?style=for-the-badge&logo=google)](https://ai.google.dev/)
[![Fivetran](https://img.shields.io/badge/Fivetran-MCP_Server-0073FF?style=for-the-badge)](https://fivetran.com/)
[![BigQuery](https://img.shields.io/badge/BigQuery-Analytics-669DF6?style=for-the-badge&logo=google-cloud)](https://cloud.google.com/bigquery)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org/)

[🚀 Live Demo](https://fairpay-ai-307483004262.us-central1.run.app) • [🎥 Demo Video](https://vimeo.com/1199498932) • [📖 Source Code](./fairpay_agent/agent.py)

---

## Inspiration

Every year, companies lose top talent because they don't know if they're paying fairly — until it's too late. HR teams rely on outdated compensation surveys, gut feelings, and static spreadsheets to make decisions that directly affect people's livelihoods and organizational competitiveness.

The data exists. The Bureau of Labor Statistics publishes detailed wage data for 830+ occupations across 400+ metro areas. Internal HRIS systems already track every employee's pay. But connecting these two worlds — reliably, repeatedly, and at the speed of business — is a manual, error-prone process that most companies do once a year at best.

We asked: **What if an AI agent could continuously verify data freshness, benchmark against authoritative market data, detect compensation misalignment, and present explainable recommendations — all while keeping humans in the loop for every impactful decision?**

---

## What it does

FairPay AI is a multi-agent system that answers one critical question:

> **"Are we paying our people fairly compared to the market?"**

The system:

- **Validates data pipelines** — checks that both market and internal compensation data are fresh and healthy using Fivetran MCP before any analysis begins
- **Benchmarks compensation** — compares internal pay against U.S. Bureau of Labor Statistics OEWS data by role, geography, and seniority level using a 5-level competitiveness scale
- **Detects market misalignment** — automatically escalates when employees are paid significantly below market (compa-ratio < 0.85), with severity levels for critical (<0.75) and high (0.75-0.85) gaps
- **Generates explainable reports** — produces executive-ready narratives with deterministic confidence scores, data coverage summaries, and recommended actions with named owners
- **Requires human approval** — the HITL escalation gate ensures no HR-impacting action is taken without explicit authorization
- **Handles dirty data** — uses SAFE_CAST patterns to gracefully process suppressed (`#`, `*`) values in government statistical datasets

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SequentialAgent (deterministic)              │
│                  No LLM controls the flow — only reasoning      │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Step 0:     │    │  Step 1:     │    │  Step 2:     │       │
│  │  Input       │───▶│  Data Health │───▶│  Data Gap    │       │
│  │  Router      │    │  Agent       │    │  Detector    │       │
│  │              │    │              │    │              │       │
│  │ gemini-2.5   │    │ gemini-2.5   │    │ gemini-2.5   │       │
│  │ -flash       │    │ -flash       │    │ -pro         │       │
│  │              │    │              │    │              │       │
│  │ classify     │    │ Fivetran MCP │    │ BigQuery     │       │
│  │ intent       │    │ 7 tools      │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│  state["user_intent"] state["health_report"] state["gap_report"]│
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────┐               │
│  │  Step 3:     │    │  Step 4:                 │               │
│  │  Benchmarking│───▶│  Narrative + HITL Gate   │               │
│  │  Agent       │    │                          │               │
│  │              │    │  gemini-2.5-pro          │               │
│  │ gemini-2.5   │    │                          │               │
│  │ -flash       │    │  escalate_market_        │               │
│  │              │    │  misalignment            │               │
│  │ BigQuery +   │    │  (compa < 0.85 only)     │               │
│  │ SAFE_CAST    │    │                          │               │
│  └──────────────┘    └──────────────────────────┘               │
│         │                          │                            │
│         ▼                          ▼                            │
│  state["benchmark_results"]  state["final_report"]              │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Roster

| Agent | Model | Role | Tools | Runs Every Turn? |
|-------|-------|------|-------|-----------------|
| Input Router | gemini-2.5-flash | Validates & classifies user intent | None | ✅ Yes |
| Data Health | gemini-2.5-flash | Fivetran pipeline health check | Fivetran MCP (7 tools) | ❌ Cached after first turn |
| Data Gap Detector | gemini-2.5-pro | Finds missing market benchmarks | `query_bigquery` | ❌ Cached after first turn |
| Benchmarking | gemini-2.5-flash | Computes compa-ratio, confidence | `query_bigquery` + SAFE_CAST | ✅ Yes |
| Narrative | gemini-2.5-pro | Executive report + HITL gate | `escalate_market_misalignment` | ✅ Yes |

### Data Flow

```
User query
    │
    ▼
Input Router → classifies intent → state["user_intent"]
    │
    ▼
Data Health → Fivetran MCP list_connections → state["health_report"]
    │                                           (cached on repeat turns)
    ▼
Data Gap → BigQuery: HRIS vs BLS coverage → state["gap_report"]
    │                                           (cached on repeat turns)
    ▼
Benchmarking → BigQuery: market + internal → state["benchmark_results"]
    │            compa_ratio, confidence score
    ▼
Narrative → Executive report + HITL escalation → state["final_report"]
```

---

## How we built it

**Agent Framework**: Google ADK v2.1.0 with `SequentialAgent` for deterministic orchestration. No LLM decides which agent runs — the pipeline is fixed and predictable.

**Dual-Model Strategy**: Gemini 2.5 Flash for tool-calling agents (speed + cost efficiency) and Gemini 2.5 Pro for reasoning agents (accuracy + prose quality). Each agent has tuned `GenerateContentConfig` — temperature 0.1 for tools, 0.5 for narrative.

**Fivetran MCP Integration**: The Data Health Agent connects to Fivetran's MCP server via `StdioServerParameters`, using 7 of the 77 available tools to check connector status, last sync time, and schema health for 3 data pipelines.

**BigQuery + SAFE_CAST**: Three datasets power the analysis:
- `bls_oews_market_data` — BLS OEWS May 2025 (830+ occupations × 400+ metros)
- `internal_hris_compensation` — Synthetic HRIS for a multi-location electronics manufacturer
- `xref_position_to_soc` — Maps internal job titles to Standard Occupational Classification codes

All wage columns use `SAFE_CAST(REPLACE(REPLACE(col, '#', ''), '*', '') AS NUMERIC)` to handle suppressed government data.

**Session Caching**: A `before_agent_callback` checks if `health_report` and `gap_report` already exist in `session.state`. If so, those agents are skipped — saving ~15 seconds per subsequent query in the same session.

**Tool Scoping**: Each agent's instruction includes explicit `YOUR TOOLS` and `DO NOT USE` sections, preventing tool confusion when 77+ Fivetran MCP tools share the SequentialAgent's context.

**HITL Escalation Gate**: `escalate_market_misalignment` is called only when compa-ratio < 0.85. This is an exception gate — routine reports complete without human intervention.

---

## Challenges we ran into

**Fivetran MCP subprocess authentication**: `StdioServerParameters.env` *replaces* the entire subprocess environment rather than merging with it. The MCP server silently failed because it had no `PATH` or `HOME`. Fix: explicitly pass system environment variables alongside credentials.

**77 tools flooding Gemini's context**: When Fivetran's MCP server loaded all 77 tools into the shared SequentialAgent context, downstream agents tried to call Fivetran tools instead of their own. Fix: instruction-based tool scoping (`YOUR TOOLS / DO NOT USE`) proved more reliable than API-level filtering.

**BLS OEWS data quality**: The Bureau of Labor Statistics uses `#` for "wage exceeds $100/hour" and `*` for "estimate not releasable." These characters break `CAST` operations in BigQuery. Fix: `SAFE_CAST` with nested `REPLACE` on all 7 wage columns.

**Fivetran column renaming**: Fivetran's Google Sheets connector silently renamed `A_PCT25` to `a_pct_25` during ingestion. The Benchmarking Agent initially failed, then self-diagnosed the column name mismatch by querying `INFORMATION_SCHEMA.COLUMNS` and correcting its SQL — without human intervention.

**SequentialAgent shared state**: All sub-agents share the same `InvocationContext`, which means tool lists bleed across agents. Combined with 77 Fivetran tools, this caused the Narrative Agent to attempt Fivetran API calls instead of generating reports. Fix: explicit tool boundaries in each agent's system prompt.

**`require_confirmation` incompatibility**: ADK v2.1.0's `FunctionTool(fn, require_confirmation=True)` caused initialization errors. Fix: moved the HITL pattern to an instruction-driven approach where the agent calls the tool and presents the result for user action.

---

## Accomplishments that we're proud of

- **Self-healing SQL**: The Benchmarking Agent detected a column name discrepancy between source data and BigQuery, queried the schema, and corrected its own SQL — zero human intervention needed.

- **5-agent deterministic pipeline**: No LLM decides what runs next. The `SequentialAgent` guarantees the same execution order every time: validate → detect gaps → benchmark → report.

- **Session caching pattern**: Health and gap checks run once per session. Subsequent queries skip straight to benchmarking, reducing response time from ~45s to ~15s.

- **Production-grade data handling**: `SAFE_CAST` on all wage columns, retry logic with exponential backoff on BigQuery, and explicit NULL handling for suppressed BLS values.

- **HITL exception gate**: The escalation tool fires only on material findings (compa-ratio < 0.85). Routine reports complete end-to-end without human intervention — the agent knows when to ask and when to just deliver.

- **5-level competitiveness scale**: Goes beyond binary "above/below market" with Significantly Below [RED], Below [ORANGE], At Market [GREEN], Above [BLUE], Significantly Above [PURPLE].

- **Multi-location comparison**: A single query like "Compare FAE pay across Cupertino, Brea, and Bellevue" generates a side-by-side table with market medians, compa-ratios, and headcounts.

---

## What we learned

1. **MCP subprocess environments are isolated** — always explicitly pass `PATH`, `HOME`, and any required system variables in `StdioServerParameters.env`.

2. **Tool scoping through instructions > API-level filtering** — when 77 tools share a context, telling the LLM "DO NOT USE these tools" in the system prompt is more reliable than trying to hide them.

3. **Deterministic scoring beats LLM-generated confidence** — a rules-based confidence score (start at 5, deduct for specific conditions) produces the same output for the same input. Every time.

4. **Session state is the backbone of multi-agent data flow** — `output_key` + `{variable_name}` templating in instructions is the cleanest way to pass data between agents in a `SequentialAgent`.

5. **`SAFE_CAST` is essential for government data** — any system that ingests BLS, Census, or similar federal datasets must handle suppressed values at the SQL level, not the application level.

6. **Model selection is a cost-quality tradeoff** — Flash for tool calling (fast, cheap, good enough), Pro for reasoning and prose (slower, more expensive, noticeably better output).

---

## What's next for FairPay AI

- **Expanded HRIS integration** — Integrate with more HRIS platforms (Workday, BambooHR) via Fivetran connectors
- **Attrition prediction** - Add predictive attrition coupling (flight risk × underpayment)
- **Pay equity analysis** — gender and demographic pay gap detection when statistically significant sample sizes are available
- **Additional market data** — international benchmarks (EU, APAC) and industry-specific surveys
- **Total Reward** — Add total compensation modeling (equity, bonuses, benefits)
- **Multi-tenant architecture** — one deployment serving multiple business units with role-based access

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent Framework | Google ADK 2.1.0 | SequentialAgent orchestration |
| LLM (Tools) | Gemini 2.5 Flash | Fast tool calling, classification |
| LLM (Reasoning) | Gemini 2.5 Pro | Complex analysis, prose generation |
| Data Pipeline | Fivetran MCP Server | Pipeline health monitoring |
| MCP Protocol | Anthropic MCP SDK 1.27.2 | Agent ↔ Fivetran communication |
| Data Warehouse | Google BigQuery | Market + HRIS data queries |
| Market Data | BLS OEWS May 2025 | 830+ occupations × 400+ metros |
| Language | Python 3.12 | Runtime |
| Cloud Platform | Google Cloud | Cloud Run |

---

## Project Structure

```
fairpay-ai/
├── .env                         # Fivetran + GCP credentials
├── fairpay_agent/
│   ├── __init__.py
│   └── agent.py                 # 5-agent pipeline 
├── fivetran-mcp/
│   ├── server.py                # Fivetran MCP server
│   └── .venv/                   # MCP server dependencies
└── .venv/                       # ADK runtime environment
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/kcngkc/FairPay.git
cd fairpay-ai

# 2. Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install google-adk google-cloud-bigquery python-dotenv

# 4. Set up Fivetran MCP server
cd fivetran-mcp
python3.12 -m venv .venv
source .venv/bin/activate
pip install mcp requests fivetran-connector-sdk
deactivate
cd ..

# 5. Configure credentials
cat > .env << 'EOF'
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=fairpay-498216
GOOGLE_CLOUD_LOCATION=us-central1
FIVETRAN_API_KEY=your_key
FIVETRAN_API_SECRET=your_secret
FIVETRAN_ALLOW_WRITES=true
EOF

# 6. Authenticate to Google Cloud
gcloud auth application-default login
gcloud auth application-default set-quota-project fairpay-498216

# 7. Launch
source .venv/bin/activate
adk web .
```

Open `http://localhost:8000` → Select **fairpay_agent** → Try:

```
Are we paying our FAE Engineers in Cupertino fairly compared to market?
```

---

## Sample Output

```
## FairPay AI — Compensation Analysis Report

**Role:** FAE Manager
**SOC Code:** 11-9041 — Architectural and Engineering Managers
**Location:** Cupertino
**Compa-Ratio:** 0.74 — Significantly Below Market [RED]

🚨 MARKET MISALIGNMENT ALERT (severity: critical)
This finding has been escalated to HR leadership.
```

---

## License

[Apache 2.0](LICENSE)

---

*Built with ❤️ for the Google Cloud Rapid Agent Hackathon 2026 — Fivetran Track*
