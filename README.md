# AI Data Debugger

AI Data Debugger is a Streamlit-based application for evaluating the reliability of ML and analytics datasets. It combines deterministic data quality checks, baseline-vs-new dataset drift comparison, cleaning impact simulation, data contract export, and optional AI-assisted explanations.

The project is designed to move beyond static CSV profiling. Instead of only showing summary statistics, it helps answer operational questions that matter in ML workflows:

- Has the new batch changed compared with the historical baseline?
- Which data quality issues are most likely to affect model training or reporting?
- What should be reviewed or fixed first?
- Would a conservative cleaning pass improve the dataset health score?
- Can the current dataset profile be turned into a reusable validation contract?

## Overview

AI Data Debugger supports a repeatable dataset reliability workflow:

1. Upload a baseline CSV dataset.
2. Configure dataset roles such as target, timestamp, and entity ID.
3. Run deterministic quality checks and calculate a health score.
4. Upload a newer dataset batch for drift comparison.
5. Review issue drilldowns, recommendations, and visual diagnostics.
6. Simulate conservative cleaning and compare before/after results.
7. Export a cleaned dataset preview.
8. Export a YAML or JSON data contract.
9. Track run history locally across repeated scans.

LLM providers are optional. Detection, scoring, drift metrics, and cleaning simulation remain deterministic. AI providers are only used for explanation, remediation planning, Q&A, report generation, and code suggestions.

## Key Capabilities

- CSV profiling with shape, dtypes, missingness, unique counts, and sample rows
- Missing value, duplicate row, constant column, outlier, dtype, target imbalance, timestamp, and ID-like feature checks
- Health score from 0 to 100 with severity-based deductions
- Baseline-vs-new dataset drift comparison
- Schema drift, missing-value drift, numeric PSI drift, categorical drift, and cardinality drift detection
- ML-aware dataset role setup for target, timestamp, entity ID, protected columns, and ML-excluded columns
- Issue-level investigation cards with ML impact, business impact, recommended fixes, and example pandas code
- Cleaning simulation with before/after rows, columns, health score, and issue count
- Cleaned dataset preview export as CSV, optional Parquet, or ZIP with metadata
- Local SQLite run history without storing raw uploaded data
- Health score and issue trend charts for repeated runs
- YAML and JSON data contract export
- Optional rule-based, local Ollama, or OpenAI explanation layer

## Demo

A guided demo is available in [DEMO.md](DEMO.md).

Sample files:

- `sample_data/baseline_market_events.csv`
- `sample_data/broken_market_events.csv`

The demo scenario compares a healthy historical market-events dataset against a degraded production batch. It highlights missing-value drift, numeric distribution drift, categorical drift, unseen categories, duplicate rows, outliers, and target imbalance.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## AI Provider Options

The application works without any external AI provider. In the default mode, explanations are generated from deterministic rule-based templates.

Supported modes:

1. **Rule-based only**  
   No LLM calls. All explanations, summaries, and recommendations are generated locally from deterministic issue metadata.

2. **Local Ollama**  
   Uses a local Ollama server at `http://localhost:11434/api/chat`.

   ```bash
   ollama serve
   ollama pull llama3.1
   ```

3. **OpenAI API**  
   Uses the official OpenAI Python SDK when available. The API key can be entered in the sidebar for the current session or read from `OPENAI_API_KEY`.

   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

External AI providers do not receive raw uploaded rows. Prompts are built from compact metadata only, such as issue objects, column dtypes, missing rates, unique ratios, summary statistics, health scores, and drift summaries.

## Reliability Workflow

Before running analysis, users can optionally define dataset roles:

- target column
- timestamp column
- entity ID column
- protected columns
- columns to exclude from ML checks

These roles reduce false positives and make the checks more relevant to ML workflows. For example, a `date` column can be treated as a timestamp, and `ticker` can be treated as an entity key instead of a generic high-cardinality categorical feature.

Each analysis run is stored locally in SQLite with:

- run ID
- monitor name
- timestamp
- row and column counts
- health score
- issue counts by severity
- AI provider used
- role and configuration summary
- issue summaries as JSON

Raw uploaded datasets are not persisted in run history.

## Data Contract Export

The app can generate a validation contract from the current dataset profile. Contracts can be exported as YAML or JSON and include:

- required columns
- expected dtypes
- max missing-rate rules
- allowed values for low-cardinality categorical columns
- numeric min/max ranges
- uniqueness constraints for configured entity IDs

The generated contract is intended as a starting point for review before integration with CI, orchestration, or production data quality checks.

## Project Structure

```text
app.py
data_debugger/
  profiler.py
  checks.py
  scoring.py
  issue_catalog.py
  cleaning.py
  visualizations.py
  ui_components.py
  remediation/
  ai_insights/
  connectors/
  contracts/
  drift/
  storage/
  utils/
sample_data/
DEMO.md
requirements.txt
README.md
```

## Design Principles

- Keep detection deterministic and reproducible.
- Use LLMs only after rule-based issues and metrics already exist.
- Do not send raw uploaded rows to external AI providers.
- Treat cleaning code as preview-only unless explicitly applied by the user.
- Store run metadata and issue summaries, not raw datasets.
- Compare baseline and new datasets directly for drift analysis.
- Keep the application easy to run locally with Streamlit.

## Future Work

Planned improvements include:

- saved monitor configurations
- user-editable thresholds
- prediction-time-aware leakage checks
- production connectors for S3, Postgres, and BigQuery
- Great Expectations suite export
- scheduled runs and alerts
- richer contract editing
- train/test split diagnostics

The long-term direction is a lightweight ML data reliability tool for catching data issues before they become model or reporting failures.
