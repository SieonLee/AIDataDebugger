# AI Data Debugger

I built AI Data Debugger around a problem I kept running into: most profiling tools are good at telling you that a column has missing values, duplicates, or weird distributions, but they stop right before the useful part.

The real question is usually:

> Is this going to break my model, skew my metrics, or quietly fail when the next batch arrives?

So this project treats a dataset less like a static CSV and more like something that changes over time. It checks the data deterministically, explains why the issues matter for ML or analytics work, lets you preview cleanup impact, and keeps local run history so the same dataset can be monitored across batches.

## What It Is

AI Data Debugger is a Streamlit app for debugging ML and analytics datasets. The first version started as a CSV profiler, but it has grown into a small data reliability workflow:

- upload a baseline dataset
- define the dataset roles, like target, timestamp, and entity ID
- scan for common quality issues
- compare a newer batch against the baseline
- review drift and reliability risks
- simulate conservative cleaning
- export a cleaned dataset preview
- export a YAML or JSON data contract
- track run history over time

The app uses AI only after the deterministic checks are done. LLMs can help explain issues, draft remediation plans, answer questions, or write preview code, but they do not decide whether an issue exists.

## Why I Made It

Tools like pandas-profiling and Great Expectations are useful, but they are usually either descriptive or validation-oriented. I wanted something closer to the workflow I would want as an ML engineer:

- What changed since the last healthy batch?
- Which issues are most likely to hurt model training or reporting?
- What should I fix first?
- If I apply a safe cleanup, does the score actually improve?
- Can I turn what I learned into a contract for the next run?

That is the shape of this project. It is not trying to replace a full data observability platform. It is a focused prototype for the first few minutes of dataset debugging, when you need signal quickly and want the reasoning to be understandable.

## Current Features

The app can:

- profile a CSV dataset and show shape, dtypes, missingness, unique counts, and sample rows
- detect missing values, duplicates, constant columns, high-cardinality categoricals, numeric outliers, type mismatches, target imbalance, timestamp risks, and ID-like fields
- assign issue severity as `critical`, `warning`, or `minor`
- calculate a health score from 0 to 100
- compare a baseline dataset with a newer batch
- detect schema drift, missing-value drift, numeric PSI drift, categorical drift, and cardinality drift
- show issue drilldowns with ML impact, business impact, suggested fixes, and example pandas code
- run a conservative cleaning simulation and compare before/after score, rows, columns, and issue count
- export cleaned dataset previews as CSV, optional Parquet, or a ZIP package with cleaning metadata
- save local run history in SQLite without storing raw uploaded data
- show health score and issue trends for repeated runs under the same monitor name
- export a validation contract as YAML or JSON
- use rule-based explanations, local Ollama, or OpenAI API as the explanation layer

## Demo

There is a ready-to-run demo in [DEMO.md](DEMO.md).

Use these files:

- `sample_data/baseline_market_events.csv`
- `sample_data/broken_market_events.csv`

The story is simple: the baseline file is a healthy historical market-events dataset, and the broken file is a newer production batch with degraded quality. The demo walks through role setup, drift comparison, cleaning simulation, contract export, and run history.

## Run Locally

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL from the terminal.

## AI Providers

The app works without any LLM provider. In that mode, all explanations are generated from deterministic rules.

If you want AI-assisted explanations, the sidebar supports:

1. **Rule-based only**  
   No LLM calls. This is the safest mode and the one I usually use first in a demo.

2. **Local Ollama**  
   Uses `http://localhost:11434/api/chat`. This is useful if you want local explanations without sending metadata to an external API.

3. **OpenAI API**  
   Uses the official OpenAI Python SDK if it is installed. The app reads the key from the sidebar input first, then from `OPENAI_API_KEY`.

For OpenAI:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

For Ollama:

```bash
ollama serve
ollama pull llama3.1
```

The privacy boundary is intentional: raw uploaded rows are not sent to LLM providers. The model only receives compact metadata such as issue objects, column dtypes, missing rates, unique ratios, summary stats, health score, and drift summaries.

## Repeatable Workflow

The part I care most about is that this is not just a one-time report.

Before analysis, you can set dataset roles:

- target column
- timestamp column
- entity ID column
- protected columns
- columns to exclude from ML checks

Those roles reduce false positives. For example, `date` should be treated as a timestamp, and `ticker` in a market dataset should be treated as an entity key rather than a suspicious categorical feature.

Each run is saved locally to SQLite with:

- run ID
- monitor name
- timestamp
- row and column counts
- health score
- issue counts by severity
- AI provider used
- role/config summary
- issue summaries as JSON

Raw uploaded data is not stored.

## Data Contracts

After a scan, the app can export a validation contract as YAML or JSON. The contract includes required columns, expected dtypes, missing-rate limits, low-cardinality accepted values, numeric min/max ranges, and uniqueness constraints for entity IDs.

The generated contract is meant as a starting point. In a real workflow, a team would review and tighten it before wiring it into CI, orchestration, or a data quality monitor.

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

## Design Choices

A few decisions are deliberate:

- Detection is deterministic.
- LLMs explain and plan; they do not detect issues.
- Cleaning code generated by AI is preview-only.
- Run history stores metadata, not raw datasets.
- Drift compares baseline vs new directly instead of profiling the new batch in isolation.
- The app stays Streamlit-based so the prototype is easy to run and inspect.

## Future Work

The next things I would add are:

- saved monitor configs
- user-editable thresholds
- target-aware leakage checks using prediction time
- real S3/Postgres/BigQuery connectors
- Great Expectations export
- scheduled runs and alerts
- richer contract editing
- train/test split diagnostics

The long-term direction is a lightweight ML data reliability cockpit: not a giant platform, but a practical place to catch data problems before they become model problems.
