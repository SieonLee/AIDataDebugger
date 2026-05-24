# AI Data Debugger Demo

This is the demo flow I use to show the project as a repeatable ML data reliability workflow instead of a one-off CSV profiling tool.

## Files

Use the two sample files in `sample_data/`:

- `baseline_market_events.csv`
- `broken_market_events.csv`

The baseline file represents a healthy historical batch for a market-event prediction model. The broken file represents a newer production batch after the data started drifting and the ingestion quality got worse.

The model scenario is intentionally simple: use market features to predict whether the stock or ETF is up the next day, stored in `next_day_up`.

## What The Broken Batch Shows

The broken batch is designed to make the product light up in a realistic way:

- `volume`, `market_event`, and `log_return` have more missing values
- `close`, `volume`, and `log_return` drift numerically
- `ticker` and `market_event` shift categorically
- new market events appear, like `Fed Emergency Meeting` and `Volatility Halt`
- duplicate rows show up
- `volume` and `log_return` contain outliers
- `next_day_up` becomes heavily imbalanced

That is the story: the training data looked fine, but the new batch no longer behaves like the historical baseline.

## Walkthrough

1. Start the app.

   ```bash
   streamlit run app.py
   ```

2. Keep the AI provider on **Rule-based only** at first.

   I usually say this out loud: the checks, scores, drift metrics, and cleaning simulation are deterministic. AI can explain the results later, but it does not decide what is wrong.

3. Upload `sample_data/baseline_market_events.csv` as the baseline dataset.

4. Set the monitor name to `market_events`.

   This matters because the history view groups repeated runs by monitor name. The baseline and broken files are different CSVs, but they represent the same pipeline.

5. Open **Dataset role setup** and set:

   - target column: `next_day_up`
   - timestamp column: `date`
   - entity ID column: `ticker`

   This is a good moment to explain that the tool is not treating every column blindly. A timestamp, target, and entity key should behave differently from ordinary model features.

6. Review the first health score.

   The baseline should look relatively clean. The point is to establish a healthy reference batch.

7. Upload `sample_data/broken_market_events.csv` as the optional new dataset for comparison.

8. Open **Data Drift**.

   Show the drift score, missing-value drift table, PSI chart, and categorical drift output. This is the strongest part of the demo because it moves the product from “EDA” to “monitoring a production batch.”

9. Open **Recommendations**.

   These are ordered by severity, estimated impact, and ML risk. I frame this as the handoff from detection to action.

10. Open **Cleaning simulation**.

    Run the suggested cleaning simulation and show the before/after score, row count, column count, and remaining issue count.

11. Export the cleaned dataset preview.

    The app can download CSV, optional Parquet, or a ZIP package with cleaning metadata. I call it a preview on purpose because generated cleaning should be reviewed before becoming a production transform.

12. Open **Report** and export the data contract.

    Download the YAML or JSON contract. Point out that this turns the scan into something reusable: required columns, dtypes, missing-rate limits, numeric ranges, allowed values, and entity-key expectations.

13. Open **History**.

    The run history is local SQLite. It stores metadata and issue summaries, not raw uploaded data. After running both sample files, the health score trend should show the baseline and degraded batch under the same `market_events` monitor.

14. Optional: enable an AI provider.

    If Ollama or OpenAI is configured, use it to generate an explanation or remediation plan. The important caveat is that raw rows are still not sent to the provider.

## Short Interview Script

Here is the short version I would say in an interview:

> I built this because a lot of profiling tools stop at “this column has missing values.” That is useful, but for ML work I usually need to know whether the issue affects training, evaluation, or production reliability.
>
> In this demo I start with a healthy baseline market-events dataset. I mark `date` as the timestamp, `ticker` as the entity ID, and `next_day_up` as the target. That gives the checks some ML context and reduces false positives.
>
> Then I upload a newer production batch. The Data Drift tab compares the two datasets directly. It picks up missing-value drift, numeric PSI drift, categorical shifts, new event types, duplicate rows, outliers, and target imbalance.
>
> The key design choice is that detection stays deterministic. LLMs are only used for explanations, remediation plans, code previews, and report writing. They do not decide whether an issue exists, and raw rows are not sent to external APIs.
>
> After that I run a cleaning simulation, export a cleaned dataset preview, generate a YAML or JSON data contract, and show the History tab. That is the shift from a one-time EDA tool to a repeatable data reliability workflow.

## Things To Emphasize

When presenting it, I would avoid saying “CSV profiler” too much. The stronger framing is:

- baseline vs production batch
- deterministic data quality gates
- ML-aware roles
- schema and distribution drift
- cleaning preview with measurable score improvement
- validation contract export
- local run history
- AI for interpretation, not detection

That story lands better because it sounds like model reliability work, not just exploratory analysis.
