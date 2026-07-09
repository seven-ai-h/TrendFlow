# How to Start the TrendFlow Server

This guide covers starting, stopping, and restarting the TrendFlow dashboard.

---

## Quick Start (already set up)

```bash
streamlit run dashboard.py
```

Then open **http://localhost:8501** in your browser.

---

## First-Time Setup (fresh machine or empty database)

Run these once, in order:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Load demo data so the charts + Model Lab have something to work with
python seed_data.py

# 3. Launch the dashboard
streamlit run dashboard.py
```

The dashboard opens at **http://localhost:8501**.

---

## Restarting the Server

Stop any running instance, then start fresh:

```bash
pkill -f "streamlit run"
streamlit run dashboard.py
```

---

## Loading Real Data (instead of the demo seed)

This collects live data from Hacker News, Reddit, GitHub, Dev.to, RSS feeds,
and NewsAPI. It only works where outbound APIs are **not** blocked
(i.e. your local machine — not a restricted/remote sandbox).

```bash
python test_hn_api.py      # collect from all sources (headlines + prices) once
streamlit run dashboard.py
```

### Automatic daily collection

To keep the database growing on its own, run the scheduler. It collects once
immediately, then every 24 hours (each run only *adds* new data — it dedupes):

```bash
python scheduler.py                              # every 24 hours
TRENDFLOW_INTERVAL_HOURS=48 python scheduler.py  # every 48 hours instead
```

Leave it running in a terminal, or detach it so it keeps going in the background:

```bash
nohup python scheduler.py > scheduler.log 2>&1 &
```

Run the dashboard in a separate terminal; it always reads the latest data.

> **Optional:** create a `.env` file with `NEWS_API_KEY=...` (NewsAPI) and
> `ANTHROPIC_API_KEY=...` (to enable the 🧠 AI Insights tab). A
> `GITHUB_TOKEN=...` raises the GitHub rate limit but isn't required.

---

## When Do I Need to Restart?

| Change you made                       | What to do                                              |
| ------------------------------------- | ------------------------------------------------------- |
| New data (`seed_data.py` / collector) | Just click **🔄 Refresh Data** in the sidebar — no restart |
| Retrain the ML models                 | Click **🚀 Train / Retrain All Models** in the Model Lab tab |
| Edited Python code                    | Restart Streamlit (or use the **Rerun** button it offers) |

---

## Troubleshooting

- **"No data available" warning** → the database is empty. Run `python seed_data.py`
  (or `python test_hn_api.py` for real data).
- **Changes don't show up** → hard-refresh the browser: `Cmd/Ctrl + Shift + R`.
- **Port already in use** → kill the old process (`pkill -f "streamlit run"`) or
  run on another port: `streamlit run dashboard.py --server.port 8502`.
- **Model Lab says "not enough data"** → run `python seed_data.py` to generate a
  learnable training set, then click **Train / Retrain All Models**.
