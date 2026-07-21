"""
TrendFlow scheduler — keeps the database growing automatically.

Runs the full collection pipeline (headlines + sentiment + prices) once right
away, then on a fixed interval forever. Because the pipeline dedupes by URL and
upserts price bars, every run simply *adds* new data — history accumulates so
the models have more to learn from over time.

Usage:
    python scheduler.py                 # collect now, then every 24 hours
    TRENDFLOW_INTERVAL_HOURS=48 python scheduler.py   # every 48 hours instead

Leave it running in a terminal, or detach it:
    nohup python scheduler.py > scheduler.log 2>&1 &
"""
import os
import time
import traceback
from datetime import datetime

import schedule

from test_hn_api import run_pipeline

INTERVAL_HOURS = float(os.getenv("TRENDFLOW_INTERVAL_HOURS", "24"))


def job():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 60}\n[{stamp}] Starting scheduled collection\n{'=' * 60}")
    try:
        run_pipeline()
        print(f"[{datetime.now():%H:%M:%S}] Collection finished OK.")
    except Exception:
        # never let one bad run kill the loop
        print(f"[{datetime.now():%H:%M:%S}] Collection failed:")
        traceback.print_exc()
    nxt = schedule.next_run()
    if nxt:
        print(f"Next run scheduled for {nxt:%Y-%m-%d %H:%M:%S}.")


def main():
    print(f"TrendFlow scheduler starting — interval: every {INTERVAL_HOURS} hours.")
    print("Running the first collection now…")
    job()  # collect immediately so there's fresh data on startup

    schedule.every(INTERVAL_HOURS).hours.do(job)
    print("\nScheduler is running. Press Ctrl+C to stop.\n")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nScheduler stopped.")


if __name__ == "__main__":
    main()
