import schedule
import time
from datetime import datetime
from test_hn_api import test_hacker_news_api

def job():
    """Run the data collection"""
    try:
        print(f"\n{'='*50}")
        print(f"Starting collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        
        test_hacker_news_api()
        
        print(f"\n{'='*50}")
        print(f"Collection completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"ERROR during collection: {e}")
        import traceback
        traceback.print_exc()

# Schedule to run every hour
schedule.every(1).hours.do(job)

print("="*60)
print("AUTOMATED DATA COLLECTOR STARTED")
print("="*60)
print(f"Collection frequency: Every 1 hour")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Press Ctrl+C to stop")
print("="*60)

# Run immediately on startup
print("\nRunning initial collection...")
job()

# Then run on schedule
print("\nScheduler active. Waiting for next scheduled run...")
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute