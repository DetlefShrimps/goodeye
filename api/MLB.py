import pandas as pd
import schedule
import time
from pybaseball import statcast
from datetime import datetime, timedelta
import threading
from pybaseball import cache

cache.enable()

def get_real_time_statcast_data():
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    data = statcast(start_dt=yesterday, end_dt=today)
    return data

def save_data():
    data = get_real_time_statcast_data()
    data.to_csv('/home/jesse/g00d3y3/data/baseball_savant.csv', index=False)
    data.to_excel('/home/jesse/g00d3y3/data/baseball_savant.xlsx', index=False)
    print(f"Data saved at {datetime.now()}")

def job():
    save_data()

def main():
    schedule.every(15).seconds.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    # Run the main function in a separate thread
    daemon_thread = threading.Thread(target=main)
    daemon_thread.daemon = True
    daemon_thread.start()

    # Keep the main thread running
    while True:
        time.sleep(1)
