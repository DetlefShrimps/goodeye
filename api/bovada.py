import requests
import pandas as pd
import logging
import time  # Importing the time module
import json  # Importing json for pretty printing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BovadaScraper:
    def __init__(self):
        self.host = "https://www.bovada.lv"
        self.mlb_path = "/services/sports/event/v2/events/A/description/baseball/mlb"
        self.max_retries = 5
        self.retry_delay = 10  # seconds
        self.headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Mobile/15E148 Safari/604.1",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",  # Do Not Track Request Header
            "Referer": "https://www.bovada.lv/"
        }

    def fetch_data(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            logging.info(f"Fetched data from {url}")
            if response.text:
                return response.json()
            else:
                logging.error(f"Empty response from {url}")
                return None
        except requests.RequestException as e:
            logging.error(f"Error fetching data from {url}: {e}")
            return None

    def parse_json(self, data):
        try:
            # Print the structure of the JSON data for debugging
            logging.debug("JSON data structure:\n" + json.dumps(data, indent=2))
            
            events = []
            for event in data[0]['events']:  # Access the first element and then 'events'
                event_data = {
                    'id': event['id'],
                    'description': event['description'],
                    'start_time': event['startTime'],
                    'competitors': [{'id': comp['id'], 'name': comp['name'], 'home': comp['home']} for comp in event['competitors']],
                    'markets': []
                }
                for group in event['displayGroups']:
                    for market in group['markets']:
                        market_data = {
                            'market_key': market['description'],
                            'outcomes': [{'description': outcome['description'], 'price': outcome['price'], 'handicap': outcome['price'].get('handicap')} for outcome in market['outcomes']]
                        }
                        event_data['markets'].append(market_data)
                events.append(event_data)
            logging.info(f"Parsed {len(events)} events from JSON data")
            return events
        except Exception as e:
            logging.error(f"Error parsing JSON data: {e}")
            logging.debug(f"Problematic JSON data: {json.dumps(data, indent=2)}")
            return None

    def verify_data(self, events):
        if not events:
            return False
        required_keys = {'id', 'description', 'start_time', 'competitors', 'markets'}
        for event in events:
            if not all(key in event for key in required_keys):
                logging.error(f"Missing keys in event: {event}")
                return False
        return True

    def save_data(self, events):
        df = pd.DataFrame(events)
        df.to_csv('mlb_bets.csv', index=False)
        logging.info("Data saved successfully to mlb_bets.csv")

    def retry_strategy(self, url):
        retries = 0
        while retries < self.max_retries:
            data = self.fetch_data(url)
            if data:
                events = self.parse_json(data)
                if events and self.verify_data(events):
                    return events
            retries += 1
            logging.info(f"Retrying... ({retries}/{self.max_retries})")
            time.sleep(self.retry_delay)
        return None

    def scrape_bovada(self):
        mlb_url = f"{self.host}{self.mlb_path}"
        mlb_bets = self.retry_strategy(mlb_url)

        if mlb_bets:
            self.save_data(mlb_bets)
        else:
            logging.error("Failed to fetch and verify all required bets after multiple retries.")

if __name__ == "__main__":
    scraper = BovadaScraper()
    scraper.scrape_bovada()
