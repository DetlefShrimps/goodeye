import requests
import pandas as pd
import logging
import time
import json
import os
import zipfile
import schedule

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

    def retry_strategy(self, url):
        retries = 0
        while retries < self.max_retries:
            data = self.fetch_data(url)
            if data:
                events = self.parse_json(data)
                if events:
                    return events
            retries += 1
            logging.info(f"Retrying... ({retries}/{self.max_retries})")
            time.sleep(self.retry_delay)
        return None

    def scrape_bovada(self):
        mlb_url = f"{self.host}{self.mlb_path}"
        mlb_bets = self.retry_strategy(mlb_url)

        if mlb_bets:
            self.create_excel_files(mlb_bets)
            self.create_zip_file()
        else:
            logging.error("Failed to fetch and verify all required bets after multiple retries.")

    def create_excel_files(self, data):
        self.parse_american_lines_basic(data)
        self.parse_american_lines_detailed(data)
        self.parse_american_lines_grouped(data)
        self.parse_american_lines_pivot(data)

    def parse_american_lines_basic(self, data):
        american_lines = []
        for event in data[0]['events']:
            for group in event['displayGroups']:
                for market in group['markets']:
                    for outcome in market['outcomes']:
                        if 'price' in outcome and 'american' in outcome['price']:
                            american_lines.append({
                                'Event ID': event['id'],
                                'Event Description': event['description'],
                                'Competitor': outcome['description'],
                                'American Line': outcome['price']['american']
                            })

        df = pd.DataFrame(american_lines)
        df.to_excel('data/american_lines_basic.xlsx', index=False)

    def parse_american_lines_detailed(self, data):
        american_lines = []
        for event in data[0]['events']:
            for group in event['displayGroups']:
                for market in group['markets']:
                    for outcome in market['outcomes']:
                        if 'price' in outcome and 'american' in outcome['price']:
                            for comp in event['competitors']:
                                american_lines.append({
                                    'Event ID': event['id'],
                                    'Event Description': event['description'],
                                    'Start Time': event['startTime'],
                                    'Competitor ID': comp['id'],
                                    'Competitor Name': comp['name'],
                                    'Competitor Home': comp['home'],
                                    'Market': market['description'],
                                    'American Line': outcome['price']['american']
                                })

        df = pd.DataFrame(american_lines)
        df.to_excel('data/american_lines_detailed.xlsx', index=False)

    def parse_american_lines_grouped(self, data):
        events_list = []
        for event in data[0]['events']:
            markets_list = []
            for group in event['displayGroups']:
                for market in group['markets']:
                    outcomes_list = []
                    for outcome in market['outcomes']:
                        if 'price' in outcome and 'american' in outcome['price']:
                            outcomes_list.append({
                                'Competitor': outcome['description'],
                                'American Line': outcome['price']['american']
                            })
                    markets_list.append({
                        'Market': market['description'],
                        'Outcomes': pd.DataFrame(outcomes_list)
                    })
            events_list.append({
                'Event ID': event['id'],
                'Event Description': event['description'],
                'Start Time': event['startTime'],
                'Markets': pd.DataFrame(markets_list)
            })

        df = pd.DataFrame(events_list)
        df.to_excel('data/american_lines_grouped.xlsx', index=False)

    def parse_american_lines_pivot(self, data):
        records = []
        for event in data[0]['events']:
            for group in event['displayGroups']:
                for market in group['markets']:
                    for outcome in market['outcomes']:
                        if 'price' in outcome and 'american' in outcome['price']:
                            records.append({
                                'Event ID': event['id'],
                                'Event Description': event['description'],
                                'Start Time': event['startTime'],
                                'Market': market['description'],
                                'Competitor': outcome['description'],
                                'American Line': outcome['price']['american']
                            })

        df = pd.DataFrame(records)
        pivot_df = df.pivot_table(index=['Event ID', 'Event Description', 'Start Time'], columns='Market', values='American Line', aggfunc='first')
        pivot_df.to_excel('data/american_lines_pivot.xlsx')

    def create_zip_file(self):
        with zipfile.ZipFile('data/bovada.zip', 'w') as zf:
            zf.write('data/american_lines_basic.xlsx')
            zf.write('data/american_lines_detailed.xlsx')
            zf.write('data/american_lines_grouped.xlsx
