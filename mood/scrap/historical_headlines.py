"""Get historical headlines """

import os
import requests
from datetime import datetime


def headlines_from_newsdata_io(api_key, start_date, end_date):
    """Get headlines from NewsData.io within a specified date range."""
    url = f'https://newsdata.io/api/1/archive?apikey={api_key}&q=finance&from_date={start_date}&to_date={end_date}&language=en'

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to retrieve data: {response.status_code}")
        try:
            error_data = response.json()
            print("Error details:", error_data)
        except ValueError:
            print("Response content is not valid JSON")
            print(response.text)
        return []

    try:
        data = response.json()
    except ValueError:
        print("Failed to parse response JSON")
        return []

    headlines = [article['title'] for article in data.get('results', [])]

    if not headlines:
        print("No news data found for the specified date range.")

    return headlines


def _headlines_from_newsdata_io_example():
    api_key = os.environ[
        'NEWSDATA_API_KEY'
    ]  # get one for free here: https://newsdata.io/api-key
    # --> But the
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    news_data = headlines_from_newsdata_io(api_key, start_date, end_date)

    if news_data:
        print(f"Fetched {len(news_data)} news headlines from NewsData.io:")
        for headline in news_data:
            print(headline)
    else:
        print("No news data fetched.")
