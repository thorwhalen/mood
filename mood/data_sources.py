"""Data sources."""

import requests
from bs4 import BeautifulSoup


def headlines_from_yahoo_finance():
    """Get headlines from Yahoo Finance."""
    url = 'https://finance.yahoo.com/news/'
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve data")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all 'a' tags that contain 'href' attribute and 'h3' tag inside them
    news_items = soup.find_all('a', href=True)

    for item in news_items:
        # Check if the 'a' tag contains an 'h3' tag with text
        headline_tag = item.find('h3')
        if headline_tag:
            headline = headline_tag.text.strip()
            if headline:
                yield headline


# alias for forwards compatibility
# (might get our headlines from somewhere else (possibly multiple places) in the future)
headlines = headlines_from_yahoo_finance


