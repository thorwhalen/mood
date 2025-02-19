"""Sentiment analysis tools """

import json
from dol import Pipe

from mood.util import add_egress


@add_egress(json.loads)
def sentiment_analysis(headlines):
    """Get sentiment analysis for headlines."""
    from oa import ask

    if not isinstance(headlines, str):  # Pattern: ingress
        headlines = '\n\n'.join(
            headlines
        )  # join with two newlines to separate headlines
    headline_sentiments = ask.ai.sentiment_analysis(headlines)
    return headline_sentiments
