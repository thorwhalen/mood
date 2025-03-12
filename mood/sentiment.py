"""Sentiment analysis tools"""

from functools import lru_cache
import json
from dol import Pipe

from mood.util import add_egress


@add_egress(json.loads)
def sentiment_analysis(headlines):
    """Get sentiment analysis for headlines."""
    from oa import ask

    if not isinstance(headlines, str):  # Pattern: ingress
        headlines = "\n\n".join(
            headlines
        )  # join with two newlines to separate headlines
    headline_sentiments = ask.ai.sentiment_analysis(headlines)
    return headline_sentiments


@lru_cache(maxsize=1)
def _flair_text_classifier(model="en-sentiment"):
    from flair.models import TextClassifier

    return TextClassifier.load(model)


def flair_sentiment_score(text: str, model="en-sentiment") -> float:
    """
    Analyzes the sentiment of the input text using Flair's pre-trained sentiment model.

    This function returns a sentiment score as a float:
      - Positive sentiment is returned as a positive value.
      - Negative sentiment is returned as a negative value.

    Example:
      Input: "I absolutely love this product!"
      Output: 0.99 (approximately, for a strong positive sentiment)

    :param text: The input string to analyze.
    :return: Sentiment score (positive for positive sentiment, negative for negative sentiment).
    """
    from flair.data import Sentence

    sentence = Sentence(text)

    _flair_text_classifier(model).predict(sentence)

    # The classifier adds a label to the sentence; we assume the first label is the sentiment result.
    label = sentence.labels[0]

    if label.value.upper() == "POSITIVE":
        return label.score
    elif label.value.upper() == "NEGATIVE":
        return -label.score
    else:
        # Fallback in case an unexpected label is returned.
        return 0.0
