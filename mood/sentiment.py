"""Sentiment analysis tools"""

from functools import lru_cache

# Prompt for headline sentiment scoring. LLM generation is routed through ``aix``
# (the multi-provider facade) so the model/provider is switchable from one seam.
SENTIMENT_ANALYSIS_PROMPT = (
    "You are an expert sentiment analysis system, specialized in financial news "
    "and trading headlines.\n\n"
    "I will give you a list of headlines and you will output a json that contains "
    "these headlines as keys,\n"
    "and a natural number between -9 and 9 that describes the sentiment of the "
    "headline,\n"
    "where -9 represents the most negative, 0 is neutral, and 9 is the most "
    "positive.\n\n"
    "HEADLINES:\n{headlines}\n"
)


def _make_sentiment_analyzer(*, model: str = None):
    """Build the headline-sentiment function on :mod:`aix` (lazy import).

    Returns a ``headlines_str -> {headline: score}`` callable. ``aix`` is imported
    here, on first use, so importing :mod:`mood` stays offline. ``output_schema``
    makes ``aix`` request and parse JSON robustly (tolerating fenced output), so the
    result is already a parsed dict.
    """
    import aix

    return aix.prompt_func(
        SENTIMENT_ANALYSIS_PROMPT,
        output_schema=dict,
        name="sentiment_analysis",
        model=model,
    )


def sentiment_analysis(headlines, *, analyze=None):
    """Score the sentiment of financial-news headlines.

    ``headlines`` is a single string or an iterable of headline strings. Returns a
    ``{headline: score}`` mapping where each score is an integer in ``[-9, 9]``.

    ``analyze`` is an injectable ``headlines_str -> dict`` callable; when omitted it
    is built lazily on :mod:`aix` (so importing this module stays offline).
    """
    if not isinstance(headlines, str):  # Pattern: ingress
        # join with two newlines to separate headlines
        headlines = "\n\n".join(headlines)
    if analyze is None:
        analyze = _make_sentiment_analyzer()
    return analyze(headlines=headlines)


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
