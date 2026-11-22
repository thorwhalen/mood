# mood.sentiment

Sentiment analysis tools

### Functions

| [`flair_sentiment_score`](#mood.sentiment.flair_sentiment_score)(text[, model])         | Analyzes the sentiment of the input text using Flair's pre-trained sentiment model.   |
|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [`sentiment_analysis`](#mood.sentiment.sentiment_analysis)(headlines, \*[, analyze]) | Score the sentiment of financial-news headlines.                                      |

### mood.sentiment.flair_sentiment_score(text, model='en-sentiment')

Analyzes the sentiment of the input text using Flair’s pre-trained sentiment model.

This function returns a sentiment score as a float:

> - Positive sentiment is returned as a positive value.
> - Negative sentiment is returned as a negative value.

### Example

Input: “I absolutely love this product!”
Output: 0.99 (approximately, for a strong positive sentiment)

* **Parameters:**
  **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The input string to analyze.
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  Sentiment score (positive for positive sentiment, negative for negative sentiment).

### mood.sentiment.sentiment_analysis(headlines, , analyze=None)

Score the sentiment of financial-news headlines.

`headlines` is a single string or an iterable of headline strings. Returns a
`{headline: score}` mapping where each score is an integer in `[-9, 9]`.

`analyze` is an injectable `headlines_str -> dict` callable; when omitted it
is built lazily on `aix` (so importing this module stays offline).
