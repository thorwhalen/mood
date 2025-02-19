"""

Sentiment analysis for stock market data.

>>> from mood import headlines_mood
>>> headlines_mood()  # doctest: +SKIP
{"Yaccarino shakes up X amid Musk's pressure on costs, FT says": -3,
 'Coup-hit Niger was betting on a China-backed oil pipeline as a lifeline. Then the troubles began': -7,
 'A Mexico City neighborhood keeps the iconic Volkswagen Beetle alive': 1,
 ...
 "Here's the Average Social Security Benefit at Age 62 -- and Why It's Not the Best News for Retirees": -5,
 'Analyst Report: Mitsubishi UFJ Financial Group, Inc.': 0,
 'Forget NextEra Energy. Buy This Magnificent Dividend King Instead': 6}
"""

from mood.sentiment import sentiment_analysis
from mood.tools import headlines_mood, headlines, search_and_save_news
