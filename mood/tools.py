"""Tools for sentiment analysis"""

from dol import Pipe
from mood.data_sources import headlines
from mood.sentiment import sentiment_analysis


headlines_mood = Pipe(headlines, sentiment_analysis)
