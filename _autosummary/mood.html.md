# mood

Sentiment analysis for stock market data.

```pycon
>>> from mood import headlines_mood
>>> headlines_mood()
{"Yaccarino shakes up X amid Musk's pressure on costs, FT says": -3,
 'Coup-hit Niger was betting on a China-backed oil pipeline as a lifeline. Then the troubles began': -7,
 'A Mexico City neighborhood keeps the iconic Volkswagen Beetle alive': 1,
 ...
 "Here's the Average Social Security Benefit at Age 62 -- and Why It's Not the Best News for Retirees": -5,
 'Analyst Report: Mitsubishi UFJ Financial Group, Inc.': 0,
 'Forget NextEra Energy. Buy This Magnificent Dividend King Instead': 6}
```

### Modules

| [`dataset_makers`](mood.dataset_makers.html.md#module-mood.dataset_makers)     | Tools to make datasets for training and testing models (for semantic attribute etc.)   |
|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`learn_sentiments`](mood.learn_sentiments.html.md#module-mood.learn_sentiments) | Learning sentiments from data.                                                         |
| [`model_analysis`](mood.model_analysis.html.md#module-mood.model_analysis)     | Tools for model analysis                                                               |
| [`mood_modeling`](mood.mood_modeling.html.md#module-mood.mood_modeling)       | Mood Modeling Module                                                                   |
| [`scripts`](mood.scripts.html.md#module-mood.scripts)                   | Scripts for mood.                                                                      |
| [`sentiment`](mood.sentiment.html.md#module-mood.sentiment)               | Sentiment analysis tools                                                               |
| [`tools`](mood.tools.html.md#module-mood.tools)                       | Tools for sentiment analysis                                                           |
| [`util`](mood.util.html.md#module-mood.util)                         | Utils for mood                                                                         |
