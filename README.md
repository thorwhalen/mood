# mood

Investigations in financial sentiment analysis

To install:	```pip install mood```

Note: Requires an OpenAI API Token. 
[How can to get one of those?](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)

For more NLP work, see [idiom](https://github.com/thorwhalen/idiom/) and [lexis](https://github.com/thorwhalen/lexis/).

## Getting Started with Mood Models

The `mood` package provides a complete pipeline for training machine learning models to detect semantic attributes (or "moods") in text. Here's the typical workflow. 
If you already have your training data, you can skip steps 1 and 2. 

### 1. Define Your Semantic Attributes

Start by defining the semantic attributes you want to model. These are the "moods" or characteristics you want to detect in text:

```python
semantic_attributes = {
    "diplomatic_politeness": "Diplomatic Politeness (Degree of formality, civility, and ceremonial tone)",
    "military_intensity": "Military Reference Intensity (Mentions of soldiers, arms, tactics, or losses)",
    "sentiment_polarity": "Sentiment Polarity (Positive or negative affective tone)",
    # ... add more attributes as needed
}
```

### 2. Generate Training Data

Use the AI-powered dataset generation to create labeled training examples:

```python
from mood.dataset_makers import make_semantic_attributes_dataset

# Generate training data for your semantic attributes
make_semantic_attributes_dataset(
    semantic_attributes=semantic_attributes,
    store="path/to/save/datasets",  # or use a MutableMapping
    n_examples=1000,  # total examples per attribute
    batch_size=100,  # examples per API call
    verbose=2,  # show progress
)
```

This will create text segments with scores (0-5) for each semantic attribute, saved as text files with format:
```
0 The meeting was dull and unproductive, leaving everyone feeling dissatisfied.
3 The negotiations showed moderate diplomatic courtesy between the parties.
5 Your Excellency, it is indeed our greatest honor to welcome you here today.
```

### 3. Parse and Prepare Data

Convert the raw text data into structured format and compute embeddings:

```python
from mood.dataset_makers import parsed_lines
import pandas as pd
from oa.batch_embeddings import compute_embeddings

# Parse the generated text files
dataset = pd.DataFrame(list(parsed_lines(dataset_content)))
# Results in: DataFrame with 'score' and 'segment' columns

# Compute embeddings for the text segments
embeddings = compute_embeddings(
    segments=dataset.segment.to_list(), batch_size=1000, verbosity=2
)

# Combine into final training dataset
training_data = pd.DataFrame(
    {"score": dataset.score, "segment": dataset.segment, "embedding": embeddings}
)
```

### 4. Train Mood Models

Use the `MoodModelingManager` to train and evaluate multiple types of models:

```python
from mood.mood_modeling import MoodModelingManager

# Create manager with your training data
manager = MoodModelingManager(
    df=training_data,
    embedding_col="embedding",
    score_col="score",
    verbose=2,  # Show detailed progress
)

# Train and evaluate models with single train/test split
results = manager.train_and_evaluate()

# For more stable metrics, run cross-validation
cv_results = manager.cross_validate_models(n_splits=10)

# Get performance summary
summary = manager.get_model_summary(use_cv=True)
print(summary)
```

### 5. Select and Deploy Best Model

```python
# Fit final models on all data
manager.fit_final_models()

# Get the best model based on performance metric
best_model_name, best_model = manager.get_best_model(
    metric="spearman",  # or 'accuracy', 'f1', etc.
    use_cv=True,
)

print(f"Best model: {best_model_name}")

# Use the model to predict mood scores for new text
new_embeddings = compute_embeddings(["Your new text here"])
mood_scores = manager.predict_mood(new_embeddings)
print(f"Mood scores: {mood_scores}")  # Values between 0 and 1
```

### Model Types Available

The system automatically trains several types of models:

- **Numerical Regression**: Ridge regression, SVR for continuous score prediction
- **Binary Classification**: Logistic regression, SVM for high/low classification  
- **Ordinal Regression**: Specialized models for ordered categories (requires `mord` package)

Each model type handles the data differently:
- Numerical models predict continuous scores
- Binary models classify as high vs. low mood
- Ordinal models respect the ordered nature of the scores

### Output Interpretation

All final models output **mood scores between 0 and 1**, where:
- 0 = completely lacks the semantic attribute
- 1 = maximally expresses the semantic attribute
- Values are ordinally aligned with training scores

### Advanced Features

**Custom Model Configuration:**
```python
# Define custom models
custom_models = {
    "custom_ridge": {
        "data_type": "numerical",
        "model_class": Ridge,
        "model_params": {"alpha": 0.5},
        "max_dims": 200,
        "output_transform": "sigmoid",
    }
}

manager = MoodModelingManager(df=data, models=custom_models)
```

**Performance Analysis:**
```python
# Analyze model performance across different metrics
from mood.model_analysis import analyze_all

classifier_stats, regression_stats = compute_model_stats(model_stats)
report = analyze_all(classifier_stats, regression_stats)
```

## Quick Example: Current Headlines Sentiment

For a quick start, try analyzing current headlines:

```python
from mood import headlines_mood

# Get current headlines with sentiment scores
sentiment_scores = headlines_mood()
print(sentiment_scores)
```

This returns a dictionary mapping headlines to sentiment scores from -10 to +10.d

Investigations in financial sentiment analysis

To install:	```pip install mood```

Note: Requires an OpenAI API Token. 
[How can to get one of those?](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)

For more NLP work, see [idiom](https://github.com/thorwhalen/idiom/) and [lexis](https://github.com/thorwhalen/lexis/).




# Examples

## current headlines and their sentiment scores


```python

>>> from mood import headlines_mood
>>> headlines_mood()
{"Yaccarino shakes up X amid Musk's pressure on costs, FT says": -3,
 'Coup-hit Niger was betting on a China-backed oil pipeline as a lifeline. Then the troubles began': -7,
 'A Mexico City neighborhood keeps the iconic Volkswagen Beetle alive': 1,
 'Bitter political fight in Bolivia is paralyzing the government as unrest boils over economic crisis': -8,
 'Credit’s Strong Run Stumbles for First Time This Year': -4,
 'Former Stark Chairman in Thai Detention After Months on the Run': -5,
 '2 Millionaire-Maker Technology Stocks': 5,
 'Want to Earn More in the Stock Market With Less Effort? Try This Simple Strategy': 4,
 '3 Reasons to Buy Nvidia Stock Before June 26': 7,
 "Cathie Wood Says Software Is the Next Big AI Opportunity -- 2 Super Stocks You'll Wish You'd Bought Today if She's Right": 8,
 'Companies that give money to Democrats are beating Republican donors in the stock market': 2,
 'Is Buying Stocks When the S&P 500 Hits a New All-Time High a Smart Strategy? History Provides a Clear Answer.': 2,
 'Q1 Design Software Earnings: Unity (NYSE:U) Earns Top Marks': 4,
 'Social Networking Q1 Earnings: Snap (NYSE:SNAP) Simply the Best': 6,
 "A Look Back at Footwear Stocks' Q1 Earnings: Steven Madden (NASDAQ:SHOO) Vs The Rest Of The Pack": -1,
 "A Look Back at Heavy Machinery Stocks' Q1 Earnings: Oshkosh (NYSE:OSK) Vs The Rest Of The Pack": -1,
 'Unpacking Q1 Earnings: Keurig Dr Pepper (NASDAQ:KDP) In The Context Of Other Beverages and Alcohol Stocks': -2,
 'Q1 Earnings Outperformers: Agilysys (NASDAQ:AGYS) And The Rest Of The Vertical Software Stocks': 3,
 'What do homebuilders want? Immigration reform': 0,
 'Examining the steep rise in monthly auto loan payments': -2,
 'Cannes Lions 2024: What marketers are saying': 1,
 "Sports has to feed into a streamer's 'ecosystem': Kevin Mayer": 0,
 'Taylor Swift vs. Central Banks: One Swiftie calls ‘BS’': -2,
 "'There are bargains out there' as summer travel season heats up": 3,
 'This week in Bidenomics: Pile on the debt': -3,
 'Netflix flirts with all-time highs as investors cheer ad momentum, foray into live sports': 8,
 'Existing home sales decline in May as home prices reach record high': -4,
 "Regulators find weakness in 'living wills' from BofA, Citi, Goldman, and JPM": -6,
 'How to watch and listen to Yahoo Finance': 0,
 "Why Nvidia's 'gravy train' could come to 'screeching halt' after a volatile trading week": -7,
 'Trump’s campaign now has a cash advantage over Biden': 1,
 'The Anti-Altman’s Hail Mary Pitch to Investors': -3,
 "Tesla's Autonomous Strategy to be Key to EV Maker Reaching $1 Trillion-Plus Valuation, Wedbush Says": 7,
 'Alaska Airlines, Flight Attendants Union Reach Tentative Deal': 6,
 'China’s 618 online shopping event marks first-ever sales drop': -7,
 'Apple Throws Down the Gauntlet to European Regulators': -3,
 'Apple’s AI Features Won’t Be Offered in European Union Because of New Laws': -5,
 'Equity Markets Mostly Lower as Existing Home Sales Drop': -4,
 'Oil Rig Count Falls by Three This Week, Baker Hughes Says': -2,
 'Dubai Real Estate’s Resilience May Signal End of Boom-Bust Cycle': 5,
 'Montana aims to remind seniors about its ‘reverse annuity mortgage’': 0,
 'Desmarais-Backed Mortgage Firm Nesto Acquires Lender CMLS': 3,
 'Record high prices, rising mortgage rates depress US home sales': -8,
 'US home sales fall for the 3rd straight month in May amid rising mortgage rates, record-high prices': -8,
 "On wealthy Martha's Vineyard, costly housing is forcing workers out and threatening public safety": -7,
 "Mortgage rates decline for third consecutive week — 'bodes well for the housing market'": 4,
 'Congress should adopt mortgage interest tax credit: CHLA': 3,
 'Savings interest rates today, June 22, 2024 (best account provides 5.30% APY)': 4,
 'How to pay off your house faster with biweekly mortgage payments': 5,
 'The best credit cards for vacations for June 2024': 5,
 "Climate change makes India's monsoons erratic. Can farmers still find a way to prosper?": -2,
 'CDK Hackers Want Millions in Ransom to End Car Dealership Outage': -7,
 'S&P 500 Trading Volume Spikes at Wall Street Close: Markets Wrap': 0,
 'Bitcoin Could Hit $500,000 by October 2025, According to This Billionaire Investor': 7,
 'This Is Why Altcoin Investors Struggle Despite Bitcoin, Ether Sitting Near Yearly Highs': -3,
 'BitoGroup partners with Far Eastern International Bank to launch first crypto-friendly bank account': 2,
 'Cathie Wood sells $13 million of a struggling tech stock': -6,
 'Nvidia Stock Gets Hit With Bearish Reversal. If You Have Big Profits, This Is What You Should Do.': -5,
 "Single Mom With $1.3 Million From Divorce Can't Afford $8,000 Monthly Rent, Dave Ramsey Says It's Time To Move": -6,
 'Morningstar | A Weekly Summary of Stock Ideas and Developments in the Companies We Cover': 0,
 '5 Little-Known Perks of a Costco Membership': 4,
 'Billionaire David Tepper Goes Bargain Hunting: 6 Stocks He Just Bought': 5,
 'Palantir Inks Deal With Starlab. Is the Stock Ready to Head to the Stars?': 6,
 "Here's the Average Social Security Benefit at Age 62 -- and Why It's Not the Best News for Retirees": -5,
 'Analyst Report: Mitsubishi UFJ Financial Group, Inc.': 0,
 'Forget NextEra Energy. Buy This Magnificent Dividend King Instead': 6}
```

