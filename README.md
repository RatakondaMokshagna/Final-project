# Text Mining and Sentiment Analysis Based Intelligent System for Monitoring and Evaluating Public Health Awareness Campaigns

An end-to-end NLP system that monitors public sentiment and detects misinformation in COVID-19 tweets, then turns the results into actionable recommendations for public health communication teams, delivered through an interactive **Streamlit** dashboard.

> **Authors:** Dr. C. Priyadharsini (Assistant Professor), T Dyana Vardhan Reddy (RA2211026050024), Ratakonda Mokshagna (RA2211026050011)
> SRM Institute of Science and Technology, Tiruchirappalli, India

---

## Overview

The COVID-19 pandemic produced an "infodemic": a flood of accurate and false information on social media. Public health authorities need tools to track how people feel and to catch harmful misinformation quickly. Most published research stops at model evaluation. This project goes further by combining four capabilities in one deployable tool:

1. **Sentiment classification** (negative / neutral / positive) with an LSTM
2. **Misinformation detection** with TF-IDF + Logistic Regression
3. **Temporal spike detection** on negative sentiment (mean + 2σ)
4. **Rule-based policy recommendations** for campaign managers

## Key Results

| Component | Model | Result |
|---|---|---|
| Sentiment classification | LSTM (128 units) | **98.79%** test accuracy, **0.99** macro F1 |
| Misinformation detection | TF-IDF + Logistic Regression | **99.95%** test accuracy |
| Spike detection | μ + 2σ on daily negative counts | **8** spike dates, aligned with documented pandemic events |
| Policy engine | Threshold-based rules | Recommended "Release official fact-check response" (26.74% negative, 0.49% misinformation) |

Per-class results (test set of 81,983 tweets):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.99 | 0.98 | 0.98 | 22,043 |
| Neutral | 0.99 | 0.99 | 0.99 | 34,505 |
| Positive | 0.98 | 0.99 | 0.98 | 25,435 |

## Dataset

Three public COVID-19 Twitter datasets were combined:

- April – June 2020 (initial lockdowns)
- August – September 2020 (reopening, second-wave concerns)
- April – June 2021 (vaccine rollout, new variants)

After concatenation and deduplication: **409,915 tweets**, with a label split of 26.74% negative, 42.08% neutral and 31.18% positive. Each record has the cleaned tweet text, a sentiment label, a timestamp, engagement metrics (retweets, favorites) and a location field.

**Engineered features**
- `engagement_score` = `retweet_count` + `favorite_count`
- `weighted_sentiment` = sentiment label × engagement score

## Methodology

### Sentiment pipeline (LSTM)
- Keras `Tokenizer`, vocabulary of 10,000 terms, sequences padded or truncated to 100 tokens
- 80:20 train/test split (`random_state=42`): 327,932 train / 81,983 test, plus a 20% validation split during training
- Architecture: `Embedding(10000, 128)` → `LSTM(128)` → `Dropout(0.5)` → `Dense(64, ReLU)` → `Dropout(0.3)` → `Dense(3, Softmax)`
- Adam optimizer (lr = 0.001), sparse categorical cross-entropy, batch size 64, up to 8 epochs, early stopping (patience = 2) on validation loss
- Trained on a Google Colab T4 GPU

### Misinformation pipeline
- Labels generated from a domain keyword lexicon: `fake, hoax, myth, rumor, 5g, conspiracy, plandemic, bioweapon`
- TF-IDF (5,000 features, English stop-words removed) → Logistic Regression (L2, C = 1.0, max_iter = 1000)
- 0.49% of tweets (about 2,009) flagged as misinformation

### Spike detection
Daily negative tweet counts are aggregated; any day exceeding **mean + 2 standard deviations** is flagged.

| Date | Negative tweets |
|---|---|
| 25 Apr 2020 | 1,724 |
| 4 May 2020 | 1,820 |
| 9 May 2020 | 1,675 |
| 27 Sep 2020 | 2,017 |
| 30 Sep 2020 | 1,740 |
| 1 Oct 2020 | 1,624 |
| 25 May 2021 | 1,698 |
| 3 Jun 2021 | 1,677 |

### Policy recommendation engine
1. Negative sentiment > 40%: launch a targeted public awareness campaign immediately
2. Negative sentiment ≤ 40% but spikes detected: release an official fact-check response
3. Otherwise: campaign is performing within normal bounds

## Dashboard Features

**Single-message analysis**
- Free-text input with instant sentiment and misinformation prediction
- Downloadable result CSV

**Bulk campaign monitoring** (upload a CSV)
- KPI cards: total tweets, negative %, positive %, average engagement
- Interactive Plotly charts: sentiment distribution and temporal trends
- Regional breakdown table
- Spike detection results with flagged dates
- Policy recommendation
- Downloadable processed dataset

**Required CSV columns:** `clean_tweet`, `sentiment`, `created_at`, `retweet_count`, `favorite_count`, `place`

## Tech Stack

Python · TensorFlow / Keras · scikit-learn · Pandas · NumPy · Plotly · Streamlit · joblib

## Project Structure

> Adjust file names to match your repository.

```
├── app.py                      # Streamlit dashboard
├── notebooks/
│   └── training.ipynb          # Model training and evaluation (Colab)
├── models/
│   ├── lstm_model.keras        # Trained LSTM
│   ├── tokenizer.joblib        # Keras tokenizer
│   ├── label_encoder.joblib    # Label encoder
│   ├── tfidf_vectorizer.joblib # TF-IDF vectorizer
│   └── misinfo_clf.joblib      # Logistic Regression classifier
├── data/                       # Datasets (not included, see Dataset section)
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

## Limitations

- **Misinformation labels are keyword-derived**, so the 99.95% accuracy reflects agreement with the lexicon and will miss evolved or implicit misinformation.
- **Sentiment labels were not independently validated.** The very high LSTM accuracy may partly reflect the regularity of automated labelling, so expect lower performance on independently annotated data.
- **English only.** Coverage of other languages (Hindi, Tamil, Telugu, etc.) is future work.
- **Policy thresholds** (40% negativity, 2σ spikes) follow domain conventions and have not been empirically calibrated.

## Future Work

- Neural misinformation classifier trained on annotated corpora (e.g. CONSTRAINT-COVID)
- BERTweet or other transformer encoders for sentiment
- Multilingual support (mBERT / XLM-RoBERTa)
- Retrieval-augmented generation for evidence-grounded policy reports
- Graph-based misinformation propagation analysis
- Correlation with epidemiological data and a formal user study with health communication officers

## Citation

If you use this work, please cite:

```
C. Priyadharsini, T. Dyana Vardhan Reddy, and R. Mokshagna, "Text Mining and Sentiment
Analysis Based Intelligent System for Monitoring and Evaluating Public Health Awareness
Campaigns," SRM Institute of Science and Technology, Tiruchirappalli, India.
```

## License

Add your preferred license here (e.g. MIT).

## Acknowledgements

Datasets derived from publicly available COVID-19 Twitter corpora, including the work of A. Lamsal. Built with TensorFlow, scikit-learn and Streamlit.
