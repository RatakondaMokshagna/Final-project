import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Health Campaign Monitor", layout="wide")

st.title("Text Mining and Sentiment Analysis for Public Health Awareness Campaigns")

# ----------------------------
# AUTO REFRESH
# ----------------------------
refresh_interval = st.sidebar.slider("Auto Refresh (seconds)", 0, 60, 0)

if refresh_interval > 0:
    time.sleep(refresh_interval)
    st.rerun()

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model("sentiment_model.keras")
    tokenizer = joblib.load("tokenizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    misinfo_model = joblib.load("misinfo_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tokenizer, label_encoder, misinfo_model, tfidf

model, tokenizer, label_encoder, misinfo_model, tfidf = load_models()

MAX_LEN = 100

# ----------------------------
# CSV CONVERTER
# ----------------------------
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# ----------------------------
# REAL-TIME ANALYSIS
# ----------------------------
st.header("Real-Time Analysis")

user_input = st.text_area("Enter Tweet / Public Message")

if st.button("Analyze"):

    if user_input.strip() != "":

        seq = tokenizer.texts_to_sequences([user_input.lower()])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        pred = model.predict(padded)
        sentiment = label_encoder.inverse_transform([np.argmax(pred)])[0]

        misinfo_pred = misinfo_model.predict(tfidf.transform([user_input]))[0]

        col1, col2 = st.columns(2)

        col1.metric("Sentiment", sentiment)

        if misinfo_pred == 1:
            col2.error("Misinformation Detected")
        else:
            col2.success("No Misinformation")

        # DOWNLOAD SINGLE RESULT
        result_df = pd.DataFrame({
            "text": [user_input],
            "sentiment": [sentiment],
            "misinformation": [misinfo_pred]
        })

        csv_single = convert_df(result_df)

        st.download_button(
            label="Download Result",
            data=csv_single,
            file_name="prediction.csv",
            mime="text/csv"
        )

# ----------------------------
# DASHBOARD SECTION
# ----------------------------
st.markdown("---")
st.header("Campaign Dashboard")

file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    required_cols = ['clean_tweet','sentiment','created_at',
                     'retweet_count','favorite_count','place']

    if not all(col in df.columns for col in required_cols):
        st.error("Dataset must contain required columns")
    else:

        df = df[required_cols]
        df.dropna(inplace=True)

        df['engagement'] = df['retweet_count'] + df['favorite_count']

        # ----------------------------
        # KPI METRICS
        # ----------------------------
        st.subheader("Key Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Tweets", len(df))
        col2.metric("Negative %",
                    round((df['sentiment']=='neg').mean()*100,2))
        col3.metric("Positive %",
                    round((df['sentiment']=='pos').mean()*100,2))
        col4.metric("Avg Engagement",
                    round(df['engagement'].mean(),2))

        # ----------------------------
        # SENTIMENT DISTRIBUTION
        # ----------------------------
        st.subheader("Sentiment Distribution")

        fig = px.bar(
            x=df['sentiment'].value_counts().index,
            y=df['sentiment'].value_counts().values,
            labels={'x':'Sentiment','y':'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------
        # TREND ANALYSIS
        # ----------------------------
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

        trend = df.groupby(
            df['created_at'].dt.date
        )['sentiment'].value_counts().unstack().fillna(0)

        st.subheader("Trend Analysis")
        st.line_chart(trend)

        # ----------------------------
        # REGION ANALYSIS
        # ----------------------------
        st.subheader("Regional Insights")

        region = df.groupby('place')['sentiment'].value_counts().unstack().fillna(0)
        st.dataframe(region.head())

        # ----------------------------
        # SPIKE DETECTION
        # ----------------------------
        negative_daily = df[df['sentiment']=='neg'] \
            .groupby(df['created_at'].dt.date).size()

        mean = negative_daily.mean()
        std = negative_daily.std()
        threshold = mean + 2 * std

        spikes = negative_daily[negative_daily > threshold]

        st.subheader("Spike Detection")

        if len(spikes) > 0:
            st.error("Negative Sentiment Spike Detected")
            st.write(spikes)
        else:
            st.success("No major spikes detected")

        # ----------------------------
        # POLICY RECOMMENDATION
        # ----------------------------
        st.subheader("Policy Recommendation")

        neg_pct = (df['sentiment']=='neg').mean()*100

        if neg_pct > 40:
            st.warning("Launch awareness campaign immediately")
        elif len(spikes) > 0:
            st.warning("Release clarification campaign")
        else:
            st.success("Campaign performing well")

        # ----------------------------
        # DOWNLOAD DATASET RESULTS
        # ----------------------------
        csv = convert_df(df)

        st.download_button(
            label="Download Processed Data",
            data=csv,
            file_name='analysis_results.csv',
            mime='text/csv'
        )