import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import re


# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("product_reviews.csv")
    return df


df = load_data()

# ------------------- DATA CLEANING -------------------

# 1️⃣ Remove Duplicates
df.drop_duplicates(inplace=True)

# 2️⃣ Handle Missing Values
df.dropna(subset=["Review_Text"], inplace=True)  # Drop rows with missing reviews


# 3️⃣ Clean Text (Remove extra spaces, convert to lowercase, remove special characters)
def clean_text(text):
    text = text.strip().lower()  # Convert to lowercase and remove spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text


df["Review_Text"] = df["Review_Text"].apply(clean_text)

# 4️⃣ Handle Outliers (Ratings should be between 1-5)
df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]


# ------------------- SENTIMENT ANALYSIS -------------------

# Function to classify sentiment based on rating
def classify_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"


df["Sentiment"] = df["Rating"].apply(classify_sentiment)

# ------------------- STREAMLIT DASHBOARD -------------------

st.title("📊 Sentiment Analysis of Product Reviews")
st.write("Performing data cleaning and sentiment analysis on customer reviews.")

# # Display Dataset
# if st.checkbox("Show Cleaned Dataset"):
#     st.dataframe(df)

# Sentiment Count
sentiment_counts = df["Sentiment"].value_counts()

# Pie Chart
fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", colors=["green", "blue", "red"])
ax.set_title("Overall Sentiment Distribution")
st.pyplot(fig)

# Bar Chart
fig, ax = plt.subplots()
ax.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "blue", "red"])
ax.set_title("Overall Sentiment Count")
ax.set_ylabel("Number of Reviews")
st.pyplot(fig)

# Filter by Product
product_list = df["Product_Name"].unique()
selected_product = st.selectbox("Select a Product to View Reviews:", product_list)
filtered_df = df[df["Product_Name"] == selected_product]

# Display Reviews
st.write(f"Reviews for **{selected_product}**:")
st.dataframe(filtered_df[["Review_Text", "Rating", "Sentiment"]])

# ------------------- OVERALL SENTIMENT FOR SELECTED PRODUCT -------------------

# Determine the most frequent sentiment
if not filtered_df.empty:
    overall_sentiment = filtered_df["Sentiment"].value_counts().idxmax()
    st.header(f"📝 Overall Sentiment for {selected_product}: **{overall_sentiment}**")
if st.checkbox("Show Cleaned Dataset"):
    st.dataframe(df)