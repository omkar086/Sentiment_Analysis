import os
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob

# Setup
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load data
df = pd.read_csv('vaccination_tweets.csv')
df = df[['text']].dropna().drop_duplicates()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+|http\S+", '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

# Clean text
df['clean_text'] = df['text'].apply(preprocess)

# Create labels using polarity
df['polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

def map_sentiment(p):
    if p >= 0.2:
        return 'Positive'
    elif p <= -0.2:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['polarity'].apply(map_sentiment)

# Vectorize text
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model and vectorizer
os.makedirs('model', exist_ok=True)

with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
