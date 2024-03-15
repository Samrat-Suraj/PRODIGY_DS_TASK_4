import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

def load_dataset(file_path):
    """Load the dataset from the given file path."""
    return pd.read_csv(file_path)

def clean_text(text):
    """Clean the text by removing Twitter handles, special characters, and short words."""
    text = re.sub(r'@[\w]*', '', text)  # Remove Twitter handles
    text = re.sub('[^a-zA-Z#]', ' ', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if len(word) > 3])  # Remove short words
    return text

def tokenize_and_stem(text):
    """Tokenize the text and stem each word."""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in text.split()]

def visualize_word_cloud(data, title=''):
    """Visualize word cloud for the given data."""
    all_words = " ".join([sentence for sentence in data])
    wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def extract_hashtags(tweets):
    """Extract hashtags from tweets."""
    hashtags = []
    for tweet in tweets:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.extend(ht)
    return hashtags

def plot_top_hashtags(hashtags, title=''):
    """Plot the top hashtags."""
    freq = nltk.FreqDist(hashtags)
    d = pd.DataFrame({'Hashtag': list(freq.keys()), 'Count': list(freq.values())})
    d = d.nlargest(columns='Count', n=10)
    plt.figure(figsize=(15, 9))
    sns.barplot(data=d, x='Hashtag', y='Count')
    plt.title(title)
    plt.show()

def preprocess_data(df):
    """Preprocess the data."""
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    df['tokenized_tweet'] = df['clean_tweet'].apply(tokenize_and_stem)
    df['clean_tweet'] = df['tokenized_tweet'].apply(lambda x: ' '.join(x))
    return df

def feature_extraction(df):
    """Perform feature extraction using CountVectorizer."""
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=1000, stop_words='english')
    bow = bow_vectorizer.fit_transform(df['clean_tweet'])
    return bow

def train_model(X_train, y_train):
    """Train the logistic regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print F1 score and accuracy."""
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"F1 Score: {f1}")
    print(f"Accuracy Score: {accuracy}")

# Load the dataset
file_path = 'Twitter_sentiment.csv'
twitter_df = load_dataset(file_path)

# Preprocess the data
twitter_df = preprocess_data(twitter_df)

# Visualize word cloud for all tweets
visualize_word_cloud(twitter_df['clean_tweet'], title='Word Cloud for All Tweets')

# Split data into positive and negative tweets
positive_tweets = twitter_df[twitter_df['label'] == 0]['clean_tweet']
negative_tweets = twitter_df[twitter_df['label'] == 1]['clean_tweet']

# Visualize word cloud for positive tweets
visualize_word_cloud(positive_tweets, title='Word Cloud for Positive Tweets')

# Visualize word cloud for negative tweets
visualize_word_cloud(negative_tweets, title='Word Cloud for Negative Tweets')

# Extract hashtags
positive_hashtags = extract_hashtags(positive_tweets)
negative_hashtags = extract_hashtags(negative_tweets)

# Plot top hashtags for positive tweets
plot_top_hashtags(positive_hashtags, title='Top Hashtags for Positive Tweets')

# Plot top hashtags for negative tweets
plot_top_hashtags(negative_hashtags, title='Top Hashtags for Negative Tweets')

# Feature extraction
X = feature_extraction(twitter_df)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, twitter_df['label'], random_state=42, test_size=0.25)

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
evaluate_model(model, X_test, y_test)
