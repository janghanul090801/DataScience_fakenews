import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('WELFake_Dataset.csv')
df = df.dropna(subset=['text'])
df['label'] = df['label'].apply(lambda x: 1-x)

X = df['text']
y = df['label']

custom_stopwords = [
    'via',
    'said',
    'follow',
    'reuters',
    'image',
    'featured'
]

vectorizer = TfidfVectorizer(
    stop_words=custom_stopwords
)

X = vectorizer.fit_transform(X)

LR = LogisticRegression()
LR.fit(X, y)
