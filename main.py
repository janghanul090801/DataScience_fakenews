import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# 데이터 - 가짜 뉴스 및 진짜 뉴스 데이터
df = pd.read_csv('WELFake_Dataset.csv')
# 결측치 삭제
df = df.dropna(subset=['text'])
# 원본 데이터가 필드가 반전되어있어 필드 반전
df['label'] = df['label'].apply(lambda x: 1-x)

# 독립 변수와 종속 변수 분리
X = df['text']
y = df['label']

# 학습 데이터와 테스트 데이터 분리
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    random_state=801,
    stratify=df['label']
)

# 특정 단어가 너무 많은 영향을 미치지 않도록 stopword 지정
custom_stopwords = [
    'reuters',
    'via',
    'said',
    'follow',
    'image',
    'featured'
]

# vectorizer init
vectorizer = TfidfVectorizer(
    stop_words=custom_stopwords
)

vectorizer.fit(X)

# text 데이터 벡터화
X_train = vectorizer.transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# 모델 학습
LR = LogisticRegression()
LR.fit(X_train, y_train)

confusion_matrix(y_test, LR.predict(X_test))
print(accuracy_score(y_test, LR.predict(X_test)))
print(classification_report(y_test, LR.predict(X_test)))

# 테스트 데이터1 - 가짜 뉴스
fake = pd.read_csv('Fake.csv')['text']

fake_predict = LR.predict(vectorizer.transform(fake))
# 간단하게 정확도 표시
print(((fake_predict == 0).sum())/(len(fake_predict)))

# 테스트 데이터2 - 진짜 뉴스
true = pd.read_csv('True.csv')['text']

true_predict = LR.predict(vectorizer.transform(true))
# 간단하게 정확도 표시
print(((true_predict == 1).sum())/(len(true_predict)))

# 테스트 데이터3 - 다른 뉴스 데이터
news = pd.read_csv('news.csv')
news = news.dropna(subset=['text'])

news_predict = LR.predict(vectorizer.transform(news['text']))
# 간단하게 정확도 표시
print(((news_predict == news['label']).sum())/(len(news_predict)))
