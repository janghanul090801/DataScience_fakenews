import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('news.csv')
df.info()

X = df['text']
y = df['label']

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    random_state=1221,
    stratify=df['label']
)

# TODO : 백터화 로직 공부하기 ( 백터화 기준, 너무 단순화 되지 않았는지 알아오기 )
# TODO : 다른 모델 낋여와서 결과 비교하기
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

LR = LogisticRegression()
LR.fit(X_train, y_train)

confusion_matrix(y_test, LR.predict(X_test))
accuracy_score(y_test, LR.predict(X_test))
print(classification_report(y_test, LR.predict(X_test)))