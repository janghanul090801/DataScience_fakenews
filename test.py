# 다른 모델로 해본 테스트
# ai-generated

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('WELFake_Dataset.csv')
df = df.dropna(subset=['text'])
df['label'] = df['label'].apply(lambda x: 1-x)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    random_state=801,
    stratify=df['label']
)

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# -------------------------------
# Logistic Regression
# -------------------------------
lr = LogisticRegression(
    max_iter=1000,
    random_state=801
)

lr.fit(X_train, y_train)

print("===== Logistic Regression =====")
print(classification_report(
    y_test,
    lr.predict(X_test)
))

# -------------------------------
# Decision Tree
# -------------------------------
dt = DecisionTreeClassifier(
    random_state=801,
    max_depth=50
)

dt.fit(X_train, y_train)

print("===== Decision Tree =====")
print(classification_report(
    y_test,
    dt.predict(X_test)
))

# -------------------------------
# SVM
# -------------------------------
svm = LinearSVC(
    random_state=801
)

svm.fit(X_train, y_train)

print("===== Linear SVM =====")
print(classification_report(
    y_test,
    svm.predict(X_test)
))

# -------------------------------
# 외부 데이터 평가
# -------------------------------

fake = pd.read_csv("Fake.csv")["text"]
true = pd.read_csv("True.csv")["text"]

fake_vec = vectorizer.transform(fake)
true_vec = vectorizer.transform(true)

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Linear SVM": svm
}

for name, model in models.items():
    fake_pred = model.predict(fake_vec)
    true_pred = model.predict(true_vec)

    fake_acc = (fake_pred == 0).mean()
    true_acc = (true_pred == 1).mean()

    print(f"\n===== {name} =====")
    print(f"Fake 정확도 : {fake_acc:.4f}")
    print(f"True 정확도 : {true_acc:.4f}")



