# train_and_save_model.py (run this once)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('Datasets.csv')

def apply_response(Label):
    if Label == 0:
        return 0  # True Content
    elif Label == 1:
        return 1  # False Content

df['results'] = df['Label'].apply(apply_response)

cv = CountVectorizer()
X = df['content']
y = df['results']
X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

models = []
NB = MultinomialNB()
NB.fit(X_train, y_train)
models.append(('naive_bayes', NB))

lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, y_train)
models.append(('svm', lin_clf))

reg = LogisticRegression(random_state=0, solver='lbfgs')
reg.fit(X_train, y_train)
models.append(('logistic', reg))

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
models.append(('DecisionTreeClassifier', dtc))

classifier = VotingClassifier(models)
classifier.fit(X_train, y_train)

# Save vectorizer + model
joblib.dump((cv, classifier), "content_model.pkl")
print("Model saved as content_model.pkl")
