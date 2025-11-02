from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

from api.utils.preprocess import load_data, preprocess_text

df = load_data("Project/data/fake_news/financeiros", "Project/data/real_news/financeiros")

df["processed_text"] = df["texto"].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["processed_text"], df["label"], test_size=0.3, random_state=42, stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("svm", CalibratedClassifierCV(LinearSVC(), method="sigmoid", cv=5))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

dump(model, "Project/api/model/model.pkl")
print("Modelo salvo em model/model.pkl")
