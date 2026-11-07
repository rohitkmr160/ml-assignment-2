"""Train 5 classifiers on the UCI Breast Cancer Wisconsin (Diagnostic) dataset,
evaluate them, and save models + test data for the Streamlit app."""
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

RANDOM_STATE = 42

# Step 1: dataset (569 instances, 30 features, binary target)
data = load_breast_cancer(as_frame=True)
df = data.frame            # 'target' column: 0 = malignant, 1 = benign
df.to_csv("breast_cancer_full.csv", index=False)
X, y = df.drop(columns="target"), df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Step 2: models (scaling inside a pipeline so the saved file is self-contained)
models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "KNN": Pipeline([("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    joblib.dump(model, f"model/{name.lower().replace(' ', '_')}.joblib")

results = pd.DataFrame(rows).round(4)
results.to_csv("model/metrics_summary.csv", index=False)
print(results.to_string(index=False))

# Test data (features + target) to upload in the Streamlit app
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print("test_data.csv:", test_df.shape, "| train:", X_train.shape)
