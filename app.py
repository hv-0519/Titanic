"""
app.py  -  Flask backend for Titanic Survival Prediction
=========================================================
Install:  pip install -r requirements.txt
Run locally:  python app.py
Deploy on Render: gunicorn app:app
"""

import os, warnings
import numpy as np
import pandas as pd
import seaborn as srs

from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════
print("Loading Titanic dataset...")
df = srs.load_dataset("titanic")

# Drop columns we don't need
df.drop(columns=['deck', 'embark_town', 'alive', 'adult_male', 'class', 'who'], inplace=True)

# ── Fix ALL missing values BEFORE any encoding or astype ──────
df['age'].fillna(df['age'].mean(), inplace=True)
df['fare'].fillna(df['fare'].mean(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['alone'].fillna(False, inplace=True)

# Drop any remaining rows with NaN (safety net)
df.dropna(inplace=True)

# ── Encode categorical columns ────────────────────────────────
le = LabelEncoder()
df['sex']      = le.fit_transform(df['sex'].astype(str))
df['embarked'] = le.fit_transform(df['embarked'].astype(str))

# Convert booleans and everything to int
df = df.apply(lambda col: col.map(lambda x: int(bool(x)) if isinstance(x, (bool, np.bool_)) else x))
df = df.astype(float).astype(int)

# ── Verify no NaN remains ─────────────────────────────────────
assert df.isnull().sum().sum() == 0, "NaN values still present after cleaning!"

X = df.drop(columns=['survived'])
y = df['survived']

FEATURE_NAMES = list(X.columns)
print(f"Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ══════════════════════════════════════════════════════════════
#  TRAIN MODELS
# ══════════════════════════════════════════════════════════════
print("Training models...")

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred1 = log_model.predict(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
Y_pred2 = knn_model.predict(X_test)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred3 = dt_model.predict(X_test)

print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred1))
print("KNN Confusion Matrix:\n",               confusion_matrix(y_test, Y_pred2))
print("Decision Tree Confusion Matrix:\n",     confusion_matrix(y_test, y_pred3))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred1))
print("KNN Accuracy:",                 accuracy_score(y_test, Y_pred2))
print("Decision Tree Accuracy:",       accuracy_score(y_test, y_pred3))

# ══════════════════════════════════════════════════════════════
#  PRE-COMPUTE METRICS
# ══════════════════════════════════════════════════════════════
MODELS = {
    "logistic":      log_model,
    "knn":           knn_model,
    "decision_tree": dt_model,
}
MODEL_LABELS = {
    "logistic":      "Logistic Regression",
    "knn":           "K-Nearest Neighbors",
    "decision_tree": "Decision Tree",
}

METRICS = {}
for name, model in MODELS.items():
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm      = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    knn_k = []
    for k in range(1, 21):
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train, y_train)
        knn_k.append(round(m.score(X_test, y_test), 4))

    METRICS[name] = {
        "train_accuracy":      round(model.score(X_train, y_train) * 100, 2),
        "test_accuracy":       round(model.score(X_test,  y_test)  * 100, 2),
        "roc_auc":             round(float(roc_auc), 4),
        "confusion_matrix":    cm.tolist(),
        "fpr":                 [round(v, 6) for v in fpr.tolist()],
        "tpr":                 [round(v, 6) for v in tpr.tolist()],
        "knn_k_scores":        knn_k,
        "feature_importances": model.feature_importances_.tolist()
                               if hasattr(model, "feature_importances_")
                               else [0.0] * len(FEATURE_NAMES),
    }

print("✅ All models ready.")

# ══════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data       = request.get_json(force=True)
        features   = data.get("features", [])
        model_name = data.get("model_name", "logistic")

        if model_name not in MODELS:
            return jsonify({"error": f"Invalid model. Choose: {list(MODELS.keys())}"}), 400
        if len(features) != len(FEATURE_NAMES):
            return jsonify({"error": f"Need {len(FEATURE_NAMES)} features, got {len(features)}"}), 400

        X_in  = np.array(features, dtype=float).reshape(1, -1)
        model = MODELS[model_name]
        pred  = int(model.predict(X_in)[0])
        proba = model.predict_proba(X_in)[0]

        return jsonify({
            "prediction":  "Survived" if pred == 1 else "Did Not Survive",
            "survived":    pred,
            "probability": round(float(proba[pred]) * 100, 2),
            "model_used":  MODEL_LABELS[model_name],
            "model_key":   model_name,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/model-metrics")
def model_metrics():
    return jsonify({
        k: {
            "model_name":     MODEL_LABELS[k],
            "train_accuracy": METRICS[k]["train_accuracy"],
            "test_accuracy":  METRICS[k]["test_accuracy"],
            "roc_auc":        METRICS[k]["roc_auc"],
        }
        for k in METRICS
    })

@app.route("/roc-data/<model_name>")
def roc_data(model_name):
    if model_name not in METRICS:
        return jsonify({"error": "Model not found"}), 404
    m = METRICS[model_name]
    return jsonify({
        "model_name": MODEL_LABELS[model_name],
        "fpr": m["fpr"], "tpr": m["tpr"], "auc": m["roc_auc"],
    })

@app.route("/confusion-matrix/<model_name>")
def confusion_matrix_data(model_name):
    if model_name not in METRICS:
        return jsonify({"error": "Model not found"}), 404
    return jsonify({
        "model_name":       MODEL_LABELS[model_name],
        "confusion_matrix": METRICS[model_name]["confusion_matrix"],
        "labels":           ["Did Not Survive", "Survived"],
    })

@app.route("/knn-k-scores")
def knn_k_scores():
    return jsonify({
        "scores":  METRICS["knn"]["knn_k_scores"],
        "k_range": list(range(1, 21)),
    })

@app.route("/feature-importance")
def feature_importance():
    return jsonify({
        "features":    FEATURE_NAMES,
        "importances": METRICS["decision_tree"]["feature_importances"],
    })

@app.route("/features")
def get_features():
    return jsonify({"features": FEATURE_NAMES, "count": len(FEATURE_NAMES)})

# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)