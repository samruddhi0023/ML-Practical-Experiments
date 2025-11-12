import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ------------------ LOAD DATASET ------------------
iris = load_iris()
X = iris.data
y = iris.target

# ------------------ SPLIT DATA ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------ STANDARDIZE DATA ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ DEFINE CLASSIFICATION MODELS ------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# ------------------ TRAIN AND EVALUATE MODELS ------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)
    })

# ------------------ DISPLAY RESULTS ------------------
df_results = pd.DataFrame(results)
print("\nPerformance Comparison of Classification Algorithms:\n")
print(df_results.sort_values(by="Accuracy", ascending=False))

# ------------------ VISUALIZE RESULTS ------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.barh(df_results["Model"], df_results["Accuracy"], color='skyblue')
plt.xlabel("Accuracy Score")
plt.ylabel("Algorithm")
plt.title("Comparison of Classification Algorithms (Iris Dataset)")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ------------------ CONCLUSION ------------------
print("\nConclusion: Thus, we compared multiple classification algorithms and observed their respective accuracies and evaluation metrics.")