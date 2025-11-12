# Experiment No. 4
# Title: Implement Logistic Regression Algorithm on a Suitable Dataset
# Aim: To learn about supervised regression technique named Logistic Regression

# ------------------ IMPORT REQUIRED LIBRARIES ------------------
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ LOAD DATASET ------------------
# Using the famous Iris dataset (multiclass classification)
iris = load_iris()
X = iris.data       # Independent features
y = iris.target     # Target classes

# ------------------ SPLIT DATA INTO TRAIN AND TEST ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------ TRAIN LOGISTIC REGRESSION MODEL ------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ------------------ MAKE PREDICTIONS ------------------
y_pred = model.predict(X_test)

# ------------------ EVALUATE MODEL ------------------
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------ VISUALIZE CONFUSION MATRIX ------------------
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title("Confusion Matrix for Logistic Regression (Iris Dataset)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

# ------------------ CONCLUSION ------------------
print("Conclusion: Thus, we have successfully implemented Logistic Regression using Python.")
