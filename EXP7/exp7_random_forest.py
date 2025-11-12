import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ LOAD DATASET ------------------
iris = load_iris()
X = iris.data
y = iris.target

# ------------------ SPLIT INTO TRAIN AND TEST SETS ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------ CREATE AND TRAIN THE RANDOM FOREST MODEL ------------------
# n_estimators = number of trees in the forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ------------------ MAKE PREDICTIONS ------------------
y_pred = rf_model.predict(X_test)

# ------------------ EVALUATE MODEL PERFORMANCE ------------------
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# ------------------ VISUALIZE CONFUSION MATRIX ------------------
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title("Random Forest Classifier - Confusion Matrix (Iris Dataset)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

# ------------------ FEATURE IMPORTANCE VISUALIZATION ------------------
importances = rf_model.feature_importances_
feature_names = iris.feature_names

plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=feature_names, color="green")
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ------------------ CONCLUSION ------------------
print("Conclusion: Thus, we have implemented the Random Forest Classifier using Python.")