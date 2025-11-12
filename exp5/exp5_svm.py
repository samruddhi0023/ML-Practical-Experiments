import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ LOAD DATASET ------------------
# We'll use the Iris dataset again (3 classes)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# ------------------ SPLIT THE DATA ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------ CREATE AND TRAIN THE MODEL ------------------
# kernel='linear' can be changed to 'rbf', 'poly', 'sigmoid' for different decision boundaries
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# ------------------ MAKE PREDICTIONS ------------------
y_pred = svm_model.predict(X_test)

# ------------------ EVALUATE MODEL ------------------
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# ------------------ VISUALIZE CONFUSION MATRIX ------------------
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Greens", fmt='d',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title("SVM Classification - Confusion Matrix (Iris Dataset)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

# ------------------ CONCLUSION ------------------
print("Conclusion: Thus, we have implemented the Support Vector Machine (SVM) algorithm for classification using Python.")