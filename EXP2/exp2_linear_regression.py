import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('TkAgg')

# ------------------ CREATE A SIMPLE DATASET ------------------
# Example: Predict Salary (dependent variable) based on Years of Experience (independent variable)
data = {
    'Years_of_Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [30000, 35000, 40000, 43000, 48000, 50000, 55000, 60000, 63000, 70000]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# ------------------ SEPARATE INDEPENDENT AND DEPENDENT VARIABLES ------------------
X = df[['Years_of_Experience']]  # Independent variable
y = df['Salary']                 # Dependent variable

# ------------------ CREATE AND TRAIN MODEL ------------------
model = LinearRegression()
model.fit(X, y)

# ------------------ MAKE PREDICTIONS ------------------
y_pred = model.predict(X)

# ------------------ DISPLAY RESULTS ------------------
print("\nRegression Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mean_squared_error(y, y_pred))
print("R² Score:", r2_score(y, y_pred))

# ------------------ VISUALIZE THE REGRESSION LINE ------------------
plt.scatter(X, y, color='blue', label='Actual Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title("Simple Linear Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.show()

# ------------------ CONCLUSION ------------------
print("\nConclusion: Thus, we have implemented Simple Linear Regression and visualized the best-fit line.")