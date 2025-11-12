import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ CREATE A SAMPLE DATASET ------------------
# Example: Predicting CO2 Emissions based on Engine Size, Cylinders, and Fuel Consumption
data = {
    'Engine_Size': [1.6, 2.0, 2.4, 3.0, 3.5, 4.0, 4.5, 5.0],
    'Cylinders': [4, 4, 4, 6, 6, 8, 8, 8],
    'Fuel_Consumption': [8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 14.0],
    'CO2_Emissions': [150, 170, 190, 220, 240, 265, 290, 310]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# ------------------ DEFINE INDEPENDENT (X) AND DEPENDENT (y) VARIABLES ------------------
X = df[['Engine_Size', 'Cylinders', 'Fuel_Consumption']]
y = df['CO2_Emissions']

# ------------------ SPLIT THE DATA INTO TRAINING AND TESTING SETS ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------ CREATE AND TRAIN THE MODEL ------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------ MAKE PREDICTIONS ------------------
y_pred = model.predict(X_test)

# ------------------ EVALUATE THE MODEL ------------------
print("\nRegression Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ------------------ VISUALIZE RESULTS ------------------
# Visualize how predicted vs actual values compare
plt.scatter(y_test, y_pred, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.grid(True)
plt.show()

# ------------------ CONCLUSION ------------------
print("\nConclusion: Thus, we have implemented Multiple Linear Regression using Python.")