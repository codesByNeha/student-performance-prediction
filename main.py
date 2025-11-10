#  STUDENT PERFORMANCE PREDICTION PROJECT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import joblib

data = pd.read_csv("archive/student-por.csv")

print(" Dataset loaded successfully!")
print("Shape of dataset:", data.shape)
print("\nFirst 5 rows:\n", data.head())

print("\n Columns in dataset:\n", data.columns)
print("\nData Types:\n", data.dtypes)
print("\nMissing Values:\n", data.isnull().sum())

le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

print("\n Encoding complete. Encoded data preview:\n")
print(data.head())

correlation = data.corr()
print("\n🔹 Correlation with G3 (Final Grade):\n")
print(correlation["G3"].sort_values(ascending=False))

plt.figure(figsize=(12, 8))
sns.heatmap(correlation, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap of Features")
plt.show()

X = data.drop('G3', axis=1)
y = data['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n Data successfully split!")
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Metrics:")
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 2))

sorted_index = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_index]
y_pred_sorted = y_pred[sorted_index]

x = np.arange(25)
width = 0.4

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, y_test_sorted[:25], width=width, color='skyblue', alpha=0.8, label='Actual Grades')
plt.bar(x + width/2, y_pred_sorted[:25], width=width, color='orange', alpha=0.7, label='Predicted Grades')
plt.xlabel("Student Index")
plt.ylabel("Final Grade (G3)")
plt.title("Actual vs Predicted Student Final Grades (First 25 Students)")
plt.legend()
plt.tight_layout()
plt.show()

x = np.arange(25)
width = 0.4

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, y_test_sorted[:25], width=width, color='lightcoral', alpha=0.8, label='Actual')
plt.bar(x + width/2, y_pred_sorted[:25], width=width, color='mediumseagreen', alpha=0.8, label='Predicted')
plt.xlabel("Student Index")
plt.ylabel("Grades (G3)")
plt.title("Vertical Regression Bar Chart - Actual vs Predicted Grades (First 25 Students)")
plt.legend()
plt.tight_layout()
plt.show()

strong_features = correlation["G3"][correlation["G3"].abs() > 0.1].index
print("\n🔹 Selected Strong Features:\n", strong_features)

X_strong = data[strong_features].drop('G3', axis=1)
y = data['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X_strong, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_strong = r2_score(y_test, y_pred)
print("\n R² Score after Feature Selection:", round(r2_strong, 2))

poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)
y_poly_pred = poly_model.predict(X_test)
r2_poly = r2_score(y_test, y_poly_pred)

print("\n R² Score with Polynomial Regression (Degree 2):", round(r2_poly, 2))

print("\n Model Comparison:")
print(f"Linear Regression R²: {round(r2, 2)}")
print(f"Feature-Selected Linear Regression R²: {round(r2_strong, 2)}")
print(f"Polynomial Regression R²: {round(r2_poly, 2)}")

joblib.dump(model, "student_grade_predictor.pkl")
print("\n 7-feature model saved as 'student_grade_predictor.pkl' successfully!")

joblib.dump(list(X_strong.columns), "model_features.pkl")
print(" Saved feature names for Streamlit frontend!")

errors = y_test.values - y_pred
x = np.arange(len(errors))

plt.figure(figsize=(12, 6))
plt.bar(x, errors, color=np.where(errors >= 0, 'limegreen', 'tomato'), alpha=0.8)
plt.axhline(y=0, color='black', linewidth=1.2, linestyle='--')
plt.title("Prediction Error Distribution (Actual - Predicted Grades)")
plt.xlabel("Student Index")
plt.ylabel("Error Value (Difference)")
plt.tight_layout()
plt.show()


