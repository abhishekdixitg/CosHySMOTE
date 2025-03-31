
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Feature: Random values between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Target: Linear relation with some noise


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LinearRegression();

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# Step 7: Visualize Results
plt.scatter(X_test, y_test, color="blue", label="Actual Data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.title("Linear Regression Model")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.show()

# Step 8: Display Model Parameters
print(f"Intercept (b0): {model.intercept_[0]:.2f}")
print(f"Coefficient (b1): {model.coef_[0][0]:.2f}")
