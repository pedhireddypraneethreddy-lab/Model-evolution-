import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 2, 3, 4],
    "Marks": [50, 55, 65, 70, 80, 85, 90, 45, 55, 60]
}
df = pd.DataFrame(data)

# X and y
X = df[["Study_Hours"]]
y = df["Marks"]

# Model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Evaluation
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("MAE:", mae)
print("R2 Score:", r2)

# Graph 1: Actual vs Predicted
plt.scatter(y, y_pred)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.show()

# Graph 2: Regression Line
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Regression Line")
plt.show()
