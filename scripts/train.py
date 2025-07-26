from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import joblib

data = fetch_california_housing()
X, y = data.data, data.target

model = LinearRegression().fit(X, y)
joblib.dump(model, "sk_model.joblib")
print("Model trained and saved to sk_model.joblib")