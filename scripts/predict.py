from sklearn.datasets import fetch_california_housing
import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load("sk_model.joblib")
y_pred = model.predict(X_test)
print("R^2 score on test set:", r2_score(y_test, y_pred))