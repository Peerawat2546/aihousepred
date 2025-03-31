import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

file_path = "Housing.csv"
df = pd.read_csv(file_path)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, drop_first=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

X = df.drop('price', axis=1).values
y = df['price'].values

lsvc = LinearSVC(C=0.2, penalty="l1", dual=False, max_iter=5000).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
selected_features = df.drop('price', axis=1).columns[model.get_support()].tolist()
print("Selected Features:", selected_features)

X = X_new.astype(float)
y = y.astype(float)

class HousePricePredictor:
    def __init__(self, X, y):
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.y = y.reshape(-1, 1)
        self.theta = self.compute_theta()

    def compute_theta(self):
        return np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y

    def predict(self, input_features):
        input_features = np.insert(input_features, 0, 1)
        return input_features @ self.theta

    def compute_r_squared(self):
        y_pred = self.X @ self.theta
        sst = np.sum((self.y - np.mean(self.y))**2)
        ssr = np.sum((self.y - y_pred)**2)
        return 1 - (ssr / sst)

    def plot_predictions(self):
        predicted_prices = self.X @ self.theta
        plt.scatter(self.y, predicted_prices, alpha=0.5, label='Actual vs Predicted')
        plt.plot(self.y, self.y, color='red', label='Perfect Prediction Line')
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs Predicted Prices")
        plt.legend()
        plt.show()

predictor = HousePricePredictor(X, y)

input = X[0]
predicted_price = predictor.predict(input)
print(f"Predicted price: {predicted_price[0]:,.2f}")

r_squared = predictor.compute_r_squared()
print(f"R-squared: {r_squared:.4f}")

predictor.plot_predictions()
