import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "Housing.csv" 
df = pd.read_csv(file_path)
X = df[['area', 'bedrooms', 'bathrooms']].values
y = df['price'].values

class HousePricePredictor:
    def __init__(self, X, y):
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.y = y.reshape(-1, 1)
        self.theta = self.compute_theta()
    
    def compute_theta(self):
        return np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
    
    def predict(self, area, bedrooms, bathrooms):
        input_features = np.array([1, area, bedrooms, bathrooms])
        return input_features @ self.theta
    
    def compute_r_squared(self):
        y_pred = self.X @ self.theta
        sst = np.sum((self.y - np.mean(self.y))**2)
        ssr = np.sum((self.y - y_pred)**2)
        r2 = 1 - (ssr / sst)
        return r2

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
example_area = 8700
example_bedrooms = 5
example_bathrooms = 2
predicted_price = predictor.predict(example_area, example_bedrooms, example_bathrooms)
print(f"Predicted price: {predicted_price[0]:,.2f}")

r_squared = predictor.compute_r_squared()
print(f"R-squared: {r_squared:.4f}")

predictor.plot_predictions()