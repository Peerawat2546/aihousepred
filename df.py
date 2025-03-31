import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "Housing.csv"
df = pd.read_csv(file_path)

binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
for col in binary_columns:
    df[col] = df[col].apply(lambda x: 1 if x == 'yes' else 0)

selected_features = ['area', 'bathrooms', 'airconditioning', 'stories', 
                     'parking', 'bedrooms', 'prefarea', 'mainroad']
X = df[selected_features].values
y = df['price'].values.reshape(-1, 1)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

X = np.hstack((np.ones((X.shape[0], 1)), X))

class HousePricePredictor:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.theta = self.compute_theta()
    
    def compute_theta(self):
        return np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
    
    def predict(self, area, bedrooms, bathrooms, stories, parking, airconditioning, prefarea, mainroad):
        input_features = np.array([[area, bathrooms, airconditioning, stories, 
                                    parking, bedrooms, prefarea, mainroad]])
        input_features = (input_features - X_mean) / X_std
        input_features = np.hstack(([1], input_features.flatten()))
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
example_features = [8700, 3, 2, 2, 5, 1, 1, 1]  
predicted_price = predictor.predict(*example_features)
print(f"Predicted price: {predicted_price[0]:,.2f}")

r_squared = predictor.compute_r_squared()
print(f"R-squared: {r_squared:.4f}")

predictor.plot_predictions()