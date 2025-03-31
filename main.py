import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "Housing.csv"  
df = pd.read_csv(file_path)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical Columns:", categorical_cols)

df = pd.get_dummies(df, drop_first=True) 

df = df.apply(pd.to_numeric, errors='coerce') 

df.fillna(df.mean(), inplace=True)

correlations = df.corr()['price'].abs().sort_values(ascending=False)
print("Feature Correlations:\n", correlations)

threshold = 0
selected_features = correlations[correlations > threshold].index.tolist()
selected_features.remove('price') 
print("Selected Features:", selected_features)

X = df[selected_features].astype(float).values  
y = df['price'].astype(float).values 

class HousePricePredictor:
    def __init__(self, X, y, feature_names):
        self.feature_names = feature_names
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

predictor = HousePricePredictor(X, y, selected_features)

input_features = np.array([6500, 4, 1, 2, 2, 5, 0, 0, 0, 0, 1, 0, 1])
predicted_price = predictor.predict(input_features)
print(f"Predicted price: {predicted_price[0]:,.2f}")

r_squared = predictor.compute_r_squared()
print(f"R-squared: {r_squared:.4f}")

predictor.plot_predictions()
