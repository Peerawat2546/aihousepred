import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

file_path = "Housing.csv"  
df = pd.read_csv(file_path)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical Columns:", categorical_cols)

df = pd.get_dummies(df, drop_first=True) 

df = df.apply(pd.to_numeric, errors='coerce') 

df.fillna(df.mean(), inplace=True)

X = df.drop(columns=['price']).astype(float).values  
y = df['price'].astype(float).values 

class HousePricePredictor:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        self.pipeline = Pipeline([
            ('feature_selection', SelectFromModel(Lasso(alpha=0.05))),
            ('regression', LinearRegression())
        ])
        self.pipeline.fit(self.X_train, self.y_train)
    
    def predict(self, input_features):
        input_features = input_features.reshape(1, -1)  
        return self.pipeline.predict(input_features)
    
    def compute_r_squared(self):
        y_pred = self.pipeline.predict(self.X_test)
        return r2_score(self.y_test, y_pred)

    def plot_predictions(self):
        y_pred = self.pipeline.predict(self.X_test)
        plt.scatter(self.y_test, y_pred, alpha=0.5, label='Actual vs Predicted')
        plt.plot(self.y_test, self.y_test, color='red', label='Perfect Prediction Line')
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
