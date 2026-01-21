# ================================
# House Price Prediction - Model Development
# ================================

# Import libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# 1. Load Dataset
# ================================
data = pd.read_csv(r'C:\Users\HP\Downloads\house-prices-advanced-regression-techniques\train.csv')

# ================================
# 2. Feature Selection
# Using ONLY 6 approved features + target
# ================================
selected_features = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "Neighborhood",
    "SalePrice"
]

data = data[selected_features]

# ================================
# 3. Handle Missing Values
# ================================
# Numerical columns
num_cols = data.select_dtypes(include=["int64", "float64"]).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

# Categorical column
data["Neighborhood"] = data["Neighborhood"].fillna(
    data["Neighborhood"].mode()[0]
)

# ================================
# 4. Encode Categorical Variable
# ================================
data = pd.get_dummies(data, columns=["Neighborhood"], drop_first=True)

# ================================
# 5. Split Features and Target
# ================================
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 6. Train Model (Random Forest)
# ================================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ================================
# 7. Model Evaluation
# ================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance (Random Forest)")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

# ================================
# 8. Save Model and Feature Columns
# ================================
joblib.dump(model, "model/house_price_model.pkl")
joblib.dump(X.columns, "model/model_columns.pkl")

print("\nModel and feature columns saved successfully!")