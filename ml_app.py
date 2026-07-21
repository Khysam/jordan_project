import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os

def run_ml_pipeline():
    # --- 1. Data Loading and Initial Preprocessing ---
    file_path = 'jordan_market_dataset_2026.csv'
    try:
        df = pd.read_csv(file_path)
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
        print("Data loaded and 'Sale_Date' converted.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory.")
        return

    # Store unique categorical values for Streamlit app
    unique_shoe_models = df['Shoe_Model'].unique().tolist()
    unique_colorways = df['Colorway'].unique().tolist()
    unique_conditions = df['Condition'].unique().tolist()

    # --- 4. Feature Engineering ---
    features = ['Shoe_Model', 'Colorway', 'Condition', 'Retail_Price_USD']
    target = 'Resale_Price_USD'
    df_model = df[features + [target]].copy()
    categorical_features = ['Shoe_Model', 'Colorway', 'Condition']
    df_model_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)
    X = df_model_encoded.drop(columns=[target])
    y = df_model_encoded[target]
    print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

    # --- 5. Model Selection and Training ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_results = {}

    # RandomForestRegressor
    print("Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    model_results['RandomForestRegressor'] = {'R2': r2_rf}
    print(f"RandomForestRegressor R2: {r2_rf:.2f}")

    # LinearRegression
    print("Training LinearRegression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    model_results['LinearRegression'] = {'R2': r2_lr}
    print(f"LinearRegression R2: {r2_lr:.2f}")

    # GradientBoostingRegressor
    print("Training GradientBoostingRegressor...")
    gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbr_model.fit(X_train, y_train)
    y_pred_gbr = gbr_model.predict(X_test)
    r2_gbr = r2_score(y_test, y_pred_gbr)
    model_results['GradientBoostingRegressor'] = {'R2': r2_gbr}
    print(f"GradientBoostingRegressor R2: {r2_gbr:.2f}")

    # --- Select the best model ---
    best_model_name = max(model_results, key=lambda name: model_results[name]['R2'])
    best_r2 = model_results[best_model_name]['R2']
    print(f"Best model: {best_model_name} with R2: {best_r2:.2f}")

    best_model = None
    if best_model_name == 'RandomForestRegressor':
        best_model = rf_model
    elif best_model_name == 'LinearRegression':
        best_model = lr_model
    elif best_model_name == 'GradientBoostingRegressor':
        best_model = gbr_model

    # --- 8. Save Model and Artifacts for Deployment ---
    model_save_path = 'best_sneaker_resale_model.pkl'
    features_save_path = 'model_features.pkl'
    unique_cat_values_path = 'unique_categorical_values.pkl'

    if best_model is not None:
        joblib.dump(best_model, model_save_path)
        joblib.dump(X.columns.tolist(), features_save_path)
        joblib.dump({
            'Shoe_Model': unique_shoe_models,
            'Colorway': unique_colorways,
            'Condition': unique_conditions
        }, unique_cat_values_path)
        print(f"Best model saved to {model_save_path}")
        print(f"Feature columns saved to {features_save_path}")
        print(f"Unique categorical values saved to {unique_cat_values_path}")
    else:
        print("No best model found to save.")

if __name__ == '__main__':
    run_ml_pipeline()
