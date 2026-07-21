"""
Pipeline training untuk model prediksi harga jual kembali (resale) sneaker
Air Jordan.

Perbaikan dibanding versi sebelumnya:
- Preprocessing (one-hot encoding) dibungkus dalam sklearn Pipeline lewat
  ColumnTransformer, bukan pd.get_dummies manual. Ini menghilangkan risiko
  mismatch kolom antara training dan inference, dan otomatis menangani
  kategori baru yang tidak pernah dilihat saat training (handle_unknown='ignore').
- Hyperparameter tuning (RandomizedSearchCV) untuk RandomForestRegressor dan
  GradientBoostingRegressor, dievaluasi dengan cross-validation.
- Evaluasi akhir memakai R2, MAE, dan RMSE pada test set yang benar-benar
  belum pernah dilihat model.
- Artefak yang disimpan jadi lebih sederhana: satu Pipeline utuh (preprocessing
  + model) sehingga app.py tidak perlu lagi merekonstruksi one-hot encoding
  secara manual.
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = 'jordan_market_dataset_2026.csv'
MODEL_SAVE_PATH = 'best_sneaker_resale_model.pkl'
FEATURES_SAVE_PATH = 'model_features.pkl'
UNIQUE_CAT_VALUES_PATH = 'unique_categorical_values.pkl'
METRICS_SAVE_PATH = 'model_metrics.pkl'

CATEGORICAL_FEATURES = ['Shoe_Model', 'Colorway', 'Condition']
NUMERICAL_FEATURES = ['Retail_Price_USD']
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = 'Resale_Price_USD'

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
    print(f"Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
    return df


def build_preprocessor():
    """ColumnTransformer: one-hot encoding untuk kategorikal, numerik dibiarkan apa adanya.

    handle_unknown='ignore' membuat pipeline tidak crash kalau suatu saat ada
    kombinasi kategori baru yang belum pernah dilihat saat training.
    """
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ],
        remainder='passthrough',  # kolom numerik (Retail_Price_USD) diteruskan apa adanya
    )


def get_search_spaces():
    """Definisi model kandidat + ruang pencarian hyperparameter."""
    return {
        'RandomForestRegressor': (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                'model__n_estimators': [100, 200, 300, 400],
                'model__max_depth': [None, 5, 10, 15, 20],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4, 8],
                'model__max_features': ['sqrt', 'log2', None],
            },
        ),
        'GradientBoostingRegressor': (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'model__max_depth': [2, 3, 4, 5],
                'model__subsample': [0.7, 0.85, 1.0],
                'model__min_samples_leaf': [1, 2, 4],
            },
        ),
    }


def run_ml_pipeline():
    df = load_data()

    # Simpan nilai unik kategorikal untuk dropdown di app.py
    unique_cat_values = {
        col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_FEATURES
    }

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} baris | Test: {X_test.shape[0]} baris")

    preprocessor = build_preprocessor()
    search_spaces = get_search_spaces()

    results = {}
    fitted_estimators = {}

    # --- Baseline tanpa tuning: LinearRegression ---
    print("\n=== Melatih baseline: LinearRegression (tanpa tuning) ===")
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression()),
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred = lr_pipeline.predict(X_test)
    results['LinearRegression'] = {
        'cv_r2_mean': None,
        'cv_r2_std': None,
        'test_r2': r2_score(y_test, y_pred),
        'test_mae': mean_absolute_error(y_test, y_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    fitted_estimators['LinearRegression'] = lr_pipeline
    print(f"  Test R2: {results['LinearRegression']['test_r2']:.4f}")

    # --- Model dengan hyperparameter tuning ---
    for model_name, (estimator, param_grid) in search_spaces.items():
        print(f"\n=== Tuning {model_name} (RandomizedSearchCV, cv=5) ===")
        start = time.time()

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', estimator),
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=20,
            cv=5,
            scoring='r2',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - start

        best_pipeline = search.best_estimator_
        y_pred = best_pipeline.predict(X_test)

        results[model_name] = {
            'cv_r2_mean': search.best_score_,
            'cv_r2_std': search.cv_results_['std_test_score'][search.best_index_],
            'test_r2': r2_score(y_test, y_pred),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'best_params': search.best_params_,
        }
        fitted_estimators[model_name] = best_pipeline

        print(f"  Selesai dalam {elapsed:.1f}s")
        print(f"  CV R2 (mean +/- std): {search.best_score_:.4f} +/- {results[model_name]['cv_r2_std']:.4f}")
        print(f"  Test R2: {results[model_name]['test_r2']:.4f}")
        print(f"  Test MAE: {results[model_name]['test_mae']:.2f}")
        print(f"  Test RMSE: {results[model_name]['test_rmse']:.2f}")
        print(f"  Best params: {search.best_params_}")

    # --- Pilih model terbaik berdasarkan Test R2 ---
    best_model_name = max(results, key=lambda name: results[name]['test_r2'])
    best_pipeline = fitted_estimators[best_model_name]
    best_metrics = results[best_model_name]

    print(f"\n=== Model terbaik: {best_model_name} (Test R2: {best_metrics['test_r2']:.4f}) ===")

    # --- Simpan artefak ---
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    joblib.dump(FEATURES, FEATURES_SAVE_PATH)
    joblib.dump(unique_cat_values, UNIQUE_CAT_VALUES_PATH)
    joblib.dump(
        {
            'best_model_name': best_model_name,
            'metrics': best_metrics,
            'all_results': results,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'features': FEATURES,
        },
        METRICS_SAVE_PATH,
    )

    print(f"\nModel disimpan ke: {MODEL_SAVE_PATH}")
    print(f"Daftar fitur disimpan ke: {FEATURES_SAVE_PATH}")
    print(f"Nilai unik kategorikal disimpan ke: {UNIQUE_CAT_VALUES_PATH}")
    print(f"Metrik evaluasi disimpan ke: {METRICS_SAVE_PATH}")


if __name__ == '__main__':
    run_ml_pipeline()
