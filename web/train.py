import os
import pandas as pd
import numpy as np
import threading
import time
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

def train_model(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    results,
    training_threshold,
    dataset_name,
    is_optimized=False,
    X_val=None,
    y_val=None
):
    y_pred = [None]
    training_time = [None]
    optimization_time = [0]
    training_completed = [False]

    def train():
        start_time = time.time()
        try:
            model_type = "Optimized" if is_optimized else "Default"

            if isinstance(model, BayesSearchCV) and is_optimized:
                # Perform parameter optimization
                opt_start = time.time()
                model.fit(X_train, y_train)
                optimization_time[0] = time.time() - opt_start
            else:
                # Default or duplicated "optimized" training (for models without optimization)
                if 'lightgbm' in model.get_params():
                    # Removed early stopping logic here
                    lgb_estimator = model.named_steps['lightgbm']
                    lgb_estimator.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

            y_pred[0] = model.predict(X_test)
            training_time[0] = time.time() - start_time
            training_completed[0] = True
        except Exception as e:
            print(f"Error training {model_name} ({model_type}): {e}")
            training_completed[0] = False

    thread = threading.Thread(target=train)
    thread.start()
    thread.join(timeout=training_threshold)

    if not training_completed[0]:
        print(f"Model {model_name} ({'Optimized' if is_optimized else 'Default'}) exceeded training time ({training_threshold}s) or failed.")
        y_pred[0] = np.nan
        training_time[0] = np.nan
    else:
        # Evaluate performance
        mse = mean_squared_error(y_test, y_pred[0])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred[0])
        r_squared = r2_score(y_test, y_pred[0])

        n = len(y_test)
        p = X_test.shape[1]
        adjusted_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1)) if n > p + 1 and p > 0 else r_squared

        result = {
            'Model': f"{model_name} ({'Optimized' if is_optimized else 'Default'})",
            'Dataset': dataset_name,
            'Parameter Type': 'Optimized' if is_optimized else 'Default',
            'Training Time (s)': training_time[0],
            'Optimization Time (s)': optimization_time[0],
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r_squared,
            'Adjusted R2 Score': adjusted_r_squared
        }

        if isinstance(model, BayesSearchCV) and is_optimized:
            result['Best Params'] = str(model.best_params_)
        elif not is_optimized:
            result['Parameters'] = str(model.get_params())

        results.append(result)

def main():
    # Default parameters
    default_params = {
        'Linear Regression': {},
        'Ridge Regression': {'ridge__alpha': 1.0},
        'Lasso Regression': {'lasso__alpha': 0.1},
        'Elastic Net Regression': {
            'elasticnet__alpha': 0.1,
            'elasticnet__l1_ratio': 0.5
        },
        'Random Forest Regression': {
            'randomforest__n_estimators': 100,
            'randomforest__max_depth': 10,
            'randomforest__min_samples_split': 2
        },
        'XGBoost Regression': {
            'xgboost__learning_rate': 0.01,
            'xgboost__max_depth': 3,
            'xgboost__n_estimators': 100
        },
        'LightGBM Regression': {}
    }

    # Parameter spaces for optimization
    param_spaces = {
        'Ridge Regression': {
            'ridge__alpha': Real(0.1, 5.0, prior='log-uniform')
        },
        'Lasso Regression': {
            'lasso__alpha': Real(0.01, 0.5, prior='log-uniform')
        },
        'Elastic Net Regression': {
            'elasticnet__alpha': Real(0.01, 0.5, prior='log-uniform'),
            'elasticnet__l1_ratio': Real(0.2, 0.8)
        },
        'LightGBM Regression': {
            'lightgbm__num_leaves': Integer(20, 40),
            'lightgbm__learning_rate': Real(0.01, 0.03, prior='log-uniform'),
            'lightgbm__n_estimators': Integer(50, 150)
        },
        'Random Forest Regression': {
            'randomforest__n_estimators': Integer(50, 100),
            'randomforest__max_depth': Categorical([5, 10]),
            'randomforest__min_samples_split': Integer(2, 4)
        },
        'XGBoost Regression': {
            'xgboost__learning_rate': Real(0.01, 0.03, prior='log-uniform'),
            'xgboost__max_depth': Integer(3, 5),
            'xgboost__n_estimators': Integer(50, 150)
        }
    }

    # Pipelines
    pipelines = {
        'Linear Regression': Pipeline([('linearregression', LinearRegression())]),
        'Ridge Regression': Pipeline([('ridge', Ridge())]),
        'Lasso Regression': Pipeline([('lasso', Lasso())]),
        'Elastic Net Regression': Pipeline([('elasticnet', ElasticNet())]),
        'LightGBM Regression': Pipeline([
            ('lightgbm', LGBMRegressor(num_leaves=31, learning_rate=0.01, n_estimators=100))
        ]),
        'Random Forest Regression': Pipeline([('randomforest', RandomForestRegressor())]),
        'XGBoost Regression': Pipeline([
            ('xgboost', xgb.XGBRegressor(use_label_encoder=False, eval_metric='rmse'))
        ])
    }

    # Set default params
    for name, pipeline in pipelines.items():
        if name in default_params and default_params[name]:
            pipeline.set_params(**default_params[name])

    # Create optimized models
    models_optimized = {}
    for name, pipeline in pipelines.items():
        if name in param_spaces:
            models_optimized[name] = BayesSearchCV(
                estimator=pipeline,
                search_spaces=param_spaces[name],
                cv=3,
                scoring='r2',
                n_jobs=1,
                verbose=1,
                n_iter=5,
                random_state=42
            )
        else:
            # If no optimization, reuse pipeline as "optimized"
            models_optimized[name] = pipeline

    training_threshold = 7200
    datasets = ['dataset1', 'dataset5']

    results_dir = "model_results"
    os.makedirs(results_dir, exist_ok=True)
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}")
        data_path = f"/home/dev/project/modelling/preprocessing/results/{dataset_name}"

        try:
            train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
            test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
            val_data = pd.read_csv(os.path.join(data_path, "val.csv"))
        except Exception as e:
            print(f"Error loading data for {dataset_name}: {e}")
            continue

        # Remove 'mssv' if present
        for df in [train_data, test_data, val_data]:
            if 'mssv' in df.columns:
                df.drop(columns=['mssv'], inplace=True)

        target_variable = 'diem_hp'
        if target_variable not in train_data.columns:
            print(f"Target variable '{target_variable}' not found in {dataset_name}. Skipping.")
            continue

        X_train = train_data.drop(columns=[target_variable]).fillna(0)
        y_train = train_data[target_variable].fillna(0)
        X_test = test_data.drop(columns=[target_variable]).fillna(0)
        y_test = test_data[target_variable].fillna(0)
        X_val = val_data.drop(columns=[target_variable]).fillna(0)
        y_val = val_data[target_variable].fillna(0)

        results = []

        # First run: Default parameters
        for model_name, pipeline in pipelines.items():
            print(f"\nTraining {model_name} (Default)...")
            train_model(
                model_name=model_name,
                model=pipeline,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                results=results,
                training_threshold=training_threshold,
                dataset_name=dataset_name,
                is_optimized=False,
                X_val=X_val,
                y_val=y_val
            )

        # Second run: Optimized parameters
        for model_name, model in models_optimized.items():
            if model_name in param_spaces:
                print(f"\nTraining {model_name} (Optimized)...")
                train_model(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    results=results,
                    training_threshold=training_threshold,
                    dataset_name=dataset_name,
                    is_optimized=True,
                    X_val=X_val,
                    y_val=y_val
                )
            else:
                # No optimization available; duplicate default run as "Optimized"
                print(f"\n{model_name} has no optimization; duplicating Default as Optimized...")
                train_model(
                    model_name=model_name,
                    model=pipelines[model_name],
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    results=results,
                    training_threshold=training_threshold,
                    dataset_name=dataset_name,
                    is_optimized=True,
                    X_val=X_val,
                    y_val=y_val
                )

        # Save results
        results_df = pd.DataFrame(results)
        dataset_results_dir = os.path.join(results_dir, dataset_name)
        os.makedirs(dataset_results_dir, exist_ok=True)
        results_file = os.path.join(dataset_results_dir, 'model_results_comparison.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Results for {dataset_name} saved to {results_file}")

if __name__ == "__main__":
    main()
