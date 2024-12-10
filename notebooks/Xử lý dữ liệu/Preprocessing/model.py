import os
import pandas as pd
import numpy as np
import re
import threading
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

# 1. Define the split_data_by_group function
def split_data_by_group(data, group_col, train_ratio, val_ratio, test_ratio):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1."

    train_set = pd.DataFrame()
    val_set = pd.DataFrame()
    test_set = pd.DataFrame()

    # Group by the specified column
    grouped = data.groupby(group_col)

    for group, group_data in grouped:
        n_samples = len(group_data)
        if n_samples < 3:
            # Assign all samples to the training set
            train_set = pd.concat([train_set, group_data], ignore_index=True)
            print(f"Group '{group}' has only {n_samples} sample(s). Assigned to training set.")
            continue

        # Proceed with splitting for groups with sufficient samples
        try:
            # Split into train and temp (val + test)
            train, temp = train_test_split(
                group_data, test_size=1-train_ratio, random_state=42, shuffle=True
            )

            # Adjust val_size to ensure val + test = (val_ratio + test_ratio)
            val_size = val_ratio / (val_ratio + test_ratio)

            # Check if temp has enough samples to split
            if len(temp) < 2:
                # Assign all temp to test set if only one sample remains
                train_set = pd.concat([train_set, train], ignore_index=True)
                test_set = pd.concat([test_set, temp], ignore_index=True)
                print(f"Group '{group}' split into {len(train)} train and {len(temp)} test sample(s).")
            else:
                # Split temp into val and test
                val, test = train_test_split(
                    temp, test_size=1-val_size, random_state=42, shuffle=True
                )
                # Append to respective sets
                train_set = pd.concat([train_set, train], ignore_index=True)
                val_set = pd.concat([val_set, val], ignore_index=True)
                test_set = pd.concat([test_set, test], ignore_index=True)
                print(f"Group '{group}' split into {len(train)} train, {len(val)} val, and {len(test)} test sample(s).")
        except ValueError as e:
            # Handle unexpected errors
            print(f"Error splitting group '{group}': {e}. Assigning all to training set.")
            train_set = pd.concat([train_set, group_data], ignore_index=True)

    return train_set, val_set, test_set

# 2. Define the column cleaning functions
def clean_column_names(df):
    # Replace any sequence of non-word characters with a single underscore
    df.columns = [
        re.sub(r'\W+', '_', col).strip('_') for col in df.columns
    ]

    # Ensure column names start with a letter by prefixing with 'f_' if necessary
    df.columns = [
        col if re.match(r'^[A-Za-z]', col) else f'f_{col}' for col in df.columns
    ]

    # Ensure uniqueness by appending suffixes to duplicate names
    seen = {}
    new_columns = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    df.columns = new_columns

    return df

def verify_column_names(df):
    problematic_cols = [
        col for col in df.columns
        if not re.match(r'^[A-Za-z]\w*$', col)
    ]
    return problematic_cols

# 3. Define the model training function
def train_model(model_name, model, X_train, y_train, X_test, y_test, results, training_threshold, dataset_name):
    y_pred = [None]
    training_time = [None]
    training_completed = [False]

    def train():
        start_time = time.time()
        try:
            print(f"Starting training for {model_name}...")
            model.fit(X_train, y_train)
            y_pred[0] = model.predict(X_test)
            training_time[0] = time.time() - start_time
            training_completed[0] = True
            print(f"Completed training for {model_name} in {training_time[0]:.2f} seconds.")
        except Exception as e:
            print(f"Error training model {model_name}: {e}")
            training_completed[0] = False

    # Run training in a separate thread
    thread = threading.Thread(target=train)
    thread.start()
    thread.join(timeout=training_threshold)

    if not training_completed[0]:
        print(f"Model {model_name} exceeded training time ({training_threshold} seconds) or encountered an error.")
        y_pred[0] = np.nan
        training_time[0] = np.nan
    else:
        # Compute metrics
        mse = mean_squared_error(y_test, y_pred[0])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred[0])
        r_squared = r2_score(y_test, y_pred[0])

        # Calculate Adjusted R-squared if possible
        n = len(y_test)
        p = X_test.shape[1]
        if n > p + 1 and p > 0:
            adjusted_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1))
        else:
            adjusted_r_squared = r_squared  # Cannot compute Adjusted R-squared

        print(f"Model {model_name} trained successfully in {training_time[0]:.2f} seconds.")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R-squared: {r_squared}")
        print(f"Adjusted R-squared: {adjusted_r_squared}")

        results.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'Training Time (s)': training_time[0],
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r_squared,
            'Adjusted R2 Score': adjusted_r_squared
        })

# 4. Main Script
def main():
    # 4.1. Define the models with limited threads for LightGBM and RandomForest
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Elastic Net Regression': ElasticNet(),
        'LightGBM Regression': LGBMRegressor(n_jobs=1, num_threads=1),  # Limit threads
        'Random Forest Regression': RandomForestRegressor(n_jobs=1, n_estimators=100, max_depth=10),  # Limit threads and complexity
        'XGBoost Regression': xgb.XGBRegressor(n_jobs=1, nthread=1),  # Limit threads
        'Support Vector Regression': SVR(),

        
    }

    # 4.2. Training threshold
    training_threshold = 3600  # seconds

    # 4.3. List of preprocessed datasets
    datasets = ['dataset5.csv','dataset1.csv', 'dataset2.csv', 'dataset3.csv', 'dataset4.csv']

    # 4.4. Results folder
    results_dir = "model_results"
    os.makedirs(results_dir, exist_ok=True)

    # 4.5. Process each dataset
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}")

        # 4.5.1. Read dataset
        try:
            data = pd.read_csv(dataset_name)
            print(f"Successfully read {dataset_name}")
        except Exception as e:
            print(f"Error reading {dataset_name}: {e}")
            continue

        # 4.5.2. Clean column names
        data = clean_column_names(data)
        print("Column names after cleaning:", data.columns.tolist())

        # 4.5.3. Verify column names
        problematic_columns = verify_column_names(data)
        if problematic_columns:
            print(f"Problematic columns after cleaning: {problematic_columns}")
            print("Further cleaning or renaming may be required.")
            # Optionally, implement additional cleaning steps or skip problematic datasets
            # For now, we'll continue assuming cleaning was sufficient
        else:
            print("All column names are clean and compatible with LightGBM.")

        # 4.5.4. Identify numerical and categorical columns
        numerical = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical = data.select_dtypes(include=['object']).columns.tolist()

        print(f"Numerical columns: {numerical}")
        print(f"Categorical columns: {categorical}")

        # 4.5.5. Define target variable
        target_variable = 'dtbhk1'  # Change to actual target variable if different
        if target_variable not in data.columns:
            print(f"Target variable '{target_variable}' not found in dataset '{dataset_name}'. Skipping this dataset.")
            continue
        if target_variable in numerical:
            numerical.remove(target_variable)
        if target_variable in categorical:
            categorical.remove(target_variable)

        # 4.5.6. Define features and target
        X = data.drop(columns=[target_variable])
        y = data[target_variable]

        # 4.5.7. Specify the group column
        group_column = 'hocky_monhoc_count'  # Replace with your actual group column name

        if group_column not in data.columns:
            print(f"Group column '{group_column}' not found in dataset '{dataset_name}'.")
            print("Proceeding with a standard train-test split without grouping.")
            # Proceed with standard split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=True
                )
                print(f"Data split into Training ({len(X_train)} samples) and Testing ({len(X_test)} samples) sets.")
            except ValueError as e:
                print(f"Error splitting data for dataset '{dataset_name}': {e}")
                continue
        else:
            print(f"Group column '{group_column}' found. Proceeding with grouped split.")
            # Split data using split_data_by_group
            train_set, val_set, test_set = split_data_by_group(
                data, group_col=group_column,
                train_ratio=0.9, val_ratio=0.05, test_ratio=0.05
            )
            print(f"Data split into Training ({len(train_set)} samples), Validation ({len(val_set)} samples), Testing ({len(test_set)} samples) sets.")

            # Define features and targets
            X_train = train_set.drop(columns=[target_variable])
            y_train = train_set[target_variable]
            X_val = val_set.drop(columns=[target_variable])
            y_val = val_set[target_variable]
            X_test = test_set.drop(columns=[target_variable])
            y_test = test_set[target_variable]

            print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Testing set: {X_test.shape}")

        # 4.5.8. Encode categorical variables (Label Encoding for consistency across models)
        if group_column in categorical:
            categorical.remove(group_column)  # Remove group column from categorical features if present

        for col in categorical:
            if col in X_train.columns:
                # Convert to 'category' dtype
                X_train[col] = X_train[col].astype('category')
                X_test[col] = X_test[col].astype('category')
                if group_column in data.columns:
                    X_val[col] = X_val[col].astype('category')

                # Label Encoding
                X_train[col] = X_train[col].cat.codes
                X_test[col] = X_test[col].cat.codes
                if group_column in data.columns:
                    X_val[col] = X_val[col].cat.codes

                print(f"Label encoded categorical column: {col}")

        # 4.5.9. Handle missing values
        if group_column not in data.columns:
            # No grouped split, standard split
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            y_train = y_train.fillna(0)
            y_test = y_test.fillna(0)
        else:
            # Grouped split includes validation
            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)
            X_test = X_test.fillna(0)
            y_train = y_train.fillna(0)
            y_val = y_val.fillna(0)
            y_test = y_test.fillna(0)

        print("All features are now numeric and missing values are handled.")

        # 4.5.10. Prepare results storage
        results = []

        # 4.5.11. Train and evaluate each model
        for model_name, model in models.items():
            print(f"\nTraining model: {model_name}")

            train_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                results=results,
                training_threshold=training_threshold,
                dataset_name=dataset_name  # Pass dataset_name here
            )

        # 4.5.12. Save results to CSV
        results_df = pd.DataFrame(results)
        dataset_results_dir = os.path.join(results_dir, os.path.splitext(dataset_name)[0])
        os.makedirs(dataset_results_dir, exist_ok=True)
        results_file = os.path.join(dataset_results_dir, 'model_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Results for {dataset_name} saved to {results_file}")

if __name__ == "__main__":
    main()
