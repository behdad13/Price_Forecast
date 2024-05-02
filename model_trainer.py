import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn.pipeline import Pipeline
import joblib
import os
from data_prep import DataProcessing


# Processing the training and testing data
def PrepareScaleData(data_processing_function, market_name, n_steps):
    """
    Loads, prepares, and scales the dataset for model training and testing.

    Parameters:
    - data_processing_function: Function to process data.
    - market_name: Market name for data processing.
    - n_steps: Number of previous price lags considered in the dataset.

    Returns:
    - X_train: Training features.
    - X_test: Test features.
    - y_train: Training target variable.
    - y_test: Test target variable.
    - scaler_X: Scaler used for the feature variables.
    - scaler_y: Scaler used for the target variable.
    """
    # Load the dataset
    data = data_processing_function(market_name, n_steps)

    # Separate the last 10% of data as the test set
    train_data = data[:int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]

    # Separate features and target
    X_train = train_data.drop(columns=['P(t)'])
    y_train = train_data['P(t)']
    X_test = test_data.drop(columns=['P(t)'])
    y_test = test_data['P(t)']

    # Normalize the features (training and test)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Normalize the target variable (training and test)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test, scaler_X, scaler_y


# model trainer
def FindBestModel(X_train, y_train):
    """
    Conducts a grid search to find the best regression model and its parameters.

    Parameters:
    - X_train: Training feature data.
    - y_train: Training target data.

    Returns:
    - best_model: The best model found during the grid search.
    """
    # Define the models, including RF, XGB, and Lasso, and Ridge
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    rf_model = RandomForestRegressor()

    # Create a pipeline for the GridSearchCV
    pipeline = Pipeline([('model', xgb_model)])

    # Create a space
    param_grid = [{
        'model': [xgb_model],
        'model__max_depth': [3, 4, 5],
        'model__learning_rate': [0.1, 0.05],
        'model__n_estimators': [100, 200, 300],
        'model__subsample': [0.8, 0.9, 1.0]
    }, {
        'model': [rf_model],
        'model__max_depth': [3, None],
        'model__n_estimators': [100, 200, 300]
    }]

    # Grid search with 5-fold time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, verbose=1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Best model found by grid search
    best_model = grid_search.best_estimator_
    print("Best model:", best_model)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score (neg_mean_squared_error):", grid_search.best_score_)
    
    return best_model


# Saving the model artifacts
def SaveModelArtifacts(model, scaler_X, scaler_y, market_name, directory='model_artifacts'):
    """
    Saves the model and scalers to the specified directory.

    Parameters:
    - model: Trained model to be saved.
    - scaler_X: Scaler used for the feature variables.
    - scaler_y: Scaler used for the target variable.
    - market_name: The name of the market to be saved.
    - directory: Directory path where the artifacts will be saved.
    """
    os.makedirs(directory, exist_ok=True)
    
    # Paths for the artifacts
    model_path = os.path.join(directory, f'best_model_{market_name}.pkl')
    scaler_X_path = os.path.join(directory, f'scaler_X_{market_name}.pkl')
    scaler_y_path = os.path.join(directory, f'scaler_y_{market_name}.pkl')
    
    # Save the artifacts
    joblib.dump(model, model_path)
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    
    print(f"Model for {market_name} saved to {model_path}")
    print(f"Scaler for X and {market_name} saved to {scaler_X_path}")
    print(f"Scaler for Y and {market_name} saved to {scaler_y_path}")


def main(market_name):
    n_step = 12
    X_train, _, y_train, _, scaler_X, scaler_y = PrepareScaleData(DataProcessing, market_name, n_step)
    
    # Find the best model (model set: XGBoost, RF, Linear Reg(Ridge), Linear Reg(Lasso))
    best_model = FindBestModel(X_train, y_train)

    # Save the model and scaler
    SaveModelArtifacts(best_model, scaler_X, scaler_y, market_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a machine learning model with data from a specific market.')
    parser.add_argument('market_name', type=str, help='The name of the market for which the data is prepared and tested')
    args = parser.parse_args()

    main(args.market_name)

# python -m model_trainer ieso