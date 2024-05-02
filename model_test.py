import joblib
import pandas as pd
import os
import argparse
from model_trainer import PrepareScaleData
from data_prep import DataProcessing

def LoadModel(model_path, scaler_y_path):
    """
    Loads a saved model and scaler.

    Parameters:
    - model_path: Path to the saved model file.
    - scaler_y_path: Path to the saved scaler file for the target variable.

    Returns:
    - best_model: The loaded best model
    - scaler_y: The loaded scaler for the target variable
    """
    # Load the model and scaler
    best_model = joblib.load(model_path)
    scaler_y = joblib.load(scaler_y_path)

    return best_model, scaler_y


def ModelTest(best_model, scaler_y, X_test, y_test):
    """
    Makes predictions on the test set and returns a DataFrame of predictions and true values.

    Parameters:
    - best_model: Loaded best model
    - scaler_y: Scaler_y to descale the results
    - X_test: Test feature data.
    - y_test: Test target data.

    Returns:
    - results_df: DataFrame containing the predictions and actual values.
    """

    # Do the prediction on the test set
    y_pred = best_model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Create DataFrame of predictions and actuals
    results_df = pd.DataFrame({
        'Timestamp': y_test.index,
        'y_pred': y_pred,
        'y_test': y_test.values
    })

    return results_df


def main(market_name):
    # Prepare and scale data
    _, X_test, _, y_test, _, _ = PrepareScaleData(DataProcessing, market_name, 12)

    # Load model and scaler
    model_path = f'model_artifacts/best_model_{market_name}.pkl'
    scaler_y_path = f'model_artifacts/scaler_y_{market_name}.pkl'
    best_model, scaler_y = LoadModel(model_path, scaler_y_path)

    # Test model and save results
    results_df = ModelTest(best_model, scaler_y, X_test, y_test)
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, f'predictions_{market_name}.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and test a machine learning model with data from a specific market.')
    parser.add_argument('market_name', type=str, help='The name of the market for which the data is prepared and tested')
    args = parser.parse_args()

    main(args.market_name)

# python -m model_test ieso