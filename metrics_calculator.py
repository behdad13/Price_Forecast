from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import argparse


def ReadResultData(market_name):
    df = pd.read_csv(f'results/predictions_{market_name}.csv')
    return df


def CalculateRmseMae(results_df):
    """
    Calculates the Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) from the results DataFrame.

    Parameters:
    - results_df: DataFrame with columns 'y_pred' for predictions and 'y_test' for actual values.

    Returns:
    - rmse: Root Mean Squared Error of the predictions.
    - mae: Mean Absolute Error of the predictions.
    """
    y_pred = results_df['y_pred']
    y_test = results_df['y_test']

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test MSE: {mse}")
    print(f"Test RMSE: {rmse}")
    print(f"Test MAE: {mae}")

    return rmse, mae


def main(market_name):
    forecast_df = ReadResultData(market_name)
    rmse, mae = CalculateRmseMae(forecast_df)
    print(f"RMSE: {rmse}, MAE: {mae}")


# For testing the code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process market data to calculate RMSE and MAE.')
    parser.add_argument('market_name', type=str, help='The name of the market to process')
    args = parser.parse_args()

    main(args.market_name)


# python -m metrics_calculator ieso