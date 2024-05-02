import pandas as pd
import plotly.graph_objects as go
import argparse


def VisualizePredictions(data, market_name):
    """
    Visualizes predicted and actual values over time using Plotly.

    Parameters:
    - data (pd.DataFrame): A DataFrame with columns 'Timestamp', 'y_pred', and 'y_test'.

    Returns:
    - Plotly graph object displaying the predictions and actual values.
    """
    
    trace1 = go.Scatter(
        x=data['Timestamp'],
        y=data['y_pred'],
        mode='lines',
        name='Predicted'
    )
    
    trace2 = go.Scatter(
        x=data['Timestamp'],
        y=data['y_test'],
        mode='lines',
        name='Actual'
    )
    
    layout = go.Layout(
        title=f'Predicted vs Actual Values Over Time in {market_name}',
        xaxis_title='Timestamp',
        yaxis_title='Values',
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type='date'
        )
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.show()


# For testing the code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process market data to calculate RMSE and MAE.')
    parser.add_argument('market_name', type=str, help='The name of the market to process')
    args = parser.parse_args()

    df = pd.read_csv(f'results/predictions_{args.market_name}.csv')
    VisualizePredictions(df, args.market_name)

# python -m result_visualization ieso