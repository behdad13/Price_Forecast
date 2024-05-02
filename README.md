**Time Series Forecasting in Electricity Markets**

This project provides a simple framework for time series forecasting in electricity markets.

Components:
- Data preprocessing
- Model trainer
- Model test runner
- Metrics calculator
- Visualization of forecasts


To run each component of the project, execute the following command:


1. data preprocessing: It shows you the processed dataset

`python -m data_prep <market_name>` like `python -m data_prep nyiso`

----
2. model trainer: It runs the model and save the scaler and best model. It includes the HPO and model selection.

`python -m model_trainer <market_name>` like `python -m model_trainer nyiso`

----

3. model test runner: It loads the best model and scaler and runs the forecast on the test set.

`python -m model_test <market_name>` like `python -m model_test nyiso`

----
4. metrics calculator: It calculates the metrics

`python -m metrics_calculator <market_name>` like `python -m metrics_calculator ieso`

----
5. visualization of forecasts: It visualize the forecasts and actuals

`python -m result_visualization <market_name>` like `python -m result_visualization ieso`


