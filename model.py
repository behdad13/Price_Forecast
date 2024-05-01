import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
from data_prep import DataProcessing

# Load your dataset
n_steps = 12
market_name = 'ieso'
data = DataProcessing(market_name, n_steps)

# Separate the last 10% of data as the test set
train_validation_data = data[:int(len(data) * 0.9)]
test_data = data[int(len(data) * 0.9):]

# Shuffle the 90% training and validation data
train_validation_data = train_validation_data.sample(frac=1).reset_index(drop=True)

# Split the remaining data into training (70% of 90%) and validation (20% of 90%)
X_train, X_val, y_train, y_val = train_test_split(
    train_validation_data.drop(columns=['P(t)']),
    train_validation_data['P(t)'],
    test_size=0.222,  # This is approximately 20% of the 90%
    random_state=42
)

# Prepare the test set
X_test = test_data.drop(columns=['P(t)'])
y_test = test_data['P(t)']

# Normalize the features (training, validation, test)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

# Normalize the target variable (training, validation, test)
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

# Define the models
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
rf_model = RandomForestRegressor()
ridge_model = Ridge()
lasso_model = Lasso()

# Create a pipeline with a placeholder for the estimator
pipeline = Pipeline([('model', xgb_model)])  # Start with any model

# Parameter grid with conditional parameters
param_grid = [{
    'model': [xgb_model],
    'model__max_depth': [3],
    'model__learning_rate': [0.1],
    'model__n_estimators': [100, 200, 300],
    'model__subsample': [0.8, 0.9, 1.0]
}, {
    'model': [rf_model],
    'model__max_depth': [3, None],
    'model__n_estimators': [100, 200, 300]
}, {
    'model': [ridge_model],
    'model__alpha': [0.1, 1, 10]  # Regularization strength
}, {
    'model': [lasso_model],
    'model__alpha': [0.1, 1, 10]  # Regularization strength
}]

# Grid search with 3-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model found by grid search
best_model = grid_search.best_estimator_

# Print best model and its parameters
best_model = grid_search.best_estimator_
print("Best model:", best_model)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score (neg_mean_squared_error):", grid_search.best_score_)

# Validation errors for each parameter combination
results = grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(params, 'has a score of:', mean_score)


# Predictions and evaluation on the test set
y_pred = best_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()  # Inverse transform predictions

mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse}')

# Save predictions and true values into a CSV
results_df = pd.DataFrame({
    'Timestamp': test_data.index,
    'y_pred': y_pred,
    'y_test': y_test.values
})
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
results_df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)

# Save the model and scaler
model_artifact_dir = 'model_artifacts'
os.makedirs(model_artifact_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(model_artifact_dir, 'best_model.pkl'))
joblib.dump(scaler_X, os.path.join(model_artifact_dir, 'scaler_X.pkl'))
joblib.dump(scaler_y, os.path.join(model_artifact_dir, 'scaler_y.pkl'))