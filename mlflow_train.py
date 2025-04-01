import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Enable Auto-logging
mlflow.sklearn.autolog()

# Load Sample Dataset (Boston Housing dataset alternative)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["target"]), df["target"], test_size=0.2, random_state=42)

# Start MLflow Experiment
mlflow.set_experiment("mlflow_model_tracking")


# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate Metrics
mse = mean_squared_error(y_test, y_pred)



with mlflow.start_run():
    # Log Metrics
    mlflow.log_metric("mse", mse)

    # Log Model
        # mlflow.sklearn.log_model(model, "linear_regression_model")
    model_uri = "linear_regression_model"
    mlflow.sklearn.log_model(model, model_uri)

    # Register Model
    model_name = "HousePricePredictor"
    model_version = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{model_uri}", model_name)

    print(f"Model Trained & Logged! MSE: {mse}")
    print("-----------------------------------------------------------------------")
    print(f"Model registered as '{model_name}', Version: {model_version.version}")

