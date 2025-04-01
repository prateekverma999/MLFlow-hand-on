import mlflow

mlflow.set_experiment("first_experiment")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.08)
    mlflow.log_artifact("mlflow-basic.py")
    print("Experiment logged!")
