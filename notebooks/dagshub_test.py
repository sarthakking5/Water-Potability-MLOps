import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/sarthakking5/Water-Potability-MLOps.mlflow")

dagshub.init(repo_owner='sarthakking5', repo_name='Water-Potability-MLOps', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)