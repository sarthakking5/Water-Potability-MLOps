import numpy as np
import pandas as pd
import pickle
import json
import mlflow
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from mlflow import log_metrics,log_params,log_artifact
import mlflow.sklearn
import dagshub
from mlflow.models import infer_signature

dagshub.init(repo_owner='sarthakking5', repo_name='Water-Potability-MLOps', mlflow=True)
mlflow.set_experiment("DVC Pipeline")
mlflow.set_tracking_uri("https://dagshub.com/sarthakking5/Water-Potability-MLOps.mlflow")

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")

def evaluation_model(model,X_test:pd.DataFrame,y_test:pd.Series,model_name:str):
    try:
        params=yaml.safe_load(open("params.yaml","r"))
        test_size=params['data_collection']["test_size"]
        n_estimators=params["model_building"]["n_estimators"]

        y_pred=model.predict(X_test)

        acc=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred)

        metrics={
        "Accuracy":acc,
        "Precision":precision,
        "F1_score":f1,
        "Recall":recall,
            }
        mlflow.log_metrics(metrics)

        mlflow.log_param("Test_size",test_size)
        mlflow.log_param("n_estimators",n_estimators)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        
        # Log confusion matrix artifact
        mlflow.log_artifact(cm_path)

        return metrics
    except Exception as e:
        raise Exception(f"Error evaluation model:{e}")

def save_metrics(metrics:dict,metrics_path:str):
    try:
        with open(metrics_path,"w") as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}:{e}")

def main():
    try:
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"
        model_name = "Best Model"

        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        
        with mlflow.start_run() as run:
            metrics=evaluation_model(model,X_test,y_test,model_name)
            save_metrics(metrics,metrics_path)

            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)

            mlflow.log_artifact(__file__)

            signature=infer_signature(X_test, model.predict(X_test))

            mlflow.sklearn.log_model(model,"Best Model", signature=signature)

            run_info={'run_id':run.info.run_id,'model_name':"Best Model"}

            reports_path="reports/run_info.json"

            with open(reports_path,"w") as file:
                json.dump(run_info,file,indent=4)
    
    except Exception as e:
        raise Exception (f"An error occurred: {e}")

if __name__=="__main__":
    main()

