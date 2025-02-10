import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import dagshub
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from mlflow.models import infer_signature

#Initialize Dagshub and setup MLflow experiment tracking
dagshub.init(repo_owner='sarthakking5', repo_name='Water-Potability-MLOps', mlflow=True)
mlflow.set_experiment("Experiment 4")
mlflow.set_tracking_uri("https://dagshub.com/sarthakking5/Water-Potability-MLOps.mlflow")

data=pd.read_csv(r"D:\projects\mlops_project\mlops_project\data\raw\water_potability.csv")

#split the dataset into training and testing

def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            mean_value=df[column].mean()
            df[column].fillna(mean_value,inplace=True)
    return df

processed_data=fill_missing_with_mean(data)
X=processed_data.drop(columns=["Potability"],axis=1)
y=processed_data['Potability']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

rf=RandomForestClassifier(random_state=42)
param_dist={
    "n_estimators":[100,200,300],
    "max_depth":[None,4,5,6,10]
}

random_search=RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

with mlflow.start_run(run_name="Random Forest Tuning") as parent_run:
    random_search.fit(X_train,y_train)

    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination{i+1}",nested=True) as child_run:
            mlflow.log_params(random_search.cv_results_['params'][i])
            mlflow.log_metric("mean_test_score",random_search.cv_results_['mean_test_score'][i])

    print("Best parameters found:",random_search.best_params_)

    mlflow.log_params(random_search.best_params_)

    best_rf=random_search.best_estimator_
    best_rf.fit(X_train,y_train)

    y_pred=best_rf.predict(X_test)

    acc=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)

    metrics={
        "Accuracy":acc,
        "Precision":precision,
        "F1_score":f1,
        "Recall":recall,
            }
    mlflow.log_metrics(metrics)

    final_data=mlflow.data.from_pandas(processed_data)

    mlflow.log_input(final_data,"processed")

    mlflow.log_artifact(__file__)

    sign=infer_signature(X_test,random_search.best_estimator_.predict(X_test))

    mlflow.sklearn.log_model(random_search.best_estimator_,"Best Model",signature=sign)

    print("Accuracy:",acc)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1_score",f1)