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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from mlflow.models import infer_signature

#Initialize Dagshub and setup MLflow experiment tracking
dagshub.init(repo_owner='sarthakking5', repo_name='Water-Potability-MLOps', mlflow=True)
mlflow.set_experiment("Experiment 3")
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

models={
    "Logistic Regression":LogisticRegression(class_weight='balanced',max_iter=100),
    "Random Forest":RandomForestClassifier(),
    "Support Vector Classifier":SVC(),
    "Decision Tree":DecisionTreeClassifier(),
    "KNN":KNeighborsClassifier(),
    "XGBoost":XGBClassifier()
}


with mlflow.start_run(run_name="Water Potability Models Experiment"):
    for model_name,model in models.items():
        with mlflow.start_run(run_name=model_name,nested=True):
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            acc=accuracy_score(y_test,y_pred)
            precision=precision_score(y_test,y_pred)
            recall=recall_score(y_test,y_pred)
            f1=f1_score(y_test,y_pred)
            model_filename=(f"{model_name.replace(' ','_')}.pkl")
            pickle.dump(model,open(model_filename,"wb"))
            y_pred=model.predict(X_test)
            
            mlflow.log_metric("accuracy",acc)
            mlflow.log_metric("precision",precision)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("f1_score",f1)

            cm=confusion_matrix(y_test,y_pred)
            plt.figure(figsize=(5,5))
            sns.heatmap(cm,annot=True)
            plt.xlabel("Prediction")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.savefig(f"confusion_matrix_{model_name.replace(' ','_')}.jpg")
            signature=infer_signature(X_test,model.predict(X_test))
            mlflow.log_artifact(f"confusion_matrix_{model_name.replace(' ','_')}.jpg")
            mlflow.sklearn.log_model(model,model_name.replace(' ','_'),signature=signature)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("author","skinger2")
            

    print("All models have been trained and logged successfully")

