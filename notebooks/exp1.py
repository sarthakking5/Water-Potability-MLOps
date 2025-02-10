import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import dagshub

#Initialize Dagshub and setup MLflow experiment tracking
dagshub.init(repo_owner='sarthakking5', repo_name='Water-Potability-MLOps', mlflow=True)
mlflow.set_experiment("Experiment1")
mlflow.set_tracking_uri("https://dagshub.com/sarthakking5/Water-Potability-MLOps.mlflow")

data=pd.read_csv(r"D:\projects\mlops_project\mlops_project\data\raw\water_potability.csv")

#split the dataset into training and testing
from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(data,test_size=0.20,random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value=df[column].median()
            df[column].fillna(median_value,inplace=True)
    return df

train_processed_data=fill_missing_with_median(train_data)
test_processed_data=fill_missing_with_median(test_data)

from sklearn.ensemble import RandomForestClassifier
import pickle

X_train=train_processed_data.drop(columns=["Potability"],axis=1)
y_train=train_processed_data["Potability"]

n_estimators=10

with mlflow.start_run():
    clf=RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train,y_train)

    pickle.dump(clf,open("model.pkl","wb"))
    X_test=test_processed_data.iloc[:,0:-1].values
    y_test=test_processed_data.iloc[:,-1].values

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    model=pickle.load(open("model.pkl","rb"))
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1_score=f1_score(y_test,y_pred)

    mlflow.log_metric("accuracy",acc)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_score",f1_score)

    mlflow.log_param("n_estimators",n_estimators)

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True)
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.jpg")

    mlflow.log_artifact("confusion_matrix.jpg")
    mlflow.sklearn.log_model(clf,"RandomForestClassifier")

    mlflow.log_artifact(__file__)
    mlflow.set_tag("author","skinger2")
    mlflow.set_tag("model","GB")

    print("Accuracy:",acc)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1_score:",f1_score)




