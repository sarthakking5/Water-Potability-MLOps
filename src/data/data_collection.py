import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

def load_params(filepath:str):
    try:
        with open(filepath,"r") as file:
            params=yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}:{e}")

def load_data(filepath:str):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error laoding parameters from {filepath}:{e}")

def split_data(data:pd.DataFrame,test_size:float):
    try:
        return train_test_split(data,test_size=test_size,random_state=42)
    except Exception as e:
        raise Exception(f"error spliting data:{e}")

def save_data(df:pd.DataFrame,filepath:str):
    try:
        df.to_csv(filepath,index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}:{e}")

def main():
    data_filepath=r"D:\projects\mlops_project\mlops_project\water_potability.csv"
    params_filepath="params.yaml"
    raw_data_path=os.path.join("data","raw")
    try:
        data=load_data(data_filepath)
        test_size=load_params(params_filepath)
        train_data,test_data=split_data(data,test_size)

        os.mkdir(raw_data_path)

        save_data(train_data,os.path.join(raw_data_path,"train.csv"))
        save_data(test_data,os.path.join(raw_data_path,"test.csv"))

    except Exception as e:
        raise Exception(f"An error occurred: {e}")

if __name__=="__main__":
    main()
