import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle

from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:  
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys) 
    
def evaluate_model(x_train, y_train, x_test, y_test, models):
    try:
        train_accuracy ={}
        test_accuracy ={}
        logging.info("Evaluate model function initiated")
        for name, model in models.items():
            scores = cross_val_score(model, x_train, y_train, cv=4, scoring='r2')
            train_accuracy[name] = scores.mean()
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            test_accuracy[name]  = r2_score(y_pred,y_test)
        logging.info("model evaluation completed")
        df_result=pd.DataFrame({
            'model' : list(models.keys()),
            'Train_accuracy': [train_accuracy[name] for name in models.keys()],
            'Test_accuracy' :[test_accuracy[name] for name in models.keys()]
        })
        logging.info("Models name list created")
        df_result = df_result.sort_values(by=['Test_accuracy'],ascending=False)
        logging.info("utils process done")
        return df_result
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)