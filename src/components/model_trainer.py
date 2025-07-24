import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("Control is on initiate_model_trainner")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models= {
                'svr' : SVR(),
                'dt'  : DecisionTreeRegressor(),
                'rf'  : RandomForestRegressor(),
                'xgb' : XGBRegressor(),
                'lg'  : LinearRegression()
                    }
            logging.info("Model initialized")
            model_report:dict=evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test =y_test, models=models)
            logging.info("Model_report collected")
            best_model_row = model_report.loc[model_report['Test_accuracy'].idxmax()]
            print(best_model_row)
            logging.info(f"score {best_model_row}")

            model_name = best_model_row['model']  # This will be 'lg'
            best_model = models[model_name]       # models['lg'] → LinearRegression()


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
       
        except Exception as e:
            raise CustomException(e, sys)
