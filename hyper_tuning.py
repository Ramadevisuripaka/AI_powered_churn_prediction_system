import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys
import seaborn as sns

# modularization
import logging
from log_code import setup_logging
logger = setup_logging('hyper_tuning')
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def hyper_parameter(Training_data_scaled, Testing_data_scaled, y_train_bal, y_test):
    """
    Applies GridSearchCV Hyperparameter Tuning on Logistic Regression.
    Finds the best parameters
    """
    try:
        logger.info(f'===================  Hyperparameter Tuning  =============================')
        logger.info(f'Training data shape : {Training_data_scaled.shape}')
        logger.info(f'Testing data shape  : {Testing_data_scaled.shape}')
        logger.info(f'y_train_bal shape   : {y_train_bal.shape}')
        logger.info(f'y_test shape        : {y_test.shape}')


        logger.info(f'================  GridSearchCV ================')

        parameters = {
            'penalty'      : ['l1', 'l2', 'elasticnet', 'None'],
            'C'            : [1.0, 2.0, 3.0],
            'dual'         : [True, False],
            'class_weight' : ['balanced', None]
        }

        logger.info(f'Parameters Grid : {parameters}')

        reg = LogisticRegression()

        grid_reg = GridSearchCV(estimator=reg, param_grid=parameters, scoring='accuracy', cv=12)

        logger.info(f'Fitting GridSearchCV ...')
        grid_reg.fit(Training_data_scaled, y_train_bal)

        best_params = grid_reg.best_params_
        logger.info(f'Best Parameters found : {best_params}')

        logger.info(f'================ Retraining with Best Parameters ================')
        logger.info(f'C : {best_params["C"]} | class_weight : {best_params["class_weight"]} | dual : {best_params["dual"]} | penalty : {best_params["penalty"]}')

        best_model_tuned = LogisticRegression(
            C            = best_params['C'],
            class_weight = best_params['class_weight'],
            dual         = best_params['dual'],
            penalty      = best_params['penalty']
        )

        best_model_tuned.fit(Training_data_scaled, y_train_bal)
        logger.info(f'Model retrained with best parameters successfully')
        logger.info(f'================ Evaluation on Test Data ================')

        y_pred = best_model_tuned.predict(Testing_data_scaled)

        acc = accuracy_score(y_test, y_pred)
        cm  = confusion_matrix(y_test, y_pred)
        cr  = classification_report(y_test, y_pred)

        logger.info(f'Test Accuracy : {acc}')
        logger.info(f'Confusion Matrix : {cm}')
        logger.info(f'Classification Report : {cr}')

        return best_model_tuned, best_params

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')