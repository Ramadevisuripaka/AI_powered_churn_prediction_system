import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys
import seaborn as sns
#modularization
import logging
from log_code import setup_logging
logger = setup_logging('feature_scaling')

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def feature_scaling(Training_data, Testing_data):
    """
    Applies feature scaling (StandardScaler) on Training and Testing data.Training_data (pd.DataFrame): Balanced training data after cat_to_num.Testing_data  (pd.DataFrame): Testing data after cat_to_num.

    """
    try:
        logger.info(f'===================  Feature Scaling  =============================')
        logger.info(f'Before applying Feature Scaling Train columns and shapes \n : {Training_data.columns} : {Training_data.shape}')
        logger.info(f'Before applying Feature Scaling Test columns and shapes \n : {Testing_data.columns} : {Testing_data.shape}')

        # ---- Standard Scaler ----
        logger.info(f'================ Applying StandardScaler ================')

        ss = StandardScaler()
        ss.fit(Training_data)  # learn mean and std from training data



        logger.info(f'After applying Feature Scaling Train columns and shapes \n : {Training_data_scaled.columns} : {Training_data_scaled.shape}')
        logger.info(f'After applying Feature Scaling Test columns and shapes \n : {Testing_data_scaled.columns} : {Testing_data_scaled.shape}')

        logger.info(f'Training data sample after scaling : \n {Training_data_scaled.head()}')
        logger.info(f'Testing data sample after scaling : \n {Testing_data_scaled.head()}')

        return Training_data_scaled, Testing_data_scaled

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')