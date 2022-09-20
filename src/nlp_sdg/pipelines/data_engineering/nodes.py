"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""
from typing import Dict
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame


def dummy_node(data: DataFrame) -> DataFrame:
    
    """Dummy node to read data

    Args:
        data: Data containing features and target.
    Returns:
        data.
    """


    return data



def clean_agreement(data: pd.DataFrame) -> pd.DataFrame:

    '''
    This function takes in a dataframe and keeps rows with pos labels more than neg labels, and
    has high agreement score (0.4)

    Args:
        source training data
    
    Returns:
        Filtered out data that has high positive community agreement with SDG labels

    '''

    data = data.loc[(data['labels_negative'] < data['labels_positive']) & (data['agreement'] >= 0.4)]
    print("success")

    return data


