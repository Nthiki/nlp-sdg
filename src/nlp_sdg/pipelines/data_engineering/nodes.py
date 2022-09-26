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