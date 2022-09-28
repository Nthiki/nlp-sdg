"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""
from typing import Dict
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
import sqlite3


def dummy_node(data: DataFrame) -> DataFrame:
    """Dummy node to read data

    Args:
        data: Data containing features and target.
    Returns:
        data.
    """

    return data


def convert_to_csv(data : DataFrame) -> DataFrame:
    connection = sqlite3.connect(data)
    cursor = connection.cursor()
    
    # Execute the query
    cursor.execute('select * from mydata')
    # Get Header Names (without tuples)
    colnames = [desc[0] for desc in cursor.description]
    # Get data in batches
    while True:
        # Read the data
        df = pd.DataFrame(cursor.fetchall())
        # We are done if there are no data
        if len(df) == 0:
            break
        # Let us write to the file
        else:
            df.to_csv(f, header=colnames)

    cursor.close()
    connection.close()
    
    return df