"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""
from typing import Dict
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

#NLTK library
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

''' ================================== 

 Dummy code

 ==================================== '''


def dummy_node(data: DataFrame) -> DataFrame:
    
    """Dummy node to read data

    Args:
        data: Data containing features and target.
    Returns:
        data.
    """


    return data

''' ================================== 

 Data engineering functions for Team A

 ==================================== '''





''' ================================== 

 Data engineering functions for Team B

 ==================================== '''




''' ================================== 

 Data engineering functions for Team C

 ==================================== '''



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

    return data

# secondary functions that help clean the SDG text data

def _lemmatize(text: str) -> str:
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(text)

#primary function

def preprocess_sdg_data(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocess data.
    
     Args:
        data: Full (all columns) training cleaned data according to agreement score
        
     Returns:
        Processed text data as pandas dataframe.
        
    '''
    
    #lemmatize the text
    data['text'] = data['text'].apply(_lemmatize)

    return data

    

''' ================================== 

                 The end 

 ==================================== '''





