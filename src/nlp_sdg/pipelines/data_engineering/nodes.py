"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""
#loading necessary libraries
from typing import Dict
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

def _clean_agreement(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes in a dataframe and filters out the rows with negative label higher than positive label,
    and have an agreement score less than 0.4
    
    Args: 
        Source training data
        
    Returns:
        Filtered data that has high positive community agreement with SDG labels
        
        
    '''
    data = data.loc[(data['labels_negative'] < data['labels_positive']) & (data['agreement'] >= 0.4)]
    return data


def _missing_data(df: pd.DataFrame) ->pd.DataFrame:
    """
    This function takes in a dataframe and filters out the rows with missing data
    
    Args: 
        Source training data
        
    Returns:
        Data without missing cells

    """
    df1 = df.dropna(subset=['text'])
    df2 = df1.dropna(subset=['sdg'])
    return df2


def _clean_article(text: str) -> str:
    
    """Converts apostrophe suffixes to words, replace webpage links with url, annotate 
    hashtags and mentions, remove a selection of punctuation, and convert all words to lower case.
    Args:
        text: 'text' article or sentence to be cleaned
    Returns:
        clean_text: clean text rid of noise 
    """
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|\
                      (?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = hero.clean(pd.Series(text))[0]  
    return text


def _lemmatize(text: str) -> str:
    """This function is responsible for Lemmatization of the text.
    Args:
        text (String): sentence containing 'text' to lemmatize (stemming)
    Returns:
        text (String): sentence with converted 'text' into lemmatized word(s)  
    """
    # removing any form of hyper link(s)
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words and len(word)>2]
    return ' '.join(text)


# Creating a function to upscale and balance the dataset
def _data_balancing(df_input: pd.DataFrame) -> pd.DataFrame:
    """
        Dataset balancing for all target variable to be equal in frequency.
    Args:
        `df` (DataFrame): pd.Series containing the target variable
    Return:
        df (DataFrame): dataframe with resample and balance dataset by upscaling.
    """
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|\
                      (?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = hero.clean(pd.Series(text))[0]  
    return text


def _lemmatize(text: str) -> str:
    """This function is responsible for Lemmatization of the text.
    Args:
        data: Data containing features and target.
    Returns:
        data.
    """


    return data