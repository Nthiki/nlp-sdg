"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""
from typing import Dict
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

#twitter cleaning
import re
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


def _clean_tweet(tweet):
    '''
    tweet: String
           Input Data
    tweet: String
           Output Data
           
    func: Removes hashtag symbol in front of a word
          Replace URLs with a space in the message
          Replace ticker symbols with space. The ticker symbols are any stock symbol that starts with $.
          Replace  usernames with space. The usernames are any word that starts with @.
          Replace everything not a letter or apostrophe with space
          Remove single letter words
          filter all the non-alphabetic words, then join them again

    '''
    
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub('\$[a-zA-Z0-9]*', ' ', tweet)
    tweet = re.sub('\@[a-zA-Z0-9]*', ' ', tweet)
    tweet = re.sub('[^a-zA-Z\']', ' ', tweet)
    tweet = re.sub(r'\s+', " ", tweet)
    tweet = ' '.join( [w for w in tweet.split() if len(w)>1] )
    
    return tweet


def _lemmatize(text: str) -> str:
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(text)



def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Function takes in the whole dataframe and carries out the following preprocessing steps:
    
    1. Removes strings etc
    2. Lemmatization and removing stopwords
    
    Then return the dataframe with an added column that has the cleaned version of the text
    '''
    data['clean_text'] = data['Text'].apply(lambda x:_clean_tweet(x))
    data['clean_text'] = data['clean_text'].apply(_lemmatize)
    #print(data['clean_text'][0])
    
    return data





''' ================================== 
 Data engineering functions for Team C
 ==================================== '''