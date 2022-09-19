"""
This is a boilerplate pipeline 'data_processing_classification'
generated using Kedro 0.18.2
"""
import pandas as pd
import numpy as np
import texthero as hero

# nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Primary Functions
def clean_agreement(data: pd.DataFrame) -> pd.DataFrame:
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

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocess data.
    
     Args:
        data: Full (all columns) training cleaned data according to agreement score
        
     Returns:
        Processed text data as pandas dataframe.
        
    '''
    #apply text hero
    data['text'] = hero.clean(data['text'])
    #lemmatize the text
    data['text'] = data['text'].apply(_lemmatize)


    return data


    #Secondary functions

def _lemmatize(text: str) -> str:
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(text)



