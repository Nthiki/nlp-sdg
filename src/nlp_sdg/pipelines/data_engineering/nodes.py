"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""
from typing import Dict
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

def dummy_node(data: DataFrame) -> DataFrame:
    """Dummy node to read data

    Args:
        data: Data containing features and target.
    Returns:
        data.
    """


    return data

def clean_agreement(data:pd.DataFrame) -> DataFrame:
    '''
    This function takes in a dataframe and keeps rows
    with positive labels more than negative labels
    and has a high agreement score (>0.4)
    
    Args:
        source training data
        
    Returns:
        Filtered out data that has positive community agreement 
        with SDG labels
    '''

    data = data.loc[(data['labels_negative'] < data['labels_positive']) & (data['agreement'] >= 0.4)]
    print("success")

    return data

def clean_text(data:pd.DataFrame) -> DataFrame:
    
    """Converts apostrophe suffixes to words, replace webpage links with url, 
    annotate hashtags and mentions, remove a selection of punctuation, and convert all words to lower case.
    Args:
        data (DataFrame): dataframe containing 'text' column to convert
    Returns:
        data (DataFrame): dataframe with converted 'text' column 
    """
    def word_lemma(words):
        lemmatizer = WordNetLemmatizer()
        lemma = [lemmatizer.lemmatize(word) for word in words]
        return ''.join([l for l in lemma])
    import string
    def remove_extras(post):
        punc_numbers = string.punctuation + '0123456789'
        return ''.join([l for l in post if l not in punc_numbers])
    
    def sdglabler(df: pd.DataFrame):
        sdgLables = {1: "No poverty", 2: "Zero Hunger", 3: "Good Health and well-being", 4: "Quality Education", 5: "Gender equality", 6: "Clean water and sanitation", 7: "Affordable and clean energy", 9: "Industry, Innovation and Infrustructure", 8: "Decent work and economic growth",
                    10: "Reduced Inequality", 13: "Climate Action", 11: "Sustainable cites and communities", 12: "Responsible consumption and production", 14: "life below water", 15: "Life on land", 16: "Peace , Justice and strong institutions", 17: "Partnership for the goals"}
        df['SDG_Labels'] = df['sdg'].map(sdgLables)

    # drop empty rows
    data = data.dropna(subset=['text'])
    data = data.dropna(subset=['sdg'])
    #Lower case
    data["text"] = data["text"].str.lower()
    #Removal of Punctuation
    data["text"] = data['text'].apply(remove_extras)
    data["text"] = data['text'].apply(word_lemma)    
    # Map the target variable name to their code for better understanding
    sdglabler(data)

    return data

def data_balancing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset balancing for all target variable to be equal in frequency.
    Args:
        data (DataFrame): pd.Series containing the target variable
    Return:
        data (DataFrame): dataframe with resample and balance dataset by upscaling.
    """
    df = data.copy()
    # setting the maximum resample size to mean of categories in 'sdg'
    class_size = int(df.sdg.value_counts().mean()) 

    target_size = df.sdg.value_counts() # getting category name and their size
    appended_target = [] # creating an empty list to append all category after resampling

    # Creating a for-loop to resample and append to a list
    for index, size in target_size.items():
        if size < class_size: # setting condition to check if to downsample or upsample
            temp_pd = resample(df[df['sdg']==index],
                              replace=True, # sample with replacement
                              n_samples=class_size, # match number in resample class
                              random_state=27)
        else:
            temp_pd = resample(df[df['sdg']==index],
                              replace=False, # sample without replacement (no need to duplicate observations)
                              n_samples=class_size, # match number in resample class
                              random_state=27)
        # Appending each category after resampling
        appended_target.append(temp_pd)
        
    # Creating a new dataframe and viewing
    df_resampled = pd.concat(appended_target, axis=0)
    
    return df_resampled