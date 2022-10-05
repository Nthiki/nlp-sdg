"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""
from typing import Dict
import numpy as np
import pandas as pd
import re
from pyspark.sql import DataFrame
import texthero as hero
import nltk
from nltk.corpus import stopwords
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

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
    df = df_input.copy()
    # setting the maximum size for each category in 'sdg'
    class_size = int(df.sdg.value_counts().max()) 

    target_size = df.sdg.value_counts() # getting category name and their size
    appended_target = [] # creating an empty list to append all category after resampling

    # Creating a for-loop to resample and append to a list
    for index, size in target_size.items():
        if size < class_size: # setting condition to check if to downsample or upsample
            temp_pd = resample(df[df['sdg']==index],
                              replace=True, # sample with replacement
                              n_samples=class_size, # match number in majority class
                              random_state=27)
        else:
            temp_pd = resample(df[df['sdg']==index],
                              replace=False, # sample with replacement (no need to duplicate observations)
                              n_samples=class_size, # match number in minority class
                              random_state=27)
    # Appending each category after resampling
        appended_target.append(temp_pd)
        
    # Creating a new dataframe and viewing
    df_resampled = pd.concat(appended_target, axis=0)
    
    return df_resampled


def osdg_preprocessed_data(data: pd.DataFrame) -> pd.DataFrame:
    '''
     This function applies all the above functions to the dataframe to make it preprocessed data.
    
     Args:
        data: Full (all columns) training cleaned data according to agreement score
        
     Returns:
        Processed text data as pandas dataframe
        
    '''
    # removing missing rows in the important columns
    data = _missing_data(data)
    #apply text hero
    data['text'] = data['text'].apply(_clean_article)
    #lemmatize the text
    data['text'] = data['text'].apply(_lemmatize)
    #clean_agreement
    data = _clean_agreement(data)
    # Resampling the dataset to balance each target
    #data = _data_balancing(data)

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
