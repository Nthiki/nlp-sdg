"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""
#from typing import Dict
#import numpy as np
from sqlite3 import Timestamp
import pandas as pd
import re
#import os
#from pyspark.sql import DataFrame
import texthero as hero
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
#from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
from datetime import date,timedelta,datetime
import snscrape.modules.twitter as sntwitter
import time 



''' ================================== 
 Data engineering functions for Team C
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
    #df2 = df1.dropna(subset=['sdg'])
    return df1


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

def article_preprocessed_data(data: pd.DataFrame) -> pd.DataFrame:
    '''
     This function applies all the above functions to the dataframe to make it preprocessed data.
    
     Args:
        data: Full (all columns) training cleaned data according to agreement score
        
     Returns:
        Processed text data as pandas dataframe
        
    '''
    # removing missing rows in the important columns
    #data = _missing_data(data)
    #apply text hero
    data['text'] = data['text'].apply(_clean_article)
    #lemmatize the text
    data['text'] = data['text'].apply(_lemmatize)
    #clean_agreement
    # Resampling the dataset to balance each target
    #data = _data_balancing(data)

    return data


''' ================================== 
 Data engineering functions for Team A
 ==================================== '''
'''
TK: As it stands, I don't see the need for this code, we can simply read from the data nodes itself and start using the data.
'''
# def convert_to_csv(data : DataFrame) -> DataFrame:
#     connection = sqlite3.connect(data)
#     cursor = connection.cursor()

#     # Execute the query
#     cursor.execute('select * from mydata')
#     # Get Header Names (without tuples)
#     colnames = [desc[0] for desc in cursor.description]
#     # Get data in batches
#     while True:
#         # Read the data
#         df = pd.DataFrame(cursor.fetchall())
#         # We are done if there are no data
#         if len(df) == 0:
#             break
#         # Let us write to the file
#         else:
#             df.to_csv(f, header=colnames)

#     cursor.close()
#     connection.close()

#     return df



''' ================================== 
 Data engineering functions for Team B
 ==================================== '''

#initial data dump to the rds
def fetch_all_tweets():
    ''' Fetch all tweets from 2017 september based on the keywords provided, it returns a dataframe'''
    tweets_list2 = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'(shell affordable energy) OR (shell clean energy) OR (shell affordable clean energy)  OR (shell decent work) OR (shell economic growth) OR (shell decent work economic growth) OR (shell industry innovation infrastructure) OR (shell innovation) OR (shell industry innovation) OR (shell responsible consumption) OR (shell responsible production) OR (shell climate action) OR (shell parternship) OR (shell partnership for goals)  lang:en since:2017-09-01 until:{date.today()}').get_items()):
        tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username,tweet.user.verified,tweet.user.location,tweet.replyCount,tweet.retweetCount,tweet.likeCount,tweet.quoteCount,tweet.url])
        df = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet_Id', 'Text', 'Username','Verified','Location','Reply_Count','Retweet_Count','Like_Count','Quote_Count','url'])
    return df

   

#For now I think it's best to work with the static version of the dataset
#
#def fetch_sectioned_tweets(max_date:Timestamp) -> pd.DataFrame:
#    
#    '''Fetch tweets added since the time the last data was fetched it returns a dataframe'''
#    #d = datetime.strptime(max_date['Datetime'].to_string(index=False)[0:-6], "%Y-%m-%d %H:%M:%S")
#    d = max_date.strftime("%Y-%m-%d %H:%M:%S")
#    d = datetime.strptime(d,"%Y-%m-%d %H:%M:%S")
#    maximum_date = int(time.mktime(d.timetuple())+1)
#   now_time = int((datetime.now()+timedelta(days=1)).timestamp())
#    sectioned_tweet_list = []
#    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'(shell affordable energy) OR (shell clean energy) OR (shell affordable clean energy)  OR (shell decent work) OR (shell economic growth) OR (shell decent work economic growth) OR (shell industry innovation infrastructure) OR (shell innovation) OR (shell industry innovation) OR (shell responsible consumption) OR (shell responsible production) OR (shell climate action) OR (shell parternship) OR (shell partnership for goals) lang:en  since_time:{maximum_date} until_time:{now_time}').get_items()):
#        sectioned_tweet_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username,tweet.user.verified,tweet.user.location,tweet.replyCount,tweet.retweetCount,tweet.likeCount,tweet.quoteCount,tweet.url])
#        df = pd.DataFrame(sectioned_tweet_list, columns=['Datetime', 'Tweet_Id', 'Text', 'Username','Verified','Location','Reply_Count','Retweet_Count','Like_Count','Quote_Count','url'])
#    return df


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

def _token_stop_pos(text):
        '''
        Maps the part of speech to words in sentences giving consideration to words that are nouns, verbs, 
        adjectives and adverbs
        '''
        pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
        tags = pos_tag(word_tokenize(text))
        newlist = []
        for word, tag in tags:
            if word.lower() not in set(stopwords.words('english')):
                newlist.append(tuple([word, pos_dict.get(tag[0])]))
        return newlist


def _lemmatize_tweets(text: str) -> str:
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    #text = text.lower() text the moment is a list and not a string - something is wrong here
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(text)

def twitter_rds_to_csv(data) -> pd.DataFrame:
    df = pd.DataFrame(data)
    return df


def preprocess_tweets(data)->pd.DataFrame:
    '''
    Function takes in the whole dataframe and carries out the following preprocessing steps:

    1. General text cleaning
    2. Part of Speech tagging
    3. Lemmatization

    Then return the dataframe
    ''' 
    #max_date = data.Datetime.max() 
    #if max_date is not None:
    #    print(f"Fetching extra records as from {max_date}")
    #    fetched_data = fetch_sectioned_tweets(max_date)
        #df=fetched_data
    #    df = data.append(fetched_data)
    #    print("Done fetching  extra records!!")
    #else:
    #    print("Fetching all records")
    #    df = fetch_all_tweets()
    #    print("Done fetching records...") 

    df = fetch_all_tweets()

    df['clean_text'] = df['Text'].apply(lambda x:_clean_tweet(x))
    df['POS tagged'] = df['clean_text'].apply(_token_stop_pos)
    df['Lemma'] = df['clean_text'].apply(_lemmatize_tweets)
    df['hashtags'] = df['Text'].apply(lambda x: " ".join ([w for w in x.split() if '#'  in w[0:3] ]))
    df['hashtags']=df['hashtags'].str.replace("[^a-zA-Z0–9]", ' ')
    df = df.loc[:,['Datetime', 'Tweet_Id','Username','Verified','Location','Reply_Count','Retweet_Count','Like_Count','Quote_Count','url','clean_text','hashtags','POS tagged','Lemma']]
    #print(f'success!, and max date is {max_date}')

    return df




