"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""
#loading necessary libraries
from typing import Dict
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
import re
import os

# libraries for NLP
#pip install nltk
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer

# libraries used for scrapping tweets
from datetime import date,timedelta,datetime
import snscrape.modules.twitter as sntwitter
import time 


def fetch_all_tweets():
    ''' Fetch all tweets from 2017 september based on the keywords provided, it returns a dataframe'''
    tweets_list2 = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'(shell affordable energy) OR (shell clean energy) OR (shell affordable clean energy)  OR (shell decent work) OR (shell economic growth) OR (shell decent work economic growth) OR (shell industry innovation infrastructure) OR (shell innovation) OR (shell industry innovation) OR (shell responsible consumption) OR (shell responsible production) OR (shell climate action) OR (shell parternship) OR (shell partnership for goals)  lang:en since:2022-09-01 until:{date.today()}').get_items()):
        tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username,tweet.user.verified,tweet.user.location,tweet.replyCount,tweet.retweetCount,tweet.likeCount,tweet.quoteCount])
        df = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet_Id', 'Text', 'Username','Verified','Location','Reply_Count','Retweet_Count','Like_Count','Quote_Count'])
    return df

def fetch_sectioned_tweets(df):
    '''Fetch tweets added since the time the last data was fetched it returns a dataframe'''
    # read the already existing csv file
    # Check maximum date
    d = datetime.strptime(df['Datetime'].agg('max')[0:-6], "%Y-%m-%d %H:%M:%S")
    max_date = int(time.mktime(d.timetuple())+1)
    now_time = int((datetime.now()+timedelta(days=1)).timestamp())
    sectioned_tweet_list = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'(shell affordable energy) OR (shell clean energy) OR (shell affordable clean energy)  OR (shell decent work) OR (shell economic growth) OR (shell decent work economic growth) OR (shell industry innovation infrastructure) OR (shell innovation) OR (shell industry innovation) OR (shell responsible consumption) OR (shell responsible production) OR (shell climate action) OR (shell parternship) OR (shell partnership for goals) lang:en  since_time:{max_date} until_time:{now_time}').get_items()):
        sectioned_tweet_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username,tweet.user.verified,tweet.user.location,tweet.replyCount,tweet.retweetCount,tweet.likeCount,tweet.quoteCount])
        df = pd.DataFrame(sectioned_tweet_list, columns=['Datetime', 'Tweet_Id', 'Text', 'Username','Verified','Location','Reply_Count','Retweet_Count','Like_Count','Quote_Count'])
    return df

def generate_hashtag_column(tweet):
         '''
        tweet: String
               Input Data
        hashtag: String
               Output Data
        func: Generates hashtag column from tweet
         '''
        #  hashtag = tweet.apply(lambda x: " ".join ([w for w in x.split() if '#'  in w[0:3] ]))
         hashtag = " ".join ([w for w in tweet.split() if '#'  in w[0:3] ])
         hashtag =tweet.replace("[^a-zA-Z0–9]", ' ')
         return hashtag


def clean_tweet(tweet):
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
        tweet = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', tweet)
        tweet = re.sub('\@[a-zA-Z0-9]*', ' ', tweet)
        tweet = re.sub('[^a-zA-Z\']', ' ', tweet)
        tweet = re.sub(r'\s+', " ", tweet)
        tweet = ' '.join( [w for w in tweet.split() if len(w)>1] )
    
        return tweet

def token_stop_pos(text):
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

def lemmatize(pos_data):
        '''
        Performs lemmatization on tokens based on its part of speech tagging 
        '''
        wordnet_lemmatizer = WordNetLemmatizer()
        lemma_rew = " "
        for word, pos in pos_data:
            if not pos:
                lemma = word
                lemma_rew = lemma_rew + " " + lemma
            else:
                lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
                lemma_rew = lemma_rew + " " + lemma
        return lemma_rew



def fetch_save_tweets():
    # Putting together everything
    file_exists = os.path.exists("/home/kedro/data/01_raw/tweets_2.csv")
    if (file_exists == True):
        print("File exist, fetching extra records")
        df1 = pd.read_csv('/home/kedro/data/01_raw/tweets_2.csv')
        df1=df1.loc[:,~df1.columns.str.contains('Unnamed')]
        fetched_data =  fetch_sectioned_tweets(df1)
        # fetched_data['clean_text'] = fetched_data['Text'].apply(lambda x:clean_tweet(x))
        # fetched_data['hashtags'] = fetched_data['Text'].apply(lambda x:generate_hashtag_column(x))
        # fetched_data = fetched_data.loc[:,['Datetime', 'Tweet_Id','Username','Verified','Location','Reply_Count','Retweet_Count','Like_Count','Quote_Count','clean_text','hashtags']]
        # df = pd.concat([fetched_data,df1])
        # bul = df.duplicated(subset='Tweet_Id', keep='first' )
        # df=df[~bul]
        df=fetched_data
        # print("Done fetching and appending extra records!!")
        print(df.columns)

    else:
        print("File does not exist, fetching all records")
        df = fetch_all_tweets()
        print("Done fetching records...")
        print("All processes are done!!")
    df['clean_text'] = df['Text'].apply(lambda x:clean_tweet(x))
    df['hashtags'] = df['Text'].apply(lambda x:generate_hashtag_column(x))
    df = df.loc[:,['Datetime', 'Tweet_Id','Username','Verified','Location','Reply_Count','Retweet_Count','Like_Count','Quote_Count','clean_text','hashtags']]

    return df



def data_preprocessing(df:pd.DataFrame)->pd.DataFrame:
    '''
    Function takes in the whole dataframe and carries out the following preprocessing steps:
    
    1. General text cleaning
    2. Part of Speech tagging
    3. Lemmatization
    
    Then return the dataframe
    '''    
    
    # df['clean_text'] = df['Text'].apply(lambda x:clean_tweet(x))
    df['POS tagged'] = df['clean_text'].apply(token_stop_pos)
    df['Lemma'] = df['POS tagged'].apply(lemmatize)
    print('success!')
    
    return df


