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
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'(shell affordable energy) OR (shell clean energy) OR (shell affordable clean energy)  OR (shell decent work) OR (shell economic growth) OR (shell decent work economic growth) OR (shell industry innovation infrastructure) OR (shell innovation) OR (shell industry innovation) OR (shell responsible consumption) OR (shell responsible production) OR (shell climate action) OR (shell parternship) OR (shell partnership for goals)  lang:en since:2017-09-01 until:{date.today()}').get_items()):
        tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username,tweet.user.verified,tweet.user.location,tweet.replyCount,tweet.retweetCount,tweet.likeCount,tweet.quoteCount,tweet.url])
        df = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet_Id', 'Text', 'Username','Verified','Location','Reply_Count','Retweet_Count','Like_Count','Quote_Count','url'])
    return df

def fetch_sectioned_tweets(max_date):
    '''Fetch tweets added since the time the last data was fetched it returns a dataframe'''
    d = datetime.strptime(max_date['Datetime'].to_string(index=False)[0:-6], "%Y-%m-%d %H:%M:%S")
    maximum_date = int(time.mktime(d.timetuple())+1)
    now_time = int((datetime.now()+timedelta(days=1)).timestamp())
    sectioned_tweet_list = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'(shell affordable energy) OR (shell clean energy) OR (shell affordable clean energy)  OR (shell decent work) OR (shell economic growth) OR (shell decent work economic growth) OR (shell industry innovation infrastructure) OR (shell innovation) OR (shell industry innovation) OR (shell responsible consumption) OR (shell responsible production) OR (shell climate action) OR (shell parternship) OR (shell partnership for goals) lang:en  since_time:{maximum_date} until_time:{now_time}').get_items()):
        sectioned_tweet_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username,tweet.user.verified,tweet.user.location,tweet.replyCount,tweet.retweetCount,tweet.likeCount,tweet.quoteCount,tweet.url])
        df = pd.DataFrame(sectioned_tweet_list, columns=['Datetime', 'Tweet_Id', 'Text', 'Username','Verified','Location','Reply_Count','Retweet_Count','Like_Count','Quote_Count','url'])
    return df

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

def data_preprocessing(max_date):
    '''
    Function takes in the whole dataframe and carries out the following preprocessing steps:
    
    1. General text cleaning
    2. Part of Speech tagging
    3. Lemmatization
    
    Then return the dataframe
    '''        
    if max_date is not None:
        print(f"Fetching extra records as from {max_date}")
        fetched_data =  fetch_sectioned_tweets(max_date)
        df=fetched_data
        print("Done fetching  extra records!!")
    else:
        print("Fetching all records")
        df = fetch_all_tweets()
        print("Done fetching records...") 
    df['clean_text'] = df['Text'].apply(lambda x:clean_tweet(x))
    df['POS tagged'] = df['clean_text'].apply(token_stop_pos)
    df['Lemma'] = df['POS tagged'].apply(lemmatize)
    df['hashtags'] = df['Text'].apply(lambda x: " ".join ([w for w in x.split() if '#'  in w[0:3] ]))
    df['hashtags']=df['hashtags'].str.replace("[^a-zA-Z0–9]", ' ')
    df = df.loc[:,['Datetime', 'Tweet_Id','Username','Verified','Location','Reply_Count','Retweet_Count','Like_Count','Quote_Count','url','clean_text','hashtags','POS tagged','Lemma']]
    print(f'success!, and max date is {max_date}')
    
    return df    