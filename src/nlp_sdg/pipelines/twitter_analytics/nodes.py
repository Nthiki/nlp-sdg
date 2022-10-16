
"""
This is a boilerplate pipeline 'twitter_analytics'
generated using Kedro 0.18.2
"""
# utilities
from typing import List, Dict
import pandas as pd
import numpy as np
import logging

#sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



''' ================================== 
     Sentiment analysis
 ==================================== '''

#simple sentiment analysis model

def _vader_sentiment_analysis(tweet):
    
    '''
    This function calculates the sentiment score and returns it    
    
    Args:
        tweet: text string  from the twitter data set
        
     Returns:
        Sentiment (pos, neg or neutral).

    '''
    analyzer = SentimentIntensityAnalyzer()
    
    vs = analyzer.polarity_scores(tweet)
    compound = vs['compound']

    if compound >= 0.5:
        return 'Positive'
    elif compound <= -0.5 :
        return 'Negative'
    else:
        return 'Neutral'

def _vader_sentiment_analysis_2(tweet):
    
    '''
    This function calculates the sentiment score and returns it    
    
    Args:
        tweet: text string  from the twitter data set
        
     Returns:
        Sentiment (pos, neg or neutral).

    '''
    analyzer = SentimentIntensityAnalyzer()
    
    vs = analyzer.polarity_scores(tweet)
    compound = vs['compound']

    return compound


#what about adding more columns here? We could add timestamps to monitor over time, etc

def label_tweet(data:pd.DataFrame) -> pd.DataFrame:

    data['sentiment'] = data['Lemma'].apply(_vader_sentiment_analysis)
    data['score'] = data['Lemma'].apply(_vader_sentiment_analysis_2)


    #data = data[['clean_text','sentiment']]
    return data