"""
This is a boilerplate pipeline 'twitter_analytics'
generated using Kedro 0.18.2
"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#this is a dummy- node
def dummy_node(data):
    print("Twitter Analytics dummy node completed")
    return 5


    
def vadersentimentanalysis(tweet):
    
    '''
    calculates the sentiment score and returns it
    '''
    analyzer = SentimentIntensityAnalyzer()

    vs = analyzer.polarity_scores(tweet)
    compound = vs['compound']
    return compound

def vader_analysis(compound):
    '''
    Maps the sentiments based on sentiment score as 'positive', 'negative' and 'neutral'
    '''
    if compound >= 0.5:
        return 'Positive'
    elif compound <= -0.5 :
        return 'Negative'
    else:
        return 'Neutral'

def label_tweet(df: pd.DataFrame) -> pd.DataFrame:
    '''function to predict sentiments using the Vader model
    
    Args: dataframe
    '''
    df['sentiment_score'] = df['Lemma'].apply(vadersentimentanalysis)
    df['sentiment'] = df['sentiment_score'].apply(vader_analysis)
    df = df[['clean_text','sentiment']]
    
    return df

