import streamlit as st
import seaborn as sns
import altair as alt
from streamlit_option_menu import option_menu

#data processing dependencies
import pandas as pd
import numpy as np
import streamlit.components.v1 as component
import requests


#data visualization dependencies
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
import time

from kedro.io import DataCatalog
import yaml

from kedro.extras.datasets.pickle import PickleDataSet

from kedro.extras.datasets.pandas import (
    CSVDataSet,
    SQLTableDataSet,
    SQLQueryDataSet,
    ParquetDataSet,
)


def main():
    """Twitter Analytics App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    with st.sidebar:
        selection = option_menu(
            menu_title="Main Menu",
            options=["Twitter Analytics","Live Feed"
                     ],
            icons=["emoji-expressionless",
                   "twitter"],
            menu_icon="cast",
            default_index=0
        )
    config = {
            "clean_tweet_data": {
                "type": "pandas.CSVDataSet",
                "filepath": "s3://internship-sdg-2022/kedro/data/02_intermediate/clean_tweet_data.csv",
                "credentials": "s3_credentials",
                "load_args": {
                    "sep": ','
                }
            },
            "labelled_twitter_data": {
                "type": "pandas.CSVDataSet",
                "filepath": "s3://internship-sdg-2022/kedro/data/03_primary/labelled_twitter_data.csv",
                "credentials": "s3_credentials",
                "load_args": {
                    "sep": ','
                }
            },
            "clean_tweet_data_csv": {
                "type": "pandas.CSVDataSet",
                "filepath": "s3://internship-sdg-2022/kedro/data/03_primary/clean_tweet_data_csv.csv",
                "credentials": "s3_credentials",
                "load_args": {
                    "sep": ','
                }
            },
        }

    #TO DO: keep this somewhere safer 
    credentials = {
        "s3_credentials": {
                "key": "AKIA5XNJCCEVDTPAHASV",
                "secret": "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"
        }
    }

    catalog = DataCatalog.from_config(config, credentials)
    #cache function that loads in data
    @st.cache(allow_output_mutation = True)
    def load_data(data_name):
        data = catalog.load(data_name)
        data.drop(data.tail(383).index,inplace=True)
        #can add extra stuff here
        return data


    data_load_state = st.text('Loading data from AWS S3...')
    data = load_data("labelled_twitter_data")
    #catalog.save("boats", df)
    data_load_state.text("")


    if selection == "Twitter Analytics":
        st.title("**TWITTER ANALYTICS**")
        




        #data['Datetime']= pd.to_datetime(data['Datetime'], utc=True).dt.date
        data['Year'] = pd.DatetimeIndex(data['Datetime']).year  
        data['Month'] = pd.DatetimeIndex(data['Datetime']).month  

        # build a datetime index from the date column
        datetime_series = pd.to_datetime(data['Datetime'])
        datetime_index = pd.DatetimeIndex(datetime_series.values)

        df=data.set_index(datetime_index)
        df.sort_index(inplace=True)

        #st.dataframe(df)

        st.markdown("Sentiment analysis refers to identifying as well as classifying the sentiments that are expressed in the text source. Tweets are often useful in generating a vast amount of sentiment data upon analysis. These data are useful in understanding the opinion of the people about a variety of topics.")
        hashtag = data.copy()
        hashtag['hashtags'] = hashtag['hashtags'].astype('str')
        hashtag['hashtags'] = hashtag['hashtags'].apply(lambda x: x.lower().split())
        hashtag = hashtag.explode('hashtags', ignore_index = True)
        hashtag["hashtags"] = hashtag["hashtags"].replace('nan', np.NaN)
        hashtag = hashtag.dropna()




        slicer1,slicer2 = st.columns(2)

        with slicer1:
            option_list = ['All'] + list(data['sentiment'].unique())
            sentiment_option = st.selectbox("Select sentiment",option_list)
            sentiment_choice = sentiment_option

        with slicer2:
            option_list = ['All'] + list(data['Year'].unique())
            year_option = st.selectbox("Select Year",option_list)
            year_choice = year_option

        @st.cache
        def get_tweet_no_metrics(sentiment_choice,year_choice ):
            

            if (sentiment_choice == 'All') & (year_choice == 'All'):
                total_tweet = data['Tweet_Id'].nunique()

            elif (sentiment_choice == sentiment_choice) & (year_choice == 'All'):
                total_tweet = data[data['sentiment'] == sentiment_choice]['Tweet_Id'].nunique()
                
            elif (sentiment_choice == 'All') & (year_choice == year_choice):
                total_tweet = data[data['Year'] == year_choice]['Tweet_Id'].nunique()

            else:
                total_tweet = data[(data['sentiment'] == sentiment_choice) & (data['Year'] == year_choice)]['Tweet_Id'].nunique()

            return total_tweet

        @st.cache
        def get_users_count(sentiment_choice,year_choice ):

                if (sentiment_choice == 'All') & (year_choice == 'All'):
                    total_user = data['Username'].nunique()

                elif (sentiment_choice == sentiment_choice) & (year_choice == 'All'):
                    total_user = data[data['sentiment'] == sentiment_choice]['Username'].nunique()
                
                elif (sentiment_choice == 'All') & (year_choice == year_choice):
                    total_user = data[data['Year'] == year_choice]['Username'].nunique()

                else:
                    total_user = data[(data['sentiment'] == sentiment_choice) & (data['Year'] == year_choice)]['Username'].nunique()

                return total_user

        @st.cache
        def get_hashtags(sentiment_choice,year_choice ):

                if (sentiment_choice == 'All') & (year_choice == 'All'):
                    unique_hashtag = hashtag['hashtags'].nunique()

                elif (sentiment_choice == sentiment_choice) & (year_choice == 'All'):
                    unique_hashtag = hashtag[hashtag['sentiment'] == sentiment_choice]['hashtags'].nunique()
                
                elif (sentiment_choice == 'All') & (year_choice == year_choice):
                    unique_hashtag = hashtag[hashtag['Year'] == year_choice]['hashtags'].nunique()

                else:
                    unique_hashtag = hashtag[(hashtag['sentiment'] == sentiment_choice) & (hashtag['Year'] == year_choice)]['hashtags'].nunique()

                return unique_hashtag



        #@st.cache
        def get_loc_metrics(sentiment_choice,year_choice):

            if (sentiment_choice == 'All') & (year_choice == 'All'):
                location_tweet = data['Location'].nunique()

            elif (sentiment_choice == sentiment_choice) & (year_choice == 'All'):
                location_tweet = data[data['sentiment'] == sentiment_choice]['Location'].nunique()
                
            elif (sentiment_choice == 'All') & (year_choice == year_choice):
                location_tweet = data[data['Year'] == year_choice]['Location'].nunique()

            else:
                location_tweet = data[(data['sentiment'] == sentiment_choice) & (data['Year'] == year_choice)]['Location'].nunique()

            return location_tweet

        #@st.cache
        def bar_plot_sentiment_year(sentiment_choice,year_choice):
            fig, axs = plt.subplots(figsize=(12, 4))

            if (sentiment_choice == sentiment_choice) & (year_choice == 'All'):
                plt.hist(data['sentiment'])
                plt.title("Distribution of Sentiment in All years")
                #plt.xlabel("Sentiment")
                plt.ylabel("Frequency")

            else:
                time_data = data[data['Year'] == year_choice]
                plt.hist(time_data['sentiment'])
                plt.title(f"Distribution of Sentiment in {year_choice}")
                #plt.xlabel("Sentiment")
                plt.ylabel("Frequency")

            return st.pyplot(fig)

        #@st.cache
        def freq_tweets(sentiment_choice,year_choice):
            fig, axs = plt.subplots(figsize=(12, 4))

            if (sentiment_choice == sentiment_choice) & (year_choice == 'All'):
                df.resample('M').size().plot(kind='line', rot=0, ax=axs)
                plt.title("Average Monthly Tweet Frequency")
                #plt.xlabel("Sentiment")
                plt.ylabel("Frequency")

            else:
                df_freq = df[df['Year'] == year_choice]
                df_freq.resample('M').size().plot(kind='line', rot=0, ax=axs)
                plt.title(f"Average Monthly Tweet Frequency in {year_choice}")
                #plt.xlabel("Sentiment")
                plt.ylabel("Frequency")

            return st.pyplot(fig)


        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Number of tweets", get_tweet_no_metrics(sentiment_choice,year_choice ))
        col2.metric("Number of locations", get_loc_metrics(sentiment_choice,year_choice ))
        col3.metric("Number of users", get_users_count(sentiment_choice,year_choice ))
        col4.metric("Number of hashtags", get_hashtags(sentiment_choice,year_choice ))


        bar_plot_sentiment_year(sentiment_choice,year_choice)

        freq_tweets(sentiment_choice,year_choice)


        fig, axs = plt.subplots(figsize=(12, 4))
        data.groupby(data["Month"])["score"].mean().plot(
            kind='bar', rot=0, ax=axs
        )
        plt.title("Average Sentiment Score per month")
        plt.xlabel("Months")
        plt.ylabel("Sentiment score")
        st.pyplot(fig)

        #sentiment analysis line plot
        fig, axs = plt.subplots(figsize=(12, 4))
        df.score.resample('M').mean().plot(
            kind='line', rot=0, ax=axs
        )
        plt.title("Average Monthly Sentiment Score")
        plt.xlabel("Time period")
        plt.ylabel("Sentiment score")
        st.pyplot(fig)
        #st.dataframe(data)


        col1, col2, col3 = st.columns([1,3,2])
        with col1:

            #twitter = Image.open('images/twitter.png')
            #st.image(twitter)

            st.markdown("<h5 style='text-align: center; color: #23395d;'>Word Frequency</h5>", unsafe_allow_html=True)
            words = data.copy()

            if (sentiment_choice == 'All') & (year_choice == 'All'):

                words["mytext_new"] = words['Lemma'].str.lower().str.replace('[^\w\s]','')
                words["mytext_new"] = words["mytext_new"].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')
                words = words.mytext_new.str.split(expand=True).stack().value_counts().reset_index() 
                words.columns = ['Word', 'Frequency'] 
                words = words.sort_values('Frequency', ascending= False).head(10)

                word_frequency_plot = px.bar(data_frame=words,
                        x='Word',
                        y='Frequency',
                        color_discrete_sequence =['#23395d']*3)

            elif (sentiment_choice == sentiment_choice) & (year_choice == 'All'):
                    
                words = words[words['sentiment'] == sentiment_choice]

                words["mytext_new"] = words['Lemma'].str.lower().str.replace('[^\w\s]','')
                words["mytext_new"] = words["mytext_new"].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')
                words = words.mytext_new.str.split(expand=True).stack().value_counts().reset_index() 
                words.columns = ['Word', 'Frequency'] 
                words = words.sort_values('Frequency', ascending= False).head(10)

                word_frequency_plot = px.bar(data_frame=words,
                        x='Word',
                        y='Frequency',
                        color_discrete_sequence =['#23395d']*3)

            elif (sentiment_choice == 'All') & (year_choice == year_choice):

                words = words[words['Year'] == year_choice]


                words["mytext_new"] = words['Lemma'].str.lower().str.replace('[^\w\s]','')
                words["mytext_new"] = words["mytext_new"].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')
                words = words.mytext_new.str.split(expand=True).stack().value_counts().reset_index() 
                words.columns = ['Word', 'Frequency'] 
                words = words.sort_values('Frequency', ascending= False).head(10)

                word_frequency_plot = px.bar(data_frame=words,
                        x='Word',
                        y='Frequency',
                        color_discrete_sequence =['#23395d']*3)

            else:

                words = words[(words['sentiment'] == sentiment_choice) & (words['Year'] == year_choice)]

                words["mytext_new"] = words['Lemma'].str.lower().str.replace('[^\w\s]','')
                words["mytext_new"] = words["mytext_new"].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')
                words = words.mytext_new.str.split(expand=True).stack().value_counts().reset_index() 
                words.columns = ['Word', 'Frequency'] 
                words = words.sort_values('Frequency', ascending= False).head(10)

                word_frequency_plot = px.bar(data_frame=words,
                        x='Word',
                        y='Frequency',
                        color_discrete_sequence =['#23395d']*3)

            st.plotly_chart(word_frequency_plot, use_container_width=False, sharing="streamlit")

    if selection == "Live Feed":
        page_options = "Tweets"
        # # selection = option_menu( menu_title=None,
        #                     options=page_options,
        #                     icons=["house", "camera-reels", "graph-up", "file-person"],
        #                     orientation='horizontal',
        #                     styles={
        #                                 "container": {"padding": "0!important", "background-color": "#FF4B4B"},
        #                                 "icon": {"color": "black", "font-size": "15px",  },
        #                                 "nav-link": {
        #                                     "font-size": "15px",
        #                                     "text-align": "center",
        #                                     "margin": "5px",
        #                                     "--hover-color": "#eee",
        #                                     "color": "white"
        #                                 },
        #                                 "nav-link-selected": {"background-color": "white", "color": "#FF4B4B"},
        #                             },
        #)

    

        #data = pd.read_csv("tweets_2.csv")
        data['timeline'] = data['url'].apply(lambda x: x.split('/sta')[0])
        st.session_state['data'] = data

        sampled = data.sample(10000)

        @st.cache
        def embed_tweet(tweet_url):
            api = "https://publish.twitter.com/oembed?url={}".format(tweet_url)
            response = requests.get(api)
            res = response.json()['html']
            return res


        tweet_url = [url for url in sampled['url']]
        timeline_url = [timeline for timeline in sampled['timeline']]

        st.markdown("# Live Feed")


        
        with st.spinner('Running...'):
                time.sleep(1)
        tweet_embed1 = embed_tweet(tweet_url[0])
        tweet_embed2 = embed_tweet(tweet_url[1])
        tweet_embed3 = embed_tweet(tweet_url[2])
        tweet_embed4 = embed_tweet(tweet_url[2])
        tweet_embed5 = embed_tweet(tweet_url[4])
        tweet_embed6 = embed_tweet(tweet_url[5])
        tweet_embed7 = embed_tweet(tweet_url[6])
        tweet_embed8 = embed_tweet(tweet_url[7])
        tweet_embed9 = embed_tweet(tweet_url[8])
        tweet_embed10 = embed_tweet(tweet_url[9])
            
        tweet_list = [tweet_embed1, tweet_embed2, tweet_embed3, tweet_embed4, tweet_embed5, tweet_embed6, tweet_embed7, tweet_embed8, tweet_embed9, tweet_embed10]
        for embed in tweet_list:
            component.html(embed, height= 500, scrolling=True)


if __name__ == '__main__':
    main()
