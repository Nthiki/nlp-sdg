import streamlit as st
import seaborn as sns
import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime




from kedro.io import DataCatalog
from kedro.config import ConfigLoader
from kedro.framework.project import settings
import yaml

from kedro.extras.datasets.pickle import PickleDataSet

from kedro.extras.datasets.pandas import (
    CSVDataSet,
    SQLTableDataSet,
    SQLQueryDataSet,
    ParquetDataSet,
)

st.markdown("# Twitter analysis")
st.sidebar.markdown("# Twitter analysis")

#update these colms using real time twitter data

col1, col2, col3 = st.columns(3)
col1.metric("Number of tweets", "16193")
col2.metric("Data coverage", "3+ years")
col3.metric("Overall sentiment", "Positive")



from kedro.extras.datasets.pickle import PickleDataSet

from kedro.extras.datasets.pandas import (
    CSVDataSet,
    SQLTableDataSet,
    SQLQueryDataSet,
    ParquetDataSet,
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
}


#retieving keys and secret
conf_path = "conf/"
conf_loader = ConfigLoader(conf_path)
conf_catalog = conf_loader.get("credentials*", "credentials*/**")


catalog = DataCatalog.from_config(config, conf_catalog)


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


col1, col2 = st.columns(2)
col1.metric("Number of tweets", get_tweet_no_metrics(sentiment_choice,year_choice ))
col2.metric("Number of locations mentioned", get_loc_metrics(sentiment_choice,year_choice ))

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

data['date'] = pd.to_datetime(data['Datetime'])


#yo = df.score.resample('M').mean()
#st.dataframe(yo)

