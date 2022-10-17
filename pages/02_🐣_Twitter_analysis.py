import streamlit as st
import seaborn as sns
import altair as alt


from kedro.io import DataCatalog
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
col1.metric("Sentiment", "Positive", "4%")
col2.metric("Number of tweets", "1,340", "202")
col3.metric("Mentions", "839", "3")



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
    #can add extra stuff here
    return data


data_load_state = st.text('Loading data from AWS S3...')
data = load_data("labelled_twitter_data")
#catalog.save("boats", df)
data_load_state.text("")

st.dataframe(data)


c = alt.Chart(data).mark_line().encode(
        alt.X("Datetime", title='Number of predictions'), alt.Y('score',title='Description of SDG'))


st.altair_chart(c, use_container_width=True)


df = load_data("clean_tweet_data")
st.dataframe(df)
