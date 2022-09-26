import streamlit as st
import plotly.express as px
import numpy as np 
import pandas as pd

from kedro.io import DataCatalog
import yaml

from kedro.extras.datasets.pandas import (
    CSVDataSet,
    SQLTableDataSet,
    SQLQueryDataSet,
    ParquetDataSet,
)

io = DataCatalog(
    {
        "sdg_text_data": CSVDataSet(filepath="data/01_raw/train.csv", load_args=dict(sep="\t"))
    }
)


#st.markdown("# UN SDG Internship Project🎈")
st.sidebar.markdown("# This feature classifies new articles")

#Title

st.title('UN SDG Internship Project')
@st.experimental_memo
#@st.cache
def load_data(data_name):
    data = io.load(data_name)
    #can add extra stuff here
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load rows of data into the dataframe.
#data_name = "sdg_text_data"
data = load_data("sdg_text_data")
# Notify the reader that the data was successfully loaded.
data_load_state.text("")


df = data.head(5)

st.subheader('Select news article')
option = st.selectbox(
    'Select a news article',
    ('Shell goes under', 'Coastal woes in Durban', 'CEO of Shell steps down'))

st.write('You selected:', option)

#When we select an article, is it possible to show the article?

st.dataframe(df)

# Histogram of SDG label distribution


st.subheader('Training data set')
st.markdown('##### SDG label distribution')

hist_values = np.histogram(data['sdg'], bins=15, range=(0,16))[0]

st.bar_chart(hist_values)

