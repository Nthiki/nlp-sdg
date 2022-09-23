import streamlit as st
import plotly.express as px
import numpy as np 

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

#Title

st.title('UN SDG Internship Project')

@st.cache
def load_data(data_name):
    data = io.load(data_name)
    #can add extra stuff here
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
#data_name = "sdg_text_data"
data = load_data("sdg_text_data")
# Notify the reader that the data was successfully loaded.
data_load_state.text("")

# Histogram of SDG label distribution

st.subheader('SDG label distribution')

hist_values = np.histogram(data['sdg'], bins=15, range=(0,16))[0]

st.bar_chart(hist_values)

