import streamlit as st
import plotly.express as px
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from kedro.io import DataCatalog
import yaml

from kedro.extras.datasets.pickle import PickleDataSet

from kedro.extras.datasets.pandas import (
    CSVDataSet,
    SQLTableDataSet,
    SQLQueryDataSet,
    ParquetDataSet,
)

io = DataCatalog(
    {
        "sdg_text_data": CSVDataSet(filepath="data/01_raw/train.csv", load_args=dict(sep="\t")),
        "osdg_preprocessed_data": CSVDataSet(filepath="data/03_primary/osdg_preprocessed_data.csv", load_args=dict(sep=',')),
        "sdg_classifier": PickleDataSet(filepath="data/06_models/sdg_classifier.pickle/2022-10-03T10.53.28.385Z/sdg_classifier.pickle", backend="pickle"),
        "vectorizer": PickleDataSet(filepath="data/06_models/vectorizer.pickle/2022-10-03T14.27.01.572Z/vectorizer.pickle", backend="pickle")
    }
)

#st.markdown("# UN SDG Internship Project🎈")
st.sidebar.markdown("# This feature classifies new articles")

#Title

classifier = io.load("sdg_classifier")
vectorizer = io.load("vectorizer")


#st.write(classifier)

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
data = load_data("osdg_preprocessed_data")
# Notify the reader that the data was successfully loaded.
data_load_state.text("")


df = data.copy()
df_new = data[11:13]

#Fit to ALL original train data and transform new data
X_tfidf = vectorizer.fit_transform(df['clean_text'])
X_new_tfidf = vectorizer.transform(df_new['clean_text'])

#predictions
st.markdown('Predictions')
y_pred = classifier.predict(X_new_tfidf)

st.write(y_pred)

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