import streamlit as st
import plotly.express as px
import numpy as np 
import pandas as pd
import time
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

#load in data

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

#helper functions

#need functions that clean incoming data

def clean_text_data(my_text):

    doc = list(my_text.split(" "))

    return my_text

def df_text_predict(df_upload):
    X = df_upload['text'].values
    #X_tfidf = vectorizer.transform(X) # need to clean data (remove numbers etc)
    #predicted = classifier.predict(X_tfidf)
    if predicted[0] == 1:
        return "This is SDG label 1"
    else:
        return "This is not SDG label 1"

def text_predict(my_text):
    doc = list(my_text.split(" "))
    doc = vectorizer.transform(doc)
    predicted = classifier.predict(doc)
    if predicted[0] == 1:
        return "This is SDG label 1"
    else:
        return "This is not SDG label 1"

#main function

def main():
    st.title('UN SDG Internship Project')
    st.markdown('## News article classification')
    
    message1 = st.text_area("Please Enter News article", "Type Here")
    if st.button("Detect SDG"):
        with st.spinner('Running model...'):
            time.sleep(1)
        #clean_message = clean_text(message1)
        result1 = text_predict(message1)
        st.success(result1)

    uploaded_file = st.file_uploader("Upload csv file containing news article data")
    if uploaded_file is not None:
        
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

        if st.button('Predict SDG'):
            result2 = df_text_predict(message1)
            st.success(result2)

     
    

if __name__ == '__main__':
    main()


#df = data.copy()
#df_new = data[11:13]

#Fit to ALL original train data and transform new data
#X_tfidf = vectorizer.fit_transform(df['clean_text'])
#X_new_tfidf = vectorizer.transform(df_new['clean_text'])

#When we select an article, is it possible to show the article?

#st.dataframe(df)

st.header('EDA')
st.subheader('Training data set')
st.markdown('##### SDG label distribution')

hist_values = np.histogram(data['sdg'], bins=15, range=(0,16))[0]

st.bar_chart(hist_values)