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

config = {
    "osdg_preprocessed_data": {
        "type": "pandas.CSVDataSet",
        "filepath": "s3://internship-sdg-2022/kedro/data/02_intermediate/osdg_preprocessed_data.csv",
        "credentials": "s3_credentials",
        "load_args": {
            "sep": ','
        }
    },
    "sdg_classifier": {
        "type": "pickle.PickleDataSet",
        "filepath": "s3://internship-sdg-2022/kedro/data/06_models/sdg_classifier.pickle",
        "credentials": "s3_credentials",
        "backend": "pickle"
    },
    "vectorizer": {
        "type": "pickle.PickleDataSet",
        "filepath": "s3://internship-sdg-2022/kedro/data/06_models/vectorizer.pickle",
        "credentials": "s3_credentials",
        "backend": "pickle"
    }, 
}

#keep this somewhere safer
credentials = {
    "s3_credentials": {
            "key": "<key>",
            "secret": "<secret>"
     }
}

catalog = DataCatalog.from_config(config, credentials)

data_load_state = st.text('Loading data from AWS S3...')
data = catalog.load("osdg_preprocessed_data")
#catalog.save("boats", df)
data_load_state.text("")
#st.dataframe(data)

classifier = catalog.load("sdg_classifier")
vectorizer = catalog.load("vectorizer")

st.sidebar.write("This feature based on natural language processing (NLP) assigns labels to text predicated on Sustainable Development Goals (SDGs).")




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
        return "Goal 1: No Poverty"
    elif predicted[0] == 2:
        return "Goal 2: Zero Hunger"
    elif predicted[0] == 3:
        return "Goal 3: Good Health and Well-being"
    elif predicted[0] == 4:
        return "Goal 4: Quality Education"
    elif predicted[0] == 5:
        return "Goal 5: Gender Equality"
    elif predicted[0] == 6:
        return "Goal 6: Clean Water and Sanitation"
    elif predicted[0] == 7:
        return "Goal 7: Affordable and Clean Energy"
    elif predicted[0] == 8:
        return "Goal 8: Decent Work and Economic Growth"
    elif predicted[0] == 9:
        return "Goal 9: Industry, Innovationa and Infrastructure"
    elif predicted[0] == 10:
        return "Goal 10: Reduced Inequality"
    elif predicted[0] == 11:
        return "Goal 11: Sustainable Cities and Communities"
    elif predicted[0] == 12:
        return "Goal 12: Responsible Consumption and Production"
    elif predicted[0] == 13:
        return "Goal 13: Climate Action"
    elif predicted[0] == 14:
        return "Goal 14: Life Below Water"
    else:
        return "Goal 15: Life on Land"

#main function

def main():
    #st.title('UN SDG Internship Project')
    st.markdown('## Text classification')
    st.markdown('#### How do public artifacts reflect Shell\'s contribution towards Sustainable Development Goals?')
    
    message1 = st.text_area("Type in or paste any text segment (e.g. publication excerpt, news article) in the text box below", "Type Here")
    if st.button("Get SDG Label"):
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

st.header('Data Analysis')
st.subheader('Training data set')
st.markdown('##### SDG label distribution')

hist_values = np.histogram(data['sdg'], bins=15, range=(0,16))[0]

st.bar_chart(hist_values)
