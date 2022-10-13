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
    "predictions": {
        "type": "pandas.CSVDataSet",
        "filepath": "s3://internship-sdg-2022/kedro/data/07_model_output/predictions.csv",
        "credentials": "s3_credentials",
        "load_args": {
            "sep": ','
        }
    },
}

#keep this somewhere safer
credentials = {
    "s3_credentials": {
            "key": "AKIA5XNJCCEVDTPAHASV",
            "secret": "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"
     }
}


details = {'sdgLables': ["No poverty", "Zero Hunger", "Good Health and well-being",
                         "Quality Education", "Gender equality", "Clean water and sanitation",
                         "Affordable and clean energy", "Decent work and economic growth",
                         "Industry, Innovation and Infrustructure", "Reduced Inequality",
                         "Sustainable cites and communities", "Responsible consumption and production",
                         "Climate Action", "life below water", "Life on land", "Peace , Justice and strong institutions",
                         "Partnership for the goals"],
           'sdg': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
           'Description': [
               'SDG 1 seeks to ‘end poverty in all its forms everywhere’, specifically by ensuring that the poor are covered by social protection systems; by securing their rights to economic resources, access to basic services and property ownership; and by building their resilience to economic, social and environmental shocks. ',
               'SDG 2 seeks to ‘end hunger, achieve food security and nutrition and promote sustainable agriculture’',
               'SDG 3 seeks to ensure healthy lives and promote well-being for all at all ages.',
               'SDG 4 seeks to ensure inclusive and equitable quality education and promote lifelong learning opportunities for all.',
               'SDG 5 seeks to achieve gender equality and empower all women and girls.',
               'SDG 6 seeks to ensure availability and sustainable management of water and sanitation for all.',
               'SDG 7 seeks to ensure access to affordable, reliable, sustainable and modern energy for all.',
               'SDG 8 seeks to promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all',
               'SDG 9 seeks to build resilient infrastructure, promote inclusive and sustainable industrialization.',
               'SDG 10 seeks to reduce inequality within and among countries.',
               'SDG 11 seeks to reduce inequality within and among countries.',
               'SDG 12 seeks to ensure sustainable consumption and production patterns.',
               'SDG 13 seeks to take urgent action to combat climate changes and its impact.',
               'SDG 14 seeks to conserve and sustainably use the oceans, seas and marine resources for sustainable development.',
               'SDG 15 seeks to protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification and half and reverse land degradation and halt biodiversity loss.',
               'SDG 16 seeks to promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels.',
               'SDG 17 seeks to strengthen the means of implementation and revitalise the global partnership for sustainable development.'],
           'Poster': ['https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/1.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/2.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/3.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/4.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/5.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/6.jpg?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/7.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/8.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/9.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/10.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/11.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/12.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/13.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/14.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/15.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/16.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/17.png?raw=true'],
           }

# creating a Dataframe object
details = pd.DataFrame(details)

def display_sdg(pred):
    """
    Takes the predicted SDG value (int) displays the SDG graphic (a clickable link), a short description
    Args:
        The predicted SDG value
    """
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f'''
            <a href="https://sdgs.un.org/goals/goal{pred['sdg']}">
               <td><img src={pred['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
            </a>''',
                    unsafe_allow_html=True
                    )
    with col2:
        #st.markdown(f" #### {pred['sdgLables']}")
        st.markdown(pred['Description'])
        st.markdown('Click on the image to find out more.')


catalog = DataCatalog.from_config(config, credentials)


#cache function that loads in data
@st.cache
def load_data(data_name):
    data = catalog.load(data_name)
    #can add extra stuff here
    return data


data_load_state = st.text('Loading data from AWS S3...')
data = load_data("osdg_preprocessed_data")
#catalog.save("boats", df)
data_load_state.text("")
#st.dataframe(data)

classifier = catalog.load("sdg_classifier")
vectorizer = catalog.load("vectorizer")

st.sidebar.write("This feature based on natural language processing (NLP) assigns labels to text predicated on Sustainable Development Goals (SDGs).")

st.markdown('# Shell UN SDG Classification')


sdg_imgs = [
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/1.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/2.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/3.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/4.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/5.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/6.jpg?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/7.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/8.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/9.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/10.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/11.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/12.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/13.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/14.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/15.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/16.png?raw=true',
    'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/17.png?raw=true'
]

st.image(sdg_imgs, width=100)



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
    df1 = details.iloc[predicted[0] - 1]
                    
    return display_sdg(df1)
    

#main function

def main():
    #st.title('UN SDG Internship Project')
    #st.markdown('### Text classification')
    st.markdown('##### How do public artifacts reflect Shell\'s contribution towards Sustainable Development Goals?')
    
    message1 = st.text_area("Type in or paste any text segment (e.g. publication excerpt, news article) in the text box below", "Type Here")
    if st.button("Get SDG Label"):
        with st.spinner('Running model...'):
            time.sleep(1)
        #clean_message = clean_text(message1)
        result1 = text_predict(message1)
        #st.success("Success")

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


data_load_state = st.text('Loading data from AWS S3...')
predictions_df = load_data("predictions")

st.write(predictions_df)

st.header('Data Analysis')
st.subheader('Training data set')
st.markdown('##### SDG label distribution')

hist_values = np.histogram(data['sdg'], bins=15, range=(0,16))[0]

st.bar_chart(hist_values)
