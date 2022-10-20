import streamlit as st
import time
import pandas as pd
import torch
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

import plotly.express as px
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
from geopy.extra.rate_limiter import RateLimiter


from kedro.io import DataCatalog
import yaml
from kedro.config import ConfigLoader


from kedro.extras.datasets.pickle import PickleDataSet

from kedro.extras.datasets.pandas import (
    CSVDataSet,
    SQLTableDataSet,
    SQLQueryDataSet,
    ParquetDataSet,
)

config = {
    "locations": {
        "type": "pandas.CSVDataSet",
        "filepath": "s3://internship-sdg-2022/kedro/data/03_primary/locations.csv",
        "credentials": "s3_credentials",
        "load_args": {
            "sep": ','
        }
    },
    "organizations": {
        "type": "pandas.CSVDataSet",
        "filepath": "s3://internship-sdg-2022/kedro/data/03_primary/organizations.csv",
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache(allow_output_mutation=True)
def load_model():
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    return t5_model

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    return t5_tokenizer


@st.cache(allow_output_mutation = True)
def load_data_locations():
    data = catalog.load("locations")
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #can add extra stuff here
    return data

@st.cache(allow_output_mutation = True)
def load_data_organisations():
    data = catalog.load("organizations")
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #can add extra stuff here
    return data

data_locations = load_data_locations()
data_org = load_data_organisations

@st.cache(allow_output_mutation=True)
def geo_code(data):
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)
    data["loc"] = data["location"].apply(geocode)
    data["point"]= data["loc"].apply(lambda loc: tuple(loc.point) if loc else None) 
    data[['lat', 'lon', 'altitude']] = pd.DataFrame(data['point'].to_list(), index=data.index)
    return data

def run_model(input_text, min=100, max=200):

    t5_model = load_model()
    t5_tokenizer = load_tokenizer()
    input_text = str(input_text).replace('\n', '')
    input_text = ' '.join(input_text.split())
    input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
    summary_task = torch.tensor([[21603, 10]]).to(device)
    input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
    summary_ids = t5_model.generate(input_tokenized, min_length=min,
                                                    max_length=max)
    output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    st.write('Summary')

    return output[0]



st.markdown("# Text comprehension")
st.sidebar.markdown("# Text comprehension")

page_options = ["Text summarisation","Q&A","Entity recognition"]
page_selection = st.sidebar.selectbox("Choose functionality", page_options)
    
if page_selection == "Text summarisation":

    st.subheader("Text summarisation")

    max = st.sidebar.slider('Select max', 100, 400, step=10, value=150)
    min = st.sidebar.slider('Select min', 10, 100, step=10, value=50)


    message1 = st.text_area("Type in or paste any text segment", "Type Here")
    if st.button("Summarize text"):
        with st.spinner('Running model...'):
            time.sleep(1)
        result1 = run_model(message1, min, max)
        st.write(result1)

elif page_selection == "Entity recognition":

    locations, organizations= st.tabs(["Locations", "Organizations"])

    with locations:
    # load data


        #function to get longitude and latitude data from country name
        geolocator = Nominatim(user_agent="http")    
        #load data
        #data = load_data("locations")
        st.dataframe(data_locations)
        data = geo_code(data_locations)
        form = st.form("location")
        locations=form.multiselect(label='Select one or more locations', options=data['location'].to_list())
        pick_all_locations = form.checkbox(' or all locations')

        # Now add a submit button to the form:
        form.form_submit_button("Submit")

        map_osm = folium.Map()
        if pick_all_locations:
            data = data[['lat', 'lon', 'counts', 'location']]
        else:
            data = data[['lat', 'lon', 'counts', 'location']].isin(locations)


        data.apply(lambda row:folium.CircleMarker(location=[row[['lat']], row[['lon']]], 
                                                    radius=row['counts'], popup=row['location'])
                                                    .add_to(map_osm), axis=1)   

        folium_static(map_osm)



    with organizations:
        # load data
        #data = load_data("organizations")

        selection = st.radio(
            "Would you like to see the most mentioned or the least mentioned organizations?",
            ('Most Mentioned', 'Least Mentioned'))

        if selection == 'Most Mentioned':
            data = data_org.sort_values(by=['counts'], ascending = [False])
        else:
            data = data.sort_values(by=['counts'])

        organization_no=st.slider(label='Select number of organizations to display.', min_value=1, max_value=len(data))



        data = data.head(organization_no)

        fig = px.bar(data, x=data["counts"], y=data["organization"], orientation='h', color="organization")

        st.plotly_chart(fig)



else:
    st.markdown("hello")