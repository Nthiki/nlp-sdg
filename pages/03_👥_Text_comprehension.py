import streamlit as st
import time
import torch
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from allennlp.predictors.predictor import Predictor
import pandas as pd
import base64
import uuid
import re
import json
import pandas as pd
import plotly.express as px
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
from geopy.extra.rate_limiter import RateLimiter



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    From: https://discuss.streamlit.io/t/a-download-button-with-custom-css/4220
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    # if pickle_it:
    #    try:
    #        object_to_download = pickle.dumps(object_to_download)
    #    except pickle.PicklingError as e:
    #        st.write(e)
    #        return None

    # if:
    if isinstance(object_to_download, bytes):
        pass

    elif isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
        custom_css
        + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
    )
    # dl_link = f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}"><input type="button" kind="primary" value="{button_text}"></a><br></br>'

    st.markdown(dl_link, unsafe_allow_html=True)

# load data
@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv('locations.csv')
    # data = data.head(10)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

@st.cache(allow_output_mutation=True)
def geo_code(data):
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)
    data["loc"] = data["location"].apply(geocode)
    data["point"]= data["loc"].apply(lambda loc: tuple(loc.point) if loc else None) 
    data[['lat', 'lon', 'altitude']] = pd.DataFrame(data['point'].to_list(), index=data.index)
    return data

@st.cache(allow_output_mutation=True)
def load_predictor():
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz")
    return predictor

@st.cache(allow_output_mutation=True)
def load_model():
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    return t5_model

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    return t5_tokenizer


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
if page_selection == "Q&A":
            st.title('Shell: Associated Questions & Answers')

            predictor = load_predictor()

            # Create a text area to input the passage.
            passage = st.text_area("Passage", """Europe’s largest oil refinery suffered a malfunction, a potential source of jitters for the continent’s refined fuels market where supply has already been hit by industrial action.

            The compressor of fluid catalytic cracker unit 2 tripped on Oct. 12 due to the loss of power supply, according to a fire safety alert from the region’s Rjinmond Veilig service. Known as FCC units, the conversion plants are typically used to make refined products such as gasoline.

            Shell Plc’s Pernis plant near Rotterdam has been flaring elevated amounts of gas following the incident, triggering 200 complaints from the public, DCMR, an environmental regulator, said in a notice on its website. The plant is also an important source of diesel within Europe.

            The continent can ill afford material disruption to refined petroleum supply, given a European Union ban on purchases from Russia that’s due to start in early February. Strikes over pay in France have knocked out a swath of the nation’s fuelmaking, crunching supply.

            BP Plc is carrying out planned work on the FCC at its Rotterdam refinery, which is next to Pernis in Europe in terms of size.

            Read our blog on the European energy crisis

            Shell said in a statement that governments have been informed about the incident, but didn’t elaborate on what processing capacity was affected or what it would mean for fuel supply.

            “I can only tell you that we expect the nuisance will continue for the time being and that try to minimize the nuisance for the people in the vicinity,” a Shell spokesman said.

            — With assistance by April Roach, Rachel Graham and Jack Wittels""")

            # Create a text input to input the question.
            question = st.text_input("Question", "What technology was used?")

            # Use the predictor to find the answer.
            result = predictor.predict(question, passage)

            # From the result, we want "best_span", "question_tokens", and "passage_tokens"
            start, end = result["best_span"]
            question_tokens = result["question_tokens"]
            passage_tokens = result["passage_tokens"]

            # We want to render the paragraph with the answer highlighted.
            # We'll do that using `st.markdown`. In particular, for each token
            # if it's part of the answer span we'll **bold** it. Otherwise we'll
            # leave it as it.
            mds = [f"**{token}**" if start <= i <= end else token
                for i, token in enumerate(passage_tokens)]

            # And then we'll just concatenate them with spaces.
            if st.button("Find Answer"):
                st.write("Answer : "+result["best_span_str"])
                st.markdown(" ".join(mds))

            title_SOre = """
            <div style="padding:10px;border-radius:10px;margin:10px;border-style:solid; border-color:#000000; padding: 1em;">
            <h3 style="color:black;text-align:center;">Click below to use a data frame that contains news stories on the Shell Corporation from various news sources. Or upload yours</h3>
            """
            st.markdown(title_SOre, unsafe_allow_html=True)

            c29, c30, c31 = st.columns([1, 6, 1])

            with c30:

                uploaded_file = st.file_uploader(
                    "",
                    key="1",
                    help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
                )

                if uploaded_file is not None:
                    file_container = st.expander("Check your uploaded .csv")
                    shows = pd.read_csv(uploaded_file)
                    # shows = shows.drop_duplicates(subset=['text'], keep='last')
                    # shows = shows.dropna(subset=['text'], inplace=True)
                    # shows = shows.reset_index(inplace=True)
                    uploaded_file.seek(0)
                    file_container.write(shows)

                else:
                    st.info(
                        f"""
                            👆 Upload a .csv file first. Sample to try: [ShellData.csv](https://drive.google.com/u/0/uc?id=1wjj2U6zEyoNOvVdlo48ElHR5b4bMO5QK&export=download)
                            """
                    )

                    st.stop()
            option = st.selectbox('Select Your questions:',('what is the cost and monetary value', 'what technology was used?', 'what resources used?'))
            st.write('You selected:', option)
            if st.button("Answer question"):

                def qesAns(df,questions = option):
                    # df = df.head(5)
                    # df = df.drop_duplicates(subset=['text'], keep='last')
                    # df = df.dropna(subset=['text'], inplace=True)
                    # df = df.reset_index(inplace=True)
                    # """question and answer node meant to produce answers to artiles in a dataframe

                    # Args:
                    #     data: Data containing a text column.
                    # Returns:
                    #     data: a dataframe answering the asked question based on the articles in each row 
                    # """
                    def qestions(df,column,question):
                        result = []
                        for i in range(len(df[column])):
                            result.append(predictor.predict(passage=df[column][i], question=question)["best_span_str"])
                        return result

                    df[questions] = qestions(df,column= "text", question = questions)
                    return df[questions]
                res = qesAns(shows,questions= option)
                st.write(res)
                df = pd.DataFrame(res)

                c29, c33, c31 = st.columns([1, 1, 2])

                with c29:

                    CSVButton = download_button(
                        df,
                        "Summarized_Dataframe.csv",
                        "Download to CSV",
                    )
                st.stop()
if page_selection == "Entity recognition":
        st.title('Shell: Associated Locations & Organizations')

        locations, organizations= st.tabs(["Locations", "Organizations"])

        with locations:


        
            #function to get longitude and latitude data from country name
            geolocator = Nominatim(user_agent="http")    
            #load data
            data = load_data()
            data = geo_code(data)
            form = st.form("location")
            locationsw=form.multiselect(label='Select one or more locations', options=data['location'].to_list())
            pick_all_locations = form.checkbox(' or all locations')

            # Now add a submit button to the form:
            form.form_submit_button("Submit")

            map_osm = folium.Map()
            if pick_all_locations:
                data = data[['lat', 'lon', 'counts', 'location']]
            else:
                data = data[['lat', 'lon', 'counts', 'location']].isin(locationsw)
                

            data.apply(lambda row:folium.CircleMarker(location=[row[['lat']], row[['lon']]], 
                                                        radius=row['counts'], popup=row['location'])
                                                        .add_to(map_osm), axis=1)   

            folium_static(map_osm)



        with organizations:
            # load data
            @st.cache(allow_output_mutation=True)
            def load_data():
                data = pd.read_csv('organizations.csv')
                lowercase = lambda x: str(x).lower()
                data.rename(lowercase, axis='columns', inplace=True)
                return data    
            
            data = load_data()

            selection = st.radio(
                "Would you like to see the most mentioned or the least mentioned organizations?",
                ('Most Mentioned', 'Least Mentioned'))

            if selection == 'Most Mentioned':
                data = data.sort_values(by=['counts'], ascending = [False])
            else:
                data = data.sort_values(by=['counts'])

            organization_no=st.slider(label='Select number of organizations to display.', min_value=1, max_value=len(data))
            
            

            data = data.head(organization_no)

            fig = px.bar(data, x=data["counts"], y=data["organization"], orientation='h', color="organization")

            st.plotly_chart(fig)
                    
# else:
#     st.markdown("hello")