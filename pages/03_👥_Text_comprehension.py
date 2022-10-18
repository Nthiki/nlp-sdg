import streamlit as st
import time
import torch
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                
else:
    st.markdown("hello")