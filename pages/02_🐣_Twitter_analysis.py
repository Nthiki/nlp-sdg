import streamlit as st

st.markdown("# Twitter analysis")
st.sidebar.markdown("# Twitter analysis")

#update these colms using real time twitter data

col1, col2, col3 = st.columns(3)
col1.metric("Sentiment", "Positive", "4%")
col2.metric("Number of tweets", "1,340", "202")
col3.metric("Mentions", "839", "3")