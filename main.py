import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from streamlit.runtime.state import SessionState, session_state
from get_data import get_relevant

st.write("Data we have:")
documents=[
                 "Virat Kohli is a great cricketer, he's one of the youngest to score a century in test cricket",
                 "Rohit Sharma is the current captain of the Indian cricket team",
                 "MS Dhoni is a former captain of the Indian cricket team, also known as the captain cool or thala",
                 "Sachin Tendulkar is the greatest batsman of all time, he's known as the god of cricket",
                 "Ravindra Jadeja is an all-rounder, he's known for his bowling and batting skills",
                 "Jasprit Bumrah is a fast bowler, he's known for his yorkers",
                 "Mark Zuckerberg is the CEO of Facebook"
]
st.write(documents)

query=st.text_area(label="Enter your question here", value="Your question e.g. What is this document about?")

st.button(label="Submit", on_click=lambda: st.session_state.update({
                 "result": get_relevant(query,documents)
}))


