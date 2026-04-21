from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st

doc=st.file_uploader(
  label="Upload file",
  type=["pdf"]
)

db=Chroma(
  collection_name="pdfs",
  embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-001-embedding")
)

if st.button("Submit to DB"):
  