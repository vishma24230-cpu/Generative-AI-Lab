import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import re

template = PromptTemplate(template='''
  You are a helpful assistant. Answer only from the context given.
If you find the context insufficient, answer 'I don't know'.

context: {context}

question: {question}  
  ''',
                          input_variables=["context", "question"])

if "link_provided" not in st.session_state:
  st.session_state.link_provided = False

if not st.session_state.link_provided:
  video_url = st.text_input(label="Please enter your video url:")

if st.button("Summarise"):
  st.session_state.link_provided = True

  # capture video part "?v=()" using regex
  match = re.search(r"watch\?v=([A-Za-z0-9\_\-]*)&", video_url)
  id_param = match.group(1)

  if id_param == None:
    st.error(
        "The input seems incorrect. Please enter the youtube video URL again or try again later."
    )
  else:

    # get the transcript
    text = YoutubeLoader(video_id=id_param)

    # create a vector store for rag
    vec_store = Chroma(collection_name="youtube transcript",
                       embedding_function=GoogleGenerativeAIEmbeddings(
                           model="gemini-001-embedding"))
    # create a retriever for the vector store
    retriever = vec_store.as_retriever(search_type='mmr', kwargs={"k": 30})

    # Add the docs to the vectorstore
    

    model = ChatHuggingFace(repo_id="meta-llama/Llama-3.2-3B-Instruct",
                            max_new_tokens=512,
                            temperature=0.01,
                            repetition_penalty=1.03)

    chat_model = ChatHuggingFace(llm=model)
