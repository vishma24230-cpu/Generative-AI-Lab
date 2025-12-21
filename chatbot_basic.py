'''
In this document we try to create a simple chatbot using ChatPromptTemplate and streamlit.
'''

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

template = ChatPromptTemplate([
    ("system", "You are a helpful AI bot. Your name is Starlight."),
    ("placeholder", "{conversation}")
])

st.header("Simple chatbot")

# configuring llm and chatmodel
def chat():
    llm = HuggingFaceEndpoint(repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
                          max_new_tokens=500)
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

if "chat_model" not in st.session_state:
    st.session_state.chat_model=chat()

if "conversation" not in st.session_state:
    st.session_state.conversation=[]
else:
    st.write("Chat history:")
    with st.container(height=120):
        st.write(st.session_state.conversation)

        
prompt = None
user_chat_msg = st.text_input(label="Your message:")
if st.button("Send"):
    st.session_state.conversation.append(HumanMessage(user_chat_msg))
    prompt = template.invoke(
        {"conversation": st.session_state.conversation})
    # st.write(prompt)
    result=st.session_state.chat_model.invoke(prompt)
    st.session_state.conversation.append(AIMessage(result.content))
    st.write(result.content)
    
