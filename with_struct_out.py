from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, TypedDict, Annotated
from pydantic import BaseModel
import streamlit as st

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.header('Structuring result')

st.write('1) Typing library (using TypedDict)')

st.write('''
Our class looks like:

class Data(TypedDict):
Nation: Annotated[str, "Name the country involved"]
Capital: Annotated[Optional[str], "Extract the name of the country"]
Name: Annotated[Optional[str], "Name of the person who asked the question"]

We pass the class with model.with_structured_output() as a parameter.
''')


class Data(TypedDict):

  Nation: Annotated[str, "Name the country involved"]
  Capital: Annotated[Optional[str], "Extract the name of the country"]
  Name: Annotated[Optional[str], "Name of the person who asked the question"]


result = chat_model.with_structured_output(Data).invoke("capital of india")
st.write("Result :", result)

st.write("2) With Pydantic library")

st.write('''
Our class looks like

class StructData(BaseModel):
Nation: str
Capital: Optional[str]

like before we pass the class to the model.with_structured_output

''')


class StructData(BaseModel):

  Nation: str
  Capital: Optional[str]


res = chat_model.with_structured_output(StructData).invoke("capital of india")

st.write("result :", res)
