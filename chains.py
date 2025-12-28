from langchain_core import output_parsers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import MergedResult
import streamlit as st
from langchain_core.runnables import RunnableParallel

st.header('Chains')

st.text(
    "This allows us to connect one component to another in a linear fashion.")

st.subheader("1) Chains using `|` (pipe)")

with st.container():
  with st.container():
    st.code(body='''
      prompt = PromptTemplate(template='Tell me about {topic}',
        input_variables=['topic'])

      model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
               kwargs={'max_new_tokens': 100})

      parser = StrOutputParser()

      chain = prompt | model | parser

      chain.invoke({'topic': 'cricket'})
      ''',
            language='python',
            line_numbers=True)

    if st.button('Run'):
      prompt = PromptTemplate(template='Tell me about {topic} in 5 sentences',
                              input_variables=['topic'])

      model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                     model_kwargs={'max_output_tokens': 500})

      parser = StrOutputParser()

      chain = prompt | model | parser

      result = chain.invoke({'topic': 'cricket'})

      st.text(result)

st.subheader("2) Parallel chains")
st.text("Chain 1 and chain 2 runs in parallel.")
with st.container():

  # st.code(body='''
  # ''', language='Python', line_numbers=True)

  essay_prompt = PromptTemplate(template='''
    You are an AI assistant. Generate an essay within int(2000*3/2) words on the topic {topic}.
    ''',
                                input_variables=["topic"])

  quiz_prompt = PromptTemplate(template='''
  You are an AI assistant:
  Generate a quiz based on the topic.

  topic: {topic}
  ''',
                               input_variables=["topic"])

  final_prompt = PromptTemplate(template='''
  You are an AI assistant. Look the the given essay and quiz, and add information to the essay if any quiz questions are unrelated.
  Finally format the final essay and quiz into parts and give the final output.

  essay: 
  {essay}

  quiz: 
  {quiz}
  ''',
                                input_variables=["essay", "quiz"])

  model = ChatGoogleGenerativeAI(model='gemini-2.5-flash',
                                 model_kwargs={"max_output_tokens": 2000})

  parser = StrOutputParser()

  st.text(
      "Enter a topic you want to study. The AI assistant genrates an essay which you can rerad, and then generates a quiz you can answer."
  )

  topic = st.text_area("Enter a topic you want to study.")

  if st.button("Send"):
    if topic.strip() == "":
      st.error("Please enter a valid topic.")
    elif len(topic.strip()) <= 3:
      st.error("Please describe the topic a bit more.")
    else:
      parallel_chain = RunnableParallel({
          "essay": essay_prompt | model | parser,
          "quiz": quiz_prompt | model | parser
      })
      merge_chain = final_prompt | model | parser
      final_chain = parallel_chain | merge_chain | parser

      st.text(final_chain.invoke({"topic": topic}))
