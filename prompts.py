'''
based on 'prompts' lecture by campusX (text-based only).
'''

from langchain_core.runnables import chain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import streamlit as st

st.header('Document research tool')
# -> static prompts, gives lots of control to user, chamces of misuse and inconvenience to user

# user_input = st.text_input(label="Enter your prompt")

# -> dynamic prompts
template = PromptTemplate(template='''
              You are an academic writing assistant.

              Paper:
              "{paper_title}"

              Task:
              Write a response about the selected paper.

              Tone & Style:
              "{style}"

              Length Requirement:
              "{length}"

              Instructions:
              - Base the content strictly on the chosen paper.
              - Do not mention unrelated research.
              - Match the exact tone specified.
              - Respect the length constraint closely.
              - Use clear, academic language.
              - Do not add headings unless necessary.
              - Avoid bullet points unless the style implies analysis.

              Output:
              Provide only the final text.
''',
                          input_variables=['paper_title', 'style', 'length'])
paper_title = st.selectbox(
    'Paper title', ["Title ...", "Word2vec", "Attention is all you need"])
length = st.selectbox("Length", [
    "Style...", "Short (50-60 words)", "Medium (70-100 words)",
    "Long (150-200 words)"
]),
style = st.selectbox("Style", [
    "Style ...", "Commentary (Positive tone)", "Commentary (Neutral tone)",
    "Commentary (Negatiev tone)", "Critical analysis (neutral tone)",
    "Crititcal analysis (negative tone)"
])
llm = HuggingFaceEndpoint(repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
                          max_new_tokens=300)
model = ChatHuggingFace(llm=llm)

if st.button('Send', shortcut="Enter"):
    chain = template | model
    result = chain.invoke({
        'paper_title': paper_title,
        'length': length,
        'style': style
    })
    st.write(result.content)
