'''
In this document we understand the basics of the similarity scores rag architecture uses to get relevant data from a list of documents. The actual implementation of RAG will be done in a later python file.

RAG is based on three things:

# A Query vector -> The query vector for which we want to find similar documents in vector space.
# A vector db -> a DB where the information e.g documents have been converted to vectors are stored.
# A retriever -> The retriever matches the query vector to the vector db's documents and gets the most similar vectors.

In this python file we have made a simple simulation of this by

# Taking documents (list of strings) -> converting them to vector space -> returns a list of vector representations of all the documents
# Take a query -> convert it to vector representation -> Find vector from list of vectors most similar to it -> return that vector and its similarity score.

For similarity score, we use cosine_similarity here.

Made a visual representation of the idea using streamlit.

Run using:
streamlit run rag_basics.py
 '''

from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from streamlit.runtime.state import session_state


def get_relevant(query, documents):
  '''
  INPUT:
  query: the query for which similar vector is to be found from the documents,
  documents: the list of documents from which similar vector to the query is to be found,

  OUTPUT:
  return most similar item in document,  similarity score of most similar item in document
  '''
  embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
  query_embedding = embeddings.embed_query(query)
  document_embeddings = embeddings.embed_documents(documents)
  similarities = cosine_similarity([query_embedding], document_embeddings)[0]
  similarities = sorted(list(enumerate(similarities)), key=lambda x: x[1])
  return documents[similarities[-1][0]], similarities[-1][1]


def test_fun():
  documents = [
      "Virat Kohli is a great cricketer, he's one of the youngest to score a century in test cricket",
      "Rohit Sharma is the current captain of the Indian cricket team",
      "MS Dhoni is a former captain of the Indian cricket team, also known as the captain cool or thala",
      "Sachin Tendulkar is the greatest batsman of all time, he's known as the god of cricket",
      "Ravindra Jadeja is an all-rounder, he's known for his bowling and batting skills",
      "Jasprit Bumrah is a fast bowler, he's known for his yorkers",
      "Mark Zuckerberg is the CEO of Facebook"
  ]
  print(get_relevant("Who is virat kohli", documents))


if __name__ == "__main__":
  # test_fun() -> uncomment and check input to test the fgun
  st.write("Data we have:")
  documents = [
      "Virat Kohli is a great cricketer, he's one of the youngest to score a century in test cricket",
      "Rohit Sharma is the current captain of the Indian cricket team",
      "MS Dhoni is a former captain of the Indian cricket team, also known as the captain cool or thala",
      "Sachin Tendulkar is the greatest batsman of all time, he's known as the god of cricket",
      "Ravindra Jadeja is an all-rounder, he's known for his bowling and batting skills",
      "Jasprit Bumrah is a fast bowler, he's known for his yorkers",
      "Mark Zuckerberg is the CEO of Facebook"
  ]
  st.write(documents)

  query = st.text_area(label="Enter your question here",
                       value="Your question e.g. What is this document about?")

  if "result" not in st.session_state:
    st.session_state.result = ""

  st.button(label="Submit",
            on_click=lambda: st.session_state.update(
                {"result": get_relevant(query, documents)}))
  st.button(label="reset",
            on_click=lambda: st.session_state.update({"result": ""}))
  st.write(f"Most relevant vector: {st.session_state.result}")
