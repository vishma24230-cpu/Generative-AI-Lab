from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

def get_relevant(query, documents):
  '''
  INPUT:
  query: the query for which similar vector is to be found from the documents,
  documents: the list of documents from which similar vector to the query is to be found,

  OUTPUT:
  return most similar item in document,  similarity score of most similar item in document
  '''
  embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
  query_embedding=embeddings.embed_query(query)
  document_embeddings=embeddings.embed_documents(documents) 
  similarities=cosine_similarity([query_embedding],document_embeddings)[0]
  similarities=sorted(list(enumerate(similarities)), key= lambda x:x[1])
  return documents[similarities[-1][0]], similarities[-1][1]

if __name__=="__main__":
  documents=[
                   "Virat Kohli is a great cricketer, he's one of the youngest to score a century in test cricket",
                   "Rohit Sharma is the current captain of the Indian cricket team",
                   "MS Dhoni is a former captain of the Indian cricket team, also known as the captain cool or thala",
                   "Sachin Tendulkar is the greatest batsman of all time, he's known as the god of cricket",
                   "Ravindra Jadeja is an all-rounder, he's known for his bowling and batting skills",
                   "Jasprit Bumrah is a fast bowler, he's known for his yorkers",
                   "Mark Zuckerberg is the CEO of Facebook"
  ]
  print(get_relevant("Who is virat kohli", documents))