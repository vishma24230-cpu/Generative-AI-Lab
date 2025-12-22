from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI

embed_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# MAKE A NEW STORE BEFORE PASSING THE DOCUMENTS
# store = Chroma(collection_name="coll", embedding_function=embed_model)

docs = list()
docs.append(
    Document(page_content='virat kohli is a cricketer',
             metadata={'author': 'cricbuzz'}))
docs.append(
    Document(page_content='rohit sharma is a cricketer',
             metadata={'author': 'cricbuzz'}))
docs.append(
    Document(page_content='sachin tendulkar is a cricketer',
             metadata={'author': 'cricbuzz'}))
docs.append(
    Document(
        page_content=
        'dhoni is a cricketer, dhoni is also what we call the rich in bengali and it has been a long tradition to tease people as `dhoni manush` to annoy people about their lavish spending habits.',
        metadata={'author': 'cricketinfo'}))
docs.append(
    Document(
        page_content=
        'dhoni is a captain, he has had an injury several times but still came out strong. Theres a meme going around about Dhoni coming to save CSK even after death.',
        metadata={'author': 'cricketinfo'}))
# ADD DOCUMENTS TO THE STORE
# store.add_documents(docs)

# GET K MOST SIMILAR ITEMS FROM THE STORE BASED ON THE QUERY
# print(store.similarity_search('who is dhoni', k=2))

# GET K MOST SIMILAR ITEMS FROM THE STORE BASED ON THE QUERY WITH THEIR RELEVANCE SCORES
# print(store.similarity_search_with_relevance_scores('who is dhoni', k=2))

# CREATE A STORE USING .from_documents()
store = Chroma.from_documents(documents=docs,
                              embedding=embed_model,
                              collection_name="cricketers")

retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', max_tokens=50)
multiquery_ret = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

context_comp_ret = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(llm=llm),
    base_retriever=retriever)

# print(retriever.invoke("who is dhoni"))

print("Result for multiquery retriever")
print(multiquery_ret.invoke("Who is MS Dhoni"))
print("=" * 10)
print("Result for context compression retriever")
print(context_comp_ret.invoke("Who is MS Dhoni"))
