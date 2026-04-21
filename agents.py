from langchain.agents import create_agent
from langchain_core.prompt_values import PromptValue
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langsmith import Client

client=Client()

template=hub.pull()
search_tool = DuckDuckGoSearchRun()

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                    max_output_tokens=200)

agent = create_agent(model=chat_model,
                     tools=[search_tool],
                     system_prompt='You are a helpful assistant.')

result = agent.invoke(
    {"messages": [{
        "role": "user",
        "content": "price of doge"
    }]})