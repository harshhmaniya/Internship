import os
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


llm = ChatOllama(model='llama3.2')

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful AI Assistant. Follow these guidelines:
            1. First search user input in google search.
            2. If you get any linkedin profile link then return the first link with Name.
            4. if you can't get any profile link then tell user that you cannot find linkedin profile.

            Remember to choose appropriate tool based on user query."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tavily = TavilySearchResults()

tools = [tavily]

agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

response = agent_executor.invoke(
    {
        "input": "Harsh Maniya",
    }
)
print(response['output'])
