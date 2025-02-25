from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain import hub
from langchain.agents import AgentExecutor
import warnings
warnings.filterwarnings('ignore')

# Define LLM
llm = ChatOllama(model='llama3.2')

# Wikipedia Wrapper
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1,
                                       doc_content_chars_max=200)
# Wikipedia Tool
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

# Arxiv Wrapper
arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1,
                                    doc_content_chars_max=200)
# Arxhiv Tool
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

# WebBaseLoader to Scrap Langsmith Website
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()

# split into documents of 1000 characters
documents = RecursiveCharacterTextSplitter(
    chunk_overlap=200,
    chunk_size=1000,
    add_start_index=True
).split_documents(docs)

# Create Chroma Vector Store and Add Embedded Documents
vector_db = Chroma.from_documents(embedding=OllamaEmbeddings(model='llama3.2'),
                                  documents=documents)

# retriever to retrieve embedded documents
retriever = vector_db.as_retriever()

# making tool out of it
retrieval_tool = create_retriever_tool(retriever=retriever,
                                       name="LangSmith Search",
                                       description="Search Langsmith Documentation and Provide Accurate Answers.")

# Defining all the tools
# First Agent will use Arxiv then Wikipedia then Retriever Tool
tools = [arxiv_tool, wiki_tool, retrieval_tool]

# Pulling prompt from langchain hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Defining Agent
agent = create_openai_tools_agent(llm=llm,
                                  tools=tools,
                                  prompt=prompt)

# Define Executes for Agent that executes prompts
agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               verbose=True)

# Invoking Agent Executor
answer = agent_executor.invoke({"input": "What is Attention Mechanism?"})
print(answer['output'])
