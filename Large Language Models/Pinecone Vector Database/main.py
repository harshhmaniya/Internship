import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
pinecone_api = os.getenv("PINECONE_API_KEY")

llm = ChatOllama(model='llama3.2')

prompt = ChatPromptTemplate.from_messages(
    [
        ('system',"""You are an helpful AI assistant,
        that gives answer based on context after proper thinking and research of user question.
        you are a professional at this thing remember that.
        Context : {context}
        """),
        ('user','Question : {input}')
    ]
)

embeddings = OllamaEmbeddings(model='mxbai-embed-large')


def load_split_docs(file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(file_path=file_path)

    data = loader.load_and_split(
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
    )

    return data

def add_to_database(index_name, embedding_name, documents_to_add):
    pc = Pinecone(api_key=pinecone_api)

    index = pc.Index(index_name)

    database = PineconeVectorStore(
        index=index,
        embedding=embedding_name
    )
    database.add_documents(documents=documents_to_add)

    return database

documents = load_split_docs(
    file_path="press-release-q1fy25.pdf",
    chunk_size=1000,
    chunk_overlap=200
)

vector_store = add_to_database(
    index_name="example-01",
    embedding_name=embeddings,
    documents_to_add=documents
)

retriever = vector_store.as_retriever()

combine_docs = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

retrieval_chain = create_retrieval_chain(
    retriever,
    combine_docs
)

response = retrieval_chain.invoke({"input":"Key Highlights of this document"})
print(response['answer'])
