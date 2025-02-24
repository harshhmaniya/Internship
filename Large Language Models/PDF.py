from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


file_path = "Sample PDFs/nke-10k-2023.pdf"

loader = PyPDFLoader(file_path)
llm = OllamaLLM(model='llama3.2')
embeddings = OllamaEmbeddings(model='llama3.2')

vector_store = Chroma(
    collection_name='Example_Collection',
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

documents = loader.load()
print(len(documents))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200,
                                               add_start_index=True)

chunks = []
for doc in documents:
    chunks.extend(text_splitter.create_documents([doc.page_content]))

vector_store.add_documents(chunks)


def retrieve(query):
    retrieved_docs = vector_store.similarity_search(query)
    return retrieved_docs


def generate(query, context):
    prompt = {
        "question": query,
        "context": "\n\n".join([doc.page_content for doc in context])
    }
    print("Extracted prompt:", prompt)
    response = llm.invoke(str(prompt))
    return response


# Test the RAG system
query = "What is the main topic of the document?"
retrieved_context = retrieve(query)
response = generate(query, retrieved_context)
print(response)
