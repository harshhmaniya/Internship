from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

file_path = "SCET Internship Policy_2024_25_onwards.pdf"  # pdf file path
embeddings = OllamaEmbeddings(model="llama3.2")  # embedding model


def load_pdf_metadata(file):
    loader = PyPDFLoader(file)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    return pages


pages = load_pdf_metadata(file_path)


def print_metadata(pdf_file):
    print(len(pdf_file))
    print(f"{pdf_file[0].metadata}\n")
    print(pdf_file[0].page_content)


# print_metadata(pages) --> prints pdf data

vector_store = InMemoryVectorStore.from_documents(pages, embeddings)
docs = vector_store.similarity_search("Internship Guidelines", k=3)
for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')