from src.config import PDF_FILE_PATH, MODEL_NAME, PERSIST_DIRECTORY, COLLECTION_NAME
from src.document_loader import DocumentLoader
from src.rag_system import RAGSystem

def main():
    # Initialize the document loader and load/split documents.
    doc_loader = DocumentLoader(PDF_FILE_PATH)
    documents = doc_loader.load_and_split()
    print(f"Loaded {len(documents)} document chunks.")

    # Initialize the RAG system.
    rag_system = RAGSystem(
        model=MODEL_NAME,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )

    # Add the document chunks to the vector store.
    rag_system.add_documents(documents)

    # Run a sample query.
    query = "What is the main topic of the document?"
    retrieved_context = rag_system.retrieve(query)
    response = rag_system.generate(query, retrieved_context)
    print("Response:", response)

if __name__ == '__main__':
    main()