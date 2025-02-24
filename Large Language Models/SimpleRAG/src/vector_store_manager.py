from langchain_chroma import Chroma

class VectorStoreManager:
    def __init__(self, collection_name: str, embedding_function, persist_directory: str):
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )

    def add_documents(self, documents):
        """Adds a list of documents (chunks) to the vector store."""
        self.vector_store.add_documents(documents)

    def similarity_search(self, query: str):
        """Performs similarity search on the vector store."""
        return self.vector_store.similarity_search(query)