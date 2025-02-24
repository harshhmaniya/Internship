from langchain_ollama import OllamaLLM, OllamaEmbeddings
from src.vector_store_manager import VectorStoreManager

class RAGSystem:
    def __init__(self, model: str, persist_directory: str, collection_name: str):
        self.llm = OllamaLLM(model=model)
        self.embeddings = OllamaEmbeddings(model=model)
        self.vector_store_manager = VectorStoreManager(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def add_documents(self, documents):
        """Add document chunks to the vector store."""
        self.vector_store_manager.add_documents(documents)

    def retrieve(self, query: str):
        """Retrieve relevant document chunks for the query."""
        return self.vector_store_manager.similarity_search(query)

    def generate(self, query: str, context) -> str:
        """
        Build a prompt from the query and retrieved context,
        then generate a response using the LLM.
        """
        prompt = {
            "question": query,
            "context": "\n\n".join([doc.page_content for doc in context])
        }
        print("Extracted prompt:", prompt)
        response = self.llm.invoke(str(prompt))
        return response