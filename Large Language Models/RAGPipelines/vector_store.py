from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class VectorStore:
    def __init__(self, documents, LLM_Model):
        self.LLM_Model = LLM_Model
        self.documents = documents
        self.embeddings = OllamaEmbeddings(model=self.LLM_Model)
        self.vector_store = Chroma(embedding_function=self.embeddings)
        self.vector_store.add_documents(documents)

    def get_retriever(self):
        return self.vector_store.as_retriever()
