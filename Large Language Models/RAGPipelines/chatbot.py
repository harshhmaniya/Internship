from pdf_processor import PDFProcessor
from vector_store import VectorStore
from retriever import Retriever


class ChatBot:
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.pdf_processor = PDFProcessor(file_path="NIPS-2017-attention-is-all-you-need-Paper.pdf")
        self.documents = self.pdf_processor.load_and_split()
        self.vector_store = VectorStore(self.documents, self.llm_model)
        self.retriever = Retriever(self.vector_store.get_retriever(), self.llm_model)

    def ask_question(self, query):
        return self.retriever.get_answer(query)
