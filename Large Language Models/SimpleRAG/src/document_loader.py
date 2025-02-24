from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.file_path = file_path
        self.loader = PyPDFLoader(file_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )

    def load_and_split(self):
        """Loads the PDF and splits it into text chunks."""
        documents = self.loader.load()
        chunks = []
        for doc in documents:
            chunks.extend(self.text_splitter.create_documents([doc.page_content]))
        return chunks