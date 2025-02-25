from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        chunk_size = 1000
        chunk_overlap = 200
        loader = PyPDFLoader(file_path=self.file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )

        return text_splitter.split_documents(documents)
