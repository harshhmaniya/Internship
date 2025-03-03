import os
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()


embeddings = OllamaEmbeddings(model='mxbai-embed-large')

pinecone_api = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api)
index = pc.Index(name='example-01')
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

app = FastAPI()

@app.post("/upload")
async def upload_document(user_id : str, file : UploadFile):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    loader = PyPDFLoader(file_path=tmp_path)
    splitted_data = loader.load_and_split(
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
    )

    os.remove(tmp_path)

    vector_store.add_documents(documents=splitted_data,
                               namespace=user_id,)

    return {"status": "success", "message": "Document processed successfully"}


@app.post("/qa")
async def qa(user_id : str, question : str):
    vector_data = vector_store.from_existing_index(
        index_name="example-01",
        namespace=user_id,
        embedding=embeddings
    )

    retriever = vector_data.as_retriever()

    llm = ChatOllama(model='llama3.2')

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', """You are an helpful AI assistant,
            that gives answer based on context after proper thinking and research of user question.
            you are a professional at this thing remember that.
            Context : {context}
            """),
            ('user', 'Question : {input}')
        ]
    )

    document_chain = create_stuff_documents_chain(llm=llm,
                                                  prompt=prompt)

    rag_chain = create_retrieval_chain(retriever=retriever,
                                       combine_docs_chain=document_chain)

    response = rag_chain.invoke({"input": question})

    return {"answer": response["answer"]}
