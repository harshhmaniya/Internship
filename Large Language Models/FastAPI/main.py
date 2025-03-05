import os
import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

bedrock = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1'
)

embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id="amazon.titan-embed-text-v2:0"
)

pc = Pinecone()
index = pc.Index(name='example-01')
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

app = FastAPI()


@app.post("/upload")
async def upload_document(user_id: str, file: UploadFile):
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

    for doc in splitted_data:
        doc.metadata["user_id"] = user_id

    vector_store.add_documents(documents=splitted_data)

    return {"status": "success", "message": "Document processed successfully"}


@app.post("/qa")
async def qa(user_id: str, question: str):
    vector_data = vector_store.from_existing_index(
        index_name="example-01",
        embedding=embeddings
    )

    retriever = vector_data.as_retriever(
        search_kwargs={
            "filter": {"user_id": {"$eq": user_id}},
            "k": 10
        }
    )

    llm = ChatBedrockConverse(
        model="meta.llama3-70b-instruct-v1:0",
        client=bedrock
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', """You are an AI assistant whose responses must be based exclusively on the information contained in the context provided:

                        {context}

                        Instructions:
                        1. When a query is provided, search the above context for the answer.
                        2. If you find relevant information that fully answers the query, provide a detailed and accurate response using that information.
                        3. If the answer to the query is not found within the provided context or the query concerns topics outside their scope, respond with exactly:
                           "NO ANSWER FOUND"  """),
            ('user', 'Question : {input}')
        ]
    )

    document_chain = create_stuff_documents_chain(llm=llm,
                                                  prompt=prompt)

    rag_chain = create_retrieval_chain(retriever=retriever,
                                       combine_docs_chain=document_chain)

    response = rag_chain.invoke({"input": question})

    return {"answer": response["answer"]}
