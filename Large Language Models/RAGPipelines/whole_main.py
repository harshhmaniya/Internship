from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Initialize the LLM and its embeddings
llm = OllamaLLM(model="deepseek-r1")
embeddings = OllamaEmbeddings(model="deepseek-r1")

# Load the PDF document and split it into text chunks
loader = PyPDFLoader(file_path="NIPS-2017-attention-is-all-you-need-Paper.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
splitted_docs = text_splitter.split_documents(documents)

# Initialize the vector store and add the document chunks
vector_store = Chroma(embedding_function=embeddings)
vector_store.add_documents(splitted_docs)

# Create a prompt for the retrieval chain
prompt = ChatPromptTemplate.from_template(
    """
    Answer The Following question only on provided context.
    Think step by step before giving answer.
    you will get rewarded if the answer is correct
    Context : {context}
    Question : {input}
    """
)

# Create a retriever from the vector store
retriever = vector_store.as_retriever()

# Create the document combination chain and retrieval chain
stuff_doc_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, stuff_doc_chain)

# Define the user query and invoke the retrieval chain to get the context
query = "Who is the author of Attention is all you need Research Paper"
result = retrieval_chain.invoke({"input": query}).get("context", "No relevant context was found.")
print(result)


# Compose the LLM and human prompt using the pipe operator and invoke the chain
chain = prompt | llm
final_answer = chain.invoke({"context": result, "input": query})

print("Final Answer:", final_answer)
