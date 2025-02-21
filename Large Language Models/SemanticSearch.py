from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Step-1 Load PDF file page by page
file_path = "Sample PDFs/nke-10k-2023.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()
print("Total Pages :", len(docs))

# Step-2 Split Contents of Pages by Finite Number of Characters

# In-our case Splitting is done by every 1000 Characters, overlap is 200 characters
# Index of first char in split is added to metadat
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200,
                                               add_start_index=True)

splitted_doc = text_splitter.split_documents(docs)
print("Total Generated Splits :", len(splitted_doc))


# Alternatively Loading and splitting can be done in 1 step
''' loader = PyPDFLoader(file_path)
docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000,
                                                            chunk_overlap=200,
                                                            add_start_index=True))
print("Total Generated Splits :", len(docs)) '''

# Step-3 Convert docs into vectors to perform semantic search
embeddings = OllamaEmbeddings(model="llama3.2")

# Checking if generated vectors is of same size
vector_1 = embeddings.embed_query(splitted_doc[0].page_content)
vector_2 = embeddings.embed_query(splitted_doc[1].page_content)
assert len(vector_2) == len(vector_1), "Not correctly embedded"

# Size of embedded Vectors
print("Length of Generated Vectors :", len(vector_1))


# Step-4 Storing all this in a vector store
vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=splitted_doc)

# Step-5 Performing Semantic Search

# Return documents based on similarity with query
results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")
print("Document Retrieved with Similarity Search -->")
print(results[0])

# Return documents with similarity score
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print("Document Retrieved with Similarity Search with Score -->")
print("Score :", score)
print(doc)

# Return documents based on similarity to an embedded query
embedding_query = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding_query)
print("Document Retrieved with Similarity Search with Embedded Query -->")
print(results[0])
