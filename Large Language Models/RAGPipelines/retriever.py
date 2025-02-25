from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM


class Retriever:
    def __init__(self, retriever, llm_model):
        self.llm_model = llm_model
        self.llm = OllamaLLM(model=self.llm_model)
        self.prompt = ChatPromptTemplate.from_template(
            """
            Answer The Following question only on provided context.
            Think step by step before giving answer.
            You will get rewarded if the answer is correct.
            Context: {context}
            Question: {input}
            """
        )
        self.stuff_doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, self.stuff_doc_chain)

    def get_answer(self, query):
        return self.retrieval_chain.invoke({"input": query})
