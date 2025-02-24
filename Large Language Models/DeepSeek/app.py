from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

st.title("Chat with Deepseek-R1")
input_text = st.text_input("Enter Prompt")


llm = ChatOllama(model='deepseek-r1')

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant, respond to user kindly and formally."),
        ("user", "Question:{question}")
    ]
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
