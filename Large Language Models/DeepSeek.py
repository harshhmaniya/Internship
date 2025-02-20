import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Title for the Streamlit app
st.title("Chat App by Harsh Maniya")

llm = OllamaLLM(model="deepseek-r1")

prompt_template = PromptTemplate.from_template("User: {query}\nDeepSeek R1:")

user_query = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_query:

        formatted_prompt = prompt_template.format(query=user_query)
        response = llm.invoke(formatted_prompt)
        # Display the response in the app
        st.write("Response:", response)
    else:
        st.write("Please enter a query to continue.")