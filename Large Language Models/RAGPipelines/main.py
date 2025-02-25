from chatbot import ChatBot
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM


if __name__ == "__main__":
    chatbot = ChatBot(llm_model='llama3.2')
    query = "Who is the author of Attention is all you need Research Paper?"
    result = chatbot.ask_question(query)
    print(result)

    human_prompt = ChatPromptTemplate.from_template(
        """
        Based on the following context, provide a clear and human-friendly answer to the question.
        Context: {context}
        Question: {input}
        """
    )

    llm = OllamaLLM(model='llama3.2')

    chain_human = human_prompt | llm
    final_answer = chain_human.invoke({"context": result, "input": query})

    print("Final Answer:", final_answer)
