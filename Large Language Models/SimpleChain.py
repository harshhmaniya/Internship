from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

prompt_temp = ChatPromptTemplate.from_messages(
    [
        ('system',
         'You are an AI agent Called {agent_name} Helper designed by {developer_name} and you help everyone'),
        ('human',
         '{query}')
    ]
)

chain = prompt_temp | llm
response = chain.invoke(
    {
        'agent_name': 'Maniya Bot',
        'developer_name': 'Harsh Maniya',
        'query': 'Who are you?'
    }
).content
print(response)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human",
         "{input}")
    ]
)

chain_2 = prompt | llm
response = chain_2.invoke(
    {
        "input_language": "English",
        "output_language": "Hindi",
        "input": "I love programming.",
    }
).content
print(response)
