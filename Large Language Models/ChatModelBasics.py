from langchain_ollama import ChatOllama

llm = ChatOllama(model='llama3.2')

messages = [{"role": "user",
             "content": "What are the benefits of using chat models?"}]

response = llm.invoke(messages)
print("Assistant:", response.content)

messages.append({"role": "assistant",
                 "content": response.content})
messages.append({"role": "user",
                 "content": "Can you give me an example?"})

# Invoke again with updated messages
response = llm.invoke(messages)
print("Assistant:", response.content)
