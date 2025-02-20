from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

llm = OllamaLLM(model = "llama3.2")
#response = llm.invoke('Who are you?')
#print(response)

prompt_temp = PromptTemplate.from_template("Tell me about {topic}")
real_prompt = prompt_temp.format(topic = "cat")

response = llm.invoke(real_prompt)
print(response)