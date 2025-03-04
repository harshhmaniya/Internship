import os
from langchain_aws import ChatBedrockConverse # , BedrockEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import boto3

load_dotenv()
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

bedrock = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1'
)


llm = ChatBedrockConverse(
    model="meta.llama3-70b-instruct-v1:0",
    client=bedrock,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# embeddings = BedrockEmbeddings(
#     client=bedrock,
#     model_id="amazon.titan-embed-text-v2:0"
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """You are a helpful AI Assistant that gives user answer based on the provided context.
#             You first analyze the user query and then provide the answer based on the context.
#             context : {context}"""
#         ),
#         ("human", "User query : {input}")
#     ]
# )

response = llm.invoke("Testing")
print(response.content)