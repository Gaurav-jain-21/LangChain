from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
chat_template= ChatPromptTemplate([
    ('system', "You are a helpful assistant that provides information about {domain}."),
    ('human', "Can you tell me about {topic} in {domain}?"),
])

prompt= chat_template.invoke({'domain':'cricket', 'topic':"Dusra"})

print(prompt)