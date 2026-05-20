from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()
chat_template = ChatPromptTemplate([
    ('system','you are helpful {domain} assistant'),
    ('human','Explain in simple terms, what is {topic}?')
])

prompt= chat_template.invoke({'domain':"cricket", 'topic':"LBW"})

print(prompt)