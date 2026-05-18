from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
 
load_dotenv()
llm=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",task="text-generation")
model=ChatHuggingFace(llm=llm)
chat_history=[
    SystemMessage(content="you are a helpful assistant")
]
while True:
    user_input= input("you: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input== "exit":
        break
    result= model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:",result.content)
print(chat_history)    
