from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7
)

result = llm.invoke("What is the capital of France?")

print(result.content)