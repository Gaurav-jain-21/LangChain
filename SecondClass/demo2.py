from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

llm= HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
model= ChatHuggingFace(llm=llm)

class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "The overall sentiment of the review, e.g., positive, negative, neutral"]


structured_output =model.with_structured_output(Review)
result=structured_output.invoke("""
the hardware is great, but the software is feels bloated. there are too many pre-installed apps that i can't remove. also the ui looks outdated compared to other brands. Hoping for a software update to fix this.
""")
print(result)