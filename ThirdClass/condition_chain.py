from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()

llm1= HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
model1= ChatHuggingFace(llm=llm1)
parser= StrOutputParser()
class Feedback(BaseModel):
    sentiment: Literal["positive","negative"]=Field(description="Give the sentiment of the feedback")
parser2=PydanticOutputParser(pydantic_object=Feedback)   

prompt1= PromptTemplate(
    template='classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables= {'format_instruction':parser2.get_format_instructions}
)
 
classifier_chain = prompt1 | model1 | parser2

prompt2= PromptTemplate(
    template 
)

branch_chain= RunnableBranch(
    (condition, chain),
    (condition, chain)
)
