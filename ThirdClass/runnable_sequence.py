from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
load_dotenv()

prompt1= PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)
prompt2= PromptTemplate(
    template="explain the following joke - {text}",
    input_variables=['text']
)
llm1= HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
model= ChatHuggingFace(llm=llm1)

parser=StrOutputParser()

chain= RunnableSequence(prompt1, model, parser, prompt2, model, parser)
print(chain.invoke({'topic':"AI"}))