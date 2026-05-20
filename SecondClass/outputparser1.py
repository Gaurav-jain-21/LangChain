from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm= HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
model= ChatHuggingFace(llm=llm)

#1st prompt-> detailded report
template1 = PromptTemplate(
    template="Write a detailed report on the {topic}.",
    input_variables=['topic']
)

#2nd prompt-> summary of the report
template2= PromptTemplate(
    template="summarize in 5 line of the following {report}",
    input_variables=['report']
)

prompt1= template1.invoke({'topic':'black hole'})
report= model.invoke(prompt1)
prompt2= template2.invoke({'report':report.content})
summary=model.invoke(prompt2)
print(summary.content)
