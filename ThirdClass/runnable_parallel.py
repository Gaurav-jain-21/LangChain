from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

prompt1= PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables= ['topic']
)
prompt2=PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()
llm1= HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
model1= ChatHuggingFace(llm=llm1)

parallel_chain= RunnableParallel(
    {
        'tweet':RunnableSequence(prompt1, model1, parser),
        'linkedin':RunnableSequence(prompt2, model1,parser)
    }
)
result=parallel_chain.invoke({'topic':"Ai"})
print(result)