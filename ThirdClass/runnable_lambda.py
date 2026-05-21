from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough

def word_count(text):
    return len(text.split())
load_dotenv()
prompt= PromptTemplate(
    template= "write a joke about {topic}",
    input_variables=['topic']
)
llm= HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
model= ChatHuggingFace(llm=llm)
parser= StrOutputParser()
joke_gen_chain= RunnableSequence(prompt, model,parser)

parallel_chain= RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'word_count':RunnableLambda(word_count)
    }
)

final_chain= RunnableSequence(joke_gen_chain, parallel_chain)
result=final_chain.invoke({'topic':"AI"})
print(result)