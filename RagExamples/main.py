from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

loader=PyPDFLoader("Tutorial_EDIT.pdf")
docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits=text_splitter.split_documents(docs)
print(f"Split the document into {len(splits)} chunks")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore=FAISS.from_documents(documents=splits, embedding=embeddings)

retriever= vectorstore.as_retriever(search_kwargs={"k":3})


llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="conversational",
    temperature=0.1,
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=llm_endpoint)

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {input}

Answer:"""

prompt = PromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "What is the main topic of this document?"
response = rag_chain.invoke(question)

print(response)
