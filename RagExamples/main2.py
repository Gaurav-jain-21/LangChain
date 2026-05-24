from langchain_community.document_loaders import GitLoader
from dotenv import load_dotenv
load_dotenv()
repo_url="https://github.com/Gaurav-jain-21/Talent-Talk-the-Freelancing-Web-Site-for-Student.git"
local_path="./temp_repo"

print("cloning repository")
loader=GitLoader(
    clone_url=repo_url,
    repo_path=local_path,
    branch="main",
    file_filter= lambda file_path: file_path.endswith(".java")
)

docs=loader.load()
print(f"loaded {len(docs)} java files from the repository")

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

java_splitter= RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA,
    chunk_size=1000,
    chunk_overlap=200
)

splits= java_splitter.split_documents(docs)
print(f"split the code into {len(splits)}")

embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore=FAISS.from_documents(documents=splits, embedding=embeddings)
retriever=vectorstore.as_retriever(search_kwargs={"k":4})
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    temperature=0.1,
    max_new_tokens=512,
)

template = """You are a Senior Software Engineer. Use the following code snippets to answer the junior developer's question. 
Explain the code clearly, and if the answer isn't in the provided code, state that you cannot find it in this repository.

Code Context: {context}

Question: {input}

Explanation:"""
prompt = PromptTemplate.from_template(template)
def format_docs(docs):
    return "\n\n".join(f"File: {doc.metadata['file_path']}\n{doc.page_content}" for doc in docs)

# Build the Chain
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "What are the main functions defined in this repository and what do they do?"

print("\nAnalyzing code...")
response = rag_chain.invoke(question)

print("\n--- Senior Dev Response ---")
print(response)