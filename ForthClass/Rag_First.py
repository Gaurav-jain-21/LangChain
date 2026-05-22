from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader # <-- FIXED IMPORT
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import sys # Needed for sys.exit()

# Load environment variables (Make sure your .env has HF_TOKEN=...)
load_dotenv()

pdf_path = input("Enter the path to your pdf file: ")
if not os.path.exists(pdf_path):
    print("Error: File not found. Please check the path and try again.")
    sys.exit() # <-- FIXED: Stop the script if the file doesn't exist

print("\nPhase Ingestion...")
print("Reading and splitting PDF...")

# Load and split
loader = PyPDFLoader(pdf_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed and store
embeddings = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# Initialize LLM
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")

template = """You are a helpful assistant. Answer the user's question using ONLY the provided context below. 
    If the context does not contain the answer, say "I cannot find that in the document."

    Context:
    {context}

    Question: {question}
    Answer:"""

# Cleaner way to initialize the prompt template
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\nReady! Ask your questions.")

# Chat loop
while True:
    user_question = input("🧑 User: ")
    if user_question.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    if not user_question.strip():
        continue

    print("🤖 AI is thinking...")
    try:
        response = rag_chain.invoke(user_question)
        print(f"\n🤖 AI: {response.strip()}\n")
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")