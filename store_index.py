from src.helper import repo_ingestion, load_repo, text_splitter, load_embeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
import os

load_dotenv()

GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# url ="https://github.com/entbappy/End-to-end-Medical-Chatbot-Generative-AI"

# repo_ingestion(url)

documents = load_repo("repo/")

text_chunks = text_splitter(documents)

embeddings = load_embeddings()

#Storing vector in chromadb
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
vectordb.persist()

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8}
)
