from langchain.vectorstores import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import repo_ingestion
from flask import Flask, render_template, jsonify, request

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings




app = Flask(__name__)


load_dotenv()


GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

persist_directory ="db" 

# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template("""
You are an expert code assistant.

Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and technically.
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
# qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":3}), memory=memory)



@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")

    return jsonify({"response": str(user_input) })



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

    # result = qa(input)
    # print(result['answer'])
    # return str(result["answer"])
    result = rag_chain.invoke(input)
    print(result)
    return result


    
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)