from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
vector_db_path = "vectorstores/db_faiss"
app = Flask(__name__)

def load_llm():
    llm = Ollama(model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k":3}, max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs={"prompt":prompt})
    return llm_chain

def read_vectors_db():
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = FAISS.load_local(vector_db_path,embedding_model, allow_dangerous_deserialization=True)
    return db

db = read_vectors_db()
llm = load_llm()
template = """Bạn là 1 trợ lý ảo hãy sử dụng thông tin sau đây để trả lời ngắn gọn cho câu hỏi dưới đây bằng tiếng Việt:
{context}

Câu hỏi: {question}


"""
prompt = create_prompt(template)
llm_chain = create_qa_chain(prompt, llm, db)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    response = llm_chain.invoke({"query": question})
    print(response)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)