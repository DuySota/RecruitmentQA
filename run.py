import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
import re
import uuid
import firebase_admin
from firebase_admin import credentials, db
from utils.connect import connect
from utils.retrieve import retrieve
from env import URL, PORT

from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

from typing import List
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Load blog post

from langchain_community.embeddings import HuggingFaceEmbeddings
from flask import Flask, request, jsonify

# Khởi tạo Firebase Admin SDK với tệp tin credentials của bạn
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': URL
})

model_name = "all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=embedding_model
)
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=db3,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)


########################################################################



llm = Ollama(model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)



class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


output_parser = LineListOutputParser()
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt
template = """Bạn là 1 trợ lý ảo hãy sử dụng thông tin sau đây để trả lời ngắn gọn cho câu hỏi dưới đây bằng tiếng Việt:
{context}

Câu hỏi: {question}


"""
prompt = create_prompt(template)
llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents = False,
        chain_type_kwargs={"prompt":prompt})

def vote(data: gr.LikeData):
    if isinstance(data.value, str):
        if data.liked:
            print("You upvoted this response: " + data.value)
        else:
            print("You downvoted this response: " + data.value)
    else:
        print("Error: data.value is not a string.")


logged_in = False
def login(username, password):
    global logged_in
    result = retrieve(db, username, password)
    if result:
        logged_in = True
        return "Login successfully!"

    return "Incorrect username or password."

def signup(username, password, confirmpassword):
    user_id = str(uuid.uuid4())
    print("signup: ", username)
    print("signup: ", password)
    print("signup: ", confirmpassword)
    # Check if the password contains both letters and numbers
    if not re.match(r'^(?=.*[A-Za-z])(?=.*\d)', password):
        return "Password must contain at least one letter and one number."
    
    # Check if the password matches the confirmation password
    if password != confirmpassword:
        return "Passwords do not match. Please confirm your password correctly."
    # connect(db, user_id, username, password)

    return connect(db, user_id, username, password)
def logout():
    global logged_in
    logged_in = False
    print("You have been logged out.")
    return "You have been logged out."
css = """
.button textarea {font-size: 20px !important}
.clear button {background-color: #ff822d !important}
.buttonlogin:hover { background-color: #6366f1; color: white}
.buttonsignup:hover { background-color: red; color: white}
.signin .svelte-1gfkn6j{ color: red !important}
.signin .svelte-1gfkn6j {color: red !important}
.chatbot {
  background: linear-gradient(to right, rgb(95 109 255), rgb(200 254 123)); 
  color: white;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
}
.markdown h1{
    overflow: hidden;
    white-space: nowrap;
}

.animate-text {
    animation: slide-out 5s linear forwards;
}

@keyframes slide-out {
    from {
        transform: translateX(0%);
    }
    to {
        transform: translateX(100%);
    }
}

"""


with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown('# Recruitment Chatbot System', elem_classes="markdown")
    annouce = gr.Textbox(label="announcement")
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                inputlogins = [
                gr.Textbox(placeholder="What is your name?", label="username"),
                gr.Textbox(placeholder="Password", label="password", type="password")
                ]
            with gr.Row():
                btnlogin = gr.Button("Log in", elem_classes="buttonlogin")
                btnlogin.click(fn=login, inputs=inputlogins, outputs=annouce)
        with gr.Column(scale=4):
            with gr.Row():
                inputsignups = [
                gr.Textbox(placeholder="Type your name", label="username", elem_classes="signin"),
                gr.Textbox(placeholder="Password", label="password", elem_classes="signin", type="password"),
                gr.Textbox(placeholder="Confirm password", label="confirm password", elem_classes="signin", type="password")
                ]
            with gr.Row():
                btnsignup = gr.Button("Sign up", elem_classes="buttonsignup")
                btnsignup.click(fn=signup, inputs=inputsignups, outputs=annouce)
        with gr.Column(scale=1):
            btnlogout = gr.Button("Log out", elem_classes="buttonsignup")
            btnlogout.click(fn=logout, outputs=annouce)

            


    chatbot = gr.Chatbot(elem_classes="chatbot")
    msg = gr.Textbox(elem_classes="button", placeholder="Chat with me!")
    gr.Examples(examples=["Làm sao để ứng viên không nhạy cảm với bài test", 
                          "Thu nhập của App deveploper khoảng bao nhiêu nhỉ?"
                          ],
                           inputs=[msg])
    clear = gr.ClearButton([msg, chatbot], elem_classes="clear")

    def respond(message, chat_history):
        if not logged_in:
            best_ans = "Bạn cần đăng nhập trước khi bắt đầu phiên hỏi nè 😜!"
        else: 
            response = llm_chain.invoke({"query": message})
            best_ans = response['result']
        chat_history.append((message, best_ans))

        return "", chat_history
        # return "", subchatbot


    chatbot.like(vote, None, None)
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    

if __name__ == "__main__":
    demo.launch(share=True, server_port=PORT)


	

