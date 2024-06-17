import gradio as gr
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
import json
import numpy as np
import re
import uuid
import firebase_admin
from firebase_admin import credentials, db
from utils.connect import connect
from utils.retrieve import retrieve
from env import URL, PORT


# Khởi tạo Firebase Admin SDK với tệp tin credentials của bạn
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': URL
})



# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
vector_db_path = "vectorstores/db_faiss"
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
def load_llm():
    llm = Ollama(model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    return llm

def embed_question(question):
    sentences = [question]
    batch_dict = tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    print(embeddings.shape)
    numpy_array = embeddings.detach().numpy()[0]
    return numpy_array



llm = load_llm()
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


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
                          "Thu nhập của App deveploper khoảng bao nhiêu nhỉ?",
                          "Lấy ví dụ cách doanh nghiệp làm truyền thông nội bộ"
                          ],
                           inputs=[msg])
    clear = gr.ClearButton([msg, chatbot], elem_classes="clear")

    def respond(message, chat_history):
        
        if not logged_in:
            best_ans = "Bạn cần đăng nhập trước khi bắt đầu phiên hỏi nè 😜!"
        else: 
            numpy_array = embed_question(question=message)
            max_sim = -1
            max_index = -1
            for i, data_i in enumerate(data):
                data_i["emb_ques"] = np.array(data_i["emb_ques"])
                sim = cosine_similarity(numpy_array, data_i["emb_ques"])
                print(f"Similarity with index {i}: {sim}")

                # Tìm chỉ số có similarity cao nhất
                if sim > max_sim:
                    max_sim = sim
                    max_index = i
            context = data[max_index]["full_context"]
            score = max_sim
            prompt_template = "{question}"
            prompt = PromptTemplate(
                input_variables=["question"], template=prompt_template
            )
            chain = prompt | llm
            print("\n===================================================================")
            print("score: ", score)
            print("context: ", context)
            if len(chat_history)==0:
                if score>=0.9:
                    full_question = f'''Sử dụng thông tin này "{context}"
                    
                    Câu hỏi: {message}
                    Trả lời câu hỏi sau đây bằng tiếng Việt, nếu không có câu trả lời liên quan tới thông tin được cung cấp thì không cần dựa vào thông tin được cung cấp để trả lời,
                    Trả lời ngắn gọn không cần lặp lại câu hỏi:
                    '''
                else:
                    full_question = f'''
                        {message}
                    '''
            else:
                last_dialog = chat_history[-1]
                last_dialog_ = {"question": last_dialog[0], "answer": last_dialog[1]}
                if score>=0.9:
                    full_question = f'''
                    Đây là mẩu hội thoại ban đầu: 
                    "Người hỏi: {last_dialog_["question"]}
                    Người đáp: {last_dialog_["answer"]}"

                    Sử dụng thông tin này "{context}"
                    
                    Câu hỏi: "{message}"
                    Trả lời câu hỏi sau đây bằng tiếng Việt, nếu không có câu trả lời liên quan tới thông tin được cung cấp thì không cần dựa vào thông tin được cung cấp để trả lời,
                    Trả lời ngắn gọn không cần lặp lại câu hỏi:
                    '''
                else:
                    full_question = f'''
                        Bạn là một trợ lý ảo
                        Đây là mẩu hội thoại ban đầu: 
                        "Người hỏi: {last_dialog_["question"]}
                        Người đáp: {last_dialog_["answer"]}"
                        Hãy đáp lại câu: "{message}"
                    '''
            response = chain.invoke(full_question)
            best_ans = response
        chat_history.append((message, best_ans))
        print(chat_history)
        return "", chat_history
        # return "", subchatbot


    chatbot.like(vote, None, None)
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    

if __name__ == "__main__":
    demo.launch(share=True, server_port=PORT)


	

