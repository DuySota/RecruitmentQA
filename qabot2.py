from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.schema import Document
from typing import List
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
# Load blog post
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from flask import Flask, request, jsonify
loader = TextLoader("data/txt/data1.txt")
data = loader.load()
loader = TextLoader("data/txt/data2.txt")
data2 = loader.load()

docs = data + data2

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
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
db2 = Chroma.from_documents(docs, embedding_model, persist_directory="./chroma_db")
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=db3,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
retriever.add_documents(docs, ids=None)

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
question = "Kỹ sư bảo mật hệ thống có mức lương bao nhiêu?"
llm_chain.invoke(question)
print(vectorstore.similarity_search("Nhóm ngành khát nhân lực nhất trong năm 2024?"))
app = Flask(__name__)
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