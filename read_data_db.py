from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
vector_db_path = "vectorstores/db_faiss"
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
query = "Làm sao để ứng viên không nhạy cảm với bài test đầu vào?"
docs = db.similarity_search(query)
docs_and_scores = db.similarity_search_with_score(query)
print(docs)
print(docs_and_scores[0][0].page_content)


###########################################################




# ########################################################################
# from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain_community.vectorstores import Chroma

# from langchain.storage import InMemoryStore
# from langchain.retrievers import ParentDocumentRetriever
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # Load blog post

# from langchain_community.embeddings import HuggingFaceEmbeddings

# model_name = "all-MiniLM-L6-v2"
# model_kwargs = {'device': 'cuda'}
# encode_kwargs = {'normalize_embeddings': False}
# embedding_model = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )
# child_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
# parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
# vectorstore = Chroma(
#     collection_name="full_documents", embedding_function=embedding_model
# )
# db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
# query = "Cách để ứng viên không bị nhạy cảm khi test bài kiểm tra đầu vào"
# docs = db3.similarity_search(query)
# # store = InMemoryStore()
# # retriever = ParentDocumentRetriever(
# #     vectorstore=db3,
# #     docstore=store,
# #     child_splitter=child_splitter,
# #     parent_splitter=parent_splitter
# # )

# # retrieved_docs = retriever.invoke("Cách để ứng viên không bị nhạy cảm khi test bài kiểm tra đầu vào")
# # print(retrieved_docs)
# print(docs)