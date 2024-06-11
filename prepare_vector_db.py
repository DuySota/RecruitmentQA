from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

def create_db_from_text():
    with open('data/txt/origin/all.txt', 'r', encoding='utf-8') as file:
        raw_text = file.read()
    print(raw_text)
    
    text_splitter = CharacterTextSplitter(
        separator = "\n\n",
        chunk_size = 2048,
        chunk_overlap = 100,
        length_function = len
    )
    
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # text_splitter = SemanticChunker(embeddings=embedding_model, breakpoint_threshold_type="interquartile")
    chunks = text_splitter.split_text(raw_text)
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)

def create_db_from_files():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls= PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = FAISS.from_documents(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)

create_db_from_text()