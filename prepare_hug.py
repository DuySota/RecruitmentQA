from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F
import json
import numpy as np

#Mean Pooling - Take attention mask into account for correct averaging
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity
# Sentences we want sentence embeddings for

# with open('data/txt/origin/all.txt', 'r', encoding='utf-8') as file:
#         raw_text = file.read()

# text_splitter = CharacterTextSplitter(
#     separator = "\n\n",
#     chunk_size = 2048,
#     chunk_overlap = 100,
#     length_function = len
# )



# chunks = text_splitter.split_text(raw_text)
# print("chunk: ", len(chunks))
# # Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
# sentences = chunks
# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)

# # Perform pooling
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# # Normalize embeddings
# sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
# numpy_array = sentence_embeddings.numpy()
# print("Sentence embeddings:")
# print(numpy_array.shape)

# data = [{"context": chunks[i],
#           "score": numpy_array[i].tolist()} for i in range(len(chunks))]


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
# numpy_array = embed_question(question="Lấy ví dụ cách doanh nghiệp làm truyền thông nội bộ")
# max_sim = -1
# max_index = -1
# for i, data_i in enumerate(data):
#     data_i["score"] = np.array(data_i["score"])
#     sim = cosine_similarity(numpy_array, data_i["score"])
#     print(f"Similarity with index {i}: {sim}")

#     # Tìm chỉ số có similarity cao nhất
#     if sim > max_sim:
#         max_sim = sim
#         max_index = i
# print(data[max_index]["context"])



import pandas as pd

# Đọc file Excel
df = pd.read_excel('data/ThesisData.xlsx')

data = []

# In ra các trường question và answer
for index, row in df.iterrows():
    question = str(row['question']) if not pd.isna(row['question']) else ''
    answer = str(row['answer']) if not pd.isna(row['answer']) else ''
    emb_ques = embed_question(question)
    data.append({
        "question": question, 
        "answer": answer, 
        "emb_ques": emb_ques.tolist(),
        "full_context": "Câu hỏi: " + question + "\n\n" + "Trả lời: " + answer
    })


with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f)

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(data)









