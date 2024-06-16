from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("ThuanPhong/sentence_CafeBERT")
# Run inference
sentences = [
    'Một cậu bé cưỡi trên chiếc thuyền lướt sóng của mình xuyên qua làn nước biển.',
    'Thằng bé ngồi trên bãi biển.',
    'Trong bối cảnh tranh chấp căng thẳng về phân chia chủ quyền vùng biển, đã có quan điểm của một số học giả – sử gia đề xuất đổi tên biển thành "biển Đông Nam Á" ("Southeast Asia Sea") hay biển Đông Nam châu Á (South East Asia Sea) - là một tên gọi trung lập.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 256]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]