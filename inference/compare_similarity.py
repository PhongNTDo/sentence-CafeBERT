from sentence_transformers import SentenceTransformer


# Download from the 🤗 Hub
model = SentenceTransformer("ThuanPhong/sentence_CafeBERT")

def get_embedding(sentences: list):
    # Run inference
    embeddings = model.encode(sentences)
    return embeddings


samples = [["Để trở thành một doanh nhân thành công thì tôi nên học thương mại hay khoa học?",
            "Cả thương mại và kỹ thuật đều góp phần tạo nên thành công cho một người làm kinh doanh."],
           ["Ông Hai sau khi tưới hoa người vườn đã vào nhà cùng xem tivi với các cháu của mình.",
            "Ông Hai đã đến công viên với các cháu vào buổi chiều."],
           ["Nội dung của hợp đồng đã được công ty A ký kết với công ty B là gì?",
            "Hai công ty A và B đã ký vào hợp đồng gì?"]]
for sentences in samples:   
    # Get the similarity scores for the embeddings
    embeddings = get_embedding(sentences)
    similarities = model.similarity(embeddings, embeddings)
    similarity = similarities[0][0].item()
    print("Sentence 1: ", sentences[0])
    print("sentence 2: ", sentences[1])
    print("Similarity: ", similarity)
    print("-" * 10)
