from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Example model with embedding size of 384

def get_embedding(text):
    return embedding_model.encode([text])[0]

def get_texts_embedding(text_list):
    return embedding_model.encode(text_list)

def main():
    example_text = ["This is an example text.", 
                    "This is another example text."
                    ]
    example_embedding = get_texts_embedding(example_text)
    print("Example embedding shape:", example_embedding.shape)

if __name__ == "__main__":
    main()