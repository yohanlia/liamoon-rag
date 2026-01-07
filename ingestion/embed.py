from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1",
        device = "cpu",
        tokenizer_kwargs= {"padding_side": "left"},
        trust_remote_code=True
)

def embed_documents(documents):
    """
    Embed one or more documents using the Nomic Embed Text v1 model.
    """
    # Encode the documents
    embeddings = MODEL.encode(documents, show_progress_bar=True)

    # ✅ Convert NumPy arrays to lists for JSON serialization
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()

    return embeddings

def embed_query(query: str):
    """
    Embed a single user query for vector search.
    """
    embedding = MODEL.encode([query], show_progress_bar=False)[0]
    return embedding.tolist() if hasattr(embedding, "tolist") else embedding