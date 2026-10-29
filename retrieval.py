from sentence_transformers import SentenceTransformer, util as st_util
import os 
import torch


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# 1. Load the embedding model
def load_embedding_model():
    print("Loading embedding model...")
    return SentenceTransformer(MODEL_NAME)


# 3. Prepare every document: title + complete abstract
def prepare_document_texts(corpus):
    document_ids = list(corpus.keys())

    document_texts = [
        f"Title: {corpus[doc_id]['title']}\n"
        f"Abstract: {corpus[doc_id]['text']}"
        for doc_id in document_ids
    ]
    return document_ids, document_texts

# 4. Generate all document embeddings
def encode_document(model, document_texts):
    print("Generating document embeddings...")
    return model.encode(
        document_texts,
        batch_size=32,
        convert_to_tensor=True,
        show_progress_bar=True,
    )


# 5. Generate embeddings for all 300 claims
def generate_query_embeddings(model, queries):
    print("Generating query embeddings...")

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    return query_embeddings

# 6. Compare every claim with every document
def calculate_similarities(query_embeddings, document_embeddings):
    print("Calculating similarities...")

    similarity_matrix = st_util.cos_sim(
        query_embeddings,
        document_embeddings,
    )
    return similarity_matrix


def retrieve_documents(
    claim,
    model,
    corpus,
    document_ids,
    document_embeddings,
    top_k=TOP_K,
):
    """Retrieve the most similar documents for a claim."""
    query_embedding = model.encode(
        claim,
        convert_to_tensor=True,
    )

    scores = st_util.cos_sim(
        query_embedding,
        document_embeddings,
    )[0]

    top_results = scores.topk(
        min(top_k, len(document_ids))
    )

    results = []

    for index, score in zip(
        top_results.indices,
        top_results.values,
    ):
        doc_id = document_ids[index.item()]
        document = corpus[doc_id]

        results.append({
            "document_id": doc_id,
            "title": document["title"],
            "abstract": document["text"],
            "similarity": score.item(),
        })

    return results



def save_embeddings(
    document_ids,
    document_embeddings,
    file_path="data/document_embeddings.pt",
):
    """Save document IDs and embeddings to disk."""

    torch.save(
        {
            "document_ids": document_ids,
            "embeddings": document_embeddings.cpu(),
        },
        file_path,
    )

    print(f"Embeddings saved to {file_path}")


def load_embeddings(
    file_path="data/document_embeddings.pt",
):
    """Load saved document embeddings if available."""

    if not os.path.exists(file_path):
        print("No saved embeddings found.")
        return None, None

    try:
        saved_data = torch.load(
            file_path,
            map_location="cpu",
            weights_only=True,
        )

        print("Saved embeddings loaded successfully.")

        return (
            saved_data["document_ids"],
            saved_data["embeddings"],
        )

    except (EOFError, RuntimeError, KeyError) as error:
        print(f"Could not load saved embeddings: {error}")
        print("Embeddings will be regenerated.")

        return None, None
