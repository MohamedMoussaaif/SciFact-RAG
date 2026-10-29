
from data_loader import load_scifact_dataset
from retrieval import (
    load_embedding_model,
    prepare_document_texts,
    encode_document,
    retrieve_documents,
    save_embeddings,
    load_embeddings,
)
from evaluation import evaluate_retrieval
from generation import (
    create_gemini_client,
    generate_answer,
)


TOP_K = 5


def main():
    # 1. Load the embedding model
    print("Loading embedding model...")
    model = load_embedding_model()

    # 2. Load the SciFact dataset
    print("Loading SciFact dataset...")
    corpus, queries, qrels = load_scifact_dataset()

    print(f"Documents loaded: {len(corpus)}")
    print(f"Claims loaded: {len(queries)}")

    # 3. Prepare the scientific documents
    saved_document_ids, document_embeddings = load_embeddings()

    if document_embeddings is None:

        print("No saved embeddings found.")
        print("Generating document embeddings...")

        document_ids, document_texts = prepare_document_texts(corpus)

        document_embeddings = encode_document(
            model,
            document_texts,
        )

        save_embeddings(
            document_ids,
            document_embeddings,
        )

        saved_document_ids = document_ids

    print(
        "Document embedding shape:",
        document_embeddings.shape,
    )

    # 5. Evaluate retrieval across all claims
    print("Evaluating retrieval...")

    metrics = evaluate_retrieval(
        model=model,
        queries=queries,
        qrels=qrels,
        document_ids=saved_document_ids,
        document_embeddings=document_embeddings,
        top_k=TOP_K,
    )

    # 6. Display evaluation results
    print("\n=== RETRIEVAL EVALUATION ===")
    print(f"Total claims evaluated: {metrics['total_claims']}")
    print(
        "Claims with relevant documents:",
        metrics["claims_with_relevant_docs"],
    )
    print(
        f"Precision@{TOP_K}: "
        f"{metrics['precision']:.4f}"
    )
    print(
        f"Recall@{TOP_K}:    "
        f"{metrics['recall']:.4f}"
    )
    print(f"MRR:               {metrics['mrr']:.4f}")








    # 7. Generate an answer for one example claim

    query_id = "1"
    claim = queries["1"]


    print("\n=== RAG GENERATION ===")
    print("Claim:", claim)

    # Retrieve the top 5 relevant documents
    retrieved_documents = retrieve_documents(
        claim=claim,
        model=model,
        corpus=corpus,
        document_ids=saved_document_ids,
        document_embeddings=document_embeddings,
        top_k=TOP_K,
    )

    print("\n=== RETRIEVED DOCUMENTS ===")

    for i, document in enumerate(retrieved_documents, start=1):
        print(f"\nEvidence {i}")
        print("Document ID:", document["document_id"])
        print("Title:", document["title"])
        print("Similarity:", round(document["similarity"], 4))


    # Connect to Gemini
    client = create_gemini_client()

    # Generate an evidence-based answer
    answer = generate_answer(
        client=client,
        claim=claim,
        retrieved_documents=retrieved_documents,
    )

    print("\n=== GEMINI ANSWER  ===")
    print(answer)




if __name__ == "__main__":
    main()
