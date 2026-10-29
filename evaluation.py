
from sentence_transformers import util as st_util


def evaluate_retrieval(
    model,
    queries,
    qrels,
    document_ids,
    document_embeddings,
    top_k=5,
):
    """Calculate Precision@k, Recall@k, and MRR."""

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print("Generating query embeddings...")

    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    print("Calculating similarities...")

    similarity_matrix = st_util.cos_sim(
        query_embeddings,
        document_embeddings,
    )

    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    queries_with_relevant_docs = 0

    for i, query_id in enumerate(query_ids):
        relevant_ids = {
            doc_id
            for doc_id, relevance
            in qrels.get(query_id, {}).items()
            if relevance > 0
        }

        top_results = similarity_matrix[i].topk(
            min(top_k, len(document_ids))
        )

        retrieved_ids = [
            document_ids[index.item()]
            for index in top_results.indices
        ]

        hits = sum(
            doc_id in relevant_ids
            for doc_id in retrieved_ids
        )

        precision_scores.append(
            hits / len(retrieved_ids)
        )

        if relevant_ids:
            queries_with_relevant_docs += 1
            recall_scores.append(
                hits / len(relevant_ids)
            )

        reciprocal_rank = 0.0

        for rank, doc_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            if doc_id in relevant_ids:
                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    mean_precision = (
        sum(precision_scores) / len(precision_scores)
    )

    mean_recall = (
        sum(recall_scores) / len(recall_scores)
        if recall_scores else 0.0
    )

    mrr = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
    )

    return {
        "total_claims": len(query_ids),
        "claims_with_relevant_docs": (
            queries_with_relevant_docs
        ),
        "precision": mean_precision,
        "recall": mean_recall,
        "mrr": mrr,
        "top_k": top_k,
    }
