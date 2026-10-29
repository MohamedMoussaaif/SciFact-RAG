# SciFact RAG — Scientific Claim Verification Using Retrieval-Augmented Generation

## Overview

SciFact RAG is a Retrieval-Augmented Generation (RAG) system designed to analyze scientific claims using evidence retrieved from scientific abstracts.

The project combines semantic search with a Large Language Model (LLM) to identify relevant scientific documents and generate evidence-based explanations.

Instead of asking an LLM to answer a scientific question using only its pretrained knowledge, the system retrieves relevant documents from the SciFact corpus and provides their content as context. The LLM then analyzes the relationship between the claim and the retrieved evidence.

The main objective is to build an understandable and evaluable RAG pipeline while exploring how semantic retrieval and generative AI can work together for scientific fact verification.

## Objectives

- Build a complete RAG pipeline using Python.
- Retrieve scientific documents based on semantic similarity.
- Generate document embeddings using a pretrained Sentence Transformer.
- Use Gemini as the LLM for evidence-based claim analysis.
- Provide explanations and references to retrieved scientific documents.
- Evaluate retrieval performance using standard information retrieval metrics.
- Explore the limitations of semantic retrieval and LLM-based scientific reasoning.

## Architecture

The system consists of four main stages:

1. **Data ingestion:** Load scientific abstracts, claims, and relevance judgments from SciFact.
2. **Embedding generation:** Convert scientific abstracts into numerical vector representations.
3. **Semantic retrieval:** Retrieve the most relevant documents for a given claim using cosine similarity.
4. **Answer generation:** Provide the retrieved abstracts and claim to Gemini to generate an evidence-based verdict and explanation.

```text
                 Scientific Claim
                        |
                        v
              Sentence Transformer
                Query Embedding
                        |
                        v
             Semantic Similarity Search
                        |
                        v
                Top-K Documents
                        |
                        v
             Scientific Abstracts
                        |
                        v
                  Gemini API
                        |
                        v
               Generated Answer
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
        Verdict     Explanation     Sources
```

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| BEIR | Dataset loading and retrieval evaluation |
| SciFact | Scientific claims and abstract corpus |
| Sentence Transformers | Document and query embeddings |
| all-MiniLM-L6-v2 | Pretrained embedding model |
| PyTorch | Tensor representation and embedding persistence |
| Google Gemini API | Evidence analysis and answer generation |
| python-dotenv | Secure environment variable management |

## Dataset

The project uses the SciFact dataset distributed through the BEIR benchmark.

SciFact is a scientific fact-verification dataset containing claims and scientific abstracts. It provides relevance judgments that indicate which documents are relevant to each claim.

The dataset used in this project contains:

| Component | Number |
|---|---:|
| Scientific abstracts | 5,183 |
| Test claims | 300 |

The dataset provides three important components:

- **Corpus:** Scientific documents containing titles and abstracts.
- **Queries:** Scientific claims to be verified.
- **Qrels:** Relevance judgments linking claims to relevant documents.

The original SciFact release also contains evidence annotations, including supporting and contradicting evidence sentences. These annotations are distinct from BEIR's relevance judgments.

Dataset resources:

- [SciFact original repository](https://github.com/allenai/scifact)
- [BEIR benchmark](https://github.com/beir-cellar/beir)

## Pipeline Implementation

### 1. Data Loading

The first stage loads the SciFact corpus, queries, and relevance judgments.

The BEIR `GenericDataLoader` is used to load the test split.

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

data_path = util.download_and_unzip(
    DATASET_URL,
    "data"
)

corpus, queries, qrels = GenericDataLoader(
    data_folder=data_path
).load(split="test")
```

Each scientific document contains a title and an abstract.

### 2. Document Preparation

Before generating embeddings, each document is prepared by combining its title and abstract.

The title provides additional context, while the abstract contains the scientific information needed for evidence retrieval.

The resulting document representation follows this structure:

```text
Title: Scientific document title

Abstract: Complete scientific abstract
```

Each abstract is kept as one document rather than being divided into smaller chunks.

This decision preserves the scientific context of each publication and allows retrieved documents to be provided directly to the LLM.

### 3. Embedding Generation

The project uses the pretrained model:

`sentence-transformers/all-MiniLM-L6-v2`

This model transforms text into dense numerical vectors.

Each scientific abstract is encoded into a 384-dimensional embedding.

For the complete SciFact corpus, the resulting embedding matrix has the following shape:

```text
(5183, 384)
```

The model is not fine-tuned in this initial implementation. It is used directly to generate semantic representations of documents and claims.

#### Embedding Persistence

Generating embeddings for the entire corpus at every execution is unnecessary.

Therefore, document embeddings and their corresponding document IDs are saved locally using PyTorch serialization.

On subsequent executions, the system loads the previously generated embeddings instead of encoding the documents again.

This reduces repeated computation and improves execution efficiency.

If the corpus or embedding model changes, the cached embeddings must be regenerated to maintain consistency.

### 4. Semantic Retrieval

When a user submits a scientific claim, the same Sentence Transformer model converts it into a query embedding.

The system then calculates cosine similarity between the query embedding and all document embeddings.

The cosine similarity formula is:

\[
\operatorname{sim}(q,d)=
\frac{q\cdot d}{\|q\|\|d\|}
\]

Where:

- \(q\) represents the claim embedding.
- \(d\) represents a scientific document embedding.
- The similarity score measures the semantic proximity between the two vectors.

The documents are ranked according to their similarity scores, and the system retrieves the top-K documents.

The initial configuration uses:

```python
TOP_K = 5
```

For each retrieved document, the system keeps:

- Document ID
- Title
- Abstract
- Similarity score

These fields preserve the connection between retrieved evidence and its original source.

### 5. Retrieval Evaluation

Retrieval quality is evaluated using the relevance judgments provided by SciFact.

The evaluation is performed across the 300 test claims.

Three metrics are calculated:

**Precision@K**

Measures the proportion of retrieved documents that are relevant.

\[
Precision@K =
\frac{\text{Relevant documents in top K}}{K}
\]

**Recall@K**

Measures the proportion of all relevant documents that are successfully retrieved.

\[
Recall@K =
\frac{\text{Relevant documents in top K}}
{\text{Total relevant documents}}
\]

**Mean Reciprocal Rank (MRR)**

Measures how highly the first relevant document appears in the retrieved ranking.

\[
MRR =
\frac{1}{N}
\sum_{i=1}^{N}
\frac{1}{rank_i}
\]

Here, \(rank_i\) is the rank of the first relevant document for query \(i\). If no relevant document is retrieved, its reciprocal rank is zero.

#### Initial Retrieval Results

Using `all-MiniLM-L6-v2` with \(K=5\):

| Metric | Result |
|---|---:|
| Precision@5 | 0.1647 |
| Recall@5 | 0.7418 |
| MRR | 0.5959 |

These results represent the initial semantic retrieval baseline.

They provide a starting point for future experiments involving alternative embedding models, retrieval strategies, and ranking improvements.

### 6. Answer Generation with Gemini

After retrieval, the system sends the scientific claim and retrieved abstracts to the Gemini API.

The project uses the `gemini-3.1-flash-lite` model.

Gemini receives:

- The original scientific claim.
- The titles of the retrieved documents.
- The complete abstracts of the retrieved documents.

The prompt instructs the model to:

1. Analyze the relationship between the claim and the retrieved scientific evidence.
2. Use only the supplied abstracts.
3. Avoid inventing scientific facts or references.
4. Distinguish between supporting and contradicting evidence.
5. Return NOT ENOUGH EVIDENCE when the retrieved context is insufficient.
6. Identify the document sources used in its explanation.

The generated answer follows three sections:

```text
VERDICT:
SUPPORT / REFUTE / NOT ENOUGH EVIDENCE

EXPLANATION:
Evidence-based explanation of the verdict.

SOURCES:
Document IDs and titles used in the explanation.
```

An important design principle is that semantic similarity does not establish scientific truth. Retrieved documents provide potentially relevant context, but the LLM must examine their actual content before producing a conclusion.

The system's verdict should therefore be interpreted as an assessment based on the retrieved context, not as a definitive scientific conclusion.

## Project Structure

```text
scifact-rag/
│
├── data/
│   ├── scifact/
│   │   ├── corpus.jsonl
│   │   ├── queries.jsonl
│   │   └── qrels/
│   │       ├── train.tsv
│   │       └── test.tsv
│   │
│   └── document_embeddings.pt
│
├── data_loader.py
├── retrieval.py
├── evaluation.py
├── generation.py
├── main.py
│
├── .env
├── .gitignore
└── README.md
```

### Main Python Modules

| File | Responsibility |
|---|---|
| `data_loader.py` | Downloads and loads the SciFact dataset |
| `retrieval.py` | Loads the embedding model, prepares documents, generates and stores embeddings, and retrieves documents |
| `evaluation.py` | Calculates Precision@K, Recall@K, and MRR |
| `generation.py` | Connects to Gemini and generates evidence-based answers |
| `main.py` | Orchestrates the complete pipeline |

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd scifact-rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Get an API key from [Google AI Studio](https://aistudio.google.com/).

Never commit your API key to GitHub.

### 5. Run the application

```bash
python main.py
```

On the first execution, the system downloads the SciFact dataset, generates document embeddings, and saves them locally.

On subsequent executions, previously generated embeddings are loaded from disk.

The pipeline then evaluates retrieval performance and generates an answer for the selected scientific claim.

## Current Limitations

Although this project implements a complete initial RAG pipeline, several limitations remain:

- **Embedding quality:** The initial embedding model is a general-purpose sentence embedding model and may not capture specialized scientific terminology accurately.
- **Retrieval precision:** Some retrieved documents are semantically related to a claim without providing direct scientific evidence.
- **Context limitations:** The LLM only receives the top-K retrieved abstracts. Relevant evidence outside these documents may be missed.
- **LLM reliability:** Gemini can misunderstand scientific concepts or generate conclusions that are not fully supported by the retrieved context.
- **Evaluation separation:** BEIR relevance judgments evaluate document retrieval, not whether Gemini's generated scientific verdict is correct.
- **Citation verification:** The generated source references should be checked against the actual retrieved documents.

## Future Improvements

Potential directions for extending the project include:

- Experimenting with scientific embedding models such as SPECTER2.
- Comparing different embedding models using the same retrieval evaluation procedure.
- Implementing a reranking stage to improve the ordering of retrieved documents.
- Exploring hybrid retrieval combining semantic and keyword search.
- Evaluating generated verdicts against SciFact's original evidence and stance annotations.
- Introducing structured JSON outputs for more reliable answer processing.
- Persisting embeddings and metadata in a vector database such as FAISS or Chroma.
- Building a FastAPI backend to expose the RAG pipeline through REST endpoints.

## Learning Outcomes

This project explores the practical implementation of a complete RAG architecture, including:

- Scientific document preprocessing.
- Dense vector representations and semantic similarity.
- Information retrieval and ranking evaluation.
- Integration of pretrained embedding models.
- LLM prompting and evidence-grounded generation.
- Separation of retrieval quality from generation quality.
- Reproducibility and efficient persistence of generated embeddings.

The project provides a foundation for experimenting with more advanced retrieval and generation techniques in scientific information retrieval.

---

**Project:** Scientific Claim Verification Using Retrieval-Augmented Generation

**Status:** Initial RAG pipeline implemented.
