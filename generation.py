
import os

from dotenv import load_dotenv
from google import genai


MODEL_NAME = "gemini-3.1-flash-lite"


def create_gemini_client():
    """Create a Gemini API client using the environment variable."""

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Check your .env file."
        )

    return genai.Client(api_key=api_key)


def generate_answer(client, claim, retrieved_documents):
    """Generate an evidence-based answer for a scientific claim."""

    evidence = ""

    for i, document in enumerate(retrieved_documents, start=1):
        evidence += (
            f"\nEvidence {i}\n"
            f"Document ID: {document['document_id']}\n"
            f"Title: {document['title']}\n"
            f"Abstract: {document['abstract']}\n"
        )

    prompt = f"""
You are a scientific evidence analysis assistant.

Your task is to evaluate the following scientific claim
using ONLY the provided scientific abstracts.

CLAIM:
{claim}

RETRIEVED EVIDENCE:
{evidence}

Instructions:

1. Analyze whether the evidence supports or contradicts the claim.
2. Use only information explicitly available in the abstracts.
3. Do not invent scientific facts or references.
4. Do not treat semantic similarity as proof.
5. If the evidence is insufficient or inconclusive,
   choose NOT ENOUGH EVIDENCE.
6. Identify the document IDs that support your explanation.

Return your answer using this exact structure:

VERDICT:
[SUPPORT / REFUTE / NOT ENOUGH EVIDENCE]

EXPLANATION:
[Your explanation based on the evidence]

SOURCES:
[List the document IDs and titles you used]
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text
    