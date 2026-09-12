import math
from pathlib import Path

import httpx
from pypdf import PdfReader


PDF_PATH = Path("data/paper.pdf")

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3:4b-instruct"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 3


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)

    pages_text = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def get_embedding(text: str) -> list[float]:
    response = httpx.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=120.0
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    return dot_product / (magnitude_a * magnitude_b)


def retrieve_chunks(
    query: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]],
    top_k: int = TOP_K,
) -> list[str]:

    query_embedding = get_embedding(query)

    results = []

    for chunk, embedding in zip(chunks, chunk_embeddings):
        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        results.append((similarity, chunk))

    results.sort(reverse=True)

    return [chunk for _, chunk in results[:top_k]]


def ask_llm(query: str, context_chunks: list[str]) -> str:

    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
Use only the context below to answer the question.
If the answer is not contained in the context, say that you cannot find it in the document.

CONTEXT:
{context}

QUESTION:
{query}
"""

    response = httpx.post(
        OLLAMA_CHAT_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a document question-answering assistant. "
                        "Answer clearly and concisely using only the provided context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=120.0
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


print("Reading PDF...")

text = extract_pdf_text(PDF_PATH)
chunks = split_text_into_chunks(text)

print(f"Number of chunks: {len(chunks)}")
print("Creating embeddings...")

chunk_embeddings = []

for index, chunk in enumerate(chunks):
    chunk_embeddings.append(get_embedding(chunk))
    print(f"Embedded chunk {index + 1}/{len(chunks)}")


query = "What is the capital of France?"

relevant_chunks = retrieve_chunks(
    query,
    chunks,
    chunk_embeddings
)

print("\n--- RETRIEVED CONTEXT ---\n")

for index, chunk in enumerate(relevant_chunks, start=1):
    print(f"Result {index}:")
    print(chunk)
    print("\n" + "=" * 80 + "\n")


answer = ask_llm(
    query,
    relevant_chunks
)

print("\n--- FINAL ANSWER ---\n")
print(answer)