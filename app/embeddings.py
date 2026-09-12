import math
from pathlib import Path

import httpx
from pypdf import PdfReader


PDF_PATH = Path("data/paper.pdf")

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


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


text = extract_pdf_text(PDF_PATH)
chunks = split_text_into_chunks(text)

print(f"Number of chunks: {len(chunks)}")
print("Creating embeddings...")

chunk_embeddings = []

for index, chunk in enumerate(chunks):

    embedding = get_embedding(chunk)

    chunk_embeddings.append(embedding)

    print(f"Embedded chunk {index + 1}/{len(chunks)}")


query = "What limit was obtained for the branching fraction of the rare decay?"

query_embedding = get_embedding(query)

results = []

for index, (chunk, embedding) in enumerate(
    zip(chunks, chunk_embeddings)
):

    similarity = cosine_similarity(
        query_embedding,
        embedding
    )

    results.append(
        (similarity, index, chunk)
    )


results.sort(reverse=True)

print("\n--- TOP 3 RESULTS ---\n")

for similarity, index, chunk in results[:3]:

    print(
        f"Chunk {index + 1} | Similarity: {similarity:.4f}"
    )

    print(chunk)

    print("\n" + "=" * 80 + "\n")