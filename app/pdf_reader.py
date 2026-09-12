from pathlib import Path

from pypdf import PdfReader


PDF_PATH = Path("data/paper.pdf")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)

    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
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

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


text = extract_pdf_text(PDF_PATH)

chunks = split_text_into_chunks(text)

print(f"Total text length: {len(text)} characters")
print(f"Number of chunks: {len(chunks)}")

print("\n--- CHUNK 1 ---\n")
print(chunks[0])

print("\n--- CHUNK 2 ---\n")
print(chunks[1])