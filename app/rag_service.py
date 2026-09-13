import os
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://127.0.0.1:6333",
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434",
)

PDF_PATH = BASE_DIR / "data" / "paper.pdf"

COLLECTION_NAME = "paper_chunks"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3:4b-instruct"

VECTOR_SIZE = 1024

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


embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_URL,
)

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
)

client = QdrantClient(url=QDRANT_URL)


if not client.collection_exists(COLLECTION_NAME):

    print("Qdrant collection not found.")
    print("Creating RAG index...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    text = extract_pdf_text(PDF_PATH)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = text_splitter.create_documents([text])

    print(f"Created {len(documents)} chunks.")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    vector_store.add_documents(documents)

    print("Documents stored in Qdrant.")

else:

    print("Existing Qdrant collection found.")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def answer_question(question: str) -> str:

    relevant_documents = vector_store.similarity_search(
        question,
        k=TOP_K,
    )

    context = "\n\n---\n\n".join(
        document.page_content
        for document in relevant_documents
    )

    prompt = f"""
Use only the context below to answer the question.

If the answer is not contained in the context,
say that you cannot find it in the document.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = llm.invoke(
        [
            (
                "system",
                "You are a document question-answering assistant. "
                "Answer clearly and concisely using only the provided context.",
            ),
            (
                "human",
                prompt,
            ),
        ]
    )

    return response.content