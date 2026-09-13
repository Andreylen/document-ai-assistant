from pathlib import Path

from pypdf import PdfReader

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path("data/paper.pdf")

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


print("Reading PDF...")

text = extract_pdf_text(PDF_PATH)


# 1. SPLITTING

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

documents = text_splitter.create_documents([text])

print(f"Number of chunks: {len(documents)}")


# 2. EMBEDDINGS

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# 3. VECTOR STORE

print("Creating embeddings and vector store...")

vector_store = InMemoryVectorStore.from_documents(
    documents,
    embedding=embeddings
)


# 4. USER QUESTION

query = "What integrated luminosity was used in this study?"


# 5. RETRIEVAL

relevant_documents = vector_store.similarity_search(
    query,
    k=TOP_K
)

print("\n--- RETRIEVED CONTEXT ---\n")

for index, document in enumerate(relevant_documents, start=1):
    print(f"Result {index}:")
    print(document.page_content)
    print("\n" + "=" * 80 + "\n")


# 6. BUILD CONTEXT

context = "\n\n---\n\n".join(
    document.page_content
    for document in relevant_documents
)


# 7. LLM

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)


# 8. PROMPT

prompt = f"""
Use only the context below to answer the question.
If the answer is not contained in the context, say that you cannot find it in the document.

CONTEXT:
{context}

QUESTION:
{query}
"""


# 9. GENERATION

response = llm.invoke(
    [
        (
            "system",
            "You are a document question-answering assistant. "
            "Answer clearly and concisely using only the provided context."
        ),
        (
            "human",
            prompt
        )
    ]
)


print("\n--- FINAL ANSWER ---\n")
print(response.content)