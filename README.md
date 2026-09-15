# Document AI Assistant

A local document question-answering application built with FastAPI, LangChain, Qdrant, PostgreSQL, Docker, and a Qwen large language model served through Ollama.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from a PDF document and generate grounded answers based only on the retrieved context.

## Architecture

```text
User
  |
  v
FastAPI
  |
  +--> Qdrant
  |      |
  |      +--> semantic vector search
  |
  +--> Ollama
  |      |
  |      +--> Qwen LLM
  |      +--> Qwen embeddings
  |
  +--> PostgreSQL
         |
         +--> query history
```

## Main Features

- PDF question answering using RAG
- Semantic search with Qdrant
- Local Qwen LLM through Ollama
- Embeddings generated locally
- FastAPI REST API
- PostgreSQL query history
- Docker-based service orchestration
- Git and GitHub version control

## Technology Stack

- Python
- FastAPI
- LangChain
- Qdrant
- PostgreSQL
- SQL
- Ollama
- Qwen
- Docker
- Git / GitHub

## API

The application provides the following main endpoints:

### `POST /ask`

Accepts a question, retrieves relevant document chunks, generates an answer using the local LLM, and stores the interaction in PostgreSQL.

Example request:

```json
{
  "question": "What integrated luminosity was used in the study?"
}
```

### `GET /history`

Returns previously submitted questions, generated answers, response times, and timestamps.

## Requirements

Before starting the application, install:

- Docker Desktop
- Ollama

Pull the required Ollama models:

```bash
ollama pull qwen3:4b-instruct
ollama pull qwen3-embedding:0.6b
```

## Setup

Clone the repository:

```bash
git clone https://github.com/Andreylen/document-ai-assistant.git
cd document-ai-assistant
```

Create a local `.env` file based on `.env.example`.

Place the PDF document at:

```text
data/paper.pdf
```

Make sure Ollama is running.

Then start the application:

```bash
docker compose up -d --build
```

Open the interactive FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## Stopping the Application

```bash
docker compose stop
```

To start it again:

```bash
docker compose up -d
```

## How RAG Works

1. The PDF text is extracted and split into overlapping chunks.
2. Each chunk is converted into an embedding using a local embedding model.
3. The embeddings are stored in Qdrant.
4. A user question is converted into an embedding.
5. Qdrant retrieves the most relevant document chunks.
6. The retrieved text is passed to the Qwen LLM as context.
7. The generated answer and response time are stored in PostgreSQL.

## Project Status

The main local application is complete and includes:

- RAG-based document retrieval
- local LLM inference
- persistent vector storage
- SQL query history
- REST API
- Docker containerization

Possible future improvements include cloud deployment, authentication, support for multiple documents, and a dedicated web interface.