import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:4b-instruct"


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Document AI Assistant is running!"}


@app.post("/ask")
def ask(request: QuestionRequest):

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "Answer clearly and concisely."
            },
            {
                "role": "user",
                "content": request.question
            }
        ],
        "stream": False
    }

    try:
        response = httpx.post(
            OLLAMA_URL,
            json=payload,
            timeout=120.0
        )

        response.raise_for_status()

    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not communicate with Ollama: {error}"
        )

    data = response.json()

    return {
        "question": request.question,
        "answer": data["message"]["content"]
    }