import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.database import save_query, get_query_history
from app.rag_service import answer_question


app = FastAPI(
    title="Document AI Assistant",
    description="RAG-based document question-answering API",
)


BASE_DIR = Path(__file__).resolve().parent


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Document AI Assistant is running!",
    }


@app.get("/history")
def history(
    limit: int = Query(default=10, ge=1, le=100)
):
    try:
        return get_query_history(limit)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        start_time = time.perf_counter()

        answer = answer_question(request.question)

        response_time_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        save_query(
            question=request.question,
            answer=answer,
            response_time_ms=response_time_ms,
        )

        return {
            "question": request.question,
            "answer": answer,
            "response_time_ms": response_time_ms,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )