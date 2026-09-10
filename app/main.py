from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Document AI Assistant is running!"}


@app.post("/ask")
def ask(request: QuestionRequest):
    return {
        "question": request.question,
        "answer": f"You asked: {request.question}"
    }