import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_rejects_empty_or_whitespace_questions(question):
    response = client.post("/ask", json={"question": question})

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."
