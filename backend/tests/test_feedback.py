"""Tests for feedback submission endpoint."""
from core.database import ChatHistory, Feedback


def test_submit_feedback(client, db_session, test_user, test_session):
    chat_entry = ChatHistory(
        user_id=test_user.id,
        session_id=test_session.id,
        question="Test Question",
        answer="Test Answer",
        confidence_score=0.9
    )
    db_session.add(chat_entry)
    db_session.commit()
    db_session.refresh(chat_entry)

    payload = {
        "history_id": chat_entry.id,
        "rating": 5,
        "comment": "Great answer!"
    }

    response = client.post("/api/feedback", json=payload)
    assert response.status_code == 200
    assert response.json()["message"] == "Feedback received successfully"

    feedback_in_db = db_session.query(Feedback).filter(
        Feedback.chat_history_id == chat_entry.id
    ).first()
    assert feedback_in_db is not None
    assert feedback_in_db.rating == 5


def test_feedback_invalid_id(client, test_user):
    payload = {"history_id": 9999, "rating": 1}
    response = client.post("/api/feedback", json=payload)
    assert response.status_code == 404