# Feedback Data Models Documentation

## 1. Overview
The Feedback system captures user satisfaction metrics for AI-generated responses. It links a 1-5 star rating to a specific chat history entry, allowing for quality assessment of the confidence scoring engine.

---

## 2. Database Schema (SQLAlchemy)

### Table: `feedback`
This table stores the persistent feedback records.

| Column Name       | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **`id`** | `Integer` | `Primary Key` | Unique identifier for the feedback entry. |
| **`chat_history_id`** | `Integer` | `ForeignKey`, `Unique` | Links to the `chat_history` table. One-to-One relationship. |
| **`rating`** | `Integer` | `Not Null` | User rating value (1-5). |
| **`comment`** | `String` | `Nullable` | Optional text feedback from the user. |

### Relationships
* **ChatHistory (One-to-One):** Each `Feedback` entry is linked to exactly one `ChatHistory` record.
* **User (Indirect):** Feedback is linked to a User through the `ChatHistory` table.

---

## 3. API Data Models (Pydantic)

These models define the structure of JSON data exchanged between the Frontend and Backend.

### Request Model: `FeedbackRequest`
**Endpoint:** `POST /api/feedback`

```json
{
  "history_id": 123,
  "rating": 5,
  "comment": "Excellent answer, very accurate."
}