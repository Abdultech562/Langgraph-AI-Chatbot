from langchain_core.chat_history import InMemoryChatMessageHistory
from db import users_col
import uuid


def get_session_history(username, session_id):
    hist = InMemoryChatMessageHistory()
    user_doc = users_col.find_one({"username": username})
    if user_doc:
        session = next((s for s in user_doc.get("sessions", []) if s["id"] == session_id), None)
        if session:
            for msg in session.get("messages", []):
                hist.add_user_message(msg["user"])
                hist.add_ai_message(msg["bot"])
    return hist


def save_session_history(username, session_id, hist):
    msgs = [
        {
            "user": hist.messages[i].content,
            "bot": hist.messages[i + 1].content if i + 1 < len(hist.messages) else ""
        }
        for i in range(0, len(hist.messages), 2)
    ]
    users_col.update_one(
        {"username": username, "sessions.id": session_id},
        {"$set": {"sessions.$.messages": msgs}}
    )


def create_new_session(username, first_message=None):
    user_doc = users_col.find_one({"username": username})
    session_id = str(uuid.uuid4())

    if first_message:
        session_name = first_message.strip().capitalize()[:40]
    else:
        count = len(user_doc.get('sessions', [])) + 1 if user_doc else 1
        session_name = f"New Chat ({count})"

    users_col.update_one(
        {"username": username},
        {"$push": {"sessions": {"id": session_id, "name": session_name, "messages": []}}}
    )

    return session_id, session_name


def get_user_sessions(username):
    user_doc = users_col.find_one({"username": username})
    return user_doc.get("sessions", []) if user_doc else []


def save_user_name(username, name):
    users_col.update_one({"username": username}, {"$set": {"name": name}})
