import gradio as gr

def respond(message, chat_history, username, session_id):
    query_log[-1] = {  # update the last query with all metrics
        "query": message,
        "retrieved_docs_count": len(retrieved_docs),
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "ragas": ragas
    }

    print(f"Faithfulness: {faithfulness:.2f}, Relevance: {relevance:.2f}, RAGAS: {ragas:.2f}")

    # Auto-update chat title from first message (and update sidebar)
    user_doc = users_col.find_one({"username": username})
    updated = False

    for s in user_doc.get("sessions", []):
        if s["id"] == session_id and (s["name"].startswith("New Chat") or not s["name"]):
            new_name = message[:40].capitalize()
            users_col.update_one(
                {"username": username, "sessions.id": session_id},
                {"$set": {"sessions.$.name": new_name}}
            )
            updated = True
            break

    hist.add_ai_message(response_text)
    save_session_history(username, session_id, hist)

    chat_history += [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_text}
    ]

    # If name was updated, refresh dropdown choices dynamically
    if updated:
        sessions = get_user_sessions(username)
        choices = [s["name"] for s in sessions]
        return chat_history, "", gr.update(choices=choices, value=new_name)
    else:
        return chat_history, "", gr.update()
