import os
import gradio as gr
from loader import load_and_chunk_pdf
from embeddings_store import setup_embeddings_and_vectorstore
from retriever import rag_retrieve_factory, generate_response_factory
from ui import initialize_ui_helpers
from sessions import get_user_sessions
import ui as ui_module
import rag_system


def main():
    # Load and chunk PDF
    chunks, keyword_page_index = load_and_chunk_pdf()

    # Setup embeddings and vectorstore
    embeddings, vectorstore = setup_embeddings_and_vectorstore(chunks)

    # Initialize UI helpers with vectorstore and embeddings
    initialize_ui_helpers(vectorstore, embeddings, keyword_page_index)

    # Build Gradio interface (same layout as original code)
    with gr.Blocks() as demo:
        gr.Markdown("## 🤖 Memory Chatbot")

        # Login page
        with gr.Group(visible=True) as login_page:
            username_input = gr.Textbox(label="Username")
            password_input = gr.Textbox(label="Password", type="password")
            login_button = gr.Button("Login")
            register_button = gr.Button("Register")
            login_status = gr.Textbox(label="Status", interactive=False)

        # Chat page
        with gr.Group(visible=False) as chat_page:
            gr.Markdown("### 💬 Chat Interface")

            with gr.Row():
                with gr.Column(scale=1, min_width=220):
                    gr.Markdown("### 💾 Sessions")
                    session_dropdown = gr.Dropdown(label="Select Chat Session", choices=[], value=None)
                    with gr.Row():
                        new_chat_button = gr.Button("➕ New Chat")
                        delete_chat_button = gr.Button("🗑️ Delete Chat")
                        logout_button = gr.Button("🚪 Logout", variant="secondary")

                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(label="Chat History", type="messages", height=500)
                    with gr.Row():
                        message_input = gr.Textbox(
                            label="Type your message here",
                            placeholder="Press Enter or click Send...",
                            scale=4
                        )
                        send_button = gr.Button("📨 Send", variant="secondary", scale=1)

        username_state = gr.State()
        session_id_state = gr.State()

        # ------------------ FUNCTION HANDLERS ------------------

        def handle_login(username, password):
            return gr.update(visible=True), gr.update(visible=False), "", None, None

    def select_session(username, session_name):
            sessions = get_user_sessions(username)
            session = next((s for s in sessions if s["name"] == session_name), None)
            if session:
                hist = get_session_history(username, session["id"])
                msgs = [
                    {"role": "user", "content": m.content} if i % 2 == 0 else
                    {"role": "assistant", "content": m.content}
                    for i, m in enumerate(hist.messages)
                ]
                return msgs, session["id"]
            return [], None

    def handle_new_chat(username):
            session_id, session_name = create_new_session(username)
            sessions = get_user_sessions(username)
            choices = [s["name"] for s in sessions]
            return gr.update(choices=choices, value=session_name), session_id

    def handle_delete_chat(username, session_name):
            if not session_name:
                return gr.update(choices=[], value=None)
            users_col.update_one({"username": username}, {"$pull": {"sessions": {"name": session_name}}})
            sessions = get_user_sessions(username)
            choices = [s["name"] for s in sessions]
            new_value = choices[0] if choices else None
            return gr.update(choices=choices, value=new_value)

        # ------------------ BUTTON ACTIONS ------------------

    login_button.click(
            handle_login,
            inputs=[username_input, password_input],
            outputs=[login_status, login_page, chat_page, session_dropdown, username_state, session_id_state]
        )

    register_button.click(
            handle_register,
            inputs=[username_input, password_input],
            outputs=[login_status, login_page, chat_page, session_dropdown, username_state, session_id_state]
        )

    logout_button.click(
            handle_logout,
            outputs=[login_page, chat_page, login_status, username_state, session_id_state]
        )

    session_dropdown.change(
            select_session,
            inputs=[username_state, session_dropdown],
            outputs=[chatbot, session_id_state]
        )

    new_chat_button.click(
            handle_new_chat,
            inputs=[username_state],
            outputs=[session_dropdown, session_id_state]
        )

    delete_chat_button.click(
            handle_delete_chat,
            inputs=[username_state, session_dropdown],
            outputs=[session_dropdown]
        )

    message_input.submit(
            ui_module.respond,
            inputs=[message_input, chatbot, username_state, session_id_state],
            outputs=[chatbot, message_input, session_dropdown]
        )

    send_button.click(
            ui_module.respond,
            inputs=[message_input, chatbot, username_state, session_id_state],
            outputs=[chatbot, message_input, session_dropdown]
        )

        # ------------------ LAUNCH APP ------------------
    demo.launch()


if __name__ == "__main__":
    main()
