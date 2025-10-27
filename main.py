#--------------------Libraries Import-------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import gradio as gr
import os
from pymongo import MongoClient
import bcrypt
import uuid

#--------------Load API-------------
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

if not gemini_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")
if not mongo_uri:
    raise ValueError("MONGO_URI not found in .env file!")

#---------------Connect to MongoDB------------------
client = MongoClient(mongo_uri)
db = client["chatbot_db"]
users_col = db["users"]

# ----------- RAG Setup: Load Nestlé Financial Statements PDF -----------
nestle_pdf_path = "NESTLE-financial-statements-2024.pdf"

try:
    loader = PyMuPDFLoader(nestle_pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from Nestlé financial report.")
except Exception as e:
    print(f"⚠️ Error loading Nestlé PDF: {e}")

# ----------- Split the Nestlé document into smaller chunks -----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # around 1 000 characters per chunk
    chunk_overlap=150,   # small overlap keeps context connected
)

chunks = text_splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} text chunks for RAG retrieval.")

# ----------- Generate embeddings for each chunk -----------
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=gemini_key
)

print("✅ Embedding model loaded successfully!")

# ----------- Create a FAISS vector store for quick retrieval -----------
vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ FAISS vector database created successfully!")

# Create a retriever for the chatbot to use
rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ RAG retriever is ready to use!")

#---------------Connect Gemini AI-------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_key,
    temperature= 0.5
)

# Chat Memory Setup
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
    msgs = [{"user": hist.messages[i].content,
             "bot": hist.messages[i + 1].content if i + 1 < len(hist.messages) else ""}
            for i in range(0, len(hist.messages), 2)]
    users_col.update_one(
        {"username": username, "sessions.id": session_id},
        {"$set": {"sessions.$.messages": msgs}}
    )

def create_new_session(username, first_message=None):
    """Create a new chat session with an auto title from first message."""
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

# Define prompt (same)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly chatbot that remembers everything user says, including their name."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Define combined chain
chain = prompt | llm

# Build LangGraph
workflow = StateGraph(dict)  # state graph works on dict inputs/outputs

# ----------- RAG Node: Retrieve relevant Nestlé financial info -----------
def rag_retrieve(state):
    """RAG node: retrieves relevant context from Nestlé financial data."""
    query = state["message"]

    # Search in vector store
    docs = rag_retriever.invoke(query)
    print("\n--- Retrieved context ---")
    for i, d in enumerate(docs, 1):
        print(f"[Chunk {i}] {d.page_content[:300]}...\n")
    print("--------------------------\n")

    context_text = "\n\n".join([d.page_content for d in docs])

    # Save context for LLM
    state["context"] = context_text
    return state

def generate_response(state):
    """LLM node: takes input, optional context, returns response text."""
    user_input = state["message"]
    chat_history = state["history"]
    context = state.get("context", "")

    # Add context to the user input if available
    if context:
        combined_input = f"Use the following Nestlé financial data to answer accurately:\n\n{context}\n\nQuestion: {user_input}"
    else:
        combined_input = user_input

    response = chain.invoke({
        "input": combined_input,
        "history": chat_history
    })

    state["response"] = response.content
    return state

def check_name_or_greeting(state):
    """Handle name/greeting messages manually before calling LLM."""
    message_lower = state["message"].lower().strip()
    user_doc = users_col.find_one({"username": state["username"]})
    stored_name = user_doc.get("name") if user_doc else None
    response_text = None

    if "my name is" in message_lower:
        extracted_name = message_lower.split("my name is")[-1].strip().split()[0].capitalize()
        users_col.update_one({"username": state["username"]}, {"$set": {"name": extracted_name}})
        response_text = f"It's great to meet you, {extracted_name}! I'll remember your name."
    elif any(p in message_lower for p in [
        "what is my name", "what's my name", "whats my name",
        "do you know my name", "who am i", "tell me my name"
    ]):
        if stored_name:
            response_text = f"Your name is {stored_name}!"
        else:
            response_text = "I don't know your name yet! What should I call you?"
    elif any(greet in message_lower.split() for greet in ["hello", "hi", "hey"]):
        if stored_name:
            response_text = f"Hello {stored_name}! How can I help you today?"
        else:
            response_text = "Hello! I don’t know your name yet. What should I call you?"

    state["manual_response"] = response_text
    return state

def choose_next(state):
    """Decide next step based on whether a manual response exists."""
    if state.get("manual_response"):
        state["response"] = state["manual_response"]
        return "manual_done"
    return "use_llm"

# Add nodes
# ----------- Add RAG Node and Update Workflow -----------
# Add nodes
workflow.add_node("check_command", check_name_or_greeting)
workflow.add_node("rag_retrieve", rag_retrieve)
workflow.add_node("generate_llm", generate_response)

# --- Conditional routing function ---
def choose_next(state):
    """Decide next step based on manual or financial question."""
    if state.get("manual_response"):
        state["response"] = state["manual_response"]
        return "manual_done"

    msg = state["message"].lower()
    finance_keywords = [
        "nestlé", "sales", "revenue", "profit", "income", "assets",
        "liabilities", "financial", "balance sheet", "statement", "cash flow"
    ]

    # If message has finance-related keywords, use RAG
    if any(word in msg for word in finance_keywords):
        return "use_rag"
    return "use_llm"

# --- Graph edges ---
workflow.set_entry_point("check_command")
workflow.add_conditional_edges("check_command", choose_next, {
    "manual_done": END,
    "use_rag": "rag_retrieve",
    "use_llm": "generate_llm"
})
workflow.add_edge("rag_retrieve", "generate_llm")
workflow.add_edge("generate_llm", END)

# Memory for checkpointing (optional but recommended)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Login Functions
def register_user(username, password):
    if users_col.find_one({"username": username}):
        return "Username already exists!"
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users_col.insert_one({
        "username": username,
        "password": hashed,
        "name": None,
        "sessions": []
    })
    return "User registered successfully!"

def login_user(username, password):
    user = users_col.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        return True
    return False

# Helper function to save name persistently
def save_user_name(username, name):
    users_col.update_one({"username": username}, {"$set": {"name": name}})

# Gradio Chat Functions
def respond(message, chat_history, username, session_id):
    if not message.strip():
        return chat_history, ""

    hist = get_session_history(username, session_id)
    hist.add_user_message(message)
    user_doc = users_col.find_one({"username": username})
    stored_name = user_doc.get("name") if user_doc else None

    message_lower = message.lower().replace("’", "'").strip()
    response_text = None

    if "my name is" in message_lower:
        extracted_name = message_lower.split("my name is")[-1].strip().split()[0].capitalize()
        users_col.update_one({"username": username}, {"$set": {"name": extracted_name}})
        stored_name = extracted_name
        response_text = f"It's great to meet you, {stored_name}! I'll remember your name."

    elif any(phrase in message_lower for phrase in [
        "what is my name", "what's my name", "whats my name",
        "do you know my name", "who am i", "tell me my name", "remember my name"
    ]):
        if stored_name:
            response_text = f"Your name is {stored_name}!"
        else:
            response_text = "I don't know your name yet! You haven't told me. What should I call you?"

    elif any(greet in message_lower.split() for greet in ["hello", "hi", "hey"]):
        if stored_name:
            response_text = f"Hello {stored_name}! How can I help you today?"
        else:
            response_text = "Hello! I don’t know your name yet. What should I call you?"

    # AI response if none of the above
    if response_text is None:
        state = {
            "username": username,
            "message": message,
            "history": hist.messages
        }
        result = app.invoke(
            state,
            config={"configurable": {"thread_id": session_id, "checkpoint_ns": "chatbot_memory"}}
        )
        response_text = result["response"]

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

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Memory Chatbot")

    # login page
    with gr.Group(visible=True) as login_page:
        username_input = gr.Textbox(label="Username")
        password_input = gr.Textbox(label="Password", type="password")
        login_button = gr.Button("Login")
        register_button = gr.Button("Register")
        login_status = gr.Textbox(label="Status", interactive=False)

    # chat page
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
                    message_input = gr.Textbox(label="Type your message here", placeholder="Press Enter or click Send...", scale=4)
                    send_button = gr.Button("📨 Send", variant="secondary", scale=1)

    username_state = gr.State()
    session_id_state = gr.State()

    def handle_login(username, password):
        if login_user(username, password):
            sessions = get_user_sessions(username)
            choices = [s["name"] for s in sessions]
            first_value = choices[0] if choices else None
            return ("Login successful!", gr.update(visible=False), gr.update(visible=True),
                    gr.update(choices=choices, value=first_value), username, first_value)
        else:
            return ("Invalid username or password!", gr.update(visible=True), gr.update(visible=False),
                    gr.update(choices=[]), None, None)

    def handle_register(username, password):
        msg = register_user(username, password)
        if "successfully" in msg.lower():
            return (msg, gr.update(visible=False), gr.update(visible=True),
                    gr.update(choices=[]), username, None)
        return msg, gr.update(visible=True), gr.update(visible=False), gr.update(choices=[]), None, None

    def handle_logout():
        return gr.update(visible=True), gr.update(visible=False), "", None, None

    def select_session(username, session_name):
        sessions = get_user_sessions(username)
        session = next((s for s in sessions if s["name"] == session_name), None)
        if session:
            hist = get_session_history(username, session["id"])
            msgs = [{"role": "user", "content": m.content} if i % 2 == 0 else
                    {"role": "assistant", "content": m.content} for i, m in enumerate(hist.messages)]
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

    login_button.click(handle_login,
        inputs=[username_input, password_input],
        outputs=[login_status, login_page, chat_page, session_dropdown, username_state, session_id_state])
    register_button.click(handle_register,
        inputs=[username_input, password_input],
        outputs=[login_status, login_page, chat_page, session_dropdown, username_state, session_id_state])
    logout_button.click(handle_logout,
        outputs=[login_page, chat_page, login_status, username_state, session_id_state])
    session_dropdown.change(select_session,
        inputs=[username_state, session_dropdown],
        outputs=[chatbot, session_id_state])
    new_chat_button.click(handle_new_chat,
        inputs=[username_state],
        outputs=[session_dropdown, session_id_state])
    delete_chat_button.click(handle_delete_chat,
        inputs=[username_state, session_dropdown],
        outputs=[session_dropdown])
    message_input.submit(respond,
        inputs=[message_input, chatbot, username_state, session_id_state],
        outputs=[chatbot, message_input, session_dropdown])
    send_button.click(respond,
        inputs=[message_input, chatbot, username_state, session_id_state],
        outputs=[chatbot, message_input, session_dropdown])

demo.launch()
