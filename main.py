#--------------------Libraries Import-------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import gradio as gr
import os
from pymongo import MongoClient
import bcrypt
import uuid
import re
#--------------- Load API Keys ---------------
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

if not gemini_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")
if not mongo_uri:
    raise ValueError("MONGO_URI not found in .env file!")

#--------------- Connect to MongoDB ---------------
client = MongoClient(mongo_uri)
db = client["chatbot_db"]
users_col = db["users"]

# ----------- RAG Setup: Load Nestlé Financial Statements PDF -----------
nestle_pdf_path = "NESTLE-financial-statements-2024.pdf"

if not os.path.exists(nestle_pdf_path):
    raise FileNotFoundError(
        f"❌ {nestle_pdf_path} not found! Please place the file in the correct folder."
    )

# ---------------- PDF Loading (Unstructured Only) ----------------
print("🔍 Loading PDF using PyMuPDFLoader...")
loader = PyMuPDFLoader(nestle_pdf_path)
documents_raw = loader.load()
print(f"✅ PyMuPDFLoader: successfully loaded {len(documents_raw)} pages.")

# ---------------- Split the Nestlé document into smaller chunks ----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = []
for i, doc in enumerate(documents_raw):
    page_content = doc.page_content.strip()
    if not page_content:
        continue
    split_texts = text_splitter.split_text(page_content)
    for chunk_text in split_texts:
        chunks.append(Document(
            page_content=chunk_text,
            metadata={"page": i + 1, "section": "Financial Statement"}
        ))

print(f"✅ Created {len(chunks)} text chunks from Nestlé financial report.")

keyword_page_index = {}

for doc in chunks:
    page = doc.metadata.get("page")
    text = doc.page_content.lower()

    # Extract financial keywords
    keywords = re.findall(r'\b(net profit|total revenue|equity|cash|income|assets|liabilities|shareholders)\b', text)

    for kw in keywords:
        if kw not in keyword_page_index:
            keyword_page_index[kw] = set()
        keyword_page_index[kw].add(page)

print("✅ Keyword-to-page index created.")
print(keyword_page_index)

# ---------------- Load Embedding Model ----------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=gemini_key
)
print("✅ Embedding model loaded successfully!")

# ---------------- Chroma Vectorstore Setup ----------------
chroma_path = "chroma_db"

if os.path.exists(chroma_path):
    print("💾 Loading existing Chroma vector database...")
    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    print("✅ Existing Chroma vector database loaded successfully!")
else:
    print("🧠 Creating new Chroma vector database (first-time embedding)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    print("✅ Chroma vector database created and automatically saved to disk!")

#--------------- Connect Gemini AI -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_key,
    temperature=0.7
)

# ---------------- Chat Memory Setup ----------------
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

# ---------------- Professional RAG System Prompt ----------------
SYSTEM_PROMPT = """
SYSTEM:
You are **Nestlé Financial Insight Assistant**, an AI system specialized in analyzing and summarizing corporate financial documents.
Your task is to answer questions using only the verified information retrieved from the Nestlé Annual Financial Report (2024).

### Guidelines:
1. Use **only** the provided CONTEXT from the financial report. Never use external or assumed knowledge.
2. When the context includes **tables or tabular numeric data**, interpret values **column-wise** — keep each column internally consistent.
   - Example: if a table shows “2022 | 2023 | 2024” with numbers, never mix values across years.
   - Preserve the structure of percentages, units, and figures exactly as written.
3. Always base your answers on the exact figures, tables, or text shown in the context.
4. If comparing financial trends (e.g., 2023 vs. 2024), use only data explicitly available in the CONTEXT.
5. If partial data is available, summarize what is known and state that certain details are not mentioned.
6. Maintain a **clear, factual, professional** tone suitable for corporate financial analysis — no speculation or generic commentary.
7. Only if the context contains **no relevant information at all**, respond exactly with:
   "I don't have this information in the document."
8. If the user’s question includes a specific date or year (e.g., “as of January 1, 2024” or “latest figures”), use the most recent data available in the CONTEXT. Otherwise, do not assume or infer any date
9.If the user asks about a term or keyword that exists anywhere in the document, acknowledge it and provide any nearby or related information from the CONTEXT — do not say “I don’t have this information” if the term appears in the document text
Now, carefully interpret the CONTEXT below. If tabular data appears, treat each row and column logically before answering the user’s question.

---
CONTEXT:
{context_text}

---
USER QUESTION:
{user_input}

---
FINAL ANSWER:
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# ---------------- Graph Setup ----------------
workflow = StateGraph(dict)

def format_doc_for_context(d: Document):
    meta = d.metadata or {}
    page = meta.get("page")
    header = f"Page {page or 'unknown'}:\n"
    return header + d.page_content

rag_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})


def rag_retrieve(state):
    query = state["message"]
    docs = rag_retriever.invoke(query)  # RAG retrieval
    for d in docs:
        print(f"Page {d.metadata.get('page')} | Table: {d.metadata.get('is_table', False)}")

    context_text = "\n\n".join([format_doc_for_context(d) for d in docs[:5]])

    # Save retrieved documents in state for metrics
    state["context"] = context_text
    state["retrieved_docs"] = docs[:5]  # <-- Add this line

    return state

def generate_response(state):
    user_input = state["message"]
    try:
        docs = rag_retriever.get_relevant_documents(user_input)
    except Exception:
        docs = rag_retriever._get_relevant_documents(user_input, run_manager=None)
    top_docs = docs[:5]
    context_text = "\n\n".join([format_doc_for_context(d) for d in top_docs])
    combined_input = f"""{SYSTEM_PROMPT}

Here is the relevant section from the Nestlé document:
{context_text}

Question from User: {user_input}
"""
    response = llm.invoke(combined_input)
    state["context_used"] = top_docs
    return {"response": response.content}

# Manual command check (greeting/name memory)
def check_name_or_greeting(state):
    message_lower = state["message"].lower().strip()
    user_doc = users_col.find_one({"username": state["username"]})
    stored_name = user_doc.get("name") if user_doc else None
    response_text = None

    if "my name is" in message_lower:
        extracted_name = message_lower.split("my name is")[-1].strip().split()[0].capitalize()
        users_col.update_one({"username": state["username"]}, {"$set": {"name": extracted_name}})
        response_text = f"Nice to meet you, {extracted_name}. I'll remember your name."
    elif any(p in message_lower for p in ["what is my name", "who am i", "do you know my name"]):
        response_text = f"Your name is {stored_name}!" if stored_name else "I don't know your name yet. What should I call you?"
    elif any(greet in message_lower.split() for greet in ["hello", "hi", "hey"]):
        response_text = f"Hello {stored_name}!" if stored_name else "Hello! What should I call you?"

    state["manual_response"] = response_text
    return state

def choose_next(state):
    query = state["message"].lower()
    if any(word in query for word in ["nestle", "report", "financial", "balance", "income", "cash", "equity", "statement", "note", "2024"]):
        return "use_rag"
    return "use_llm"

# Graph Nodes
workflow.add_node("check_command", check_name_or_greeting)
workflow.add_node("rag_retrieve", rag_retrieve)
workflow.add_node("generate_llm", generate_response)

workflow.set_entry_point("check_command")
workflow.add_conditional_edges("check_command", choose_next, {
    "manual_done": END,
    "use_rag": "rag_retrieve",
    "use_llm": "generate_llm"
})
workflow.add_edge("rag_retrieve", "generate_llm")
workflow.add_edge("generate_llm", END)

# Memory & App Compile
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

def extract_doc_ids(docs):
    """
    Convert a list of retrieved or relevant docs into a set of page numbers or IDs.
    Handles both Document objects and ints (page numbers).
    """
    ids = set()
    for d in docs:
        if hasattr(d, "metadata"):  # It's a Document
            page = d.metadata.get("page")
            if page is not None:
                ids.add(page)
        elif isinstance(d, int):  # Already a page number
            ids.add(d)
        else:
            # Fallback for any other object types (optional)
            ids.add(str(d))
    return ids

def extract_facts(text):
    """
    Extract facts from text.
    For simplicity, this example extracts numbers, percentages, and monetary values.
    Returns a set of strings.
    """
    # Extract numbers (integers, decimals)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    # Extract percentages
    percentages = re.findall(r'\b\d+(?:\.\d+)?%', text)
    # Extract monetary values like $123, 123 USD, etc.
    money = re.findall(r'\$\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?USD', text, flags=re.IGNORECASE)

    # Combine all facts into a set
    facts = set(numbers + percentages + money)
    return facts

# Compute precision and recall for RAG retrieved documents
def compute_precision_recall(retrieved_docs, relevant_docs):
    retrieved_set = extract_doc_ids(retrieved_docs)
    relevant_set = extract_doc_ids(relevant_docs)

    if not retrieved_set or not relevant_set:
        return 0.0, 0.0

    true_positives = retrieved_set.intersection(relevant_set)
    precision = len(true_positives) / len(retrieved_set)
    recall = len(true_positives) / len(relevant_set)

    return precision, recall

# ---------------- Helper: Compute MRR ----------------
def compute_mrr(retrieved_docs, relevant_docs):
    retrieved_pages = list(extract_doc_ids(retrieved_docs))
    relevant_pages = extract_doc_ids(relevant_docs)

    for idx, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return 1.0 / idx
    return 0.0

def compute_faithfulness(answer, retrieved_docs):
    answer_facts = extract_facts(answer)  # implement extract_facts
    doc_facts = set()
    for doc in retrieved_docs:
        doc_facts.update(extract_facts(doc.page_content))
    matched = sum(1 for fact in answer_facts if fact in doc_facts)
    return matched / max(1, len(answer_facts))

def compute_relevance(answer, query, retrieved_docs):
    # Simple keyword overlap or embedding similarity
    keywords = set(query.lower().split())
    context_text = " ".join([d.page_content for d in retrieved_docs]).lower()
    matched = sum(1 for word in keywords if word in context_text)
    return matched / max(1, len(keywords))

def compute_ragas(precision, recall, faithfulness, relevance, weights=None):
    """
    Compute RAG Accuracy Score (RAGAS) as a weighted sum of metrics.
    Default weights = equal (0.25 each)
    """
    if weights is None:
        weights = {"precision": 0.25, "recall": 0.25, "faithfulness": 0.25, "relevance": 0.25}

    ragas = (
            precision * weights["precision"] +
            recall * weights["recall"] +
            faithfulness * weights["faithfulness"] +
            relevance * weights["relevance"]
    )
    return ragas


def get_relevant_docs_for_query(query, retrieved_docs, embeddings, threshold=0.6):
    """
    Determine relevant documents for a query based on cosine similarity of embeddings.
    Returns only the docs with similarity above the threshold.
    """
    # Create embedding for the query
    query_embedding = embeddings.embed_query(query)

    relevant_docs = []
    for doc in retrieved_docs:
        doc_embedding = embeddings.embed_query(doc.page_content)
        similarity = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(doc_embedding).reshape(1, -1)
        )[0][0]

        if similarity >= threshold:
            relevant_docs.append(doc)

    return relevant_docs

#----------Globally Query Log-------------
query_log = []
# Gradio Chat Functions
def respond(message, chat_history, username, session_id):
    # ---------------- Step 0: Log user query ----------------
    if message.strip():  # only log non-empty messages
        query_log.append(message)

    # ---------------- Existing logic ----------------
    if not message.strip():
        return chat_history, ""

    # Load session history
    hist = get_session_history(username, session_id)
    hist.add_user_message(message)

    # RAG retrieval
    state = {
        "username": username,
        "message": message,
        "history": hist.messages
    }
    state = rag_retrieve(state)
    retrieved_context = state.get("context", "")
    retrieved_docs = state.get("retrieved_docs", [])
    print(f"Retrieved context for this query:\n{retrieved_context[:500]}...")

    query_lower = message.lower()

    # Find which pages should be retrieved based on keywords in the query
    relevant_pages = set()
    for kw, pages in keyword_page_index.items():
        if kw in query_lower:
            relevant_pages.update(pages)

    retrieved_pages = [d.metadata.get("page") for d in retrieved_docs]

    if relevant_pages:
        precision, recall = compute_precision_recall(retrieved_pages, relevant_pages)
        mrr = compute_mrr(retrieved_pages, relevant_pages)
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, MRR: {mrr:.2f}")
    else:
        print("⚠️ No matching keywords found in index for this query. Metrics skipped.")
    # ------------------- Evaluate RAG metrics -------------------
    relevant_docs = get_relevant_docs_for_query(message, retrieved_docs, embeddings, threshold=0.75)

    precision, recall = compute_precision_recall(retrieved_docs, relevant_docs)
    mrr = compute_mrr(retrieved_docs, relevant_docs)

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, MRR: {mrr:.2f}")
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

        combined_input = f"""{SYSTEM_PROMPT}

    Here is the relevant section from the Nestlé document:
    {retrieved_context}

    Question from User: {message}
    """
        response = llm.invoke(combined_input)
        response_text = response.content

        faithfulness = compute_faithfulness(response_text, retrieved_docs)
        relevance = compute_relevance(response_text, message, retrieved_docs)
        RAGAS = (faithfulness + relevance + (precision + recall) / 2) / 3

        print(f"Faithfulness: {faithfulness:.2f}, Relevance: {relevance:.2f}, RAGAS: {RAGAS:.2f}")

        ragas = compute_ragas(
            precision=precision,  # make sure you have computed precision
            recall=recall,  # and recall
            faithfulness=faithfulness,
            relevance=relevance
        )

        print(f"Faithfulness: {faithfulness:.2f}, Relevance: {relevance:.2f}, RAGAS: {ragas:.2f}")

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

# Gradio Interface (unchanged behavior)
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
            first_name = choices[0] if choices else None
            first_id = sessions[0]["id"] if sessions else None
            return ("Login successful!", gr.update(visible=False), gr.update(visible=True),
                    gr.update(choices=choices, value=first_name), username, first_id)
        else:
            return ("Invalid username or password!", gr.update(visible=True), gr.update(visible=False),
                    gr.update(choices=[]), None, None)

    def handle_register(username, password):
        msg = register_user(username, password)
        if "successfully" in msg.lower():
            return (msg, gr.update(visible=False), gr.update(visible=True),
                    gr.update(choices=[], value=None), username, None)
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
