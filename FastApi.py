# fastapi_rag_api.py
import os
import re
import uuid
import uvicorn
import numpy as np
from typing import List, Optional, Any, Dict, Set
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import bcrypt

# Langchain / LLM / Vector libs
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- Load environment -----------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file!")
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file!")

# ----------------- MongoDB connection -----------------
client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
users_col = db["users"]

# ----------------- PDF / RAG setup -----------------
NESTLE_PDF = "NESTLE-financial-statements-2024.pdf"
if not os.path.exists(NESTLE_PDF):
    raise FileNotFoundError(f"{NESTLE_PDF} not found. Place file in project folder.")

print("Loading PDF...")
loader = PyMuPDFLoader(NESTLE_PDF)
documents_raw = loader.load()
print(f"Loaded {len(documents_raw)} pages.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# split into Document objects with metadata
chunks: List[Document] = []
for i, doc in enumerate(documents_raw):
    page_content = doc.page_content.strip()
    if not page_content:
        continue
    split_texts = text_splitter.split_text(page_content)
    for text in split_texts:
        chunks.append(Document(page_content=text, metadata={"page": i + 1, "section": "Financial Statement"}))

print(f"Created {len(chunks)} chunks.")

# build keyword -> pages index for simple heuristics
keyword_page_index: Dict[str, Set[int]] = {}
for d in chunks:
    page = d.metadata.get("page")
    text = d.page_content.lower()
    keywords = re.findall(r'\b(net profit|total revenue|equity|cash|income|assets|liabilities|shareholders|revenue|profit)\b', text)
    for kw in keywords:
        keyword_page_index.setdefault(kw, set()).add(page)
print("Keyword page index ready.")

# ----------------- Embeddings & Vector DB (Chroma) -----------------
print("Loading embeddings model...")
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=GEMINI_KEY)
chroma_path = "chroma_db"

if os.path.exists(chroma_path):
    print("Loading existing Chroma DB...")
    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    print("Chroma loaded.")
else:
    print("Creating Chroma DB (this may take a while)...")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=chroma_path)
    print("Chroma DB created and persisted.")

# set retriever
rag_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# ----------------- LLM Setup -----------------
print("Initializing Gemini LLM...")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_KEY, temperature=0.7)

SYSTEM_PROMPT = """
SYSTEM:
You are Nestlé Financial Insight Assistant. Answer using ONLY the supplied Nestlé annual report (2024).
Follow strict guidelines about tables, units, and factual answers as described in the original prompt.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
chain = prompt | llm

# ----------------- LangGraph workflow -----------------
workflow = StateGraph(dict)

def format_doc_for_context(d: Document) -> str:
    meta = d.metadata or {}
    page = meta.get("page")
    header = f"Page {page or 'unknown'}:\n"
    return header + d.page_content

def rag_retrieve(state: dict) -> dict:
    query = state["message"]
    docs = rag_retriever.invoke(query)
    context_text = "\n\n".join([format_doc_for_context(d) for d in docs[:10]])
    state["context"] = context_text
    state["retrieved_docs"] = docs[:10]
    return state

def generate_response(state: dict) -> dict:
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

def check_name_or_greeting(state: dict) -> dict:
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

def choose_next(state: dict) -> str:
    query = state["message"].lower()
    if any(word in query for word in ["nestle", "report", "financial", "balance", "income", "cash", "equity", "statement", "note", "2024"]):
        return "use_rag"
    return "use_llm"

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

memory = MemorySaver()
app_compiled = workflow.compile(checkpointer=memory)

# ----------------- Chat memory helpers -----------------
def get_session_history(username: str, session_id: str) -> InMemoryChatMessageHistory:
    hist = InMemoryChatMessageHistory()
    user_doc = users_col.find_one({"username": username})
    if user_doc:
        session = next((s for s in user_doc.get("sessions", []) if s["id"] == session_id), None)
        if session:
            for msg in session.get("messages", []):
                hist.add_user_message(msg["user"])
                hist.add_ai_message(msg["bot"])
    return hist

def save_session_history(username: str, session_id: str, hist: InMemoryChatMessageHistory):
    msgs = [{"user": hist.messages[i].content,
             "bot": hist.messages[i + 1].content if i + 1 < len(hist.messages) else ""}
            for i in range(0, len(hist.messages), 2)]
    users_col.update_one({"username": username, "sessions.id": session_id},
                         {"$set": {"sessions.$.messages": msgs}})

def create_new_session(username: str, first_message: Optional[str] = None):
    user_doc = users_col.find_one({"username": username})
    session_id = str(uuid.uuid4())
    if first_message:
        session_name = first_message.strip().capitalize()[:40]
    else:
        count = len(user_doc.get('sessions', [])) + 1 if user_doc else 1
        session_name = f"New Chat ({count})"
    users_col.update_one({"username": username},
                         {"$push": {"sessions": {"id": session_id, "name": session_name, "messages": []}}})
    return session_id, session_name

def get_user_sessions(username: str):
    user_doc = users_col.find_one({"username": username})
    return user_doc.get("sessions", []) if user_doc else []

# ----------------- Auth -----------------
def register_user(username: str, password: str) -> str:
    if users_col.find_one({"username": username}):
        return "Username already exists!"
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users_col.insert_one({"username": username, "password": hashed, "name": None, "sessions": []})
    return "User registered successfully!"

def login_user(username: str, password: str) -> bool:
    user = users_col.find_one({"username": username})
    return bool(user and bcrypt.checkpw(password.encode(), user["password"]))

def save_user_name(username: str, name: str):
    users_col.update_one({"username": username}, {"$set": {"name": name}})

# ----------------- Metrics / utils -----------------
def extract_doc_ids(docs: List[Any]) -> Set[Any]:
    ids = set()
    for d in docs:
        if hasattr(d, "metadata"):
            page = d.metadata.get("page")
            if page is not None:
                ids.add(page)
        elif isinstance(d, int):
            ids.add(d)
        else:
            ids.add(str(d))
    return ids

def extract_facts(text: str) -> Set[str]:
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    percentages = re.findall(r'\b\d+(?:\.\d+)?%', text)
    money = re.findall(r'\$\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?USD', text, flags=re.IGNORECASE)
    return set(numbers + percentages + money)

def compute_precision_recall(retrieved_docs: List[Any], relevant_docs: List[Any]):
    retrieved_set = extract_doc_ids(retrieved_docs)
    relevant_set = extract_doc_ids(relevant_docs)
    if not retrieved_set or not relevant_set:
        return 0.0, 0.0
    tp = retrieved_set.intersection(relevant_set)
    precision = len(tp) / len(retrieved_set)
    recall = len(tp) / len(relevant_set)
    return precision, recall

def compute_mrr(retrieved_docs: List[Any], relevant_docs: List[Any]):
    retrieved_pages = list(extract_doc_ids(retrieved_docs))
    relevant_pages = extract_doc_ids(relevant_docs)
    for idx, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return 1.0 / idx
    return 0.0

def compute_faithfulness(answer: str, retrieved_docs: List[Any]):
    answer_facts = extract_facts(answer)
    doc_facts = set()
    for doc in retrieved_docs:
        doc_facts.update(extract_facts(doc.page_content))
    matched = sum(1 for fact in answer_facts if fact in doc_facts)
    return matched / max(1, len(answer_facts))

def compute_relevance(answer: str, query: str, retrieved_docs: List[Any]):
    keywords = set(query.lower().split())
    context_text = " ".join([d.page_content for d in retrieved_docs]).lower()
    matched = sum(1 for w in keywords if w in context_text)
    return matched / max(1, len(keywords))

def compute_ragas(precision, recall, faithfulness, relevance, weights=None):
    if weights is None:
        weights = {"precision": 0.25, "recall": 0.25, "faithfulness": 0.25, "relevance": 0.25}
    ragas = (precision * weights["precision"] +
             recall * weights["recall"] +
             faithfulness * weights["faithfulness"] +
             relevance * weights["relevance"])
    return ragas

def get_relevant_docs_for_query(query: str, retrieved_docs: List[Any], embeddings_model, threshold=0.6):
    # embedding-based similarity for filtering relevant docs
    query_embedding = embeddings_model.embed_query(query)
    relevant = []
    for doc in retrieved_docs:
        doc_embedding = embeddings_model.embed_query(doc.page_content)
        sim = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(doc_embedding).reshape(1, -1))[0][0]
        if sim >= threshold:
            relevant.append(doc)
    return relevant

# query log
query_log: List[dict] = []

# ----------------- FastAPI app & models -----------------
app = FastAPI(title="Nestle-RAG-FastAPI")

# CORS (allow local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class Credentials(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    username: str
    session_id: str
    message: str

class NewSessionRequest(BaseModel):
    username: str
    first_message: Optional[str] = None

# ----------------- API endpoints -----------------
@app.post("/register")
def api_register(creds: Credentials):
    msg = register_user(creds.username, creds.password)
    if "successfully" in msg.lower():
        return {"status": "success", "message": msg}
    raise HTTPException(status_code=400, detail=msg)

@app.post("/login")
def api_login(creds: Credentials):
    if login_user(creds.username, creds.password):
        return {"status": "success", "message": "Login successful"}
    raise HTTPException(status_code=401, detail="Invalid username or password")

@app.get("/sessions/{username}")
def api_get_sessions(username: str):
    return get_user_sessions(username)

@app.post("/sessions/new")
def api_create_session(req: NewSessionRequest):
    sid, sname = create_new_session(req.username, req.first_message)
    return {"id": sid, "name": sname}

@app.delete("/sessions/{username}/{session_name}")
def api_delete_session(username: str, session_name: str):
    users_col.update_one({"username": username}, {"$pull": {"sessions": {"name": session_name}}})
    return {"status": "success", "deleted": session_name}

@app.get("/keywords")
def api_keywords():
    # Return the keyword -> pages mapping (convert sets to lists)
    return {k: sorted(list(v)) for k, v in keyword_page_index.items()}

@app.get("/")
def root():
    return {"message": "Memory Chatbot API is running! Use /docs to test endpoints."}

@app.get("/query_log")
def api_query_log(limit: int = 50):
    return query_log[-limit:]

@app.post("/chat")
def api_chat(req: ChatRequest):
    message = req.message
    username = req.username
    session_id = req.session_id

    # basic validation
    if not message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # log query
    query_log.append({"query": message})

    # load session history
    hist = get_session_history(username, session_id)
    hist.add_user_message(message)

    # RAG retrieval via rag_retrieve
    state = {"username": username, "message": message, "history": hist.messages}
    state = rag_retrieve(state)
    retrieved_context = state.get("context", "")
    retrieved_docs = state.get("retrieved_docs", [])

    # compute metrics vs keyword index (optional)
    query_lower = message.lower()
    relevant_pages = set()
    for kw, pages in keyword_page_index.items():
        if kw in query_lower:
            relevant_pages.update(pages)
    retrieved_pages = [d.metadata.get("page") for d in retrieved_docs]

    if relevant_pages:
        precision_simple, recall_simple = compute_precision_recall(retrieved_pages, list(relevant_pages))
        mrr_simple = compute_mrr(retrieved_pages, list(relevant_pages))
    else:
        precision_simple = recall_simple = mrr_simple = 0.0

    # compute relevance/faithfulness against filtered relevant docs (embedding threshold)
    relevant_docs = get_relevant_docs_for_query(message, retrieved_docs, embeddings, threshold=0.75)
    precision_e, recall_e = compute_precision_recall(retrieved_docs, relevant_docs)
    mrr_e = compute_mrr(retrieved_docs, relevant_docs)

    # name/greeting checks (manual)
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
        response_text = f"Your name is {stored_name}!" if stored_name else "I don't know your name yet! What should I call you?"
    elif any(greet in message_lower.split() for greet in ["hello", "hi", "hey"]):
        response_text = f"Hello {stored_name}! How can I help you today?" if stored_name else "Hello! I don’t know your name yet. What should I call you?"

    # If not manual, call LLM
    if response_text is None:
        combined_input = f"""{SYSTEM_PROMPT}

Here is the relevant section from the Nestlé document:
{retrieved_context}

Question from User: {message}
"""
        response = llm.invoke(combined_input)
        response_text = response.content

        # compute eval metrics
        faithfulness = compute_faithfulness(response_text, retrieved_docs)
        relevance = compute_relevance(response_text, message, retrieved_docs)
        ragas_val = compute_ragas(precision_e, recall_e, faithfulness, relevance)

        # update last query log entry with metrics
        query_log[-1] = {
            "query": message,
            "retrieved_docs_count": len(retrieved_docs),
            "precision": precision_e,
            "recall": recall_e,
            "mrr": mrr_e,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "ragas": ragas_val
        }

    # update session name if it's a new chat
    user_doc = users_col.find_one({"username": username})
    updated = False
    for s in user_doc.get("sessions", []):
        if s["id"] == session_id and (s["name"].startswith("New Chat") or not s["name"]):
            new_name = message[:40].capitalize()
            users_col.update_one({"username": username, "sessions.id": session_id},
                                  {"$set": {"sessions.$.name": new_name}})
            updated = True
            break

    hist.add_ai_message(response_text)
    save_session_history(username, session_id, hist)

    # return structured payload
    return {
        "response": response_text,
        "retrieved_context_preview": retrieved_context[:1000],
        "retrieved_docs_count": len(retrieved_docs),
        "metrics": {
            "precision_simple": precision_simple,
            "recall_simple": recall_simple,
            "mrr_simple": mrr_simple,
            "precision_embed": precision_e,
            "recall_embed": recall_e,
            "mrr_embed": mrr_e,
        },
        "session_updated": updated
    }

# ----------------- Run server when executed -----------------
def main():
    uvicorn.run("FastApi:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
