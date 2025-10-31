from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from rag_system import prompt, llm
from embeddings_store import setup_embeddings_and_vectorstore
from loader import load_and_chunk_pdf


def format_doc_for_context(d: Document):
    meta = d.metadata or {}
    page = meta.get("page")
    header = f"Page {page or 'unknown'}:\n"
    return header + d.page_content


# rag_retriever will be created by passing the vectorstore created in main
def rag_retrieve_factory(vectorstore):
    rag_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    def rag_retrieve(state):
        query = state["message"]
        docs = rag_retriever.invoke(query)  # RAG retrieval
        for d in docs:
            print(f"Page {d.metadata.get('page')} | Table: {d.metadata.get('is_table', False)}")

        context_text = "\n\n".join([format_doc_for_context(d) for d in docs[:5]])

        # Save retrieved documents in state for metrics
        state["context"] = context_text
        state["retrieved_docs"] = docs[:5]

        return state

    return rag_retrieve


def generate_response_factory(vectorstore):
    rag_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    def generate_response(state):
        user_input = state["message"]
        try:
            docs = rag_retriever.get_relevant_documents(user_input)
        except Exception:
            docs = rag_retriever._get_relevant_documents(user_input, run_manager=None)

        top_docs = docs[:5]
        context_text = "\n\n".join([format_doc_for_context(d) for d in top_docs])

        combined_input = f"""{prompt.system.content}\n\nHere is the relevant section from the Nestlé document:\n{context_text}\n\nQuestion from User: {user_input}\n"""

        # Here you might want to pass combined_input to llm for response generation
        # Example:
        # response = llm.invoke(combined_input)
        # state["response"] = response

        return state

    return generate_response
