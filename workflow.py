from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from retriever import rag_retrieve_factory, generate_response_factory

def build_workflow(checkpointer=None):
 workflow = StateGraph(dict)


 # Nodes are lightweight wrappers; actual functions created in main using vectorstore
 workflow.add_node("check_command", lambda s: s)
 workflow.add_node("rag_retrieve", lambda s: s)
 workflow.add_node("generate_llm", lambda s: s)


 workflow.set_entry_point("check_command")
 workflow.add_conditional_edges("check_command", lambda s: "use_rag", {
 "manual_done": END,
 "use_rag": "rag_retrieve",
 "use_llm": "generate_llm"
 })
 workflow.add_edge("rag_retrieve", "generate_llm")
 workflow.add_edge("generate_llm", END)


 memory = MemorySaver() if checkpointer is None else checkpointer
 app = workflow.compile(checkpointer=memory)
 return app

