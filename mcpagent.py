from langgraph.graph import StateGraph, END
from mcptools import (
    mcp_load_pdf,
    mcp_load_repo,
    mcp_load_arxiv,
    mcp_qa_with_history,
    mcp_summary
)

# -------------------------
# Load Node
# -------------------------
def load_node(state):

    if state.get("context"):
        return state

    source = state["source"]
    path = state["input_path"]

    if source == "pdf":
        state["context"] = mcp_load_pdf.func(path)

    elif source == "repo":
        state["context"] = mcp_load_repo.func(path)

    elif source == "arxiv":
        data = mcp_load_arxiv.func(path)

        state["context"] = data["content"]
        state["metadata"] = data["metadata"]

    return state


# -------------------------
# QA Node (FIXED 🔥)
# -------------------------
def qa_node(state):
    query = state["query"].lower()
    metadata = state.get("metadata", {})

    # -------- HARD RULE (NO LLM) --------
    if "title" in query:
        state["output"] = metadata.get("title", "Title not available")
        return state

    if "author" in query:
        authors = metadata.get("authors", [])
        state["output"] = ", ".join(authors) if authors else "Authors not available"
        return state

    if "abstract" in query:
        state["output"] = metadata.get("abstract", "Abstract not available")
        return state

    # -------- LLM QA --------
    context = state.get("context", "")
    history = state.get("history_text", "")

    answer = mcp_qa_with_history.func(
        f"{state['query']}|||{context}|||{history}"
    )

    state["output"] = answer

    # update history
    state["history_text"] = history + f"\nUser: {state['query']}\nAssistant: {answer}"

    return state


# -------------------------
# Summary Node
# -------------------------
def summary_node(state):
    state["output"] = mcp_summary.func(state["context"])
    return state


# -------------------------
# Router
# -------------------------
def router(state):
    if "summary" in state["query"].lower():
        return "summary"
    return "qa"


# -------------------------
# Graph
# -------------------------
def build_graph():
    builder = StateGraph(dict)

    builder.add_node("load", load_node)
    builder.add_node("qa", qa_node)
    builder.add_node("summary", summary_node)

    builder.set_entry_point("load")

    builder.add_conditional_edges("load", router)
    builder.add_edge("qa", END)
    builder.add_edge("summary", END)

    return builder.compile()


graph = build_graph()


def run_mcp_agent(state):
    return graph.invoke(state)