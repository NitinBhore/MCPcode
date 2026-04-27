import streamlit as st
from mcpagent import run_mcp_agent

st.set_page_config(page_title="MCP Research Chatbot", layout="wide")
st.title("🧠 MCP Research Assistant")

# -------------------------
# Session State
# -------------------------
if "context" not in st.session_state:
    st.session_state.context = None

if "metadata" not in st.session_state:
    st.session_state.metadata = {}

if "history_text" not in st.session_state:
    st.session_state.history_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "source" not in st.session_state:
    st.session_state.source = None

if "input_path" not in st.session_state:
    st.session_state.input_path = None


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("📄 Load Paper")

    option = st.radio("Choose Source", ["Upload PDF", "Local Repo", "arXiv"])

    if option == "Upload PDF":
        file = st.file_uploader("Upload PDF", type="pdf")
        if file:
            path = file.name
            with open(path, "wb") as f:
                f.write(file.read())
            st.session_state.input_path = path
            st.session_state.source = "pdf"

    elif option == "Local Repo":
        path = st.text_input("Folder path")
        if path:
            st.session_state.input_path = path
            st.session_state.source = "repo"

    elif option == "arXiv":
        url = st.text_input("arXiv URL")
        if url:
            st.session_state.input_path = url
            st.session_state.source = "arxiv"

    if st.button("🚀 Load Paper"):
        result = run_mcp_agent({
            "input_path": st.session_state.input_path,
            "source": st.session_state.source,
            "query": "load",
            "history_text": ""
        })

        st.session_state.context = result["context"]
        st.session_state.metadata = result.get("metadata", {})
        st.session_state.history_text = ""
        st.session_state.messages = []

        st.success("Paper loaded!")


# -------------------------
# Chat Interface
# -------------------------
st.subheader("💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything about the paper...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    state = {
        "query": user_input,
        "context": st.session_state.context,
        "metadata": st.session_state.metadata,
        "history_text": st.session_state.history_text
    }

    result = run_mcp_agent(state)

    answer = result["output"]

    st.session_state.history_text = result.get("history_text", "")
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)