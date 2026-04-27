from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models.ollama import ChatOllama
import os
import requests
import re

llm = ChatOllama(model="llama3.2:latest", temperature=0)

# -------------------------
# PDF Loader
# -------------------------
@tool
def mcp_load_pdf(file_path: str) -> str:
    """Load PDF and return text content"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    return "\n\n".join([c.page_content for c in chunks])


# -------------------------
# Repo Loader
# -------------------------
@tool
def mcp_load_repo(folder_path: str) -> str:
    """Load all PDFs from a folder"""
    text_data = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            text_data.append(mcp_load_pdf.func(full_path))

    return "\n\n".join(text_data)


# -------------------------
# arXiv Loader (FIXED 🔥)
# -------------------------
@tool
def mcp_load_arxiv(arxiv_url: str) -> dict:
    """Download arXiv paper and extract metadata + content"""

    if not arxiv_url:
        return {"error": "No URL"}

    arxiv_id = arxiv_url.split("/")[-1]

    # -------- Metadata via API --------
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(api_url).text

    title_match = re.search(r"<title>(.*?)</title>", response, re.DOTALL)
    title = title_match.group(1).strip() if title_match else "Unknown"

    authors = re.findall(r"<name>(.*?)</name>", response)

    abstract_match = re.search(r"<summary>(.*?)</summary>", response, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ""

    # -------- Download PDF --------
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_path = "temp_arxiv.pdf"

    pdf_data = requests.get(pdf_url)
    with open(pdf_path, "wb") as f:
        f.write(pdf_data.content)

    content = mcp_load_pdf.func(pdf_path)

    return {
        "metadata": {
            "title": title,
            "authors": authors,
            "abstract": abstract
        },
        "content": content
    }


# -------------------------
# QA Tool (with history)
# -------------------------
@tool
def mcp_qa_with_history(input_text: str) -> str:
    """Answer questions using context + chat history"""

    try:
        question, context, history = input_text.split("|||")
    except:
        return "Format error"

    prompt = f"""
    You are a research assistant.

    Conversation history:
    {history}

    Context:
    {context}

    Answer ONLY from context.

    Question:
    {question}
    """

    return llm.invoke(prompt).content


# -------------------------
# Summary Tool
# -------------------------
@tool
def mcp_summary(text: str) -> str:
    """Generate structured summary"""
    prompt = f"""
    Summarize using IMRaD + Limitations + Future Work:

    {text}
    """
    return llm.invoke(prompt).content