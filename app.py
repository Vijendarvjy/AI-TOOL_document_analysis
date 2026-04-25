# ============================================================
# AI ENTERPRISE DOCUMENT RISK ANALYZER (FIXED VERSION)
# ============================================================

import streamlit as st
import json
from typing import TypedDict, Dict, Any

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Enterprise Document Risk Analyzer",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# GROQ API KEY
# ============================================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets.")
    st.stop()

# ============================================================
# OPTIONAL IMPORTS
# ============================================================
try:
    from docx import Document
    DOCX_AVAILABLE = True
except:
    DOCX_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF_IMAGE_AVAILABLE = True
except:
    PDF_IMAGE_AVAILABLE = False

try:
    from pypdf import PdfReader
except:
    try:
        from PyPDF2 import PdfReader
    except:
        PdfReader = None

from PIL import Image
# ============================================================
# REPLACE ALL LANGCHAIN IMPORTS WITH THIS BLOCK
# ============================================================

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Settings")

MODEL_NAME = st.sidebar.selectbox(
    "Select Groq Model",
    [
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]
)

# ============================================================
# LLM
# ============================================================
@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0
    )

# ============================================================
# DOCUMENT EXTRACTORS
# ============================================================
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def extract_text_from_docx(file):
    if not DOCX_AVAILABLE:
        return ""
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text_from_image(file):
    if not OCR_AVAILABLE:
        return ""
    image = Image.open(file)
    return pytesseract.image_to_string(image)

def extract_text_from_scanned_pdf(file):
    if not OCR_AVAILABLE or not PDF_IMAGE_AVAILABLE:
        return ""
    images = convert_from_bytes(file.read())
    return "\n".join([pytesseract.image_to_string(img) for img in images])

def process_uploaded_file(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()

    if ext == "pdf":
        try:
            text = extract_text_from_pdf(uploaded_file)
            if len(text.strip()) < 50:
                uploaded_file.seek(0)
                text = extract_text_from_scanned_pdf(uploaded_file)
            return text
        except:
            uploaded_file.seek(0)
            return extract_text_from_scanned_pdf(uploaded_file)

    if ext == "docx":
        return extract_text_from_docx(uploaded_file)

    if ext == "txt":
        return extract_text_from_txt(uploaded_file)

    if ext in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(uploaded_file)

    return ""

# ============================================================
# VECTOR STORE
# ============================================================
@st.cache_resource
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(docs, embeddings)

# ============================================================
# AGENTS
# ============================================================
AGENTS = {
    "Legal Agent": "Analyze legal risks and return JSON only.",
    "Finance Agent": "Analyze financial risks and return JSON only.",
    "Compliance Agent": "Analyze compliance risks and return JSON only.",
    "Operations Agent": "Analyze operational risks and return JSON only."
}

# ============================================================
# STATE
# ============================================================
class GraphState(TypedDict):
    document_text: str
    results: Dict[str, Any]

# ============================================================
# NODE
# ============================================================
def create_agent_node(agent_name, llm, retriever):

    def node(state):
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )

        prompt = f"""
        {AGENTS[agent_name]}

        Document:
        {state["document_text"][:8000]}
        """

        response = qa.invoke({"query": prompt})
        text = response.get("result", "")

        try:
            cleaned = text.strip().replace("```json", "").replace("```", "")
            result = json.loads(cleaned)
        except:
            result = {
                "risk_score": 50,
                "risk_level": "Medium",
                "findings": [text],
                "recommendations": ["Manual review required"]
            }

        state["results"][agent_name] = result
        return state

    return node

# ============================================================
# GRAPH
# ============================================================
def build_graph(llm, retriever):
    workflow = StateGraph(GraphState)

    agents = list(AGENTS.keys())

    for a in agents:
        workflow.add_node(a, create_agent_node(a, llm, retriever))

    workflow.set_entry_point(agents[0])

    for i in range(len(agents) - 1):
        workflow.add_edge(agents[i], agents[i + 1])

    workflow.add_edge(agents[-1], END)

    return workflow.compile()

# ============================================================
# UI HELPERS
# ============================================================
def get_risk_color(score):
    if score < 30:
        return "#10B981"
    if score < 60:
        return "#F59E0B"
    if score < 80:
        return "#EF4444"
    return "#991B1B"

def display_risk_card(agent, data):
    score = data.get("risk_score", 0)
    color = get_risk_color(score)

    st.markdown(f"""
    <div style="padding:20px;border-left:6px solid {color};
    background:#111827;border-radius:12px;margin:10px 0;">
    <h3 style="color:white">{agent}</h3>
    <h1 style="color:{color}">{score}%</h1>
    <p style="color:#ccc">{data.get("risk_level","Unknown")}</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN UI
# ============================================================
st.title("📄 AI Enterprise Document Risk Analyzer")

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"]
)

manual_text = st.text_area("Or Paste Document")

# ✅ SINGLE BUTTON ONLY (FIXED DUPLICATE ERROR)
run = st.button("🚀 Analyze Document", use_container_width=True)

if run:

    if not uploaded_file and not manual_text.strip():
        st.warning("Provide input first.")
        st.stop()

    with st.spinner("Analyzing..."):

        document_text = (
            process_uploaded_file(uploaded_file)
            if uploaded_file else manual_text
        )

        if not document_text.strip():
            st.error("No text found.")
            st.stop()

        llm = load_llm()
        vector_store = build_vector_store(document_text)
        retriever = vector_store.as_retriever()

        graph = build_graph(llm, retriever)

        result = graph.invoke({
            "document_text": document_text,
            "results": {}
        })

    st.success("Analysis Completed!")

    # Overall score
    scores = [v.get("risk_score", 0) for v in result["results"].values()]
    overall = sum(scores) / len(scores) if scores else 0

    st.subheader(f"📊 Overall Risk: {overall:.1f}%")

    # Results
    st.markdown("## 🤖 Agent Results")

    col1, col2 = st.columns(2)

    items = list(result["results"].items())

    for i, (name, data) in enumerate(items):
        with col1 if i % 2 == 0 else col2:
            display_risk_card(name, data)
