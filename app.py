# ============================================================
# AI ENTERPRISE DOCUMENT RISK ANALYZER
# Groq + LangChain + LangGraph + RAG
# Supports: PDF, DOCX, TXT, Images, Scanned PDFs
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
# LOAD GROQ API KEY FROM STREAMLIT SECRETS
# ============================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets.")
    st.info("Add it in Settings → Secrets")
    st.stop()

# ============================================================
# OPTIONAL IMPORTS
# ============================================================

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF_IMAGE_AVAILABLE = True
except ImportError:
    PDF_IMAGE_AVAILABLE = False

# ------------------------------------------------------------
# ROBUST PDF IMPORT (pypdf / PyPDF2)
# ------------------------------------------------------------

try:
    from pypdf import PdfReader
    PDF_LIBRARY = "pypdf"
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PDF_LIBRARY = "PyPDF2"
    except ImportError:
        PdfReader = None
        PDF_LIBRARY = None
from PIL import Image

# ============================================================
# LANGCHAIN IMPORTS (SAFE)
# ============================================================

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_CHAIN_AVAILABLE = True
except ImportError:
    GROQ_CHAIN_AVAILABLE = False

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

MODEL_NAME = st.sidebar.selectbox(
    "Select Groq Model",
    [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m"
    ]
)

if not DOCX_AVAILABLE:
    st.sidebar.warning("DOCX support disabled (python-docx missing).")

if not OCR_AVAILABLE:
    st.sidebar.warning("OCR disabled (pytesseract missing).")

if not PDF_IMAGE_AVAILABLE:
    st.sidebar.warning("Scanned PDF OCR disabled (pdf2image missing).")

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
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def extract_text_from_docx(file):
    if not DOCX_AVAILABLE:
        return "DOCX support unavailable."

    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_txt(file):
    return file.read().decode("utf-8")


def extract_text_from_image(file):
    if not OCR_AVAILABLE:
        return "OCR unavailable."

    image = Image.open(file)
    return pytesseract.image_to_string(image)


def extract_text_from_scanned_pdf(file):
    if not OCR_AVAILABLE or not PDF_IMAGE_AVAILABLE:
        return "Scanned PDF OCR unavailable."

    images = convert_from_bytes(file.read())
    text = ""

    for image in images:
        text += pytesseract.image_to_string(image) + "\n"

    return text


def process_uploaded_file(uploaded_file):
    extension = uploaded_file.name.split(".")[-1].lower()

    if extension == "pdf":
        try:
            text = extract_text_from_pdf(uploaded_file)

            if len(text.strip()) < 50:
                uploaded_file.seek(0)
                text = extract_text_from_scanned_pdf(uploaded_file)

            return text

        except Exception:
            uploaded_file.seek(0)
            return extract_text_from_scanned_pdf(uploaded_file)

    elif extension == "docx":
        return extract_text_from_docx(uploaded_file)

    elif extension == "txt":
        return extract_text_from_txt(uploaded_file)

    elif extension in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(uploaded_file)

    return ""

# ============================================================
# VECTOR DATABASE
# ============================================================

@st.cache_resource
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(
        docs,
        embeddings
    )

# ============================================================
# AGENTS
# ============================================================

AGENTS = {
    "Legal Agent": """
Analyze legal clauses, liabilities, indemnities,
litigation exposure, obligations, and penalties.

Return JSON:
{
    "risk_score": 0,
    "risk_level": "Low",
    "findings": [],
    "recommendations": []
}
""",

    "Finance Agent": """
Analyze financial exposure, hidden costs,
pricing issues, penalties, and cashflow risks.

Return JSON only.
""",

    "Compliance Agent": """
Analyze regulatory, GDPR, HIPAA, SOX, AML,
audit, governance, and policy compliance risks.

Return JSON only.
""",

    "Operations Agent": """
Analyze operational bottlenecks, SLA risks,
delivery issues, vendor dependency, and scalability.

Return JSON only.
"""
}

# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict):
    document_text: str
    results: Dict[str, Any]

# ============================================================
# AGENT NODE
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
{state["document_text"][:12000]}
"""

        response = qa.run(prompt)

        try:
            cleaned = response.strip()

            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "")
                cleaned = cleaned.replace("```", "")

            result = json.loads(cleaned)

        except Exception:
            result = {
                "risk_score": 50,
                "risk_level": "Medium",
                "findings": [response],
                "recommendations": ["Manual review recommended."]
            }

        state["results"][agent_name] = result
        return state

    return node

# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph(llm, retriever):
    workflow = StateGraph(GraphState)

    agent_names = list(AGENTS.keys())

    for agent in agent_names:
        workflow.add_node(
            agent,
            create_agent_node(agent, llm, retriever)
        )

    workflow.set_entry_point(agent_names[0])

    for i in range(len(agent_names) - 1):
        workflow.add_edge(
            agent_names[i],
            agent_names[i + 1]
        )

    workflow.add_edge(agent_names[-1], END)

    return workflow.compile()

# ============================================================
# UI HELPERS
# ============================================================

def get_risk_color(score):
    if score < 30:
        return "#10B981"
    elif score < 60:
        return "#F59E0B"
    elif score < 80:
        return "#EF4444"
    return "#991B1B"


def display_risk_card(agent, result):
    score = result.get("risk_score", 0)
    level = result.get("risk_level", "Unknown")
    color = get_risk_color(score)

    st.markdown(
        f"""
        <div style="
            background:#111827;
            padding:25px;
            border-radius:20px;
            border-left:8px solid {color};
            margin-bottom:20px;">
            <h3 style="color:white;">{agent}</h3>
            <h1 style="color:{color};">{score}%</h1>
            <h4 style="color:#D1D5DB;">{level} Risk</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander(f"View {agent} Details"):
        st.subheader("Findings")
        for item in result.get("findings", []):
            st.write(f"• {item}")

        st.subheader("Recommendations")
        for item in result.get("recommendations", []):
            st.write(f"• {item}")

# ============================================================
# MAIN UI
# ============================================================

st.title("📄 AI Enterprise Document Risk Analyzer")
st.markdown(
    "Analyze Legal, Financial, Compliance, and Operational risks using AI."
)

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"]
)

manual_text = st.text_area(
    "Or Paste Document Content",
    height=250
)

if st.button("🚀 Analyze Document", use_container_width=True):

    if not uploaded_file and not manual_text.strip():
        st.warning("Please upload a document or paste text.")
        st.stop()

    with st.spinner("Analyzing document..."):

        if uploaded_file:
            document_text = process_uploaded_file(uploaded_file)
        else:
            document_text = manual_text

        if not document_text.strip():
            st.error("No readable content found.")
            st.stop()

# ============================================================
# MAIN EXECUTION BLOCK
# ============================================================

if st.button("🚀 Analyze Document", use_container_width=True):

    # Validate Input
    if uploaded_file is None and not manual_text.strip():
        st.warning("⚠️ Please upload a document or paste text.")
        st.stop()

    with st.spinner("🔍 Analyzing document..."):

        # ----------------------------------------------------
        # Extract Document Text
        # ----------------------------------------------------
        if uploaded_file is not None:
            document_text = process_uploaded_file(uploaded_file)
        else:
            document_text = manual_text

        if not document_text.strip():
            st.error("❌ No readable content found in the document.")
            st.stop()

        # ----------------------------------------------------
        # Validate Required Libraries
        # ----------------------------------------------------
        if not LANGCHAIN_AVAILABLE:
            st.error("langchain package is not installed.")
            st.stop()

        if not GROQ_CHAIN_AVAILABLE:
            st.error("langchain-groq package is not installed.")
            st.stop()

        if not VECTOR_DB_AVAILABLE:
            st.error("langchain-community or chromadb is not installed.")
            st.stop()

        if not LANGGRAPH_AVAILABLE:
            st.error("langgraph package is not installed.")
            st.stop()

        # ----------------------------------------------------
        # Load LLM
        # ----------------------------------------------------
        llm = load_llm()

        # ----------------------------------------------------
        # Build Vector Store
        # ----------------------------------------------------
        vector_store = build_vector_store(document_text)

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

        # ----------------------------------------------------
        # Build Workflow
        # ----------------------------------------------------
        graph = build_graph(llm, retriever)

        # ----------------------------------------------------
        # Execute Analysis
        # ----------------------------------------------------
        result = graph.invoke({
            "document_text": document_text,
            "results": {}
        })

    st.success("✅ Analysis completed successfully!")

    # ========================================================
    # OVERALL RISK SCORE
    # ========================================================

    scores = [
        agent_result.get("risk_score", 0)
        for agent_result in result["results"].values()
    ]

    overall_risk = (
        sum(scores) / len(scores)
        if scores else 0
    )

    overall_color = get_risk_color(overall_risk)

    st.subheader("📊 Overall Enterprise Risk")

    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:35px;
            background:#000000;
            border-radius:25px;
            border:4px solid {overall_color};
            margin-bottom:30px;
        ">
            <h1 style="
                color:{overall_color};
                font-size:70px;
                margin:0;
            ">
                {overall_risk:.1f}%
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

 # ============================================================
# DISPLAY AGENT RESULTS
# ============================================================

st.markdown("---")
st.subheader("🤖 Agent Risk Analysis")

col1, col2 = st.columns(2)

agent_items = list(result["results"].items())

for index, (agent_name, agent_data) in enumerate(agent_items):
    if index % 2 == 0:
        with col1:
            display_risk_card(agent_name, agent_data)
    else:
        with col2:
            display_risk_card(agent_name, agent_data)
