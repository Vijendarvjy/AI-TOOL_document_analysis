# ============================================================
# SAFE OPTIONAL IMPORTS
# Replace ALL top-level imports with this block
# ============================================================

# ---------- Optional Libraries ----------
import streamlit as st
import json
from typing import TypedDict, Dict, Any

# ---------- Required Libraries ----------

from PIL import Image
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ModuleNotFoundError:
    DOCX_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ModuleNotFoundError:
    OCR_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF_IMAGE_AVAILABLE = True
except ModuleNotFoundError:
    PDF_IMAGE_AVAILABLE = False
# ============================================================
# SIDEBAR WARNINGS
# ============================================================

if not DOCX_AVAILABLE:
    st.sidebar.warning(
        "⚠️ python-docx not installed. DOCX support disabled."
    )

if not OCR_AVAILABLE:
    st.sidebar.warning(
        "⚠️ pytesseract not installed. OCR disabled."
    )


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Enterprise Document Risk Analyzer",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Configuration")

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
)

if not OCR_AVAILABLE:
    st.sidebar.warning(
        "⚠️ pytesseract not installed. Image OCR disabled."
    )

# ============================================================
# LLM
# ============================================================

@st.cache_resource
def load_llm(api_key, model):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=0
    )

# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    return text


def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(
        paragraph.text for paragraph in doc.paragraphs
    )


def extract_text_from_txt(file):
    return file.read().decode("utf-8")


def extract_text_from_image(file):
    if not OCR_AVAILABLE:
        return "OCR not available. Install pytesseract."

    image = Image.open(file)
    return pytesseract.image_to_string(image)


def extract_text_from_scanned_pdf(file):
    if not OCR_AVAILABLE or not PDF_IMAGE_AVAILABLE:
        return (
            "Scanned PDF OCR unavailable.\n"
            "Install pytesseract and pdf2image."
        )

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
# VECTOR STORE
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
Analyze legal risks, liabilities, contracts, penalties,
indemnities, and litigation exposure.

Return valid JSON:
{
    "risk_score": 0-100,
    "risk_level": "Low/Medium/High/Critical",
    "findings": ["point1", "point2"],
    "recommendations": ["action1", "action2"]
}
""",

    "Finance Agent": """
Analyze financial exposure, hidden costs,
cashflow risks, penalties, and pricing issues.

Return valid JSON.
""",

    "Compliance Agent": """
Analyze regulatory compliance, audit readiness,
GDPR, HIPAA, SOX, AML, and policy violations.

Return valid JSON.
""",

    "Operations Agent": """
Analyze operational bottlenecks, vendor dependency,
SLA risks, delivery risks, and scalability.

Return valid JSON.
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

Document Content:
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
                "recommendations": [
                    "Manual review recommended."
                ]
            }

        state["results"][agent_name] = result
        return state

    return node

# ============================================================
# BUILD GRAPH
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
        return "#00C853"
    elif score < 60:
        return "#FF9800"
    elif score < 80:
        return "#F44336"
    return "#B71C1C"


def display_risk_card(agent, result):
    score = result.get("risk_score", 0)
    level = result.get("risk_level", "Unknown")

    color = get_risk_color(score)

    st.markdown(
        f"""
        <div style="
            background:#111827;
            padding:25px;
            border-radius:18px;
            border-left:8px solid {color};
            margin-bottom:20px;
            box-shadow:0 8px 20px rgba(0,0,0,0.3);
        ">
            <h3 style="color:white;">{agent}</h3>
            <h1 style="color:{color};">{score}%</h1>
            <h4 style="color:#E5E7EB;">{level} Risk</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander(f"View {agent} Analysis"):
        st.subheader("Key Findings")
        for finding in result.get("findings", []):
            st.write(f"• {finding}")

        st.subheader("Recommendations")
        for rec in result.get("recommendations", []):
            st.write(f"• {rec}")

# ============================================================
# MAIN APP
# ============================================================

st.title("📄 AI Enterprise Document Risk Analyzer")
st.markdown(
    """
Analyze Legal, Financial, Compliance, and Operational
risks using Groq, LangChain, LangGraph, and RAG.
"""
)

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"]
)

manual_text = st.text_area(
    "Or Paste Document Text",
    height=250
)

if st.button("🚀 Analyze Document", use_container_width=True):

    if not groq_api_key:
        st.error("Please enter your Groq API Key.")
        st.stop()

    if not uploaded_file and not manual_text.strip():
        st.warning("Upload a file or paste text.")
        st.stop()

    with st.spinner("Analyzing document..."):

        if uploaded_file:
            document_text = process_uploaded_file(uploaded_file)
        else:
            document_text = manual_text

        if not document_text.strip():
            st.error("No readable text found.")
            st.stop()

        llm = load_llm(
            groq_api_key,
            model_name
        )

        vector_store = build_vector_store(document_text)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

        graph = build_graph(llm, retriever)

        result = graph.invoke({
            "document_text": document_text,
            "results": {}
        })

    st.success("Analysis Completed Successfully!")

    # Overall Risk
    scores = [
        data.get("risk_score", 0)
        for data in result["results"].values()
    ]

    overall_risk = sum(scores) / len(scores)
    overall_color = get_risk_color(overall_risk)

    st.markdown("---")
    st.subheader("📊 Overall Enterprise Risk")

    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:35px;
            border-radius:20px;
            background:#000;
            border:4px solid {overall_color};
            margin-bottom:30px;
        ">
            <h1 style="
                font-size:60px;
                color:{overall_color};
                margin:0;
            ">
                {overall_risk:.1f}%
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Agent Cards
    col1, col2 = st.columns(2)

    agents = list(result["results"].items())

    for i, (agent, data) in enumerate(agents):
        with col1 if i % 2 == 0 else col2:
            display_risk_card(agent, data)
