import streamlit as st
import os
import tempfile
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from docx import Document
from pypdf import PdfReader
from groq import Groq

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangDocument
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any
import json
import re

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
    "Enter Groq API Key",
    type="password"
)

model_name = st.sidebar.selectbox(
    "Choose Model",
    [
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]
)

# ============================================================
# INITIALIZE LLM
# ============================================================

@st.cache_resource
def initialize_llm(api_key, model):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=0
    )

# ============================================================
# DOCUMENT LOADERS
# ============================================================

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_txt(file):
    return file.read().decode("utf-8")


def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)


def extract_text_from_scanned_pdf(file):
    images = convert_from_bytes(file.read())
    text = ""

    for img in images:
        text += pytesseract.image_to_string(img) + "\n"

    return text


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

    elif ext == "docx":
        return extract_text_from_docx(uploaded_file)

    elif ext == "txt":
        return extract_text_from_txt(uploaded_file)

    elif ext in ["png", "jpg", "jpeg"]:
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

    db = Chroma.from_documents(
        docs,
        embedding=embeddings
    )

    return db

# ============================================================
# AGENT PROMPTS
# ============================================================

AGENTS = {
    "Legal Agent": """
        Analyze legal risks, liabilities, clauses, litigation exposure,
        obligations, penalties, indemnities, and enforceability.
        Return JSON:
        {
            "risk_score": number,
            "risk_level": "Low/Medium/High/Critical",
            "findings": [],
            "recommendations": []
        }
    """,

    "Finance Agent": """
        Analyze financial risks, losses, cashflow exposure,
        hidden costs, penalties, pricing issues.
        Return JSON.
    """,

    "Compliance Agent": """
        Analyze regulatory, policy, audit, GDPR, HIPAA,
        SOX, AML, and governance compliance risks.
        Return JSON.
    """,

    "Operations Agent": """
        Analyze operational bottlenecks, delivery risks,
        SLA risks, vendor dependency, scalability issues.
        Return JSON.
    """
}

# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict):
    query: str
    results: Dict[str, Any]

# ============================================================
# CREATE AGENT NODE
# ============================================================

def create_agent_node(agent_name, llm, retriever):
    def agent(state):
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )

        prompt = f"""
        {AGENTS[agent_name]}

        Document Analysis:
        {state['query']}
        """

        response = qa.run(prompt)

        try:
            parsed = json.loads(response)
        except:
            parsed = {
                "risk_score": 50,
                "risk_level": "Medium",
                "findings": [response],
                "recommendations": []
            }

        state["results"][agent_name] = parsed
        return state

    return agent

# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph(llm, retriever):
    workflow = StateGraph(GraphState)

    for agent_name in AGENTS:
        workflow.add_node(
            agent_name,
            create_agent_node(agent_name, llm, retriever)
        )

    workflow.set_entry_point("Legal Agent")

    workflow.add_edge("Legal Agent", "Finance Agent")
    workflow.add_edge("Finance Agent", "Compliance Agent")
    workflow.add_edge("Compliance Agent", "Operations Agent")
    workflow.add_edge("Operations Agent", END)

    return workflow.compile()

# ============================================================
# RISK COLOR
# ============================================================

def get_risk_color(score):
    if score < 30:
        return "green"
    elif score < 60:
        return "orange"
    elif score < 80:
        return "red"
    return "darkred"

# ============================================================
# DISPLAY CARD
# ============================================================

def display_risk_card(agent, result):
    score = result.get("risk_score", 0)
    level = result.get("risk_level", "Unknown")
    findings = result.get("findings", [])
    recommendations = result.get("recommendations", [])

    color = get_risk_color(score)

    st.markdown(f"""
    <div style="
        background:#111;
        padding:20px;
        border-radius:15px;
        border-left:8px solid {color};
        margin-bottom:20px;
    ">
        <h3 style="color:white;">{agent}</h3>
        <h1 style="color:{color};">{score}%</h1>
        <h4 style="color:white;">{level} Risk</h4>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"View {agent} Details"):
        st.write("### Findings")
        for item in findings:
            st.write(f"- {item}")

        st.write("### Recommendations")
        for item in recommendations:
            st.write(f"- {item}")

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
    height=200
)

if st.button("🚀 Analyze Document"):

    if not groq_api_key:
        st.error("Please enter Groq API Key.")
        st.stop()

    if not uploaded_file and not manual_text:
        st.warning("Upload a file or paste text.")
        st.stop()

    with st.spinner("Processing document..."):

        if uploaded_file:
            text = process_uploaded_file(uploaded_file)
        else:
            text = manual_text

        llm = initialize_llm(groq_api_key, model_name)

        vector_store = build_vector_store(text)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

        graph = build_graph(llm, retriever)

        result = graph.invoke({
            "query": text[:12000],
            "results": {}
        })

        st.success("Analysis Completed!")

        # Overall Summary
        all_scores = [
            v["risk_score"]
            for v in result["results"].values()
        ]

        overall_risk = sum(all_scores) / len(all_scores)

        st.markdown("---")
        st.subheader("📊 Overall Risk Score")

        overall_color = get_risk_color(overall_risk)

        st.markdown(f"""
        <div style="
            text-align:center;
            padding:30px;
            background:#000;
            border-radius:20px;
            border:3px solid {overall_color};
        ">
            <h1 style="color:{overall_color};">
                {overall_risk:.1f}%
            </h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Agent Results
        cols = st.columns(2)

        for idx, (agent, data) in enumerate(result["results"].items()):
            with cols[idx % 2]:
                display_risk_card(agent, data)
