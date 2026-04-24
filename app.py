import streamlit as st
from pathlib import Path

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Document Analyzer", layout="wide")

st.title("📄 AI Document Risk & Analysis System")

# =========================
# SESSION STATE INIT
# =========================
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = True


# =========================
# SAFE FILE HANDLER
# =========================
def handle_upload(key_name, label, file_types):
    return st.file_uploader(
        label,
        type=file_types,
        key=key_name
    )


# =========================
# UPLOAD SECTION
# =========================
st.header("📤 Upload Documents")

col1, col2 = st.columns(2)

with col1:
    resume_file = handle_upload(
        "resume_uploader",
        "Upload Resume",
        ["pdf", "docx", "txt"]
    )

with col2:
    jd_file = handle_upload(
        "jd_uploader",
        "Upload Job Description",
        ["pdf", "docx", "txt"]
    )

scan_file = handle_upload(
    "scan_uploader",
    "Upload Scanned Document / Image",
    ["png", "jpg", "jpeg", "pdf"]
)

# =========================
# STORE FILES SAFELY
# =========================
if resume_file:
    st.session_state.uploaded_docs["resume"] = resume_file

if jd_file:
    st.session_state.uploaded_docs["jd"] = jd_file

if scan_file:
    st.session_state.uploaded_docs["scan"] = scan_file


# =========================
# DISPLAY UPLOADED FILES
# =========================
st.subheader("📁 Uploaded Files")

if st.session_state.uploaded_docs:
    for name, file in st.session_state.uploaded_docs.items():
        st.success(f"✔ {name.upper()} uploaded: {file.name}")
else:
    st.info("No files uploaded yet.")


# =========================
# ANALYSIS BUTTON
# =========================
st.header("🧠 Analysis Engine")

if st.button("Run AI Analysis"):

    if not st.session_state.uploaded_docs:
        st.warning("Please upload documents first.")
    else:
        st.write("🔍 Processing documents...")

        # Placeholder for your AI logic
        for name, file in st.session_state.uploaded_docs.items():
            st.write(f"Analyzing: {name} → {file.name}")

        st.success("Analysis Completed ✅")


# =========================
# OPTIONAL RESET
# =========================
if st.button("Reset App"):
    st.session_state.uploaded_docs = {}
    st.rerun()
