from __future__ import annotations

from pathlib import Path

import streamlit as st

from scripts.chat import RAGAssistant
from scripts.upload_manual import build_user_index

RAW_DIR = Path("data/raw")


def get_assistant(user_id: str, car_context: str) -> RAGAssistant:
    key = f"assistant::{user_id}::{car_context.strip()}"
    cached = st.session_state.get("assistant_key")
    if cached != key:
        st.session_state["assistant"] = RAGAssistant(user_id=user_id, car_context=car_context)
        st.session_state["assistant_key"] = key
    return st.session_state["assistant"]


st.set_page_config(page_title="Car Assistant (RAG)", page_icon="🚗")
st.title("🚗 Car Assistant (RAG)")
st.caption("Upload a car manual PDF, auto-detect model, then chat with user + global knowledge.")

user_id = st.text_input("User ID", value="user1")
car_context = st.text_input("Optional car context (model/year/trim)", value="")

uploaded_file = st.file_uploader("Upload your car manual (PDF)", type=["pdf"])
if uploaded_file is not None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = RAW_DIR / uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing manual and building user index..."):
        meta = build_user_index(pdf_path=pdf_path, user_id=user_id, chunk_words=200)
    st.success("Manual processed successfully.")
    st.write(f"Detected model: `{meta['model_name']}`")
    st.write(f"Chunks indexed: `{meta['chunks']}`")
    st.session_state.pop("assistant", None)
    st.session_state.pop("assistant_key", None)

query = st.text_input("Ask about your car")
if query:
    with st.spinner("Generating answer..."):
        assistant = get_assistant(user_id=user_id, car_context=car_context)
        answer = assistant.generate_answer(query, car_context=car_context)
    st.markdown("### Answer")
    st.write(answer)
