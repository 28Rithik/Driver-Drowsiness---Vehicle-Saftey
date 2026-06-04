from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import streamlit as st

from vehicle_assistant.config import (
    CHUNKS_PATH,
    DB_PATH,
    DOCUMENTS_PATH,
    FAISS_INDEX_PATH,
    LLM_PROVIDERS,
    MANUALS_DIR,
    OBD_JSON_PATH,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_API_BASE,
    HIDE_LLM_UI,
    ensure_directories,
    get_supported_vehicles,
)
from vehicle_assistant.llm import LLMSettings
from vehicle_assistant.service import DiagnosticAssistant
from vehicle_assistant.storage import get_messages, list_sessions, register_upload


st.set_page_config(
    page_title="Vehicle Diagnostic Assistant",
    page_icon="Car Front",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 140, 66, 0.18), transparent 32%),
                radial-gradient(circle at top right, rgba(22, 163, 74, 0.14), transparent 30%),
                linear-gradient(180deg, #0b1220 0%, #111827 48%, #0f172a 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4, p, label, span, div {
            color: inherit;
        }
        .hero-card {
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(15, 23, 42, 0.72);
            backdrop-filter: blur(12px);
            padding: 1.25rem 1.35rem;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
        }
        .muted-text {
            color: #94a3b8;
        }
        .source-card {
            border: 1px solid rgba(148,163,184,0.18);
            background: rgba(2, 6, 23, 0.52);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        .stChatMessage {
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_assistant() -> DiagnosticAssistant:
    ensure_directories()
    assistant = DiagnosticAssistant(DB_PATH)
    assistant.ensure_index(force_rebuild=False)
    return assistant


def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "vehicle" not in st.session_state:
        st.session_state.vehicle = "All vehicles"
    if "provider" not in st.session_state:
        st.session_state.provider = "Extractive"
    if "model" not in st.session_state:
        st.session_state.model = ""
    if "api_base" not in st.session_state:
        st.session_state.api_base = ""
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "messages_loaded" not in st.session_state:
        st.session_state.messages_loaded = False

    # If LLM inputs are hidden, preload values from config/env
    if HIDE_LLM_UI:
        if "provider" not in st.session_state or st.session_state.provider == "Extractive":
            st.session_state.provider = "OpenAI Compatible"
        st.session_state.model = st.session_state.model or GROQ_MODEL
        st.session_state.api_base = st.session_state.api_base or GROQ_API_BASE
        st.session_state.api_key = st.session_state.api_key or GROQ_API_KEY
        st.session_state.temperature = float(st.session_state.temperature)


def render_sidebar(assistant: DiagnosticAssistant) -> None:
    with st.sidebar:
        st.markdown("## Vehicle Diagnostic Assistant")
        st.caption("RAG over OBD-II codes and workshop manuals")

        vehicles = get_supported_vehicles()
        if st.session_state.vehicle not in vehicles:
            st.session_state.vehicle = vehicles[0]
        st.session_state.vehicle = st.selectbox(
            "Vehicle",
            vehicles,
            index=vehicles.index(st.session_state.vehicle) if st.session_state.vehicle in vehicles else 0,
        )

        if not HIDE_LLM_UI:
            st.session_state.provider = st.selectbox(
                "Answer mode",
                LLM_PROVIDERS,
                index=LLM_PROVIDERS.index(st.session_state.provider)
                if st.session_state.provider in LLM_PROVIDERS
                else 0,
            )
            st.session_state.model = st.text_input("Model name", value=st.session_state.model)
            st.session_state.api_base = st.text_input("API base URL", value=st.session_state.api_base)
            st.session_state.api_key = st.text_input("API key", value=st.session_state.api_key, type="password")
            st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
        else:
            st.markdown("**LLM settings are loaded from configuration (hidden).**")
        st.session_state.top_k = st.slider("Retrieved sources", 2, 10, st.session_state.top_k)

        st.divider()
        if st.button("New chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        if st.button("Rebuild index", use_container_width=True):
            with st.spinner("Building vector index..."):
                stats = assistant.ensure_index(force_rebuild=True)
            st.success(
                f"Indexed {stats.chunks} chunks from {stats.documents} documents."
            )

        st.divider()
        st.markdown("### Upload manual PDF")
        uploaded_files = st.file_uploader(
            "Add workshop manuals or service guides",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files and st.button("Save and index uploads", use_container_width=True):
            target_vehicle = st.session_state.vehicle if st.session_state.vehicle != "All vehicles" else "Generic"
            saved_paths: list[Path] = []
            vehicle_dir = MANUALS_DIR / target_vehicle.replace(" ", "_")
            vehicle_dir.mkdir(parents=True, exist_ok=True)
            for file in uploaded_files:
                destination = vehicle_dir / file.name
                destination.write_bytes(file.getbuffer())
                register_upload(DB_PATH, file.name, target_vehicle, "manual_pdf", str(destination))
                saved_paths.append(destination)
            assistant.ensure_index(force_rebuild=True)
            st.success(f"Saved {len(saved_paths)} manual(s) and refreshed the index.")

        st.divider()
        st.markdown("### Knowledge base")
        st.write(f"OBD dataset: {OBD_JSON_PATH.name}")
        st.write(f"Index file: {FAISS_INDEX_PATH.name}")
        st.write(f"Chunk store: {CHUNKS_PATH.name}")
        st.write(f"Document manifest: {DOCUMENTS_PATH.name}")

        session_rows = list_sessions(DB_PATH)
        if session_rows:
            st.markdown("### Recent chats")
            for row in session_rows[:5]:
                label = row["title"] or row["session_id"][:8]
                if st.button(label, key=row["session_id"], use_container_width=True):
                    st.session_state.session_id = row["session_id"]
                    st.rerun()


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin:0; font-size:2.2rem;">Vehicle Diagnostic Assistant</h1>
            <p class="muted-text" style="margin:0.45rem 0 0 0;">
                Ask about OBD-II codes, workshop procedures, maintenance schedules, or vehicle-specific repairs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_messages(session_id: str) -> None:
    messages = get_messages(DB_PATH, session_id)
    if not messages:
        st.info("Start with a question like 'What does P0300 mean on a Toyota Fortuner?'")
        return
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("### Retrieved sources")
    for index, source in enumerate(sources, start=1):
        with st.container():
            st.markdown(
                f"""
                <div class="source-card">
                    <strong>Source {index}:</strong> {source.get('title', 'Untitled')}<br/>
                    <span class="muted-text">Vehicle:</span> {source.get('vehicle', 'Unknown')}<br/>
                    <span class="muted-text">Code:</span> {source.get('code') or 'N/A'}<br/>
                    <span class="muted-text">Score:</span> {source.get('score', 0.0):.3f}<br/>
                    <span class="muted-text">Location:</span> {source.get('url') or f"Page {source.get('page')}" or 'N/A'}
                </div>
                """,
                unsafe_allow_html=True,
            )
            snippet = source.get("text", "")[:500]
            if snippet:
                st.caption(snippet)


def main() -> None:
    ensure_directories()
    inject_styles()
    init_state()
    assistant = get_assistant()

    render_sidebar(assistant)
    render_header()
    render_messages(st.session_state.session_id)

    user_prompt = st.chat_input("Ask a vehicle diagnostic question")
    if user_prompt:
        settings = LLMSettings(
            provider=st.session_state.provider,
            model=st.session_state.model,
            api_base=st.session_state.api_base,
            api_key=st.session_state.api_key,
            temperature=float(st.session_state.temperature),
        )
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching manuals and building an answer..."):
                result = assistant.ask(
                    session_id=st.session_state.session_id,
                    question=user_prompt,
                    vehicle=st.session_state.vehicle,
                    settings=settings,
                    top_k=int(st.session_state.top_k),
                    session_title=user_prompt[:40],
                )
            st.markdown(result["answer"])
            render_sources(result["sources"])


if __name__ == "__main__":
    main()
