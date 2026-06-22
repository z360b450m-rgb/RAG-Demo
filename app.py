import tempfile
from pathlib import Path

import streamlit as st

from app_config import validate_config
from core.agent import RAGAgent
from core.pipeline import DirectRAGPipeline

st.set_page_config(
    page_title="RAG Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    if "pipeline" not in st.session_state:
        try:
            validate_config()
        except ValueError as e:
            st.error(f"Configuration error: {e}")
            st.stop()
        with st.spinner("Loading embedding model (first run may download ~100MB)..."):
            st.session_state.pipeline = DirectRAGPipeline()

    if "agent" not in st.session_state:
        st.session_state.agent = RAGAgent()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "context_cache" not in st.session_state:
        st.session_state.context_cache = {}

    if "use_agent" not in st.session_state:
        st.session_state.use_agent = True


def render_sidebar():
    st.sidebar.title("📚 RAG Agent")
    st.sidebar.caption("DeepSeek + ChromaDB")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Mode")
    mode = st.sidebar.radio(
        "Query mode",
        options=["Agent (Tool-calling)", "Direct RAG"],
        index=0 if st.session_state.use_agent else 1,
        help=(
            "**Agent**: LLM decides whether to retrieve documents or answer directly.\n\n"
            "**Direct RAG**: Always retrieves documents before answering."
        ),
    )
    st.session_state.use_agent = mode == "Agent (Tool-calling)"

    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Document Upload")
    uploaded = st.sidebar.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
        key="file_uploader",
    )

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded.name).suffix
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            st.sidebar.text(f"Ingesting {uploaded.name}...")
            count = st.session_state.pipeline.ingest_document(Path(tmp_path))
            if count > 0:
                st.sidebar.success(f"'{uploaded.name}' — {count} chunks indexed")
            else:
                st.sidebar.info(f"'{uploaded.name}' — no new content to index")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Indexed Documents")
    sources = st.session_state.pipeline.list_ingested_sources()
    if sources:
        for src in sources:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.text(f"• {src}")
            with col2:
                if st.button("🗑", key=f"del_{src}", help=f"Delete {src}"):
                    deleted = st.session_state.pipeline.delete_document(src)
                    st.toast(f"Removed {deleted} chunks from '{src}'")
                    st.rerun()
    else:
        st.sidebar.caption("No documents indexed yet.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.context_cache = {}
        st.session_state.agent.clear_memory()
        st.rerun()


def render_chat():
    st.title("📚 RAG Agent")
    mode_label = "Agent" if st.session_state.use_agent else "Direct RAG"
    st.caption(f"Mode: **{mode_label}** — Ask questions about your documents.")

    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            ctx_key = msg.get("context_key")
            if ctx_key and ctx_key in st.session_state.context_cache:
                with st.expander("View Retrieved Context"):
                    for j, ctx in enumerate(st.session_state.context_cache[ctx_key]):
                        st.caption(
                            f"**Source:** {ctx['source']}  |  "
                            f"**Score:** {ctx['score']}"
                        )
                        st.text(ctx["text"])


def handle_input():
    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    sources = st.session_state.pipeline.list_ingested_sources()
    if not sources:
        st.warning("Please upload at least one document first.")
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        if st.session_state.use_agent:
            with st.spinner("Thinking..."):
                try:
                    stream = st.session_state.agent.query_stream(question)
                    response = st.write_stream(stream)
                except Exception as e:
                    response = f"Error: {e}"
                    placeholder.error(response)
            msg_key = None
        else:
            try:
                stream = st.session_state.pipeline.query_stream(
                    question, st.session_state.chat_history
                )
                response = st.write_stream(stream)
            except Exception as e:
                response = f"Error: {e}"
                placeholder.error(response)

            msg_key = f"ctx_{len(st.session_state.chat_history)}"
            try:
                contexts = st.session_state.pipeline.get_retrieved_contexts(question)
                st.session_state.context_cache[msg_key] = contexts
                if contexts:
                    with st.expander("View Retrieved Context"):
                        for j, ctx in enumerate(contexts):
                            st.caption(
                                f"**Source:** {ctx['source']}  |  "
                                f"**Score:** {ctx['score']}"
                            )
                            st.text(ctx["text"])
            except Exception:
                pass

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "context_key": msg_key,
    })


def main():
    init_session_state()
    render_sidebar()
    render_chat()
    handle_input()


if __name__ == "__main__":
    main()
