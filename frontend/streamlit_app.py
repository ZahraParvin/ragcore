"""RAGCore Streamlit frontend."""
import streamlit as st
import requests
import os

API_URL = os.environ.get("RAGCORE_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAGCore", page_icon="🧠", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 RAGCore")
    st.markdown("**Multi-domain RAG assistant**")
    st.divider()

    mode = st.radio("Mode", ["💬 Document Q&A", "🔬 Agent Data Analysis"])

    st.divider()
    st.markdown("**Collections**")
    try:
        cols_resp = requests.get(f"{API_URL}/collections", timeout=5)
        if cols_resp.ok:
            cols = cols_resp.json().get("collections", [])
            if cols:
                for c in cols:
                    st.caption(f"📁 {c['name']} ({c['count']} chunks)")
            else:
                st.caption("No collections yet.")
        else:
            st.caption("API offline")
    except Exception:
        st.caption("API offline")

    st.divider()
    st.markdown("**Presets**")
    preset = st.selectbox("Domain preset", ["custom", "industrial", "financial", "general"])
    collection_name = st.text_input("Collection name", value=preset if preset != "custom" else "my_docs")

# --- Main area ---
if mode == "💬 Document Q&A":
    st.header("💬 Document Q&A")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Upload Documents")
        uploaded = st.file_uploader("PDF, CSV, or TXT", type=["pdf", "csv", "txt", "md"], accept_multiple_files=True)
        if uploaded and st.button("Ingest documents", type="primary"):
            for f in uploaded:
                with st.spinner(f"Ingesting {f.name}..."):
                    resp = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (f.name, f.getvalue(), f.type)},
                        data={"collection": collection_name},
                    )
                if resp.ok:
                    data = resp.json()
                    st.success(f"✅ {f.name}: {data['chunks_added']} chunks added")
                else:
                    st.error(f"❌ {f.name}: {resp.text}")

    with col2:
        st.subheader("Chat")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.caption(f"Sources: {', '.join(msg['sources'])}")

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    try:
                        resp = requests.post(
                            f"{API_URL}/query",
                            json={"collection": collection_name, "question": prompt, "history": history},
                            timeout=30,
                        )
                        if resp.ok:
                            data = resp.json()
                            st.markdown(data["answer"])
                            if data.get("sources"):
                                st.caption(f"Sources: {', '.join(data['sources'])}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": data["answer"],
                                "sources": data.get("sources", []),
                            })
                        else:
                            st.error(f"Error: {resp.json().get('detail', resp.text)}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

elif mode == "🔬 Agent Data Analysis":
    st.header("🔬 Agentic Data Analysis")
    st.markdown("Upload a CSV and ask analytical questions. Claude uses tools to compute statistics, detect anomalies, and find patterns.")

    uploaded_csv = st.file_uploader("Upload CSV dataset", type=["csv"])
    question = st.text_input("Question", placeholder="Are there any anomalies in the sensor readings?")

    if uploaded_csv and question and st.button("Analyze", type="primary"):
        with st.spinner("Agent thinking..."):
            resp = requests.post(
                f"{API_URL}/agent/analyze",
                files={"file": (uploaded_csv.name, uploaded_csv.getvalue(), "text/csv")},
                data={"question": question},
                timeout=60,
            )
        if resp.ok:
            data = resp.json()
            st.subheader("Answer")
            st.markdown(data["answer"])

            if data.get("tool_calls"):
                with st.expander(f"🔧 Tool calls ({len(data['tool_calls'])})"):
                    for tc in data["tool_calls"]:
                        st.markdown(f"**{tc['tool']}** — input: `{tc['input']}`")
                        st.code(tc["result"][:300])
        else:
            st.error(f"Error: {resp.text}")
