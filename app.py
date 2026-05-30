import os
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st

from rag_pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="Chat with Documents",
    page_icon="[D]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# NEW - Replace with this CSS block
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }

    /* User message bubble */
    .chat-message-user {
        background-color: #dbeafe;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #1d4ed8;
        color: #000000 !important;
    }

    .chat-message-user * {
        color: #000000 !important;
    }

    .chat-message-user strong {
        color: #1d4ed8 !important;
        font-size: 0.95em;
    }

    /* Bot message bubble */
    .chat-message-bot {
        background-color: #dcfce7;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #16a34a;
        color: #000000 !important;
    }

    .chat-message-bot * {
        color: #000000 !important;
    }

    .chat-message-bot strong {
        color: #16a34a !important;
        font-size: 0.95em;
    }

    /* Source card */
    .source-card {
        background-color: #fefce8;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 3px solid #ca8a04;
        font-size: 0.85em;
        color: #000000 !important;
    }

    .source-card * {
        color: #000000 !important;
    }

    .source-card strong {
        color: #92400e !important;
    }

    /* Metric card */
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #cbd5e1;
        color: #000000 !important;
    }

    .stAlert { padding: 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# Title
st.title("Chat with Documents")
st.markdown("### AI-Powered Document Q&A using RAG and Groq Llama3")
st.markdown("---")


# Sidebar
with st.sidebar:
    st.header("Configuration")

    # API Key input
    st.subheader("Groq API Key")
    api_key_input = st.text_input(
        "Enter your Groq API key",
        type="password",
        help="Get your free API key from https://console.groq.com/keys",
        placeholder="gsk_..."
    )

    if api_key_input:
        os.environ['GROQ_API_KEY'] = api_key_input
        st.success("API key set")
    else:
        # Try loading from .env
        load_dotenv()
        env_key = os.getenv('GROQ_API_KEY')
        if env_key and env_key != 'your_groq_api_key_here':
            os.environ['GROQ_API_KEY'] = env_key
            st.success("API key loaded from .env")
        else:
            st.warning("Please enter your Groq API key")

    st.markdown("---")

    # Model settings
    st.subheader("Model Settings")

    model_name = st.selectbox(
    "Select LLM Model",
    options=[
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "moonshotai/kimi-k2-instruct"
    ],
    index=0,
    help="gpt-oss-120b is most capable, llama-3.1-8b is fastest"
)

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of document chunks for processing"
    )

    top_k = st.slider(
        "Top K Results",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of document chunks to retrieve per query"
    )

    st.markdown("---")

    st.subheader("Model Information")
    st.info("""
    Default LLM: openai/gpt-oss-120b

    Embeddings: all-MiniLM-L6-v2

    Vector DB: FAISS

    Search: MMR (Maximal Marginal Relevance)

    Architecture: RAG (Retrieval Augmented Generation)
    """)

    st.markdown("---")

    st.subheader("How to Use")
    st.markdown("""
    1. Enter your Groq API key above

    2. Upload one or more PDF or TXT files

    3. Wait for processing to complete

    4. Start chatting with your documents

    5. View source citations for each answer
    """)

    st.markdown("---")
    st.markdown("Built with LangChain, Groq and Streamlit")


# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

if 'doc_stats' not in st.session_state:
    st.session_state.doc_stats = None


# Main tabs
tab1, tab2, tab3 = st.tabs([
    "Upload Documents",
    "Chat",
    "History"
])


# Tab 1: Upload Documents
with tab1:
    st.header("Upload Documents")
    st.markdown("Upload PDF or TXT files to chat with them.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload one or more PDF or TXT files"
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected")

        # Show file details
        for f in uploaded_files:
            size_kb = f.size / 1024
            st.markdown(f"- **{f.name}** ({size_kb:.1f} KB)")

        # Process button
        col1, col2 = st.columns([1, 3])
        with col1:
            process_btn = st.button(
                "Process Documents",
                type="primary",
                use_container_width=True
            )

        if process_btn:
            # Check API key
            groq_key = os.getenv('GROQ_API_KEY')
            if not groq_key or groq_key == 'your_groq_api_key_here':
                st.error("Please enter a valid Groq API key in the sidebar!")
            else:
                with st.spinner("Processing documents... this may take a moment"):
                    try:
                        # Save files to temp directory
                        temp_dir = tempfile.mkdtemp()
                        file_paths = []

                        for uploaded_file in uploaded_files:
                            temp_path = os.path.join(
                                temp_dir, uploaded_file.name
                            )
                            with open(temp_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            file_paths.append(temp_path)

                        # Initialize RAG pipeline
                        pipeline = RAGPipeline(
                            groq_api_key=groq_key,
                            model_name=model_name
                        )

                        # Process documents
                        stats = pipeline.process_documents(
                            file_paths=file_paths,
                            chunk_size=chunk_size,
                            chunk_overlap=200
                        )

                        # Save to session state
                        st.session_state.rag_pipeline = pipeline
                        st.session_state.documents_processed = True
                        st.session_state.doc_stats = stats
                        st.session_state.chat_history = []

                        st.success("Documents processed successfully!")

                        # Show stats
                        st.markdown("---")
                        st.subheader("Processing Results")

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric(
                                "Documents Loaded",
                                stats['total_documents']
                            )
                        with c2:
                            st.metric(
                                "Chunks Created",
                                stats['total_chunks']
                            )
                        with c3:
                            st.metric(
                                "Avg Chunk Size",
                                f"{stats['avg_chunk_size']} chars"
                            )

                        st.info(
                            "Documents are ready! "
                            "Go to the Chat tab to start asking questions."
                        )

                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        st.info(
                            "Please check your files and API key "
                            "and try again."
                        )

    else:
        st.info("Please upload one or more PDF or TXT files to get started.")

        # Show example questions
        st.markdown("---")
        st.subheader("Example Questions You Can Ask")
        st.markdown("""
        Once you upload a document you can ask questions like:

        - What is the main topic of this document?
        - Summarize the key points from chapter 1
        - What does the document say about [specific topic]?
        - List all the recommendations mentioned
        - Compare the approaches described in sections 2 and 3
        - What conclusions does the author draw?
        """)


# Tab 2: Chat
with tab2:
    st.header("Chat with Your Documents")

    if not st.session_state.documents_processed:
        st.warning(
            "Please upload and process documents first "
            "in the Upload Documents tab."
        )
    else:
        # Document info banner
        if st.session_state.doc_stats:
            stats = st.session_state.doc_stats
            st.success(
                f"Documents ready - "
                f"{stats['total_documents']} document(s), "
                f"{stats['total_chunks']} chunks indexed"
            )

        # Chat controls
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.rag_pipeline:
                    st.session_state.rag_pipeline.reset()
                st.rerun()

        st.markdown("---")

        # Display chat history
        if st.session_state.chat_history:
            for entry in st.session_state.chat_history:
                # User message
                st.markdown(
                    f"""<div class='chat-message-user'>
                    <strong style='color:#1d4ed8;'>You:</strong>
                    <span style='color:#000000;'><br>{entry['question']}</span>
                    </div>""",
                    unsafe_allow_html=True
                )

                # Bot message
                st.markdown(
                f"""<div class='chat-message-bot'>
                <strong style='color:#16a34a;'>Assistant:</strong>
                <span style='color:#000000;'><br>{entry['answer']}</span>
                </div>""",
                unsafe_allow_html=True
            )

                # Sources
                if entry.get('sources'):
                    with st.expander(
                        f"View Sources ({len(entry['sources'])} found)"
                    ):
                        for source in entry['sources']:
                            st.markdown(
                                f"""<div class='source-card'>
                                <strong style='color:#92400e;'>
                                Source {source['rank']}
                                ({source['length']} chars):
                                </strong>
                                <span style='color:#000000;'>
                                <br>{source['content'][:300]}...
                                </span>
                                </div>""",
                                unsafe_allow_html=True
                            )

                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.caption(
                        f"Response time: {entry['response_time']:.2f}s"
                    )
                with col_m2:
                    st.caption(f"Sources used: {entry['num_sources']}")
                with col_m3:
                    st.caption(
                        f"Answer length: {len(entry['answer'].split())} words"
                    )

                st.markdown("---")
        else:
            st.info(
                "No messages yet. Ask a question below to get started!"
            )

        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is this document about?",
            key="question_input"
        )

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            ask_btn = st.button(
                "Ask",
                type="primary",
                use_container_width=True
            )

        if ask_btn and question:
            if not st.session_state.rag_pipeline:
                st.error("Pipeline not initialized. Please re-upload documents.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.rag_pipeline.query(question)
                        st.session_state.chat_history.append(result)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

        elif ask_btn and not question:
            st.warning("Please enter a question first.")

        # Suggested questions
        if st.session_state.chat_history:
            pass
        else:
            st.markdown("---")
            st.subheader("Suggested Questions")

            suggested = [
                "What is the main topic of this document?",
                "Summarize the key points",
                "What are the main conclusions?",
                "What recommendations are made?",
            ]

            cols = st.columns(2)
            for i, suggestion in enumerate(suggested):
                with cols[i % 2]:
                    if st.button(
                        suggestion,
                        key=f"suggest_{i}",
                        use_container_width=True
                    ):
                        with st.spinner("Thinking..."):
                            try:
                                result = st.session_state.rag_pipeline.query(
                                    suggestion
                                )
                                st.session_state.chat_history.append(result)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")


# Tab 3: History & Analytics
with tab3:
    st.header("Conversation History and Analytics")

    if not st.session_state.chat_history:
        st.info("No conversation history yet. Start chatting first!")
    else:
        history = st.session_state.chat_history

        # Summary metrics
        st.subheader("Session Summary")

        import numpy as np

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Questions", len(history))
        with c2:
            avg_time = np.mean([h['response_time'] for h in history])
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        with c3:
            avg_sources = np.mean([h['num_sources'] for h in history])
            st.metric("Avg Sources Used", f"{avg_sources:.1f}")
        with c4:
            avg_words = np.mean(
                [len(h['answer'].split()) for h in history]
            )
            st.metric("Avg Answer Length", f"{avg_words:.0f} words")

        st.markdown("---")

        # Full history table
        st.subheader("Full Conversation History")

        import pandas as pd

        history_df = pd.DataFrame([{
            'Turn': i + 1,
            'Question': h['question'],
            'Answer': h['answer'][:150] + '...' if len(
                h['answer']
            ) > 150 else h['answer'],
            'Sources': h['num_sources'],
            'Time (s)': round(h['response_time'], 2)
        } for i, h in enumerate(history)])

        st.dataframe(history_df, use_container_width=True)

        # Download history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download History CSV",
            data=csv,
            file_name="chat_history.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # Analytics charts
        st.subheader("Analytics")

        col1, col2 = st.columns(2)

        with col1:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            turns = range(1, len(history) + 1)
            times = [h['response_time'] for h in history]

            ax.plot(turns, times, 'b-o', linewidth=2, markersize=8)
            ax.axhline(
                y=np.mean(times),
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Mean: {np.mean(times):.2f}s'
            )
            ax.set_title('Response Time per Turn', fontweight='bold')
            ax.set_xlabel('Turn', fontweight='bold')
            ax.set_ylabel('Response Time (s)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sources = [h['num_sources'] for h in history]
            word_counts = [len(h['answer'].split()) for h in history]

            ax.bar(turns, sources, color='steelblue',
                   edgecolor='black', alpha=0.7, label='Sources Used')
            ax2 = ax.twinx()
            ax2.plot(turns, word_counts, 'r-o', linewidth=2,
                     markersize=8, label='Answer Length (words)')
            ax.set_title('Sources Used and Answer Length', fontweight='bold')
            ax.set_xlabel('Turn', fontweight='bold')
            ax.set_ylabel('Sources Used', color='steelblue', fontweight='bold')
            ax2.set_ylabel(
                'Answer Length (words)', color='red', fontweight='bold'
            )
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close()


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Chat with Documents | Built with LangChain, Groq Llama3 and Streamlit</p>
    <p style='font-size: 12px;'>
        AI Project | RAG Architecture |
        Powered by Groq API
    </p>
</div>
""", unsafe_allow_html=True)