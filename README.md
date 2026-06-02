# Chat with Documents

A RAG-powered document Q&A application built with LangChain, Groq and Streamlit.
Upload any PDF or TXT file and have an intelligent conversation with it.

## Live Demo

[Open App](https://chatwithdocuments-gfwhxrmpyc6bbwubghhyn9.streamlit.app/)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

---

## Overview

Chat with Documents is an AI-powered application that lets you upload any
document and ask questions about it in natural language. Instead of reading
through long documents manually, you can simply ask questions and get
accurate answers with source citations.

The application uses Retrieval Augmented Generation (RAG), a technique that
combines the power of large language models with a custom knowledge base built
from your documents. This means the AI answers strictly from your document
content rather than relying on general knowledge, which significantly reduces
hallucinations and improves accuracy.

Use Cases:
- Research paper analysis
- Legal document review
- Study material comprehension
- Technical documentation Q&A
- Business report analysis
- Book summarization and Q&A

---

## How It Works

RAG works in 4 stages:
Stage 1: Document Processing
Your PDF/TXT
|
v
Text Extraction
|
v
Chunking (1000 char chunks, 200 char overlap)
|
v
384-dimensional Embeddings (all-MiniLM-L6-v2)
|
v
FAISS Vector Store (indexed for fast search)
Stage 2: Query Processing
User Question
|
v
Query Embedding (same model as documents)
|
v
MMR Similarity Search (fetch top 10, return best 4)
|
v
Relevant Chunks Retrieved
Stage 3: Answer Generation
Retrieved Chunks + Conversation History + Question
|
v
Groq LLM (openai/gpt-oss-120b)
|
v
Answer + Source Citations
Stage 4: Memory Management
Answer stored in ConversationBufferMemory
|
v
Used as context for follow-up questions

### Why MMR Search?

Standard similarity search returns the top K most similar chunks which
often results in redundant, overlapping content. Maximal Marginal Relevance
(MMR) balances relevance with diversity, ensuring retrieved chunks cover
different aspects of the answer.

### Why Groq?

Groq provides extremely fast inference speeds compared to other LLM providers,
making the chat experience feel responsive and natural. The
openai/gpt-oss-120b model delivers high quality answers with low latency.

---

## Features

### Core Features
- PDF and TXT file upload (single or multiple files)
- Multi-turn conversation with memory
- Source citations for every answer
- Adjustable confidence and retrieval parameters
- Multiple Groq LLM model options

### Interface Features
- Three tab layout: Upload, Chat, History
- Suggested questions for quick start
- Clear chat button to reset conversation
- Visual chat bubbles with color coding

### Analytics Features
- Session summary metrics
- Response time tracking per turn
- Sources used per answer
- Answer length analytics
- Downloadable chat history as CSV

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Groq openai/gpt-oss-120b | Answer generation |
| Embeddings | all-MiniLM-L6-v2 | Text vectorization |
| Vector DB | FAISS | Similarity search |
| Framework | LangChain v0.2 | RAG orchestration |
| Memory | ConversationBufferMemory | Chat history |
| Search | MMR | Diverse retrieval |
| Frontend | Streamlit | Web interface |
| PDF Loading | PyPDF | Document parsing |

---

## Architecture
chat-with-documents/
|
|-- app.py                    Entry point, Streamlit UI
|-- rag_pipeline.py           Core RAG logic (reusable class)
|
|-- RAGPipeline class
|   |-- get_embeddings()      Load HuggingFace model
|   |-- get_llm()             Initialize Groq LLM
|   |-- load_document()       PDF and TXT loading
|   |-- chunk_documents()     Recursive text splitting
|   |-- create_vector_store() FAISS index creation
|   |-- _build_chain()        ConversationalRetrievalChain
|   |-- process_documents()   End-to-end document pipeline
|   |-- query()               Single turn Q&A
|   -- reset()               Clear conversation memory | |-- Streamlit App (app.py)     |-- Tab 1: Upload Documents     |   |-- File uploader     |   |-- Processing pipeline     |   -- Stats display
|
|-- Tab 2: Chat
|   |-- Chat history display
|   |-- Question input
|   |-- Source expander
|   -- Suggested questions     |     -- Tab 3: History
|-- Session metrics
|-- Full history table
|-- Analytics charts
`-- CSV download

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Groq API key (free at https://console.groq.com/keys)
- 4GB RAM minimum

### Step 1: Clone Repository

```bash
git clone https://github.com/ADHIRAJ994/chat-with-documents.git
cd chat-with-documents
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Key

Create a `.env` file in the project root:
GROQ_API_KEY=your_groq_api_key_here

Or enter it directly in the app sidebar.

### Step 5: Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Usage

### Basic Usage

Enter your Groq API key in the sidebar
Go to the Upload Documents tab
Upload one or more PDF or TXT files
Click Process Documents and wait for indexing
Go to the Chat tab
Ask questions about your document
View source citations by expanding View Sources


### Advanced Usage
Adjusting Parameters (sidebar):

Model: Choose between speed and quality

openai/gpt-oss-120b: Best quality (default)
openai/gpt-oss-20b: Faster responses
llama-3.1-8b-instant: Fastest, lower quality


Temperature: Controls answer creativity

0.0 - 0.2: Focused, factual answers (recommended)
0.5 - 0.8: More creative responses


Chunk Size: Controls document splitting

500: More precise retrieval, less context
1000: Balanced (default)
2000: More context, less precise


Top K Results: Chunks retrieved per query

2: Fast but may miss context
4: Balanced (default)
8: More context, slower




### Python API

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(
    groq_api_key="your_api_key",
    model_name="openai/gpt-oss-120b"
)

# Process documents
stats = pipeline.process_documents(
    file_paths=["document.pdf"],
    chunk_size=1000,
    chunk_overlap=200
)

print(f"Indexed {stats['total_chunks']} chunks")

# Ask questions
result = pipeline.query("What is the main topic of this document?")

print(f"Answer: {result['answer']}")
print(f"Sources: {result['num_sources']}")
print(f"Time: {result['response_time']:.2f}s")

# Multi-turn conversation
result2 = pipeline.query("Can you elaborate on that?")

# Reset conversation
pipeline.reset()
```

## Configuration

`config.json` controls all pipeline settings:

```json
{
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "llm_model": "openai/gpt-oss-120b",
    "groq_model": "openai/gpt-oss-120b",
    "top_k_retrieval": 4,
    "temperature": 0.1,
    "max_tokens": 1024,
    "search_type": "mmr",
    "fetch_k": 10,
    "lambda_mult": 0.5
}
```

### Parameter Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| chunk_size | 1000 | Characters per chunk |
| chunk_overlap | 200 | Overlap between chunks |
| top_k_retrieval | 4 | Chunks retrieved per query |
| temperature | 0.1 | LLM creativity (0=focused) |
| max_tokens | 1024 | Maximum answer length |
| fetch_k | 10 | Candidates for MMR selection |
| lambda_mult | 0.5 | MMR diversity vs relevance |

---

## Supported Models

| Model | Speed | Quality | Context Window |
|-------|-------|---------|----------------|
| openai/gpt-oss-120b | Medium | Best | 8192 tokens |
| openai/gpt-oss-20b | Fast | Very Good | 8192 tokens |
| llama-3.3-70b-versatile | Medium | Very Good | 128000 tokens |
| llama-3.1-8b-instant | Fastest | Good | 128000 tokens |
| gemma2-9b-it | Fast | Good | 8192 tokens |
| moonshotai/kimi-k2-instruct | Medium | Excellent | 131072 tokens |

---

## Limitations

### Technical Limitations
- Documents are processed in memory and not persisted between sessions
- Very large documents (over 100 pages) may take 1-2 minutes to process
- Images and tables in PDFs are not currently supported
- Only English language documents are fully supported

### RAG Limitations
- Answers are limited to information in the uploaded document
- Very specific numerical data may occasionally be misread from PDFs
- Complex multi-hop reasoning across distant document sections may be imperfect
- Conversation memory is limited to current session only

### API Limitations
- Groq API has rate limits on free tier
- API key must be entered each session unless saved in .env file

---

## Future Improvements

### Short Term
- [ ] Support for Word documents (.docx)
- [ ] Support for images and tables in PDFs
- [ ] Persistent vector store across sessions
- [ ] Better handling of very large documents

### Medium Term
- [ ] Multi-language document support
- [ ] Document comparison (ask questions across multiple docs)
- [ ] Answer confidence scoring
- [ ] Export conversation as PDF report

### Long Term
- [ ] Fine-tuned embedding model for specific domains
- [ ] Integration with cloud storage (Google Drive, Dropbox)
- [ ] API endpoint for programmatic access
- [ ] Mobile-responsive UI improvements

---

## Key Concepts Learned

This project demonstrates proficiency in:

- RAG (Retrieval Augmented Generation) architecture
- Vector databases and semantic search
- LangChain framework for LLM orchestration
- FAISS for efficient similarity search
- Groq API integration
- Conversation memory management
- MMR search for diverse retrieval
- Streamlit web application development
- End-to-end AI application deployment

---

## License

MIT License
Copyright (c) 2024 Your Name
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contact

Your Name

- GitHub: https://github.com/ADHIRAJ994
- LinkedIn: https://www.linkedin.com/in/adhiraj-chakravorty-788685344/
- Email: youradhi20@gmail.com

---

## Acknowledgments

- LangChain team for the RAG framework
- Groq for fast LLM inference API
- HuggingFace for the sentence-transformers library
- Facebook Research for the FAISS library
- Streamlit for the web framework

---

Built with LangChain, Groq and Streamlit
