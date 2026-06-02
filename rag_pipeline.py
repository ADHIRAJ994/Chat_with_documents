import os
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import traceback
# Updated LangChain v0.2 imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions
based on the provided document context.

INSTRUCTIONS:
- Answer the question using ONLY the information in the context below
- If the answer is not in the context say:
  I could not find information about this in the provided documents
- Be concise and accurate
- Do not make up information or use outside knowledge

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION:
{question}

ANSWER:"""


CONDENSE_QUESTION_TEMPLATE = """Given the conversation history and a follow-up
question rephrase the follow-up question to be a standalone question that
includes all necessary context.

Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


def load_document(file_path):
    """Load document from PDF or TXT file"""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == '.pdf':
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        return pages

    elif file_path.suffix.lower() == '.txt':
        loader = TextLoader(str(file_path), encoding='utf-8')
        docs = loader.load()
        return docs

    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks, embeddings):
    """Create FAISS vector store from chunks"""

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


def load_vector_store(vector_store_path, embeddings):
    """Load existing FAISS vector store"""

    vector_store = FAISS.load_local(
        vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


def get_embeddings():
    """Load HuggingFace embedding model"""

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("Embeddings loaded successfully")

    except Exception as e:
        print("\nFULL ERROR:")
        print(traceback.format_exc())
        raise

    return embeddings


def get_llm(api_key, model_name="openai/gpt-oss-120b",
            temperature=0.1, max_tokens=1024):
    """Initialize Groq LLM"""

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return llm


class RAGPipeline:
    """Complete RAG Pipeline for Chat with Documents"""

    def __init__(self, groq_api_key, vector_store=None,
                 model_name="openai/gpt-oss-120b"):

        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.conversation_history = []

        # Load embeddings
        self.embeddings = get_embeddings()

        # Load LLM
        self.llm = get_llm(
            api_key=groq_api_key,
            model_name=model_name
        )

        # Set vector store
        self.vector_store = vector_store

        # Initialize chain components
        self.memory = None
        self.chain = None

        if vector_store is not None:
            self._build_chain()

    def _build_chain(self):
        """Build RAG chain with memory"""

        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        qa_prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=RAG_PROMPT_TEMPLATE
        )

        condense_prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template=CONDENSE_QUESTION_TEMPLATE
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            condense_question_prompt=condense_prompt,
            output_key="answer"
        )

    def process_documents(self, file_paths, chunk_size=1000,
                          chunk_overlap=200):
        """Process documents and create vector store"""

        all_documents = []

        for file_path in file_paths:
            try:
                docs = load_document(file_path)
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if not all_documents:
            raise ValueError("No documents loaded successfully")

        chunks = chunk_documents(
            all_documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.vector_store = create_vector_store(chunks, self.embeddings)
        self._build_chain()

        return {
            'total_documents': len(all_documents),
            'total_chunks': len(chunks),
            'avg_chunk_size': int(
                np.mean([len(c.page_content) for c in chunks])
            )
        }

    def query(self, question):
        """Query the RAG pipeline"""

        if self.chain is None:
            return {
                'question': question,
                'answer': 'Please upload documents first.',
                'sources': [],
                'num_sources': 0,
                'response_time': 0
            }

        start_time = time.time()

        response = self.chain({"question": question})

        response_time = time.time() - start_time

        answer = response.get('answer', 'No answer generated')
        source_docs = response.get('source_documents', [])

        sources = []
        for i, doc in enumerate(source_docs):
            sources.append({
                'rank': i + 1,
                'content': doc.page_content,
                'length': len(doc.page_content)
            })

        result = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources),
            'response_time': response_time
        }

        self.conversation_history.append(result)

        return result

    def reset(self):
        """Reset conversation memory"""

        if self.memory is not None:
            self.memory.clear()

        self.conversation_history = []

    def save_vector_store(self, path):
        """Save vector store to disk"""

        if self.vector_store is None:
            raise ValueError("No vector store to save")

        os.makedirs(path, exist_ok=True)
        self.vector_store.save_local(path)

    def load_existing_vector_store(self, path):
        """Load existing vector store"""

        self.vector_store = load_vector_store(path, self.embeddings)
        self._build_chain()