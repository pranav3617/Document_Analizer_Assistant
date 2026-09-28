import os
import ast
import operator
import uuid
import chromadb

from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# Document Loader
# ============================================================

def load_document(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")

    return loader.load()


# ============================================================
# Document Splitter
# ============================================================

def split_docs(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


# ============================================================
# Embedding Manager (shared across all sessions — stateless encoder)
# ============================================================

class EmbeddingManager:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        print("Loading embedding model:", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        print("Embedding dimensions =", self.model.get_sentence_embedding_dimension())

    def generate_embeddings(self, texts, batch_size=64):
        # Encoding thousands of chunks (e.g. a 1500-page document) in
        # one giant call spikes memory. sentence-transformers batches
        # internally too, but setting it explicitly keeps peak memory
        # predictable regardless of how many chunks come in.
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 200
        )
        return embeddings.tolist()

    def generate_query_embedding(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()


# ============================================================
# Vector Store Manager (one collection per session)
# ============================================================

class VectorStoreManager:

    def __init__(self, persist_directory="data/vector_store", collection_name="pdf_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def clear_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

    def add_documents(self, documents, embeddings):
        ids, texts, metadatas = [], [], []

        for i, document in enumerate(documents):
            ids.append(str(uuid.uuid4()))
            texts.append(document.page_content)
            metadata = document.metadata.copy()
            metadata["chunk_id"] = i
            metadatas.append(metadata)

        if len(texts) == 0:
            raise ValueError("No document chunks were generated.")

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )


# ============================================================
# RAG Retriever
# ============================================================

class RAGRetriever:

    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedding_manager.generate_query_embedding(query)

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_documents = []
        for i, text in enumerate(documents):
            retrieved_documents.append({
                "text": text,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else None
            })

        return retrieved_documents


# ============================================================
# Groq LLM (shared — client itself is stateless per call)
# ============================================================

# Model must support tool/function calling on Groq.
# "openai/gpt-oss-20b" and "llama-3.3-70b-versatile" both do.
AGENT_MODEL = "openai/gpt-oss-20b"


def create_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")

    return ChatGroq(
        model=AGENT_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY
    )


# ============================================================
# Safe Calculator (no eval() — only arithmetic is allowed)
# ============================================================

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed")

    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )

    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))

    raise ValueError("Unsupported expression")


def safe_calculate(expression):
    try:
        tree = ast.parse(expression, mode="eval")
        return _safe_eval(tree.body)
    except Exception as e:
        raise ValueError(f"Could not evaluate '{expression}': {e}")


# ============================================================
# Web Search (DuckDuckGo — no API key required)
# ============================================================

def web_search_impl(query, max_results=4):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return "No web results found."

    formatted = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        formatted.append(f"- {title}: {body} ({href})")

    return "\n".join(formatted)


# ============================================================
# Agent Loop
# Builds session-specific tools (document search is bound to
# this session's retriever via closure), binds them to the LLM,
# and loops until the model stops calling tools.
# ============================================================

MAX_AGENT_STEPS = 5


def run_agent(question, llm, retriever=None, document_loaded=False, top_k=5):

    tools = []

    if document_loaded and retriever is not None:

        @tool
        def search_documents(query: str) -> str:
            """Search the user's uploaded documents for information
            relevant to the query. Use this whenever the question could
            be about the content of files the user uploaded."""
            retrieved = retriever.retrieve(query, top_k=top_k)
            if not retrieved:
                return "No relevant content found in the uploaded documents."
            return "\n\n".join(
                f"[{doc['metadata'].get('source_file', 'document')}] {doc['text']}"
                for doc in retrieved
            )

        tools.append(search_documents)

    @tool
    def web_search(query: str) -> str:
        """Search the public web for current information, facts, or
        anything not found in the uploaded documents. Use this for
        general knowledge questions or anything time-sensitive."""
        return web_search_impl(query)

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4) / 2'.
        Only numbers and + - * / % ** are supported."""
        return str(safe_calculate(expression))

    tools.extend([web_search, calculator])

    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        "You are a helpful assistant with access to tools: "
        + ", ".join(tools_by_name.keys())
        + ". Use search_documents for questions about the user's uploaded "
        "files, web_search for general or current-events questions, and "
        "calculator for arithmetic. Only use a tool when it's actually "
        "needed — answer directly if you already know the answer. Keep "
        "answers clear and concise, and mention which source you used "
        "when it's relevant."
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]

    for _ in range(MAX_AGENT_STEPS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return response.content

        for call in response.tool_calls:
            tool_fn = tools_by_name.get(call["name"])

            if tool_fn is None:
                result = f"Unknown tool: {call['name']}"
            else:
                try:
                    result = tool_fn.invoke(call["args"])
                except Exception as e:
                    result = f"Tool error: {e}"

            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "I wasn't able to finish answering that within my tool-use limit. Try rephrasing or breaking it into a simpler question."


# ============================================================
# Per-Session RAG System
# One instance per browser session (via cookie), so two users
# uploading documents at the same time never overwrite each other.
# ============================================================

class RAGSession:

    def __init__(self, session_id, embedding_manager, llm):
        self.session_id = session_id
        self.embedding_manager = embedding_manager
        self.llm = llm

        self.vector_store = VectorStoreManager(
            persist_directory="data/vector_store",
            collection_name=f"session_{session_id}"
        )

        self.retriever = RAGRetriever(self.vector_store, self.embedding_manager)
        self.document_loaded = False
        self.loaded_files = []  # names of files currently in this session's store

    def process_documents(self, file_paths):
        """
        Process one or more files and ADD them to this session's
        existing knowledge base (does not remove previously uploaded
        files). Call clear() first if you want to start fresh.
        """
        all_chunks = []
        total_pages = 0
        processed_names = []
        skipped = []

        for file_path in file_paths:
            try:
                documents = load_document(file_path)
            except Exception as e:
                skipped.append({"file": Path(file_path).name, "reason": str(e)})
                continue

            if not documents:
                skipped.append({"file": Path(file_path).name, "reason": "empty document"})
                continue

            chunks = split_docs(documents, chunk_size=1000, chunk_overlap=100)

            if not chunks:
                skipped.append({"file": Path(file_path).name, "reason": "no extractable text"})
                continue

            # Tag every chunk with the source filename so answers can
            # eventually be traced back to a specific PDF if needed.
            source_name = Path(file_path).name
            for chunk in chunks:
                chunk.metadata["source_file"] = source_name

            all_chunks.extend(chunks)
            total_pages += len(documents)
            processed_names.append(source_name)

        if not all_chunks:
            raise ValueError(
                "No text could be extracted from the file(s) provided."
                + (f" Skipped: {skipped}" if skipped else "")
            )

        texts = [chunk.page_content for chunk in all_chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)

        # ADD to the existing collection rather than clearing it —
        # this is what lets multiple PDFs build up in one session.
        self.vector_store.add_documents(all_chunks, embeddings)

        self.retriever = RAGRetriever(self.vector_store, self.embedding_manager)
        self.document_loaded = True
        self.loaded_files.extend(processed_names)

        return {
            "pages": total_pages,
            "chunks": len(all_chunks),
            "files_processed": processed_names,
            "files_skipped": skipped,
            "total_files_in_session": len(self.loaded_files),
            "message": f"{len(processed_names)} file(s) processed successfully"
        }

    def clear(self):
        """Wipe this session's entire knowledge base to start over."""
        self.vector_store.clear_collection()
        self.retriever = RAGRetriever(self.vector_store, self.embedding_manager)
        self.document_loaded = False
        self.loaded_files = []

    def ask(self, question):
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        return run_agent(
            question,
            self.llm,
            retriever=self.retriever,
            document_loaded=self.document_loaded,
            top_k=5
        )


# ============================================================
# Session Manager — creates/looks up a RAGSession per session_id
# ============================================================

class RAGSessionManager:

    def __init__(self):
        print("\nInitializing shared embedding model and LLM...\n")
        self.embedding_manager = EmbeddingManager()
        self.llm = create_llm()
        self._sessions = {}
        print("\nRAG Session Manager ready.\n")

    def get_session(self, session_id):
        if session_id not in self._sessions:
            self._sessions[session_id] = RAGSession(
                session_id, self.embedding_manager, self.llm
            )
        return self._sessions[session_id]

    def remove_session(self, session_id):
        session = self._sessions.pop(session_id, None)
        if session:
            session.vector_store.delete_collection()


# ============================================================
# Global manager (holds the shared model + per-session stores)
# ============================================================

session_manager = RAGSessionManager()