import argparse
import os
import subprocess
import sys
import warnings

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_PYTHON = os.path.join(_BASE_DIR, ".venv", "Scripts", "python.exe")
if (
    sys.prefix == sys.base_prefix
    and os.environ.get("_RAG_VENV_REEXEC") != "1"
    and os.path.isfile(_VENV_PYTHON)
    and os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(_VENV_PYTHON)
):
    _env = os.environ.copy()
    _env["_RAG_VENV_REEXEC"] = "1"
    raise SystemExit(
        subprocess.call([_VENV_PYTHON, os.path.abspath(__file__), *sys.argv[1:]], env=_env)
    )

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# 1. Load environment variables
load_dotenv(os.path.join(_BASE_DIR, ".env"))
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise ValueError("NVIDIA_API_KEY is not set in the environment or .env file.")

# 2. Document Loading
pdf_path = os.path.join(
    _BASE_DIR,
    "downloadloader",
    "Watershed_Development_Project_Report.pdf",
)
print(f"Loading document from: {pdf_path}")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} document pages.")

# 3. Document Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} text chunks.")

# 4. Embeddings & In-Memory Vector Store (Qdrant + FastEmbed)
print("Creating embeddings and indexing into Qdrant vector store...")
embeddings = FastEmbedEmbeddings()

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="watershed_rag_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="watershed_rag_collection",
    embedding=embeddings,
)
vectorstore.add_documents(chunks)
print("Indexing complete!")

# 5. Initialize NVIDIA LLM
# Note: nemotron uses reasoning tokens, so max_completion_tokens should be >= 1024
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500,
)

# 6. Retriever & RAG Prompt
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert assistant in Watershed Development, GIS, and Remote Sensing. "
            "Answer the question accurately based ONLY on the provided context from the "
            "Watershed Development Project Report. "
            "If the answer cannot be found in the context, say that you don't know.\n\n"
            "Context:\n{context}"
        ),
    ),
    ("human", "{question}"),
])

rag_chain = prompt | llm | StrOutputParser()

# 7. Test RAG with watershed development questions
def query_rag(question: str):
    print("\n" + "="*60)
    print(f"Query: {question}")
    print("="*60)
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunk(s):")
    for i, doc in enumerate(retrieved_docs, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        display_page = page_num + 1 if isinstance(page_num, int) else page_num
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f" [{i}] (Page {display_page}): {preview}...")

    # Build context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response
    print("\nGenerating response from NVIDIA LLM...")
    response = rag_chain.invoke({"context": context, "question": question})
    print("\nAnswer:")
    print(response)
    print("="*60)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watershed Development Report RAG")
    parser.add_argument("-q", "--question", help="Ask a single question and exit")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start an interactive Q&A loop",
    )
    args = parser.parse_args()

    if args.question:
        query_rag(args.question)
    elif args.interactive:
        print("\nInteractive mode. Type 'exit' or 'quit' to stop.")
        while True:
            user_question = input("\nAsk a question: ").strip()
            if not user_question or user_question.lower() in {"exit", "quit"}:
                break
            query_rag(user_question)
    else:
        test_questions = [
            "What are the key thematic outputs and suggested technologies in the watershed development project?",
            "What are the main limitations and future scope mentioned in the watershed development report?",
        ]
        for question in test_questions:
            query_rag(question)
