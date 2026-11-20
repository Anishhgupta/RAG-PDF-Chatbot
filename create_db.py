from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import JinaEmbeddings
from dotenv import load_dotenv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("JINA_API_KEY")

if not api_key:
    raise ValueError("JINA_API_KEY not found in .env")

embeddings = JinaEmbeddings(
    jina_api_key=api_key,
    model_name="jina-embeddings-v3"
)

data = PyMuPDFLoader("peakfiction.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)


vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="invoice_collection" , persist_directory=str(
    Path(__file__).resolve().parent / "chroma db"
))