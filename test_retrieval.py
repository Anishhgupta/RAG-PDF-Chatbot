
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import JinaEmbeddings
from langchain_chroma import Chroma

# ---------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / ".env"

load_dotenv(ENV_PATH, override=True)

jina_api_key = os.getenv("JINA_API_KEY")

if not jina_api_key:
    raise ValueError("JINA_API_KEY not found in .env")

# ---------------------------------
# 2. INITIALIZE JINA EMBEDDINGS
# ---------------------------------

embeddings = JinaEmbeddings(
    jina_api_key=jina_api_key,
    model_name="jina-embeddings-v3"
)

# ---------------------------------
# 3. LOAD EXISTING CHROMA DATABASE
# ---------------------------------

DB_PATH = BASE_DIR / "chroma db"

vectorstore = Chroma(
    collection_name="invoice_collection",
    embedding_function=embeddings,
    persist_directory=str(DB_PATH)
)

print("\n========== DATABASE CHECK ==========")
print("Database path:", DB_PATH)

total_chunks = vectorstore._collection.count()

print("Total stored chunks:", total_chunks)

if total_chunks == 0:
    raise ValueError(
        "Database is empty. Check create_db.py."
    )

# ---------------------------------
# 4. CHECK IF OMORI WAS INDEXED
# ---------------------------------

print("\n========== OMORI INDEX CHECK ==========")

stored_data = vectorstore.get(
    include=["documents", "metadatas"]
)

omori_chunks = []

for text, metadata in zip(
    stored_data["documents"],
    stored_data["metadatas"]
):
    if text and "omori" in text.lower():
        omori_chunks.append((text, metadata))

print("Chunks containing OMORI:", len(omori_chunks))

for i, (text, metadata) in enumerate(omori_chunks, 1):
    print(f"\n--- OMORI Chunk {i} ---")
    print("Source:", metadata.get("source", "Unknown"))
    print("Page:", metadata.get("page", "Unknown"))
    print(text[:1000])

# ---------------------------------
# 5. TEST SIMILARITY SEARCH
# ---------------------------------

query = "How does OMORI explore memory and emotional avoidance?"

print("\n========== SIMILARITY SEARCH ==========")
print("Question:", query)

results = vectorstore.similarity_search_with_score(
    query,
    k=8
)

print("Retrieved chunks:", len(results))

for i, (doc, score) in enumerate(results, 1):
    print(f"\n--- Retrieved Chunk {i} ---")
    print("Distance:", score)
    print("Source:", doc.metadata.get("source", "Unknown"))
    print("Page:", doc.metadata.get("page", "Unknown"))
    print("Contains OMORI:", "omori" in doc.page_content.lower())
    print("Content:")
    print(doc.page_content[:1200])

# ---------------------------------
# 6. TEST ALTERNATIVE QUERIES
# ---------------------------------

queries = [
    "What is OMORI about?",
    "OMORI memory emotional avoidance",
    "What themes does OMORI explore?"
]

print("\n========== ALTERNATIVE QUERY TEST ==========")

for q in queries:
    print(f"\nQuestion: {q}")

    docs = vectorstore.similarity_search(q, k=5)

    found = False

    for doc in docs:
        if "omori" in doc.page_content.lower():
            found = True
            print("OMORI chunk retrieved!")
            print(doc.page_content[:600])
            break

    if not found:
        print("No OMORI chunk in top 5 results.")

print("\n========== TEST COMPLETED ==========")
