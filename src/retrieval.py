import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMADB_PATH, CHROMA_COLLECTION_NAME, MODEL_NAME

DOC_COLLECTION_NAME = "rag_mt_documents"


def retrieve_relevant_terms(query: str, model, collection, k: int = 5) -> list:
    """Query'ye benzer terimleri ChromaDB'den al"""

    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    retrieved_terms = []
    if results['ids'] and len(results['ids']) > 0:
        for i, doc_id in enumerate(results['ids'][0]):
            retrieved_terms.append({
                "id": doc_id,
                "term": results['documents'][0][i] if results['documents'] else "",
                "en": results['metadatas'][0][i].get('en', "") if results['metadatas'] else "",
                "distance": results['distances'][0][i] if results['distances'] else 0
            })

    return retrieved_terms


def test_retrieval():
    """Retrieval'ı test et"""

    print("🔍 Testing retrieval...")

    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME)

    test_queries = [
        "döner mil nasıl çalışır?",
        "bakım hatası tespit etme",
        "sistem kontrolü yapıldı"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        results = retrieve_relevant_terms(query, model, collection, k=3)
        for r in results:
            print(f"  ✓ {r['term']} → {r['en']} (dist: {r['distance']:.3f})")


def retrieve_document_context(query: str, model, k: int = 3) -> list:
    """PDF dokümanlarından ilgili pasajları getir."""
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    try:
        collection = client.get_collection(name=DOC_COLLECTION_NAME)
    except Exception:
        return []

    results = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=k
    )

    contexts = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            contexts.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", ""),
                "distance": results["distances"][0][i]
            })
    return contexts


if __name__ == "__main__":
    test_retrieval()
