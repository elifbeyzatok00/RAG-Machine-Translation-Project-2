"""
PDF dokümanları okur, chunk'lara böler ve ChromaDB'ye embed eder.
Mevcut terminology collection'ına ek olarak ayrı bir doc collection kullanır.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import fitz  # PyMuPDF
import re
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import CHROMADB_PATH, MODEL_NAME, DOCUMENTS_DIR

DOC_COLLECTION_NAME = "rag_mt_documents"
CHUNK_SIZE = 300        # karakter
CHUNK_OVERLAP = 60


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text("text")
        # Çoklu boşluk ve sayfa başlığı temizliği
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        pages.append(text.strip())
    doc.close()
    return "\n\n".join(p for p in pages if p)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Metni örtüşen chunk'lara böl."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start: start + size])
        if len(chunk.strip()) > 40:   # çok kısa chunk'ları atla
            chunks.append(chunk.strip())
        start += size - overlap
    return chunks


# ── main ─────────────────────────────────────────────────────────────────────

def ingest_documents(documents_dir: Path = DOCUMENTS_DIR):
    pdf_files = sorted(documents_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  data/documents/ klasöründe PDF bulunamadı.")
        return

    print(f"📂 {len(pdf_files)} PDF bulundu.")

    print("📦 Embedding modeli yükleniyor...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    try:
        client.delete_collection(name=DOC_COLLECTION_NAME)
        print("🗑️  Eski doküman collection silindi.")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=DOC_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    total_chunks = 0
    for pdf_path in pdf_files:
        print(f"\n📄 {pdf_path.name}")
        try:
            text = extract_text(pdf_path)
        except Exception as e:
            print(f"  ❌ Metin çıkarılamadı: {e}")
            continue

        chunks = chunk_text(text)
        print(f"  → {len(chunks)} chunk")

        ids, embeddings, docs, metas = [], [], [], []
        for i, chunk in enumerate(tqdm(chunks, desc="  embedding", leave=False)):
            chunk_id = f"{pdf_path.stem}_{i}"
            emb = model.encode(chunk)
            ids.append(chunk_id)
            embeddings.append(emb.tolist())
            docs.append(chunk)
            metas.append({"source": pdf_path.name, "chunk": i})

            # ChromaDB batch insert (500'lük gruplar)
            if len(ids) == 500:
                collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
                ids, embeddings, docs, metas = [], [], [], []

        if ids:
            collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)

        total_chunks += len(chunks)
        print(f"  ✅ Tamamlandı")

    print(f"\n🎉 Toplam {total_chunks} chunk, {len(pdf_files)} PDF ChromaDB'ye yüklendi.")
    print(f"   Collection: '{DOC_COLLECTION_NAME}'")
    return collection


def search_documents(query: str, k: int = 5):
    """Doküman collection'ında semantik arama."""
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_collection(name=DOC_COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=k
    )

    print(f"\n🔍 Query: {query}")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n[{i+1}] {meta['source']} | chunk {meta['chunk']} | dist: {dist:.3f}")
        print(f"     {doc[:200]}...")


if __name__ == "__main__":
    ingest_documents()

    print("\n" + "=" * 60)
    print("TEST ARAMALARI")
    print("=" * 60)
    search_documents("rulman bakımı nasıl yapılır")
    search_documents("arıza tespit yöntemleri")
    search_documents("döner mil kontrol sistemi")
