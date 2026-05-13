import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sentence_transformers import SentenceTransformer
import chromadb
from config import (
    MODEL_NAME, CHROMADB_PATH, CHROMA_COLLECTION_NAME, TERMINOLOGY_DIR
)
import json
from tqdm import tqdm


def create_embeddings():
    """ChromaDB'ye documents ve terms embed et"""

    print("📦 Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("📥 Loading terminology...")
    terms_data = []
    terms_file = TERMINOLOGY_DIR / "terms.jsonl"
    if terms_file.exists():
        with open(terms_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    terms_data.append(json.loads(line))
    else:
        print("⚠️  No terminology file found. Using sample data...")
        terms_data = [
            {"tr": "döner mil", "en": "rotating shaft"},
            {"tr": "bakım hatası", "en": "maintenance error"},
            {"tr": "sistem kontrol", "en": "system control"},
        ]

    print("🔄 Embedding terminology...")
    for i, term in enumerate(tqdm(terms_data)):
        text = f"{term['tr']} {term['en']}"
        embedding = model.encode(text)
        collection.add(
            ids=[f"term_{i}"],
            embeddings=[embedding.tolist()],
            documents=[term['tr']],
            metadatas=[{"type": "terminology", "en": term['en']}]
        )

    print(f"✅ Embedded {len(terms_data)} terminology entries")
    return collection, model


if __name__ == "__main__":
    collection, model = create_embeddings()
    print("✨ Embedding pipeline complete!")
