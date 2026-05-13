import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

DOCUMENTS_DIR = DATA_DIR / "documents"
TERMINOLOGY_DIR = DATA_DIR / "terminology"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
QUANTIZATION = True

CHROMADB_PATH = str(EMBEDDINGS_DIR / "chromadb")
CHROMA_COLLECTION_NAME = "rag_mt_collection"

BATCH_SIZE = 4
MAX_LENGTH = 512
TEMPERATURE = 0.3
TOP_P = 0.95

BLEU_SMOOTHING = "floor"

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(TERMINOLOGY_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
