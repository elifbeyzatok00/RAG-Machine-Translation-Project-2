import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    MISTRAL_MODEL, QUANTIZATION, TEMPERATURE, TOP_P,
    CHROMADB_PATH, CHROMA_COLLECTION_NAME, MODEL_NAME
)
from retrieval import retrieve_relevant_terms


def build_rag_prompt(text: str, retrieved_context: list) -> str:
    """RAG context'i kullanarak prompt oluştur"""

    close_terms = [t for t in retrieved_context if t["distance"] < 0.65][:4]
    glossary_lines = "\n".join(
        [f'  "{t["term"]}" → "{t["en"]}"' for t in close_terms]
    )

    prompt = (
        f"[INST] Translate the Turkish sentence below into English.\n"
        f"Rules:\n"
        f"1. Output ONLY the translated sentence. No explanations.\n"
        f"2. Replace each Turkish term with its exact English equivalent from this glossary:\n"
        f"{glossary_lines}\n\n"
        f"Turkish: {text}\n"
        f"English: [/INST]"
    )
    return prompt


def build_baseline_prompt(text: str) -> str:
    return (
        f"[INST] Translate this Turkish sentence to English. "
        f"Output ONLY the English translation, nothing else.\n\n"
        f"Turkish: {text}\n"
        f"English: [/INST]"
    )


def load_mistral():
    """Mistral modelini yükle (4-bit quantized on GPU)"""
    print("💻 Loading Mistral model (4-bit quantized)...")
    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    use_gpu = torch.cuda.is_available()

    if QUANTIZATION and use_gpu:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print(f"✅ Model loaded on GPU (4-bit NF4)")
    elif use_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"✅ Model loaded on GPU (float16)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL,
            torch_dtype=torch.float32,
        )
        print(f"⚠️  Model loaded on CPU (slow)")

    return tokenizer, model


def generate_translation(prompt: str, tokenizer, model) -> str:
    """Model ile çeviri üret"""
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "[/INST]" in full_text:
        translation = full_text.split("[/INST]")[-1].strip()
    else:
        translation = full_text[len(prompt):].strip()

    # İlk anlamlı satırı al, ekstra açıklamaları at
    lines = [ln.strip() for ln in translation.splitlines() if ln.strip()]
    if lines:
        # "However", "Note", "Turkish:" gibi meta satırları çıkar
        clean = []
        for ln in lines:
            if any(ln.startswith(kw) for kw in ("However", "Note", "Turkish", "Here", "Please", "I ", "In ")):
                break
            clean.append(ln)
        translation = " ".join(clean) if clean else lines[0]

    return translation


def translate_document(document: str, use_rag: bool = True):
    """Belgeyi çevir"""
    tokenizer, model = load_mistral()

    if use_rag:
        client = chromadb.PersistentClient(path=CHROMADB_PATH)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        embedding_model = SentenceTransformer(MODEL_NAME)
        retrieved_terms = retrieve_relevant_terms(document[:200], embedding_model, collection)
        prompt = build_rag_prompt(document, retrieved_terms)
    else:
        prompt = build_baseline_prompt(document)

    return generate_translation(prompt, tokenizer, model)


if __name__ == "__main__":
    tokenizer, model = load_mistral()

    test_cases = [
        "Döner mil kontrol sistemi düzenli olarak kontrol edilmelidir.",
        "Bakım hatası tespit edildiğinde preventif bakım planı devreye alınmalıdır.",
        "Kalite güvence süreci teknik dokümantasyon ile desteklenmelidir.",
    ]

    import chromadb
    from sentence_transformers import SentenceTransformer
    from config import CHROMADB_PATH, CHROMA_COLLECTION_NAME, MODEL_NAME
    from retrieval import retrieve_relevant_terms

    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    emb_model = SentenceTransformer(MODEL_NAME)

    print("\n" + "=" * 60)
    for text in test_cases:
        retrieved = retrieve_relevant_terms(text[:200], emb_model, collection, k=5)
        rag_prompt = build_rag_prompt(text, retrieved)
        baseline_prompt = build_baseline_prompt(text)

        rag_out = generate_translation(rag_prompt, tokenizer, model)
        base_out = generate_translation(baseline_prompt, tokenizer, model)

        print(f"TR : {text}")
        print(f"RAG: {rag_out}")
        print(f"BAS: {base_out}")
        print("-" * 60)
