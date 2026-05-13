import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from embedding import create_embeddings
from retrieval import test_retrieval, retrieve_relevant_terms
from evaluation import evaluate_translations


def main():
    print("🚀 RAG MT Pipeline Starting...\n")

    # Step 1: Create embeddings
    print("=" * 50)
    print("STEP 1: Creating Embeddings")
    print("=" * 50)
    collection, embedding_model = create_embeddings()

    # Step 2: Test retrieval
    print("\n" + "=" * 50)
    print("STEP 2: Testing Retrieval")
    print("=" * 50)
    test_retrieval()

    # Step 3: Quick evaluation with sample data (no Mistral needed for demo)
    print("\n" + "=" * 50)
    print("STEP 3: Evaluation with Sample Data")
    print("=" * 50)

    refs = [
        "The rotating shaft control system should be regularly inspected.",
        "Maintenance errors must be detected using fault detection methods.",
        "Quality assurance processes must be followed during production."
    ]
    rag_hyps = [
        "The rotating shaft control system should be regularly inspected.",
        "Maintenance errors should be detected using fault detection methods.",
        "Quality assurance processes must be maintained during production."
    ]
    baseline_hyps = [
        "The spinning system inspection system should be regularly checked.",
        "Service errors should be found using breakdown detection techniques.",
        "Quality processes should be maintained during production activities."
    ]
    terms = [
        {"tr": "döner mil", "en": "rotating shaft"},
        {"tr": "bakım hatası", "en": "maintenance error"},
        {"tr": "sistem kontrolü", "en": "system control"},
        {"tr": "kalite güvence", "en": "quality assurance"},
        {"tr": "arıza tespit", "en": "fault detection"},
    ]

    results = evaluate_translations(rag_hyps, baseline_hyps, refs, terms)

    print("\n" + "=" * 50)
    print("STEP 4: Retrieval Demo")
    print("=" * 50)
    test_sentence = "Döner mil kontrol sistemi düzenli olarak kontrol edilmelidir."
    retrieved = retrieve_relevant_terms(test_sentence, embedding_model, collection, k=3)
    print(f"\nQuery: {test_sentence}")
    print("Retrieved terms:")
    for r in retrieved:
        print(f"  ✓ {r['term']} → {r['en']} (dist: {r['distance']:.3f})")

    print("\n✨ Pipeline complete!")
    print(f"\n📊 Summary:")
    print(f"   RAG BLEU:      {results['rag_bleu']:.2f}")
    print(f"   Baseline BLEU: {results['baseline_bleu']:.2f}")
    print(f"   Improvement:   {results['bleu_improvement']:+.2f}")


if __name__ == "__main__":
    main()
