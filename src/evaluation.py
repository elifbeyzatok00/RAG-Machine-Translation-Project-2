import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from sacrebleu.metrics import BLEU
from config import OUTPUT_DIR


def calculate_bleu(references: list, hypotheses: list) -> float:
    """BLEU score hesapla"""
    bleu = BLEU(smooth_method="floor")
    score = bleu.corpus_score(hypotheses, [references])
    return score.score


def calculate_terminology_precision(translation: str, terminology: list) -> float:
    """Çeviride kaç tane doğru terim kullanıldığını ölç"""
    correct = sum(
        1 for term in terminology
        if term['en'].lower() in translation.lower()
    )
    return (correct / len(terminology) * 100) if terminology else 0.0


def evaluate_translations(
    rag_translations: list,
    baseline_translations: list,
    references: list,
    terminology: list
) -> dict:
    """Tüm metrikleri hesapla ve kaydet"""

    print("📊 Evaluating translations...")

    rag_bleu = calculate_bleu(references, rag_translations)
    baseline_bleu = calculate_bleu(references, baseline_translations)

    combined_rag = " ".join(rag_translations)
    combined_baseline = " ".join(baseline_translations)

    rag_term_precision = calculate_terminology_precision(combined_rag, terminology)
    baseline_term_precision = calculate_terminology_precision(combined_baseline, terminology)

    results = {
        "rag_bleu": round(rag_bleu, 2),
        "baseline_bleu": round(baseline_bleu, 2),
        "bleu_improvement": round(rag_bleu - baseline_bleu, 2),
        "rag_term_precision": round(rag_term_precision, 1),
        "baseline_term_precision": round(baseline_term_precision, 1),
        "term_improvement": round(rag_term_precision - baseline_term_precision, 1)
    }

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"BLEU Score (RAG):      {results['rag_bleu']:.2f}")
    print(f"BLEU Score (Baseline): {results['baseline_bleu']:.2f}")
    print(f"Improvement:           {results['bleu_improvement']:+.2f}")
    print(f"\nTerminology Precision (RAG):      {results['rag_term_precision']:.1f}%")
    print(f"Terminology Precision (Baseline): {results['baseline_term_precision']:.1f}%")
    print(f"Improvement:                      {results['term_improvement']:+.1f}%")
    print(f"{sep}\n")

    output_file = OUTPUT_DIR / "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to {output_file}")
    return results


if __name__ == "__main__":
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

    evaluate_translations(rag_hyps, baseline_hyps, refs, terms)
