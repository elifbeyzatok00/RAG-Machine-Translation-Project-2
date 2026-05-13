# Retrieval-Augmented Machine Translation (RAG MT)

## Türkçe Teknik Dokümanların Terminoloji Kontrollü Çevirisi

---

## 🎯 PROJE ÖZET

**RAG-based makine çevirisi sistemi** - teknik Türkçe dokümanların İngilizce'ye çevirisinde **terminoloji tutarlılığını** artırmak amacıyla geliştirilmiş.

**Teknoloji Stack**:

- **LLM**: Mistral-7B-Instruct-v0.2 (4GB quantized)
- **Embeddings**: Sentence Transformers
- **Vector DB**: ChromaDB
- **Framework**: Python 3.8+

**Hedef Çıktı**: POC + Evaluation Metrics (BLEU, Terminology Consistency)

---

## 📂 DOSYA YAPISI

```
project/
├── README.md                    (bu dosya)
├── requirements.txt             (Python dependencies)
│
├── data/
│   ├── documents/              (Turkish sample documents)
│   ├── terminology/            (Turkish-English term pairs)
│   ├── splits/                 (train/val/test indices)
│   └── embeddings/             (ChromaDB vectors)
│
└── src/
    ├── config.py               (configuration)
    ├── embedding.py            (Vector embedding + ChromaDB)
    ├── retrieval.py            (Multi-source retrieval)
    ├── translation.py          (RAG + Baseline translation)
    └── evaluation.py           (BLEU + Terminology metrics)
```

---

## 🚀 QUICK START

### 1️⃣ Kurulum

```bash
pip install -r requirements.txt
python -m spacy download tr_core_news_sm
python -m spacy download en_core_web_sm
```

### 2️⃣ ChromaDB Embedding & Retrieval

```bash
python src/embedding.py
python src/retrieval.py
```

### 3️⃣ Mistral ile Çeviri

```bash
python src/translation.py
```

### 4️⃣ Evaluation

```bash
python src/evaluation.py
```

---

## 📊 BEKLENEN SONUÇLAR

| Metrik         | Baseline | RAG  | Improvement |
| -------------- | -------- | ---- | ----------- |
| BLEU Score     | 28.5     | 32.1 | +12.6%      |
| Term Precision | 82%      | 91%  | +11%        |
| Term Recall    | 76%      | 88%  | +15.8%      |

---

## 🔧 REQUIREMENTS

- Python 3.8+
- ~4GB GPU/CPU (Mistral-7B Quantized)
- 1GB disk space
- ~10 dakika ilk çalıştırma

---

## 🎓 MACHINE TRANSLATION DERSİ BAĞLANTI

```
Neural MT              → Mistral-based çeviri
Domain Adaptation      → Terminology DB + RAG
Evaluation Metrics     → BLEU + Custom metrics
Context-Aware MT       → Retrieval-augmented translation
Terminology Mgmt       → Constraint handling
```

---

## 📞 TROUBLESHOOTING

### Mistral Model Download Hatası

- Model ilk çalıştırmada indirilir (~4GB quantized)
- İnternet bağlantısını kontrol et

### ChromaDB Hatası

```bash
pip install --upgrade chromadb
```

### Memory Hatası

- Batch size'ı azalt (`config.py` → `BATCH_SIZE = 4`)

---

## ✨ HIGHLIGHTS

✅ Lightweight architecture (4GB Mistral)  
✅ Local ChromaDB (no external APIs required)  
✅ Terminology-aware translation  
✅ Measurable improvements (BLEU + custom metrics)
