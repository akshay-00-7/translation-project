# 🌐 English → Hindi Neural Machine Translation
### A Deep Learning Project — From Scratch to Production

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-IITB%20English--Hindi-green.svg)](https://huggingface.co/datasets/cfilt/iitb-english-hindi)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow.svg)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

---

## 📌 What Is This Project?

This project builds an **English to Hindi translation system** using deep learning — completely from scratch.

Instead of just using a ready-made API, we built **4 different neural network architectures** and compared them side by side to understand how each model learns to translate language.

> Think of it like building 4 different cars and racing them to see which one drives the best.

---

## ❓ Why Was This Built?

Language translation is one of the hardest problems in AI. Hindi and English are structurally very different languages — word order, grammar, and sentence structure are all different.

The goal of this project was to:

- Understand how neural networks learn to translate text
- Compare classical RNN-based models vs modern Transformer architecture
- See how a from-scratch model compares against a production-grade pretrained model
- Build a working interactive demo where anyone can test all 4 models live

---

## 🧠 How Does It Work?

### The Big Picture

Every translation model in this project follows the **Encoder-Decoder** pattern:

```
English Sentence
      ↓
  [ ENCODER ]  →  Converts words into numbers (context vector)
      ↓
  [ DECODER ]  →  Converts numbers back into Hindi words
      ↓
Hindi Translation
```

The difference between the 4 models is **how** the encoder and decoder are built internally.

---

### 📊 Dataset — IITB English-Hindi Corpus

| Property | Value |
|---|---|
| Source | IIT Bombay (cfilt/iitb-english-hindi) |
| Training Samples | 1.6 Million sentence pairs |
| Used in Training | 30,000 sentence pairs |
| Validation Set | ~520 pairs |
| Test Set | ~2,500 pairs |

**Why only 30,000?** Training on all 1.6M sentences would take days on a free GPU. 30K gives us enough data to see real learning patterns within Colab's time limits.

---

### 🏗️ The 4 Models Built

#### 1. 🔁 RNN (Recurrent Neural Network)
The oldest and simplest architecture. Reads words one by one, left to right.

**Problem:** It struggles with long sentences because it tries to "remember" everything in a single hidden state — like trying to remember a long story using only 1 sticky note.

```
"I am eating food" → [I] → [am] → [eating] → [food] → hidden state → Hindi
```

---

#### 2. 🧱 LSTM (Long Short-Term Memory)
An improved version of RNN. Has a special **memory cell** that decides what to remember and what to forget.

**Advantage over RNN:** Much better at handling long sentences. The memory cell acts like a notepad — it can choose to keep important information for longer.

```
LSTM has:
- Forget Gate  → "Should I forget this?"
- Input Gate   → "Should I remember this?"
- Output Gate  → "What should I output now?"
```

---

#### 3. ⚡ GRU (Gated Recurrent Unit)
A simpler version of LSTM. Fewer gates, fewer parameters — but often performs just as well.

**Advantage:** Trains faster than LSTM. Good balance between speed and accuracy.

---

#### 4. 🤖 Transformer (Helsinki-NLP Pretrained)
The state-of-the-art architecture. Instead of reading words one by one, it looks at **all words at the same time** using a mechanism called **Attention**.

```
"I am eating food"
       ↓
Attention: "food" is most related to "eating", "I" is the subject...
       ↓
Generates Hindi with full context awareness
```

**Helsinki-NLP/opus-mt-en-hi** — a production-grade model trained on millions of sentence pairs, used here as the gold standard comparison.

---

### 🔤 Vocabulary System

Before any model can learn, words must be converted to numbers. We built a custom `Vocabulary` class:

```python
word2idx = {
    '<pad>': 0,   # Padding — makes all sentences same length
    '<sos>': 1,   # Start of sentence signal
    '<eos>': 2,   # End of sentence signal
    '<unk>': 3,   # Unknown word
    'hello': 4,
    'world': 5,
    ...
}
```

**English vocabulary** and **Hindi vocabulary** are built separately from the training data.

---

### 📐 Model Architecture Details

| Component | Value |
|---|---|
| Embedding Dimension | 256 |
| Hidden Dimension | 512 |
| Batch Size | 64 |
| Optimizer | Adam (lr=0.001) |
| Loss Function | CrossEntropyLoss |
| Gradient Clipping | 1.0 |
| RNN/LSTM/GRU Epochs | 10 |
| Transformer Epochs | 20 |
| GPU | NVIDIA T4 (Google Colab) |

---

### 📈 Evaluation — BLEU Score

Models are evaluated using **BLEU Score** (Bilingual Evaluation Understudy).

> BLEU measures how similar the machine-generated translation is to a human reference translation. Score ranges from 0 to 100 — higher is better.

```
BLEU = 0   → completely wrong translation
BLEU = 100 → perfect match with human translation
```

---

## 🚀 How To Run This Project

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `translation_project (4).ipynb`
3. Go to **Runtime → Change Runtime Type → GPU (T4)**
4. Run cells from top to bottom

### Option 2 — Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/akshay-00-7/translation-project.git
cd translation-project

# 2. Install dependencies
pip install torch datasets transformers gradio sacrebleu sentencepiece pickle5

# 3. Open the notebook
jupyter notebook "translation_project (4).ipynb"
```

---

## 📦 Dependencies

```
torch          — Deep learning framework
datasets       — HuggingFace dataset loader
transformers   — Pretrained model (Helsinki-NLP)
gradio         — Interactive web demo UI
sacrebleu      — BLEU score evaluation
sentencepiece  — Tokenization for pretrained model
pickle         — Saving/loading dataset from Google Drive
```

---

## 🖥️ Interactive Demo

At the end of the notebook, a **Gradio web interface** launches automatically.

You type any English sentence → all 4 models translate it simultaneously → you can compare results live.

```
Input:  "I am eating food"

RNN Output:        → मैं खाना खा रहा हूँ
LSTM Output:       → मैं भोजन खा रहा हूँ
GRU Output:        → मैं खाना खा रहा हूँ
Pretrained Output: → मैं खाना खा रहा हूँ
```

---

## 📁 Project Structure

```
translation-project/
│
├── translation_project (4).ipynb   ← Main notebook (all code here)
└── README.md                       ← You are here
```

**Google Drive (auto-created during training):**
```
/MyDrive/translation_project/
├── dataset.pkl        ← Saved IITB dataset
├── best_rnn.pt        ← Best RNN model weights
├── best_lstm.pt       ← Best LSTM model weights
├── best_GRU.pt        ← Best GRU model weights
└── best_transformer.pt← Best Transformer model weights
```

---

## 🔬 Key Concepts Learned

| Concept | What It Means |
|---|---|
| Seq2Seq | Sequence to Sequence — input sentence → output sentence |
| Encoder-Decoder | Split the translation task into two parts |
| Hidden State | The model's "memory" of what it has read |
| Attention | Ability to focus on relevant words during translation |
| BLEU Score | Standard metric for evaluating translation quality |
| Teacher Forcing | During training, feed correct previous word (not model's guess) |
| Gradient Clipping | Prevents exploding gradients during training |
| Padding | Makes all sentences same length for batch processing |

---

## ⚠️ Limitations

- Trained on only 30K out of 1.6M available sentence pairs (due to Colab time limits)
- Custom RNN/LSTM/GRU models are basic — no attention mechanism
- Model weights are not included in repo (saved to Google Drive during training)
- Long sentences (10+ words) may produce lower quality translations for scratch models

---

## 🙋 Author

**Akshay** — [@akshay-00-7](https://github.com/akshay-00-7)

Built as a learning project to understand Neural Machine Translation from the ground up.

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

> *"The best way to learn deep learning is to build things from scratch — even if a pretrained model does it better."*
