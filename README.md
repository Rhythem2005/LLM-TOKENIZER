# Data Preprocessing for LLMs

End-to-end data preprocessing pipeline for training a GPT-style language model — from raw text to training-ready batches. Covers tokenization (both from scratch and using BPE) and input-output pair generation with PyTorch.

Based on *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

---

## Project Structure

```
DATA-PREPROCESSING-LLM/
├── TOKENIZER-.ipynb      # Word-level tokenizer built from scratch
├── TOKENIZER.ipynb       # BPE tokenizer using tiktoken (GPT-2)
├── I-0-PAIR.ipynb        # Input-output pair generation + DataLoader
├── the-verdict.txt       # Source text (Edith Wharton's "The Verdict")
├── verdit-dataset.zip    # Zipped dataset archive
└── README.md
```

---

## Notebooks

### 1. `TOKENIZER-.ipynb` — Word-Level Tokenizer from Scratch

Builds a complete tokenizer without any external tokenizer library.

**Steps:**
1. Load the **WikiText-103** dataset from Hugging Face (~540M characters).
2. Split text into tokens using **regex** (handles punctuation, special characters, whitespace).
3. Build a **vocabulary** — sorted unique tokens mapped to integer IDs (608,557 unique tokens).
4. Implement `SimpleTokenizer` class with `encoder()` and `decoder()` methods.
5. Add special tokens (`<|endoftext|>`, `<|unk|>`) to handle document boundaries and unknown words.
6. Implement `SimpleTokenizerV2` — improved version that maps unknown words to `<|unk|>` instead of crashing, and cleans up punctuation spacing on decode.
7. Demonstrate encoding and decoding with multi-document text joined by `<|endoftext|>`.

---

### 2. `TOKENIZER.ipynb` — BPE Tokenizer (tiktoken)

Uses OpenAI's `tiktoken` library with the **GPT-2 BPE encoding** to tokenize text at the subword level.

**Steps:**
1. Load the GPT-2 tokenizer via `tiktoken.get_encoding("gpt2")`.
2. Encode a sample paragraph (including punctuation, numbers, newlines, and `<|endoftext|>`) into token IDs.
3. Decode token IDs back to the original text.
4. Inspect each token individually — prints every `(token_id, decoded_string)` pair to show how BPE splits words like `"uppercase"` → `["u", "pperc", "ase"]`.

---

### 3. `I-0-PAIR.ipynb` — Input-Output Pair Generation & DataLoader

Converts tokenized text into **next-token prediction pairs** and serves them as batches via PyTorch — the format required for training a GPT-style model.

**Workflow:**

```
Raw Text (the-verdict.txt)
        ↓
BPE Tokenization (tiktoken, GPT-2) → 5,145 token IDs
        ↓
Sliding Window → Input/Target Pairs
        ↓
PyTorch Dataset (DatasetV1)
        ↓
DataLoader → Batches
```

**Steps:**
1. Read `the-verdict.txt` and tokenize it with `tiktoken` (5,145 tokens).
2. Demonstrate the **next-token prediction** concept:
   ```
    and           →  established
    and established →  himself
    and established himself →  in
    and established himself in →  a
   ```
3. Define `DatasetV1` — a custom `torch.utils.data.Dataset` that uses a **sliding window** (configurable `context_length` and `stride`) to generate all `(input, target)` tensor pairs.
4. Define `create_dataloader()` — wraps `DatasetV1` + `DataLoader` with defaults: `batch_size=4`, `context_length=256`, `stride=128`, `shuffle=True`, `drop_last=True`.
5. Create a DataLoader and iterate through the first 2 batches:
   ```
   Batch 1
   Input : tensor([[   40,   367,  2885,  1464],
           [ 1807,  3619,   402,   271],
           [10899,  2138,   257,  7026],
           [15632,   438,  2016,   257]])
   Target: tensor([[  367,  2885,  1464,  1807],
           [ 3619,   402,   271, 10899],
           [ 2138,   257,  7026, 15632],
           [  438,  2016,   257,   922]])
   ```
   Each target row is the corresponding input row shifted by one token.

---

## Getting Started

**Requirements:** Python 3.10+, Jupyter Notebook

```bash
git clone https://github.com/Rhythem2005/LLM-TOKENIZER.git
cd LLM-TOKENIZER
pip install tiktoken datasets torch
jupyter notebook
```

Run cells top to bottom in each notebook.

---

## References

- *Build a Large Language Model (From Scratch)* — Sebastian Raschka
- [tiktoken](https://github.com/openai/tiktoken)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [PyTorch Data Utilities](https://pytorch.org/docs/stable/data.html)