# 📦 Data Preprocessing for LLMs — Input-Output Pair Generation

This project demonstrates how to preprocess raw text into training-ready batches for a large language model. The notebook (`I-0-PAIR.ipynb`) walks through the complete pipeline: **tokenizing text → creating input/target pairs → wrapping them in a PyTorch Dataset → loading batches with a DataLoader**.

It uses the **GPT-2 BPE tokenizer** (via `tiktoken`) and a short story (`the-verdict.txt`) as the input text.

---

## 🎯 Purpose

LLMs learn by predicting the **next token**. To train one, you need pairs of sequences where the **target is the input shifted by one position**:

```
Input :  [token_0, token_1, token_2, token_3]
Target:  [token_1, token_2, token_3, token_4]
```

This notebook shows how to build those pairs from scratch and serve them as batches, exactly the way a real training loop would consume them.

---

## 🔄 Workflow

```
Raw Text  →  BPE Tokenization (tiktoken, GPT-2)
                    ↓
          List of Token IDs
                    ↓
     Sliding Window → Input/Target Pairs
                    ↓
        Custom PyTorch Dataset (DatasetV1)
                    ↓
           DataLoader → Batches
```

---

## 📂 Project Structure

```text
DATA-PREPROCESSING-LLM/
├── I-0-PAIR.ipynb        # Main notebook (input-output pair generation)
├── TOKENIZER-.ipynb      # Tokenizer exploration notebook
├── TOKENIZER.ipynb       # Tokenizer notebook
├── the-verdict.txt       # Source text ("The Verdict" by Edith Wharton)
├── verdit-dataset.zip    # Zipped dataset archive
└── README.md
```

---

## 📓 What the Notebook Does (Step by Step)

### 1. Install Dependencies
Installs `tiktoken`, `datasets`, and `torch`.

### 2. Initialize the Tokenizer
```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
```
Loads the GPT-2 BPE tokenizer from `tiktoken`.

### 3. Load & Preview the Text
Reads `the-verdict.txt` (20,479 characters) and prints the first ~1,000 characters.

### 4. Tokenize the Text
```python
encoded_text = tokenizer.encode(raw_text)  # → 5,145 token IDs
```

### 5. Demonstrate Input → Target Pairing
With a `context_size` of 4, shows how each input window maps to a target window shifted by one token — both as raw token IDs and decoded text:

```
 and           → established
 and established → himself
 and established himself → in
 and established himself in → a
```

### 6. Define `DatasetV1` (Custom PyTorch Dataset)
A class that:
- Tokenizes the text (with `<|endoftext|>` support).
- Uses a **sliding window** with configurable `context_length` and `stride` to generate all input/target pairs.
- Returns `(input_tensor, target_tensor)` for each sample.

### 7. Define `create_dataloader` Helper
Wraps `DatasetV1` and `torch.utils.data.DataLoader` into a single function with sensible defaults:
| Parameter        | Default |
|------------------|---------|
| `batch_size`     | 4       |
| `context_length` | 256     |
| `stride`         | 128     |
| `shuffle`        | True    |
| `drop_last`      | True    |
| `num_workers`    | 0       |

### 8. Create a DataLoader and Print Batches
Creates a DataLoader (`batch_size=4`, `context_length=4`, `stride=4`, `shuffle=False`) and prints the first **2 batches**:

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

Each row in the target is the corresponding input row **shifted by one token** — this is the next-token prediction format used by GPT-style models.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Jupyter Notebook

### Installation

```bash
git clone https://github.com/Rhythem2005/DATA-PREPROCESSING-LLM.git
cd DATA-PREPROCESSING-LLM
pip install tiktoken datasets torch
jupyter notebook I-0-PAIR.ipynb
```

Run all cells from top to bottom.

---

## 📚 References

- *Build a Large Language Model (From Scratch)* — Sebastian Raschka
- [tiktoken (OpenAI)](https://github.com/openai/tiktoken)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)