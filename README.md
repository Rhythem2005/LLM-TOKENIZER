# 🧩 Simple Tokenizer from Scratch

A word-level tokenizer built from scratch in Python — no `tiktoken`, no `transformers`, just regex and dictionaries. Built to actually understand how tokenization works before using pre-built tokenizers.

Uses the **WikiText-103** dataset from Hugging Face to build the vocabulary, and supports `<|unk|>` (unknown words) and `<|endoftext|>` (document separator) tokens.

---

## ✨ Features

- Load text from WikiText-103 (Hugging Face `datasets`)
- Tokenize text using regex
- Build vocabulary (word → integer ID)
- Encode text → token IDs
- Decode token IDs → text
- Handle unknown words with `<|unk|>`
- Handle multiple documents with `<|endoftext|>`

---

## 📂 Project Structure

```text
main.ipynb
```

Walkthrough in the notebook:
1. Load dataset
2. Tokenize with regex
3. Build vocabulary
4. Basic tokenizer (encode/decode)
5. Add special tokens
6. Examples

---

## 🚀 Getting Started

```bash
git clone https://github.com/Rhythem2005/simple-tokenizer.git
cd simple-tokenizer
pip install datasets
jupyter notebook
```

Run the notebook cells in order.

---

## 💻 Usage

```python
tokenizer = SimpleTokenizer(vocab)

ids = tokenizer.encode("The quick brown fox jumps.")
text = tokenizer.decode(ids)
```

---

## ⚠️ Limitations

- Word-level only (no subword/BPE)
- Regex-based splitting
- Vocabulary grows with corpus size
- Unseen words become `<|unk|>`

---

## 📚 References

- *Build a Large Language Model (From Scratch)* — Sebastian Raschka
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)