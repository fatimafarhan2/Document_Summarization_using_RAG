# Document Summarization System with RAG

A **Retrieval-Augmented Generation (RAG)** system that processes long documents and generates concise, coherent summaries using semantic chunking, vector embeddings, and transformer-based summarization.

---

## Features

* **Multi-format Support**: Ingests PDF, TXT, and Markdown files
* **Semantic Chunking**: Smart text splitting with overlapping windows
* **Vector Embeddings**: Uses `SentenceTransformers` with FAISS indexing
* **Contextual Retrieval**: Retrieves relevant chunks using semantic search
* **LLM Summarization**: High-quality summaries using Pegasus model
* **Performance Metrics**: Tracks token usage and latency

---

## System Architecture

```
Document → Ingestion → Chunking → Embeddings → FAISS Index
                                            ↓
Summary ← LLM Generation ← Top-K Retrieval ← Semantic Search
```

---

## Installation

### Prerequisites

* Python 3.8+
* CUDA (optional for GPU acceleration)

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install sentence-transformers
pip install faiss-cpu           # or faiss-gpu for CUDA
pip install transformers
pip install PyPDF2
pip install markdown
pip install numpy
```

Or:

```bash
pip install -r req.txt
```

---

## Directory Structure

```
project/
├── app/
│   ├── dt_ingestion.py      # Document processing
│   ├── chunksplit.py        # Text chunking
│   ├── embedd_ret.py        # Embedding & retrieval
│   └── summary_gen.py       # Summary generation
├── data/                    # Input documents
├── outputs/                 # Generated summaries
├── main.py                  # Entry point
└── req.txt                  # Requirements file
```

---

## Usage

### Step 1: Prepare Documents

Place files inside the `data/` folder:

```
data/
├── my_document.pdf
├── research_paper.txt
└── notes.md
```

### Step 2: Configure `main.py`

```python
test_files = {
    "document_1": "data/my_document.pdf",
    "document_2": "data/research_paper.txt",
    "document_3": "data/notes.md"
}
```

### Step 3: Set Output File

Update file naming in `process_file()` to prevent overwrite:

```python
out = f"outputs/summary_{7}.txt"
```

### Step 4: Run the System

```bash
python main.py
```

---

## Advanced Usage

### Process Multiple Documents

```python
test_files = {
    "research_paper": "data/paper1.pdf",
    "technical_doc": "data/manual.txt",
    "meeting_notes": "data/notes.md"
}
```

### Custom Processing

```python
# Short docs
chunks = ChunkSplit(256, 25).chunk_document(text)

# Technical focus
top_chunks = engine.semantic_retrieval("Summarize technical details", topk=8)

# Longer summaries
summary, stats = Summarizer().summarize(top_chunks)
```

---

## Expected Output

### Console Log Example

```
Processing [document_1]: data/my_document.pdf
Total sentences found: 45
Total chunks created: 8
Loading SentenceTransformer model...
Generating embeddings...
Creating FAISS index...
Performing FAISS search...
Generating summary...

Token Usage:
Input Tokens: 1024
Output Tokens: 187
Latency: 3.45 seconds

Summary saved to: outputs/summary_7.txt
```

### Output Files

* `outputs/summary_7.txt`: Final summary
* \~120–250 words (configurable)

---

## Configuration Options

### Chunking Parameters

```python
ChunkSplit(chunk_size=512, chunk_overlap=50)
```

### Retrieval Parameters

```python
engine.semantic_retrieval("Summarize this document", topk=6)
```

### Summarization Parameters (`summary_gen.py`)

```python
max_length=250
min_length=120
num_beams=6
```

---

## Supported Formats

* **PDF**: `data/sample.pdf`
* **Text**: `data/sample.txt`
* **Markdown**: `data/sample.md`

---

## System Components

### 1. Document Ingestion (`dt_ingestion.py`)

* PDF: `PyPDF2`
* Text: UTF-8 Read
* Markdown: Plain text conversion

### 2. Chunking (`chunksplit.py`)

* Sentence regex splitting
* Word overlap control
* Handles variable sizes gracefully

### 3. Embedding & Retrieval (`embedd_ret.py`)

* Model: `all-MiniLM-L6-v2`
* FAISS index (inner product + L2 normalization)

### 4. Summarization (`summary_gen.py`)

* Model: `google/pegasus-cnn_dailymail`
* Beam search & repetition penalty
* Post-cleaning of output

---

## Sample Output Format

```
(r_env) G:\rag_doc>python main.py

Processing [sample_txt]: data/sample2.txt
Total chunks created: 1
Loading SentenceTransformer model...
Creating FAISS index...
Performing FAISS search...
Generating summary...

Token Usage:
Input Tokens: 268
Output Tokens: 130
Latency: 54.24 seconds
Saved summary to: outputs/summary_4.txt
```

---

## Performance Optimization

### GPU Acceleration

* Automatically detects CUDA if available
* Faster summarization

### Memory Management

* Efficient FAISS indexing
* Batching in embedding generation

### Caching

* HuggingFace cache path: `G:/hf_cache`

---

## Troubleshooting

### CUDA Out of Memory

```python
self.device = torch.device("cpu")
```

### File Not Found

* Check `main.py` paths
* Ensure documents are present

### Slow Processing

* Lower `topk` or `max_length`
* Avoid very large documents

### Debug Mode

Uncomment debug lines in `chunksplit.py`:

```python
print(f"Sentence {i + 1}: {words} words")
```

---

## Customization

### Swap Embedding Model

```python
self.modelname = "sentence-transformers/all-mpnet-base-v2"
```

### Change Summarization Model

```python
self.model_name = "facebook/bart-large-cnn"
```

### Adjust Query

```python
top_chunks = engine.semantic_retrieval("Focus on key findings", topk=8)
```

---

## System Requirements

* **RAM**: 8GB (16GB+ recommended)
* **Storage**: 2GB (for model weights)
* **GPU**: 4GB+ VRAM for CUDA

---

## License

This project uses open-source models and libraries. Please verify the license terms for commercial usage of third-party models.

---

## Support

If you're facing issues:

1. Review the Troubleshooting section
2. Ensure all dependencies are installed
3. Verify file paths and formats

---
