import os
from app.dt_ingestion import Doc_processor
from app.chunksplit import ChunkSplit
from app.embedd_ret import embedd_retrieve
from app.summary_gen import Summarizer
from transformers import logging
logging.set_verbosity_error()

def process_file(path: str, label: str):
    print(f"\nProcessing [{label}]: {path}")
    text = text = Doc_processor().load_document(path)

    chunks = ChunkSplit(512, 50).chunk_document(text)
    print(f"Total chunks: {len(chunks)}")

    engine = embedd_retrieve()
    emb = engine.gen_embedd(chunks)
    engine.create_faiss_index(emb)

    top_chunks = engine.semantic_retrieval("Summarize this document", topk=6)

    summary, stats = Summarizer().summarize(top_chunks)

    print(f"\nToken Usage:")
    print(f"Input Tokens: {stats['input_tokens']}")
    print(f"Output Tokens: {stats['output_tokens']}")
    print(f"Total Tokens: {stats['total_tokens']}")
    print(f"Latency: {stats['generation_latency_sec']} seconds")
    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/summary_{5}.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f" Saved summary to: {out}")
    print("Preview:", summary[:300], "...")

def main():
    test_files = {
        "sample_pdf": "data/sample2.pdf",  
        # "sample_txt": "data/sample2.txt",  
        # "sample_md":  "data/sample1.md"
    }

    for label, path in test_files.items():
        if os.path.exists(path):
            process_file(path, label)
        else:
            print(f"File not found: {path}")

if __name__ == "__main__":
    main()

