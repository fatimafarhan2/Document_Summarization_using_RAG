import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

from typing import List, Dict, Any, Tuple


class embedd_retrieve:
    def __init__(self):
        self.modelname = "all-MiniLM-L6-v2"
        print(f"Loading SentenceTransformer model: {self.modelname}")
        self.model = SentenceTransformer(self.modelname)
        self.ind = None
        self.chunks = []

    def gen_embedd(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        # using lst of chunk first we extract the text normalize embedding and then stoe and return them
        # the normalization is necessaryc for cosine simmilarity
        print("Generating embeddings for input chunks...")
        self.chunks = chunks
        data = [ch['text'] for ch in chunks]
        embedd = self.model.encode(data, convert_to_numpy=True, show_progress_bar=True).astype("float32")
        faiss.normalize_L2(embedd)
        self.embedd = embedd
        print(f"Generated {len(embedd)} embeddings.")
        return embedd

    def create_faiss_index(self, embedd: np.ndarray):
        # firs we get the dimension of embedding then createa flat indexflati
        # this will allow us to search vectors which are morst simmilat to the query
        print("Creating FAISS index...")
        shapes = embedd.shape[1]
        self.ind = faiss.IndexFlatIP(shapes)
        self.ind.add(embedd)
        print("FAISS index created and embeddings added.")

    def semantic_retrieval(self, ser_qry: str, topk: int = 6) -> List[Tuple[Dict[str, Any], float]]:
        # we firs convert the ser_qry into normalized embedding
        if self.ind is None:
            raise RuntimeError("Faiss indez was not created")
        print(f"Encoding search query: '{ser_qry}'")
        emb_q = self.model.encode([ser_qry], convert_to_numpy=True, show_progress_bar=True).astype("float32")
        faiss.normalize_L2(emb_q)

        # here we search in faiss for mist simmilar chunks to emb_q or ser_Qry
        # then we return the chunk and score
        print(f"Performing FAISS search for top {topk} chunks...")
        sim_Score, res = self.ind.search(emb_q, topk)
        matched_chunk = []
        for i, j in enumerate(res[0]):
            if j != -1:
                matched_chunk.append((self.chunks[j], float(sim_Score[0][i])))
                print(f"Retrieved chunk {j} with score {sim_Score[0][i]}")
        print(f"Retrieved {len(matched_chunk)} chunks.")
        return matched_chunk
