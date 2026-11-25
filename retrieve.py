import json

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, INDEX_DIR


class Retriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.embeddings = np.load(INDEX_DIR / "embeddings.npy")
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))

    def search(self, query: str, k: int = 3) -> list[dict]:
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


if __name__ == "__main__":
    retriever = Retriever()
    for query in ["How long does a project take?", "When do I pay?", "Do you sell cars?"]:
        print(f"\nQ: {query}")
        for hit in retriever.search(query, k=2):
            print(f"  {hit['score']:.2f}  [{hit['source']}] {hit['text'][:70]}...")

