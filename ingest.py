import json

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import DOCS_DIR, EMBEDDING_MODEL, INDEX_DIR


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > max_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}".strip()
    if current:
        chunks.append(current)
    return chunks


def main() -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)
    records = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            records.append({"source": path.name, "chunk_id": i, "text": chunk})

    print(f"Found {len(records)} chunks")
    if not records:
        print("No documents found. Check that data/docs contains .md files.")
        return

    embeddings = model.encode([r["text"] for r in records], normalize_embeddings=True)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Indexed {len(records)} chunks from {DOCS_DIR}")


if __name__ == "__main__":
    main()