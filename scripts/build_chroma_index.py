import json
import os
import shutil
from pathlib import Path
import chromadb
from openai import OpenAI
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
OUTPUT_DIR = ROOT / "outputs"
CORPUS_PATH = OUTPUT_DIR / "retrieval_corpus.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hw7_retrieval_corpus")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
CHROMA_RESET = os.getenv("CHROMA_RESET", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}


def load_records() -> list[dict]:
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def build_chunks(records: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for record in records:
        for index, text_chunk in enumerate(
            chunk_text(record["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        ):
            chunks.append(
                {
                    "id": f"{record['record_id']}::chunk{index:03d}",
                    "document": text_chunk,
                    "metadata": {
                        "source": record["source"],
                        "disease_name": record["disease_name"],
                        "title": record["title"],
                        **{
                            k: v
                            for k, v in (record.get("metadata") or {}).items()
                            if isinstance(v, (str, int, float, bool))
                            or v is None
                        },
                    },
                }
            )
    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    records = load_records()
    chunks = build_chunks(records)

    if CHROMA_RESET and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing_collections = {
        collection.name for collection in chroma_client.list_collections()
    }
    if COLLECTION_NAME in existing_collections and not CHROMA_RESET:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(
            json.dumps(
                {
                    "collection_name": COLLECTION_NAME,
                    "embedding_model": EMBEDDING_MODEL,
                    "num_records": len(records),
                    "num_chunks": len(chunks),
                    "stored_chunks": collection.count(),
                    "chroma_dir": str(CHROMA_DIR),
                    "reused_existing_index": True,
                    "reset_requested": CHROMA_RESET,
                },
                indent=2,
            )
        )
        return

    client = OpenAI()
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        embeddings = embed_texts(client, [item["document"] for item in batch])
        collection.add(
            ids=[item["id"] for item in batch],
            documents=[item["document"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
            embeddings=embeddings,
        )
        print(f"Indexed {batch_start + len(batch)}/{len(chunks)} chunks")

    print(
        json.dumps(
            {
                "collection_name": COLLECTION_NAME,
                "embedding_model": EMBEDDING_MODEL,
                "num_records": len(records),
                "num_chunks": len(chunks),
                "chroma_dir": str(CHROMA_DIR),
                "reused_existing_index": False,
                "reset_requested": CHROMA_RESET,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
