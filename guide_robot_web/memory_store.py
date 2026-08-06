import json
import os
import re
from pathlib import Path
import chromadb
import psycopg
from llama_cpp import Llama
from psycopg.rows import dict_row
from latency_tracker import TimerContext



ROOT_DIR = Path.cwd()
UNI_INFO_PATH = ROOT_DIR / "txt_files" / "uni_info"
CHROMA_PATH = ROOT_DIR / "guide_robot_web" / ".chroma_memory"
EMBEDDING_MODEL_PATH = "D:/models/nomic-embed-text-v1.5.Q6_K.gguf"

DB_PARAMS = {
    "dbname": "memory_agent",
    "user": "azaki",
    "password": "mackenziefoy",
    "host": "localhost",
    "port": "5432",
}

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
text_embed = None

def connect_db():
    return psycopg.connect(**DB_PARAMS)

def ensure_tables():
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mode TEXT,
                original_prompt TEXT,
                prompt TEXT,
                response TEXT
            )
            """
        )
        cursor.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS mode TEXT")
        cursor.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS original_prompt TEXT")
        conn.commit()
    conn.close()

def get_embedder():
    global text_embed

    if text_embed is None:
        text_embed = Llama(
            model_path=EMBEDDING_MODEL_PATH,
            n_ctx=2048,
            verbose=False,
            pooling_type=1,
            embedding=True,
            n_gpu_layers=32
        )

    return text_embed

def embed_text(text):
    with TimerContext("embed_text"):
        response = get_embedder().create_embedding(text)
        return response["data"][0]["embedding"]

def embed_texts(texts):
    with TimerContext("embed_texts", metadata={"count": len(texts)}):
        return [embed_text(text) for text in texts]

def _collection(name):
    return client.get_or_create_collection(name=name)

def reset_collection(name):
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    return client.create_collection(name=name)

def flatten_json(value, prefix=""):
    chunks = []

    if isinstance(value, dict):
        for key, child in value.items():
            label = f"{prefix} {key}".strip().replace("_", " ").title()
            chunks.extend(flatten_json(child, label))
    elif isinstance(value, list):
        for index, child in enumerate(value, start=1):
            label = f"{prefix} {index}".strip()
            chunks.extend(flatten_json(child, label))
    else:
        text = str(value).strip()
        if text:
            chunks.append(f"{prefix}: {text}" if prefix else text)

    return chunks

def load_uni_chunks():
    raw = UNI_INFO_PATH.read_text(encoding="utf-8")

    try:
        data = json.loads(raw)
        chunks = flatten_json(data)
    except json.JSONDecodeError:
        sections = re.split(r"\n\s*\n+", raw)
        chunks = [section.strip() for section in sections if section.strip()]

    return [chunk for chunk in chunks if len(chunk) > 20]

def populate_uni_info():
    with TimerContext("populate_uni_info"):
        collection = reset_collection("uni_info")
        chunks = load_uni_chunks()

        if not chunks:
            return {"chunks": 0}

        with TimerContext("embed_uni_chunks"):
            embeddings = embed_texts(chunks)

        collection.add(
            ids=[f"uni-{index}" for index in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source": "uni_info"} for _ in chunks],
        )
        return {"chunks": len(chunks)}

def fetch_conversations():
    ensure_tables()
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT * FROM conversations ORDER BY id ASC")
        rows = cursor.fetchall()
    conn.close()
    return rows

def store_conversation(mode, original_prompt, english_prompt, english_response):
    ensure_tables()
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            INSERT INTO conversations (mode, original_prompt, prompt, response)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (mode, original_prompt, english_prompt, english_response),
        )
        row = cursor.fetchone()
        conn.commit()
    conn.close()

    add_conversation_embedding(row["id"], english_prompt, english_response)
    return row["id"]

def add_conversation_embedding(conversation_id, english_prompt, english_response):
    with TimerContext("add_conversation_embedding"):
        collection = _collection("conversations")
        document = f"prompt: {english_prompt}\nresponse: {english_response}"
        collection.upsert(
            ids=[str(conversation_id)],
            embeddings=[embed_text(document)],
            documents=[document],
            metadatas=[{"source": "conversation"}],
        )

def rebuild_conversation_embeddings():
    with TimerContext("rebuild_conversation_embeddings"):
        collection = reset_collection("conversations")
        rows = fetch_conversations()
        if not rows:
            return {"conversations": 0}

        documents = [f"prompt: {row['prompt']}\nresponse: {row['response']}" for row in rows]

        with TimerContext("embed_conversations"):
            embeddings = embed_texts(documents)

        collection.add(
            ids=[str(row["id"]) for row in rows],
            embeddings=embeddings,
            documents=documents,
            metadatas=[{"source": "conversation"} for _ in rows],
        )
        return {"conversations": len(rows)}

def retrieve_context(english_prompt, uni_results=10, memory_results=5):
    with TimerContext("retrieve_context_total"):
        result = {"uni_context": [], "memory_context": []}

        for name, key, count in [
            ("uni_info", "uni_context", uni_results),
            ("conversations", "memory_context", memory_results),
        ]:
            try:
                with TimerContext(f"chroma_query_{name}"):
                    collection = _collection(name)
                    query = collection.query(query_embeddings=[embed_text(english_prompt)], n_results=count)
                    result[key] = query.get("documents", [[]])[0]
            except Exception:
                result[key] = []

        return result

def remove_last_conversation():
    ensure_tables()
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT MAX(id) AS id FROM conversations")
        row = cursor.fetchone()
        conversation_id = row["id"] if row else None
        if conversation_id is not None:
            cursor.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
            conn.commit()
    conn.close()

    if conversation_id is not None:
        try:
            _collection("conversations").delete(ids=[str(conversation_id)])
        except Exception:
            pass

    return {"deleted": conversation_id}

def clear_conversations():
    ensure_tables()
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM conversations")
        conn.commit()
    conn.close()
    reset_collection("conversations")
    return {"cleared": True}

def add_manual_memory(text):
    text = text.strip()
    if not text:
        raise ValueError("memory text is required")
    return store_conversation("admin", text, text, "Memory stored.")

def status():
    ensure_tables()
    rows = fetch_conversations()
    uni_chunks = 0
    memory_chunks = 0

    try:
        uni_chunks = _collection("uni_info").count()
    except Exception:
        pass

    try:
        memory_chunks = _collection("conversations").count()
    except Exception:
        pass

    return {
        "conversations": len(rows),
        "conversationEmbeddings": memory_chunks,
        "uniInfoChunks": uni_chunks,
    }
