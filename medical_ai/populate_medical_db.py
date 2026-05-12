import fitz
import psycopg
import os
import json
from tqdm import tqdm
from llama_cpp import Llama
from pgvector.psycopg import register_vector

os.environ["LLAMA_CPP_LIB"] = "0"

DB_PARAMS = {
    'dbname': 'medical_agent',
    'user': 'postgres',
    'password': 'mackenziefoy',
    'host': 'localhost',
    'port': '5432'
}

text_size = 1000
text_overlap = 200

EMBEDDING_MODEL_PATH = "D:/models/nomic-embed-text-v1.5.Q6_K.gguf"

text_embed = Llama(
    model_path=EMBEDDING_MODEL_PATH,
    n_ctx=2048,
    verbose=False,
    embedding=True
)

def connect_db():
    return psycopg.connect(**DB_PARAMS)

def text_chunker(text, size, overlap):
    chunks = []

    for i in range(0, len(text), size-overlap):
        chunk = text[i:i+size].strip()
        if len(chunk)>50:
            chunks.append(chunk)

    return chunks

def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return set(json.load(f))
    return set()

def save_cache(cache_path, cache_set):
    with open(cache_path, 'w') as f:
        json.dump(list(cache_set), f, indent=4)

def append_pdf(filepath):
    conn = connect_db()
    cur = conn.cursor()
    register_vector(conn)
    
    doc = fitz.open(filepath)
    file_name = os.path.basename(filepath)

    print(f"Processing {file_name}.")

    for page_num, page in enumerate(tqdm(doc, desc=f"Processing Pages.")):
        raw_text = page.get_text("text").replace("\0"," ")
        chunks = text_chunker(raw_text, text_size, text_overlap)

        for chunk in chunks:
            embedding = text_embed.create_embedding(chunk)['data'][0]['embedding']

            cur.execute(
                """
                INSERT INTO medical_knowledge (file_name, page_number, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (file_name, page_num, chunk, embedding)
            )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Successfully stored {file_name} in Database.")

def process_directory(directory_path):
    cache_path = os.path.join(directory_path, "processed_files.json")
    cache = load_cache(cache_path)
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return

    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(directory_path, filename)
            
            if filename in cache:
                print(f"Skipping '{filename}' - already processed.")
                continue
            
            try:
                append_pdf(filepath)
                cache.add(filename)
                save_cache(cache_path, cache)
            except Exception as e:
                print(f"Failed to process '{filename}': {e}")
if __name__=='__main__':
    process_directory("medical")