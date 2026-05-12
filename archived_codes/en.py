import os
import warnings
import ast
import numpy as np
import chromadb
import psycopg
from tqdm import tqdm
from colorama import Fore
from llama_cpp import Llama
from psycopg.rows import dict_row
from kokoro import KPipeline
import sounddevice as sd

DB_PARAMS = {
    'dbname': 'misaki_en',
    'user': 'azaki',
    'password': 'mackenziefoy',
    'host': 'localhost',
    'port': '5432'
}

QWEN_MODEL_PATH = r"D:\models\qwen\qwen2.5-7b-instruct-q4_k_m.gguf"
EMBEDDING_MODEL_PATH = "D:/models/nomic-embed-text-v1.5.Q6_K.gguf"

SAMPLE_RATE = 24000

os.environ["HF_HUB_OFFLINE"] = "1"
#os.environ["PYTHONUTF8"] = "1"
warnings.filterwarnings("ignore")

print("Loading models...")

llm = Llama(
    model_path=QWEN_MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=32,
    verbose=False
)

text_embed = Llama(
    model_path=EMBEDDING_MODEL_PATH,
    n_ctx=2048,
    verbose=False,
    embedding=True
)

pipeline = KPipeline(lang_code='a', device='cuda')

client = chromadb.Client()

system_prompt = (
    "You are an AI assistant that remembers previous conversations. "
    "If relevant past conversations are provided, use them naturally. "
    "Do not mention recalling memory explicitly."
)

convo = [{'role': 'system', 'content': system_prompt}]

def connect_db():
    return psycopg.connect(**DB_PARAMS)

def fetch_conversations():
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT * FROM conversations')
        rows = cursor.fetchall()
    conn.close()
    return rows

def store_conversations(prompt, response):
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)",
            (prompt, response)
        )
        conn.commit()
    conn.close()

def remove_last_conversation():
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute('DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)')
        conn.commit()
    conn.close()

def create_vector_db(conversations):
    name = "conversations"
    try:
        client.delete_collection(name=name)
    except:
        pass

    collection = client.create_collection(name=name)

    for c in conversations:
        text = f"prompt:{c['prompt']} response:{c['response']}"
        embedding = text_embed.create_embedding(text)['data'][0]['embedding']
        collection.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[text]
        )

def retrieve_embeddings(queries, n_results=2):
    results_set = set()
    collection = client.get_collection(name="conversations")

    for query in tqdm(queries):
        query_embedding = text_embed.create_embedding(query)['data'][0]['embedding']
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        docs = results['documents'][0]

        for doc in docs:
            if doc not in results_set:
                results_set.add(doc)

    return results_set

def create_queries(prompt):
    query_msg = (
        'You are a first principle reasoning search query AI agent. '
        'Your list of search queries will be ran on an embedding database of all your conversations '
        'you have ever had with the user. With first principles create a Python list of queries to '
        'search the embeddings database for any data that would be necessary to have access to in '
        'order to correctly respond to the prompt. Your response must be a Python list with no syntax errors. '
        'Do not explain anything and do not ever generate anything but a perfect syntax Python list'
    )
    query_convo = [
        {'role': 'system', 'content': query_msg},
        {'role': 'user', 'content': 'Write an email to my car insurance company and create a pursuasive request for them to lower my monthly rate.'},
        {'role': 'assistant', 'content': '["What is the users name?", "What is the users current auto insurance provider?", "What is the monthly rate the user currently pays for auto insurance?"]'},
        {'role': 'user', 'content': 'how can i convert the speak function in my llama3 python voice assistant to use pyttsx3'},
        {'role': 'assistant', 'content': '["Llama3 voice assistant", "Python voice assistant", "OpenAI TTS", "openai speak"]'},
        {'role': 'user', 'content': prompt}
    ]

    response = llm.create_chat_completion(messages=query_convo)
    print(Fore.YELLOW + f'\nVector database queries: {response["choices"][0]["message"]["content"]} \n')

    try:
        return ast.literal_eval(response["choices"][0]["message"]["content"])
    except:
        return [prompt]

def recall(prompt):
    queries = create_queries(prompt)
    embeddings = retrieve_embeddings(queries)
    convo.append({
        'role': 'user',
        'content': f"MEMORIES: {embeddings}\nUSER PROMPT: {prompt}"
    })

def text_chunker(token_stream):
    buffer = ""
    sentence_endings = {'.', '!', '?', '\n'}

    for token in token_stream:
        buffer += token

        if any(buffer.endswith(p) for p in sentence_endings) and len(buffer.strip()) > 5:
            yield buffer.strip()
            buffer = ""

    if buffer.strip():
        yield buffer.strip()

def generate_tts(text_chunk, voice='af_heart', speed=1.0):
    generator = pipeline(
        text=text_chunk,
        voice=voice,
        speed=speed,
        split_pattern=r'\n+'
    )

    wav = None
    for _, _, audio in generator:
        wav = audio

    if wav is None:
        raise RuntimeError("Kokoro returned no audio")

    return np.array(wav, dtype=np.float32)

def stream_and_tts(prompt, temperature=0.7):
    convo.append({'role': 'user', 'content': prompt})
    response_text = ""
    audio_outputs = []

    stream = llm.create_chat_completion(
        messages=convo,
        stream=True,
        temperature=temperature,
    )

    print(Fore.LIGHTGREEN_EX + "\nAssistant:\n")

    token_gen = (
        chunk["choices"][0]["delta"]["content"]
        for chunk in stream
        if "content" in chunk["choices"][0]["delta"]
    )

    for sentence in text_chunker(token_gen):
        print(sentence, end=" ", flush=True)
        response_text += sentence + " "

        wav = generate_tts(sentence)
        audio_outputs.append(wav)

    print("\n")

    store_conversations(prompt, response_text.strip())
    convo.append({'role': 'assistant', 'content': response_text.strip()})

    return audio_outputs

conversations = fetch_conversations()
create_vector_db(conversations)

def main():
    print("System Ready.")

    while True:
        prompt = input(Fore.WHITE + "\nUser:\n")

        if prompt == "q":
            break

        if prompt.startswith("/recall"):
            prompt = prompt[8:]
            recall(prompt)

        elif prompt.startswith("/forget"):
            remove_last_conversation()
            if len(convo) >= 2:
                convo[:] = convo[:-2]
            print("Last memory removed.")
            continue

        audio_chunks = stream_and_tts(prompt)

        if len(audio_chunks) > 0:
            full_audio = np.concatenate(audio_chunks)
            sd.play(full_audio, samplerate=SAMPLE_RATE)
            sd.wait()

        print(f"\nGenerated {len(audio_chunks)} TTS segments.\n")

if __name__ == "__main__":
    main()
