import os
import warnings
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from tqdm import tqdm
from colorama import Fore
from llama_cpp import Llama
from psycopg.rows import dict_row
from kokoro import KPipeline
import sounddevice as sd
import json
import threading
from medical_ai.bridge import text_display_queue
from codes.silero import VoiceTranscriber
import serial

try:
    serial_port = serial.Serial('COM3', 9600, timeout=5)
except Exception as e:
    print(f"Serial Error: {type(e).__name__}, {e}")

gpu_lock = threading.Lock()

DB_PARAMS = {
    'dbname': 'medical_agent',
    'user': 'azaki',
    'password': 'mackenziefoy',
    'host': 'localhost',
    'port': '5432'
}

QWEN_MODEL_PATH = r"D:\models\qwen\qwen2.5-7b-instruct-q4_k_m.gguf"
EMBEDDING_MODEL_PATH = "D:/models/nomic-embed-text-v1.5.Q6_K.gguf"
WHISPER_PATH = r"D:\models\models--Systran--faster-distil-whisper-small.en\snapshots\ef77d90526ccd62cde3808ee70626a01e5cf83e4"
VAD_PATH = r"txt_files\silero_vad.jit"

SAMPLE_RATE = 24000

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["LLAMA_CPP_LIB"] = "0"
#os.environ["PYTHONUTF8"] = "1"
warnings.filterwarnings("ignore")

print("Loading models.")

llm = Llama(
    model_path=QWEN_MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=24,
    verbose=False
)

text_embed = Llama(
    model_path=EMBEDDING_MODEL_PATH,
    n_ctx=512,
    verbose=False,
    pooling_type=1,
    embedding=True
)

pipeline = KPipeline(lang_code='a', device='cuda')

listener = VoiceTranscriber(WHISPER_PATH, VAD_PATH)
listener.start_listening()

print("Listening")

system_prompt = (
    """You are a medical triage assistant.
    Protocol:
    1. Ask brief questions to gather a complete list of symptoms (one at a time).
    2. Do not give a final diagnosis immediately; keep gathering context.
    3. Be empathetic but concise.
    4. If an Urgency Level alert is present and >= 4, advise seeking immediate professional care.
    """
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
    text_to_embed = f"prompt:{prompt} response:{response}"

    with gpu_lock:
        embedding = text_embed.create_embedding(text_to_embed)['data'][0]['embedding']

    conn = connect_db()
    register_vector(conn)

    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO conversations (prompt, response, embedding) VALUES (%s, %s, %s)",
            (prompt, response, embedding)
        )
        conn.commit()
    conn.close()

def remove_last_conversation():
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute('DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)')
        conn.commit()
    conn.close()

def retrieve_embeddings(queries, n_results=2):
    if not queries:
        return set()
    
    results_set = set()

    conn = connect_db()
    register_vector(conn)

    with conn.cursor() as cursor:
        for query in tqdm(queries):
            with gpu_lock:
                query_embedding = text_embed.create_embedding(query)['data'][0]['embedding']

            cursor.execute('''
                SELECT prompt, response FROM conversations ORDER BY embedding <=> %s::vector LIMIT %s''', (query_embedding, n_results))
            
            rows = cursor.fetchall()

            for row in rows:
                doc = f"prompt:{row[0]} response:{row[1]}"
                results_set.add(doc)

    conn.close()
    return results_set

def retrieve_medical_embeddings(queries, n_results=2):
    if not queries:
        return set()
    
    results_set = set()

    conn = connect_db()
    register_vector(conn)

    with conn.cursor() as cursor:
        for query in tqdm(queries, desc="Searching Medical Database."):
            with gpu_lock:
                query_embedding = text_embed.create_embedding(query)['data'][0]['embedding']

            cursor.execute('''
                SELECT file_name, page_number, content, (embedding <=> %s::vector) AS distance FROM medical_knowledge ORDER BY distance ASC LIMIT %s''', (query_embedding, n_results))
            
            rows = cursor.fetchall()

            for row in rows:
                distance = row[3]

                confidence_percent = (1.0 - (distance / 2.0)) * 100.0
                doc = f"file_name:{row[0]} page_number:{row[1]} content:{row[2]},confidence: {round(confidence_percent, 1)}"
                results_set.add(doc)

    print(round(confidence_percent, 1))
    conn.close()
    return results_set

def create_queries(prompt):
    query_msg = (
        'You are a routing and triage AI agent. You have TWO databases: "memory" and "medical".\n'
        'Analyze the user prompt and respond STRICTLY with a valid JSON object containing:\n'
        '1. "memory_queries": List of strings for personal history.\n'
        '2. "medical_queries": List of strings for medical facts.\n'
        '3. "urgency_level": An integer from 1 (casual chat) to 5 (medical/physical emergency).\n'
        'Do not generate any markdown, explanation, or conversational text.'
    )
    
    query_convo = [
        {'role': 'system', 'content': query_msg},
        {'role': 'user', 'content': 'What did I tell you my favorite color was?'},
        {'role': 'assistant', 'content': '{"memory_queries": ["users favorite color"], "medical_queries": [], "urgency_level": 1}'},
        {'role': 'user', 'content': 'My body feels like it got hit by a car. I am in agony.'},
        {'role': 'assistant', 'content': '{"memory_queries": ["history of body pain", "recent injuries"], "medical_queries": ["causes of severe full body pain", "trauma symptoms"], "urgency_level": 5}'},
        {'role': 'user', 'content': prompt}
    ]

    with gpu_lock:
        response = llm.create_chat_completion(messages=query_convo, temperature=0.1)
    print(Fore.YELLOW + f'\nVector database queries: {response["choices"][0]["message"]["content"]} \n')

    try:
        return json.loads(response["choices"][0]["message"]["content"])
    except Exception as e:
        print(Fore.RED + f"Error! Back to basic memory search. Error: {e}")
        return {"memory_queries": [prompt], "medical_queries": [], "urgency_level": 2}

def recall(prompt):
    queries = create_queries(prompt)
    urgency = queries.get("urgency_level",2)

    memory_info = retrieve_embeddings(queries.get("memory_queries", []))
    medical_info = retrieve_medical_embeddings(queries.get("medical_queries", []))

    context_str = ""
    if urgency >= 4:
        context_str = f"[SYSTEM ALERT: The user's query has an Urgency Level of {urgency}/5.]\n"
        context_str += "[INSTRUCTION: Provide immediate, clear advice. Avoid casual filler words. Recommend professional help if necessary.]\n\n"

    if memory_info:
        context_str += "--- PERSONAL CHAT HISTORY ---\n"
        for m in memory_info:
            context_str += f"- {m}\n"
        context_str += "\n"

    if medical_info:
        context_str += "--- AUTHORITATIVE MEDICAL KNOWLEDGE ---\n"
        for m in medical_info:
            context_str += f"- {m}\n"
        context_str += "\n"

    convo.append({
        'role': 'user',
        'content': f"MEMORIES: {context_str}\nUSER PROMPT: {prompt}"
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
    global convo
    convo.append({'role': 'user', 'content': prompt})
    if len(convo) > 7:
        convo = [convo[0]] + convo[-5:]

    response_text = ""

    with gpu_lock:
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
        send_serial()
        print(sentence, end=" ", flush=True)
        response_text += sentence + " "

        text_display_queue.put(sentence + " ")

        wav = generate_tts(sentence)
        sd.play(wav, samplerate=SAMPLE_RATE)
        
        sd.wait()

    print("\n")

    store_conversations(prompt, response_text.strip())
    convo.append({'role': 'assistant', 'content': response_text.strip()})

def generate_diagnosis():
    global convo
    print(Fore.CYAN + "\n[Generating Diagnostic Summary...]")
    
    diagnostic_prompt = """
    Analyze the preceding conversation. Provide a final medical triage summary STRICTLY in JSON format.
    Do not include any markdown, explanation, or conversational text.
    
    Use this EXACT schema:
    {
      "summary": "Brief 1-2 sentence overview of the patient's state.",
      "symptoms": ["symptom 1", "symptom 2", "symptom 3"],
      "condition": "Most likely suspected condition",
      "confidence": 72,
      "urgency_level": "Moderate",
      "urgency_color": "🟡"
    }
    """
    temp_convo = convo.copy()
    temp_convo.append({'role': 'user', 'content': diagnostic_prompt})
    
    with gpu_lock:
        response = llm.create_chat_completion(
            messages=temp_convo,
            temperature=0.1,
            stream=False
        )
        
    result_text = response["choices"][0]["message"]["content"].strip()
    
    try:
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()
            
        data = json.loads(result_text)
        
        symptoms_list = "\n• ".join(data["symptoms"])
        formatted_summary = f"{data['summary']}\n\nSymptoms:\n• {symptoms_list}"
        
        payload = {
            "summary": formatted_summary,
            "urgency": f"{data['urgency_color']} {data['urgency_level']}",
            "suspected": data["condition"],
            "confidence": data["confidence"]
        }
        
        text_display_queue.put("||JSON||" + json.dumps(payload))
        print(Fore.GREEN + "[Dashboard Updated.]")
        
        wav = generate_tts("Diagnostic summary complete. Please review the dashboard.")
        sd.play(wav, samplerate=SAMPLE_RATE)
        sd.wait()
        
    except Exception as e:
        print(Fore.RED + f"Failed to parse diagnosis. Error: {e}\noutput: {result_text}")

def send_serial():
    try:
        msg = "a" + "\n"
        serial_port.write(msg.encode())
    except Exception as e:
        print(f"Serial Error: {e}")

def text():
    while True:
        prompt = input(Fore.WHITE + "\nUser:\n")

        if prompt == "q":
            break

        if prompt == "/diagnose":
            generate_diagnosis()
            continue

        if prompt.startswith("/recall"):
            prompt = prompt[8:]
            recall(prompt)

        elif prompt.startswith("/forget"):
            remove_last_conversation()
            if len(convo) >= 2:
                convo[:] = convo[:-2]
            print("Last memory removed.")
            continue

        text_display_queue.put("||RESET||")
        stream_and_tts(prompt)

def main():
    print("System Ready.")
    while True:
        try:
            prompt = listener.get_next_text()
            print(prompt)
            if prompt.lower() in ["q.", "exit.", "bye.", "see you later."]:
                break

            elif prompt.lower() in  ["result.", "results"]:
                generate_diagnosis()
                continue

            else:
                recall(prompt)
                
            text_display_queue.put("||RESET||")
            stream_and_tts(prompt)
                
        except KeyboardInterrupt:
            break


if __name__ == "__main__":

    text()
