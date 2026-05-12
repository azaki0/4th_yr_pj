import os
import warnings
import sounddevice as sd
import chromadb
import psycopg
import ast
import time
from tqdm import tqdm
from colorama import Fore
from llama_cpp import Llama
from psycopg.rows import dict_row
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel
import torch
import threading
from queue import Queue
import re
from num2words import num2words
from silero import VoiceTranscriber

client = chromadb.Client()

system_prompt = (
    'You are an AI assistant that has memory of every conversations you have ever had with this user. '
    'On every prompt from the user, the system has checked for any relevant messages you had with the user. '
    'If any embedded previous conversations are attached, use them for context to responding to the user, '
    'if the context is relevant and useful to responding. If the recalled conversations are irrelevant, '
    'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations. '
    'Just use any useful data from the previous convarsations and respond normally as an intelligent AI assistant.'
)

convo = [{'role': 'system', 'content': system_prompt}]
audio_queue = Queue()

DB_PARAMS = {
    'dbname': 'memory_agent',
    'user': 'azaki',
    'password': 'mackenziefoy',
    'host': 'localhost',
    'port': '5432'
}

SAMPLE_RATE = 16000

os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

QWEN_MODEL_PATH = r"D:\models\qwen\qwen2.5-7b-instruct-q4_k_m.gguf"
EMBEDDING_MODEL_PATH = "D:/models/nomic-embed-text-v1.5.Q6_K.gguf"
TRANSLATER_MODEL_PATH = r"D:\models\nllb-200-distilled-600M"
device = "cpu"
TTS_MODEL_PATH = r"D:\codes\Vits_mms_finetune\finetune-hf-vits\mms-tts-mya-female-v1"
WHISPER_PATH = r"D:\models\models--Systran--faster-distil-whisper-small.en\snapshots\ef77d90526ccd62cde3808ee70626a01e5cf83e4"
VAD_PATH = r"txt_files\silero_vad.jit"

llm = Llama(
    model_path=QWEN_MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=32,
    verbose=False
)
text_embed = Llama(model_path=EMBEDDING_MODEL_PATH,n_ctx=2048,verbose=False, embedding=True)
tokenizer = AutoTokenizer.from_pretrained(TRANSLATER_MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATER_MODEL_PATH).to(device)

tts_model = VitsModel.from_pretrained(TTS_MODEL_PATH)
tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_PATH)

listener = VoiceTranscriber(WHISPER_PATH, VAD_PATH)
listener.start_listening()

print("I'm listening.")

def normalize_numwords(s):
    s = s.replace(',', '')
    s = s.replace(' and ', ' ')
    return s

number_pattern = re.compile(r'\d+(\.\d+)?(?:,\d{3})*')

def numbers_to_words(text, lang='en'):
    def repl(match):
        token = match.group(0)
        token_clean = token.replace(',', '')
        if '.' in token_clean:
            whole, frac = token_clean.split('.', 1)
            words_whole = num2words(int(whole), lang='en')
            words_frac = ' '.join(num2words(int(d), lang='en') for d in frac)
            return normalize_numwords(f"{words_whole} point {words_frac}")
        else:
            try:
                n = int(token_clean)
                words = num2words(n, lang='en')
                return normalize_numwords(words)
            except Exception:
                return token
    return number_pattern.sub(repl, text)

def audio_player():
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        sd.play(chunk, samplerate=SAMPLE_RATE)
        sd.wait()
        audio_queue.task_done()

player_thread = threading.Thread(target=audio_player, daemon=True)
player_thread.start()

def translate(text, src_lang, target_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt").to(device)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=512
    )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def connect_db():
    conn = psycopg.connect(**DB_PARAMS)
    return conn

def fetch_conversations():
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT * FROM conversations')
        conversations = cursor.fetchall()
    conn.close()
    return conversations

def store_conversations(prompt, response):
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)",(prompt,response)
        )
        conn.commit()
    conn.close()

def remove_last_conversation():
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute('DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)')
        conn.commit()
    conn.close()

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

def generate_tts(text_chunk):
    inputs = tts_tokenizer(text_chunk, return_tensors="pt")

    with torch.no_grad():
        output = tts_model(**inputs).waveform

    output_numpy = output.squeeze().cpu().numpy()

    return output_numpy

def stream_response(prompt):
    response = ''
    stream = llm.create_chat_completion(
        messages=convo,
        stream=True,
        temperature=0.7,
    )
    print(Fore.LIGHTGREEN_EX + '\nAssistant: ')

    token_gen = (
        chunk["choices"][0]["delta"]["content"]
        for chunk in stream
        if "content" in chunk["choices"][0]["delta"]
    )

    for sentence in text_chunker(token_gen):
        response += sentence + " "
        start_time = time.perf_counter()
        sentence_expanded = numbers_to_words(sentence)
        burmese_sentence = translate(sentence_expanded, "eng_Latn", "mya_Mymr")
        end_time = time.perf_counter()
        print(sentence, end=" ", flush=True)
        with open("response.txt",'a', encoding='UTF-8')as f:
            f.write(burmese_sentence +","+ str(end_time-start_time))
            f.write("\n")

        wav = generate_tts(burmese_sentence)
        audio_queue.put(wav)

    print('\n')
    store_conversations(prompt,response)
    convo.append({'role': 'assistant', 'content': response})

def create_vector_db(conversations):
    vector_db_name = 'conversations'

    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass

    vector_db = client.create_collection(name=vector_db_name)

    for c in conversations:
        serialized_convo = f"prompt:{c['prompt']} response:{c['response']}"
        response = text_embed.create_embedding(serialized_convo)
        embedding = response['data'][0]['embedding']

        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )

def retrieve_embeddings(queries, results_per_query=2):
    embeddings = set()

    for query in tqdm(queries, desc="Processing queries to vector database"):
        response = text_embed.create_embedding(query)
        query_embedding = response['data'][0]['embedding']

        vector_db = client.get_collection(name='conversations')
        results = vector_db.query(query_embeddings=[query_embedding], n_results=results_per_query)
        best_embedding = results['documents'][0]

        for best in best_embedding:
            if best not in embeddings:
                if 'yes' in classify_embedding(query, best):
                    embeddings.add(best)   

    return embeddings    

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

def classify_embedding(query, context):
    classify_msg = (
        'You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text. '
        'You will not respond as an AI assistant. You only respond "yes" or "no". '
        'Determine whether the context contains data that directly is related to the search query. '
        'If the context is seemingly exactly what the search query needs, respond "yes" if it is anything but directly '
        'related respond "no". Do not respond "yes" unless the content is highly relevant to the search query.' 
    )
    classify_convo = [
        {'role': 'system', 'content': classify_msg},
        {'role': 'user', 'content': f'SEARCH QUERY: What is the users name? \n\nEMBEDDED CONTEXT: You are Azaki. How can i help you today.'},
        {'role': 'assistant', 'content': 'yes'},
        {'role': 'user', 'content': f'SEARCH QUERY: Llama3 python Voice Assistant \n\nEMBEDDED CONTEXT: Siri is a voice assistant'},
        {'role': 'assistant', 'content': 'no'},
        {'role': 'user', 'content': f'SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context}'}
    ]
    response = llm.create_chat_completion(messages=classify_convo)

    return response["choices"][0]["message"]["content"].strip().lower()

def recall(prompt):
    queries = create_queries(prompt)
    embeddings =  retrieve_embeddings(queries)
    convo.append({'role': 'user', 'content': f'MEMORIES: {embeddings} \n\n USER PROMPT: {prompt}'})
    print(f'\n{len(embeddings)} messages:response embeddings added for context.')

#conversations = fetch_conversations()
#create_vector_db(conversations=conversations)

'''def send_serial():
    serial_connected = False
    try:
        serial_port = serial.Serial('COM3', 9600, timeout=1)
        serial_connected = True
    except Exception as e:
        serial_connected = False
        print(f"Serial Error: {type(e).__name__}, {e}")

    if serial_connected:
        try:
            msg = "a" + "\n"
            serial_port.write(msg.encode())
            print(msg)
        except Exception as e:
            print(f"Serial Error: {e}")
            serial_connected = False
    else:
        print("Send failed! Serial not connected.")'''

while True:
    prompt = input(Fore.WHITE + 'User: \n')
    #user_input = listener.get_next_text()
    #prompt = "/recall "+ user_input
    if prompt == 'q':
        break

    if prompt[:7].lower() == '/recall':
        prompt = prompt[8:]
        recall(prompt)
        stream_response(prompt)
    elif prompt[:7].lower() == '/forget':
        remove_last_conversation()
        convo = convo[:-2]
        print('\n')
    elif prompt[:9].lower() == '/memorize':
        prompt = prompt[10:]
        store_conversations(prompt, response='Memory stored.')
        print('\n')
    else:
        convo.append({'role': 'user', 'content': prompt})
        stream_response(prompt)
