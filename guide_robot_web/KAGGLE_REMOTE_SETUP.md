# Kaggle Remote Agent Setup

This branch keeps the laptop Flask app lightweight. The laptop runs:

- `/` phone display
- `/admin` prompt and language switch
- route/map rendering
- local audio playback

Kaggle runs the heavy models through `kaggle_agent_server.py` and streams JSON-line events back to the laptop.

## Laptop

Start the local display app:

```powershell
python guide_robot_web/app.py
```

Open:

- phone display: `http://localhost:5000/`
- admin panel: `http://localhost:5000/admin`

Paste the localtunnel URL from Kaggle into the admin panel's "Kaggle tunnel URL" box.

## Kaggle

Upload or copy these files together:

- `kaggle_agent_server.py`
- `navigation.py`

The server downloads default models automatically if you do not provide paths:

- English/Myanmar LLM: `Qwen/Qwen2.5-7B-Instruct-GGUF`, file `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`
- Translator: `facebook/nllb-200-distilled-600M`
- Myanmar TTS: `facebook/mms-tts-mya`

Downloaded GGUF files are cached in `/kaggle/working/model_cache` by default.

If you already attached Kaggle datasets or want different models, set optional overrides in the notebook:

```python
import os

os.environ["QWEN_MODEL_PATH_EN"] = "/kaggle/input/your-qwen-en/qwen2.5-7b-instruct-q4_k_m.gguf"
os.environ["TRANSLATER_MODEL_PATH"] = "/kaggle/input/your-nllb-200-distilled-600m"
os.environ["VITS_MODEL_PATH"] = "/kaggle/input/your-vits-mms-mya"
```

You can also override download sources instead of local paths:

```python
os.environ["QWEN_MODEL_REPO_EN"] = "Qwen/Qwen2.5-7B-Instruct-GGUF"
os.environ["QWEN_MODEL_FILE_EN"] = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
os.environ["TRANSLATER_MODEL_ID"] = "facebook/nllb-200-distilled-600M"
os.environ["VITS_MODEL_ID"] = "facebook/mms-tts-mya"
```

Run the server:

```bash
python kaggle_agent_server.py
```

In another Kaggle cell, expose port `8000` with localtunnel:

```bash
npx localtunnel --port 8000
```

Copy the generated `https://....loca.lt` URL into the laptop admin panel.

If localtunnel asks for a password, it is usually the public IP of the Kaggle runtime:

```bash
curl https://loca.lt/mytunnelpassword
```

No Kaggle secret or competition/test-only environment variable is required for the normal flow.

## Stream Protocol

Kaggle returns one JSON object per line:

```json
{"type":"reset","data":null}
{"type":"route","data":{"points":[]}}
{"type":"text","data":"hello"}
{"type":"audio","data":"base64-float32-audio","sampleRate":16000,"dtype":"float32"}
```

The laptop forwards `reset`, `route`, and `text` to the existing browser stream. It plays `audio` locally with `sounddevice`.
