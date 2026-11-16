# Sahayak: Ayushman Bharat (PM-JAY) WhatsApp Voice Bot

A multi-language WhatsApp bot that answers questions about the **Ayushman Bharat PM-JAY** health insurance scheme. Users can send **voice notes or text** in Hindi, Marathi, Gujarati, Bengali or English. The bot replies in the same language, and as a **voice note when the user sent a voice note**.

Answers come only from a local knowledge file (`ayushman_bharat.txt`). If the answer is not in the file, the bot gives a polite fallback and points to the official website and helpline.

---

## Features

- **Voice in, voice out; text in, text out.** The reply format mirrors the input.
- **5 languages:** Hindi (`hi`), English (`en`), Marathi (`mr`), Gujarati (`gu`), Bengali (`bn`).
- **Per-message language detection.** The bot replies in the language of the current message. The language button chosen at the start is only a fallback.
- **Pre-recorded greeting audio** for users who start with a voice note.
- **Language selection buttons** in the greeting.
- **Intro video** (language-specific, with a default fallback) the first time a user asks about the scheme.
- **Eligibility website button** (WhatsApp call-to-action link).
- **Quick-topic buttons:** Eligibility, Coverage, Get the card.
- **RAG over a plain text file** using local TF-IDF. No embedding API calls, so there are no rate-limit problems.
- **Provider fallbacks:**
  - Speech-to-text: Sarvam Saaras v3, then Mistral Voxtral
  - Text-to-speech: Sarvam Bulbul v3, then Microsoft Edge TTS
- **Terminal test mode** so you can try the bot without WhatsApp.

---

## Conversation Flow

1. **First message** (anything, e.g. "Hi")
   - Voice-note users get the pre-recorded greeting audio.
   - Text greeting in the user's language, plus language buttons: हिंदी | मराठी | ગુજરાતી | বাংলা | English.
   - If the first message is a real question, it is answered right away.
2. **User asks about a healthcare scheme**
   - The bot confirms it is about Ayushman Bharat PM-JAY.
   - Sends the intro video, an invitation to ask questions, the eligibility link button and the topic buttons.
3. **Follow-up questions**
   - Retrieve the most relevant chunks from `ayushman_bharat.txt`.
   - Mistral generates a short spoken-style answer in the user's language, using only that context.
   - If the answer is not in the file, the bot sends the fallback message and the website button.

Off-topic questions get a polite "I can only help with Ayushman Bharat PM-JAY" message.

---

## Project Structure

```
.
├── sahayak_bot.py                  # the bot (FastAPI webhook + CLI mode)
├── ayushman_bharat.txt      # knowledge base (paragraphs separated by blank lines)
├── embeddings_cache.pkl     # auto-generated TF-IDF cache
├── .env                     # your secrets (do not commit)
├── audio/
│   └── greeting.ogg         # greeting audio (.ogg/.mp3/.mp4/.m4a/.wav/.aac)
├── videos/
│   ├── ayushman_intro.mp4       # default video
│   ├── ayushman_intro_hi.mp4
│   ├── ayushman_intro_en.mp4
│   ├── ayushman_intro_mr.mp4
│   ├── ayushman_intro_gu.mp4
│   └── ayushman_intro_bn.mp4
└── out/                     # voice replies saved here in CLI mode
```

- Any greeting audio format works. It is converted to OGG/Opus automatically, which is what WhatsApp needs to show a playable voice note.
- If a language-specific video is missing, the bot falls back to `videos/ayushman_intro.mp4`.

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and on your `PATH`
- A WhatsApp Business Cloud API account (Meta for Developers)
- A Mistral API key
- Optional: a Sarvam API key for better Indian-language speech

Install the Python packages:

```bash
pip install fastapi uvicorn httpx python-dotenv edge-tts
```

---

## Configuration

Create a `.env` file in the project folder:

```env
# Required
MISTRAL_API_KEY=your_mistral_key
WHATSAPP_TOKEN=your_whatsapp_cloud_api_token
PHONE_NUMBER_ID=your_whatsapp_phone_number_id
VERIFY_TOKEN=any_secret_string_you_choose

# Optional
SARVAM_API_KEY=your_sarvam_key
SARVAM_TTS_MODEL=bulbul:v3
SARVAM_SPEAKER=anand
ELIGIBILITY_WEBSITE_URL=https://your-eligibility-site.example/
```

| Variable | Required | Purpose |
|---|---|---|
| `MISTRAL_API_KEY` | Yes | Chat model (`open-mistral-nemo`) and Voxtral speech-to-text |
| `WHATSAPP_TOKEN` | Yes (WhatsApp mode) | WhatsApp Cloud API access token |
| `PHONE_NUMBER_ID` | Yes (WhatsApp mode) | Your WhatsApp sender phone number ID |
| `VERIFY_TOKEN` | Yes (WhatsApp mode) | Token used to verify the webhook with Meta |
| `SARVAM_API_KEY` | No | Enables Sarvam STT/TTS as the primary audio provider |
| `SARVAM_TTS_MODEL` | No | Sarvam TTS model (default `bulbul:v3`) |
| `SARVAM_SPEAKER` | No | Sarvam voice (default `anand` for v3, `anushka` otherwise) |
| `ELIGIBILITY_WEBSITE_URL` | No | URL for the eligibility button |

> **Never commit `.env`.** Add it to `.gitignore`.

---

## Knowledge Base

Put your content in `ayushman_bharat.txt`, in English:

- Separate topics with a **blank line**. Each paragraph becomes one searchable chunk.
- Lines starting with `#` are ignored (use them for comments or headings).
- The cache (`embeddings_cache.pkl`) rebuilds automatically when the file changes.

---

## Running

### Terminal test mode (no WhatsApp needed)

```bash
python sahayak_bot.py --cli
```

- Type text to pretend that is what the user said (skips speech-to-text).
- Type `@path/to/voice.ogg` to test a real audio file.
- Voice replies are saved in `./out/`.

### WhatsApp mode

```bash
uvicorn sahayak_bot:app --port 8000
```

Then:

1. Expose your server publicly, for example with `ngrok http 8000`.
2. In Meta for Developers, open your app, then **WhatsApp > Configuration**.
3. Set the callback URL to `https://<your-public-url>/webhook` and the verify token to your `VERIFY_TOKEN`.
4. Subscribe to the **messages** webhook field.
5. Send "Hi" to your WhatsApp test number.

---

## How It Works

| Component | Role |
|---|---|
| `MistralClient` | Chat completions with rate-limit spacing and retry on 429/5xx |
| `SarvamSTT` / `VoxtralSTT` | Speech to text (with `FallbackSTT`) |
| `SarvamTTS` / `EdgeTTS` | Text to speech (with `FallbackTTS`) |
| `AudioConverter` | ffmpeg conversion to WAV (for STT) and OGG/Opus (for WhatsApp) |
| `EmbeddingRetriever` | Local TF-IDF search over the knowledge file |
| `QueryAnalyzer` | Detects language, checks whether the question is about the scheme, rewrites it as an English search query |
| `Answerer` | Generates a short answer (under 50 words) using only the retrieved context |
| `WhatsAppChannel` / `ConsoleChannel` | Send text, audio, video, buttons and link buttons |
| `VoiceBot` | Conversation logic and per-user sessions |

---

## Customization

- **Add a language:** add its code to `VALID_LANGS`, `LANG_NAMES`, `EdgeTTS.VOICES`, `SarvamTTS.LANG_CODES`, `LANGUAGES`, `LANG_BTN_TO_CODE`, `VIDEO_FILES`, and every entry in `VoiceBot.TEXTS`, `TOPICS`, `BUTTON_BODY` and the eligibility text/label dictionaries.
- **Change fixed messages:** edit the `TEXTS` dictionary in `VoiceBot`.
- **Change quick-topic buttons:** edit `TOPICS` (WhatsApp allows at most 3 reply buttons, with titles up to 20 characters).
- **Change the answering style:** edit `Answerer.SYSTEM`.

---

## Known Limitations

- Sessions are stored in memory, so they reset when the server restarts.
- Marathi and Hindi share the Devanagari script, so script-based detection treats both as Hindi. The LLM-based detection decides the final language.
- The Sarvam REST speech-to-text endpoint accepts clips up to 30 seconds.
- WhatsApp allows only 3 reply buttons per message, so the 5 language buttons are sent as two messages.

---

## Disclaimer

This bot gives general information from a local knowledge file and is not an official government service. For authoritative details, use the official Ayushman Bharat PM-JAY website or call the helpline.
