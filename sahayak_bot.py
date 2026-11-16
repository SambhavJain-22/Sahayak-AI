"""
Ayushman Bharat (PM-JAY) WhatsApp VOICE bot — multi-language, voice note in -> voice note out.

New Flow
 1. User sends ANY first message (e.g. "Hi") -> Bot sends:
      • Pre-recorded greeting audio (audio/greeting.ogg)
      • Text greeting in Hindi
      • Language-choice buttons: हिंदी | मराठी | ગુજરાતી | বাংলা | English
      • Invitation to ask via voice or text
 2. User taps a language button -> bot locks in language, asks them to speak/type.
 3. User asks about a healthcare scheme -> bot replies about Ayushman Bharat PM-JAY,
      sends the intro video, and a link button to the eligibility website.
 4. Follow-up voice/text questions -> RAG over ayushman_bharat.txt -> spoken answer
      in the user's chosen language.
    The model may only use the txt file. If it isn't there -> polite fallback.

Needs: ffmpeg on PATH, .env with MISTRAL_API_KEY, WHATSAPP_TOKEN,
       PHONE_NUMBER_ID, VERIFY_TOKEN
       Optional: SARVAM_API_KEY for better Indian-language STT/TTS
       Optional: ELIGIBILITY_WEBSITE_URL for the website button

Run (WhatsApp):   uvicorn bot:app --port 8000      (+ ngrok)
Run (terminal):   python bot.py --cli
   - type text  = pretend that is what the user said (skips speech-to-text)
   - type @file.ogg = test a real audio file (uses speech-to-text)
   - voice replies are saved as files in ./out/
"""
import argparse
import asyncio
import base64
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import edge_tts
import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import PlainTextResponse

load_dotenv()

BASE_DIR = Path(__file__).parent
KNOWLEDGE_FILE = BASE_DIR / "ayushman_bharat.txt"
EMBED_CACHE_FILE = BASE_DIR / "embeddings_cache.pkl"
VIDEO_FILE = BASE_DIR / "videos" / "ayushman_intro.mp4"  # default (Hindi/English)
# Language-specific intro videos — place files as videos/ayushman_intro_<lang>.mp4
# e.g. ayushman_intro_mr.mp4, ayushman_intro_gu.mp4, etc.
# If a file doesn't exist the bot falls back to VIDEO_FILE automatically.
VIDEO_FILES: dict[str, Path] = {
    "hi": BASE_DIR / "videos" / "ayushman_intro_hi.mp4",
    "en": BASE_DIR / "videos" / "ayushman_intro_en.mp4",
    "mr": BASE_DIR / "videos" / "ayushman_intro_mr.mp4",
    "gu": BASE_DIR / "videos" / "ayushman_intro_gu.mp4",
    "bn": BASE_DIR / "videos" / "ayushman_intro_bn.mp4",
}
# Greeting audio — can be any format: greeting.ogg / .mp3 / .mp4 / .wav / .m4a
# The bot auto-converts to OGG/Opus (required by WhatsApp) if it isn't already.
# Place the file as audio/greeting.<ext> — the first match is used.
GREETING_AUDIO_DIR = BASE_DIR / "audio"
OUT_DIR = BASE_DIR / "out"  # only used by --cli


# --------------------------------------------------------------------------
# LLM layer (chat + embeddings)
# --------------------------------------------------------------------------
class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], json_mode: bool = False,
             temperature: float = 0.0) -> str: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class MistralClient(LLMClient):
    def __init__(self, api_key: str, chat_model: str = "open-mistral-nemo",
                 embed_model: str = "mistral-embed"):
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.http = httpx.Client(
            base_url="https://api.mistral.ai/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

    # The free tier allows very few requests per second, and one WhatsApp
    # message needs 3 calls (analyse, embed, answer). So we space the calls out
    # and retry on 429 instead of failing.
    MIN_INTERVAL = 2.0  # seconds between requests
    _lock = threading.Lock()
    _last_call = 0.0

    def _post(self, path: str, body: dict, retries: int = 8):
        delay = 5.0
        for attempt in range(retries + 1):
            with MistralClient._lock:
                wait = self.MIN_INTERVAL - (time.monotonic() - MistralClient._last_call)
                if wait > 0:
                    time.sleep(wait)
                MistralClient._last_call = time.monotonic()
            r = self.http.post(path, json=body)
            if r.status_code != 429 and r.status_code < 500:
                break
            if attempt == retries:
                break
            pause = delay
            retry_after = r.headers.get("retry-after")
            if retry_after:
                try:
                    pause = max(delay, float(retry_after))
                except ValueError:
                    pass
            print(f"[Mistral] {r.status_code} on {path}, retrying in {pause:.1f}s")
            time.sleep(min(pause, 60))
            delay = min(delay * 2, 60)
        r.raise_for_status()
        return r

    def chat(self, messages, json_mode=False, temperature=0.0) -> str:
        body = {"model": self.chat_model, "messages": messages,
                "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self._post("/chat/completions", body)
        return r.json()["choices"][0]["message"]["content"].strip()

    def embed(self, texts) -> list[list[float]]:
        r = self._post("/embeddings",
                       {"model": self.embed_model, "input": texts})
        return [d["embedding"] for d in r.json()["data"]]


# --------------------------------------------------------------------------
# Audio layer: speech-to-text, text-to-speech, ffmpeg conversion
# --------------------------------------------------------------------------
class AudioConverter:
    @staticmethod
    def _run(cmd: list[str]) -> None:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {p.stderr[-300:]}")

    @classmethod
    def to_wav(cls, src: Path, dst: Path) -> Path:
        cls._run(["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)])
        return dst

    @classmethod
    def to_ogg_opus(cls, src: Path, dst: Path) -> Path:
        # OGG/Opus is what makes WhatsApp show it as a playable voice note
        cls._run(["ffmpeg", "-y", "-i", str(src), "-c:a", "libopus",
                  "-b:a", "32k", "-ar", "48000", "-ac", "1", str(dst)])
        return dst


class STT(ABC):
    @abstractmethod
    def transcribe(self, wav_path: Path) -> str: ...


class VoxtralSTT(STT):
    """Mistral Voxtral speech-to-text. Auto-detects Hindi / English."""

    def __init__(self, api_key: str, model: str = "voxtral-mini-latest"):
        self.model = model
        self.http = httpx.Client(
            base_url="https://api.mistral.ai/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )

    def transcribe(self, wav_path: Path) -> str:
        with open(wav_path, "rb") as f:
            r = self.http.post(
                "/audio/transcriptions",
                data={"model": self.model},
                files={"file": (wav_path.name, f, "audio/wav")},
            )
        if r.status_code >= 400:
            print(f"[STT ERROR] {r.status_code}: {r.text}")
            r.raise_for_status()
        return r.json().get("text", "").strip()


class TTS(ABC):
    @abstractmethod
    def speak(self, text: str, lang: str, out_ogg: Path) -> Path: ...


class EdgeTTS(TTS):
    """Free Microsoft neural voices via the edge-tts package."""
    VOICES = {
        "hi": "hi-IN-SwaraNeural",
        "en": "en-IN-NeerjaNeural",
        "mr": "mr-IN-AarohiNeural",
        "gu": "gu-IN-NiranjanNeural",
        "bn": "bn-IN-TanishaaNeural",
    }

    def speak(self, text: str, lang: str, out_ogg: Path) -> Path:
        mp3 = out_ogg.with_suffix(".mp3")
        voice = self.VOICES.get(lang, self.VOICES["en"])
        asyncio.run(edge_tts.Communicate(text, voice).save(str(mp3)))
        return AudioConverter.to_ogg_opus(mp3, out_ogg)


class SarvamSTT(STT):
    """Sarvam Saaras v3 speech-to-text, built for Indian languages.
    The REST endpoint accepts clips up to 30 seconds."""

    def __init__(self, api_key: str, model: str = "saaras:v3"):
        self.model = model
        self.http = httpx.Client(
            base_url="https://api.sarvam.ai",
            headers={"api-subscription-key": api_key},
            timeout=120,
        )

    def transcribe(self, wav_path: Path) -> str:
        data = {"model": self.model}
        if self.model == "saaras:v3":
            data["mode"] = "transcribe"  # language is auto-detected
        with open(wav_path, "rb") as f:
            r = self.http.post("/speech-to-text", data=data,
                               files={"file": (wav_path.name, f, "audio/wav")})
        if r.status_code >= 400:
            print(f"[Sarvam STT ERROR] {r.status_code}: {r.text}")
            r.raise_for_status()
        return r.json().get("transcript", "").strip()


class SarvamTTS(TTS):
    """Sarvam Bulbul v3 text-to-speech (natural Indian voices)."""
    LANG_CODES = {
        "hi": "hi-IN",
        "en": "en-IN",
        "mr": "mr-IN",
        "gu": "gu-IN",
        "bn": "bn-IN",
    }

    def __init__(self, api_key: str, model: str = "bulbul:v3",
                 speaker: str = "anand"):
        self.model = model
        self.speaker = speaker
        self.http = httpx.Client(
            base_url="https://api.sarvam.ai",
            headers={"api-subscription-key": api_key},
            timeout=120,
        )

    def speak(self, text: str, lang: str, out_ogg: Path) -> Path:
        r = self.http.post("/text-to-speech", json={
            "text": text,
            "target_language_code": self.LANG_CODES.get(lang, "en-IN"),
            "model": self.model,
            "speaker": self.speaker,
            "output_audio_codec": "mp3",
        })
        if r.status_code >= 400:
            print(f"[Sarvam TTS ERROR] {r.status_code}: {r.text}")
            r.raise_for_status()
        audio = base64.b64decode("".join(r.json()["audios"]))
        mp3 = out_ogg.with_suffix(".mp3")
        mp3.write_bytes(audio)
        return AudioConverter.to_ogg_opus(mp3, out_ogg)


class FallbackSTT(STT):
    """Try the primary provider; if it fails or returns nothing, use the backup."""

    def __init__(self, primary: STT, backup: STT):
        self.primary, self.backup = primary, backup

    def transcribe(self, wav_path: Path) -> str:
        try:
            text = self.primary.transcribe(wav_path)
            if text:
                return text
        except Exception as e:
            print(f"[STT] primary failed, using backup: {e!r}")
        return self.backup.transcribe(wav_path)


class FallbackTTS(TTS):
    def __init__(self, primary: TTS, backup: TTS):
        self.primary, self.backup = primary, backup

    def speak(self, text: str, lang: str, out_ogg: Path) -> Path:
        try:
            return self.primary.speak(text, lang, out_ogg)
        except Exception as e:
            print(f"[TTS] primary failed, using backup: {e!r}")
            return self.backup.speak(text, lang, out_ogg)


# --------------------------------------------------------------------------
# Retrieval layer (RAG over the txt file)
# --------------------------------------------------------------------------
class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 3) -> list[str]: ...


class EmbeddingRetriever(Retriever):
    """Each paragraph (blocks separated by a blank line) is one chunk.

    Uses local TF-IDF so indexing and search do not call Mistral (the free
    embed endpoint 429s). Queries are already rewritten to English by the
    analyzer, which matches ayushman_bharat.txt.
    """

    _TOKEN = re.compile(r"[a-z]+|\d+")
    _STOP = frozenset("""
        a an the and or of to for in on at by from with without as is are was
        were be been being this that these those it its their there here not
        no nor so if then than also can may must should would could will
        about under over into such per any all more most other own
        """.split())
    CACHE_MODEL = "tfidf-local"

    def __init__(self, path: Path, llm: LLMClient | None = None):
        self.chunks = self._load_chunks(path)
        if not self.chunks:
            raise ValueError(f"No content found in {path}")
        if not self._load_cache():
            self._fit(self.chunks)
            self._save_cache()
        print(f"[RAG] indexed {len(self.chunks)} chunks locally from {path.name}")

    @staticmethod
    def _load_chunks(path: Path) -> list[str]:
        text = Path(path).read_text(encoding="utf-8")
        chunks = []
        for block in re.split(r"\n\s*\n", text):
            lines = [l.strip() for l in block.splitlines()
                     if l.strip() and not l.lstrip().startswith("#")]
            if lines:
                chunks.append(" ".join(lines))
        return chunks

    def _tokens(self, text: str) -> list[str]:
        return [t for t in self._TOKEN.findall(text.lower())
                if t not in self._STOP and (len(t) > 1 or t.isdigit())]

    def _fit(self, chunks: list[str]) -> None:
        docs = [self._tokens(c) for c in chunks]
        df: dict[str, int] = {}
        for toks in docs:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        terms = sorted(df)
        self.vocab = {t: i for i, t in enumerate(terms)}
        n = len(docs)
        self.idf = [math.log((n + 1) / (df[t] + 1)) + 1.0 for t in terms]
        self.vectors = [self._tfidf(toks) for toks in docs]

    def _tfidf(self, toks: list[str]) -> list[float]:
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec = [0.0] * len(self.vocab)
        for t, c in tf.items():
            i = self.vocab.get(t)
            if i is not None:
                vec[i] = (1.0 + math.log(c)) * self.idf[i]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def _load_cache(self) -> bool:
        if not EMBED_CACHE_FILE.exists():
            return False
        try:
            with open(EMBED_CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if (data.get("model") == self.CACHE_MODEL
                    and data.get("chunks") == self.chunks
                    and data.get("vectors")
                    and data.get("vocab") is not None
                    and data.get("idf") is not None
                    and len(data["vectors"]) == len(self.chunks)):
                self.vocab = data["vocab"]
                self.idf = data["idf"]
                self.vectors = data["vectors"]
                return True
        except Exception as e:
            print(f"[RAG] cache unused: {e}")
        return False

    def _save_cache(self) -> None:
        try:
            with open(EMBED_CACHE_FILE, "wb") as f:
                pickle.dump({
                    "model": self.CACHE_MODEL,
                    "chunks": self.chunks,
                    "vocab": self.vocab,
                    "idf": self.idf,
                    "vectors": self.vectors,
                }, f)
        except Exception as e:
            print(f"[RAG] could not save cache: {e}")

    @staticmethod
    def _cosine(a, b) -> float:
        return sum(x * y for x, y in zip(a, b))

    def search(self, query: str, k: int = 3) -> list[str]:
        qv = self._tfidf(self._tokens(query))
        scored = sorted(
            ((self._cosine(qv, v), c) for v, c in zip(self.vectors, self.chunks)),
            reverse=True,
        )
        return [c for _, c in scored[:k]]


# --------------------------------------------------------------------------
# AI helpers
# --------------------------------------------------------------------------
@dataclass
class Analysis:
    about_scheme: bool
    language: str   # "hi" (Hindi / Hinglish) or "en"
    query_en: str   # the question as a short English search query


class QueryAnalyzer:
    # Valid language codes understood by the bot
    VALID_LANGS = {"hi", "en", "mr", "gu", "bn"}

    PROMPT = (
        "You analyse the transcript of a voice message sent to a WhatsApp bot "
        "about the Indian health scheme Ayushman Bharat (PM-JAY). The transcript "
        "may be in Hindi (Devanagari), Hinglish, English, Marathi, Gujarati, or "
        "Bengali, and may contain speech-recognition mistakes. "
        "Return JSON only, in this shape: "
        '{"about_scheme": true, "language": "hi", "query_en": "..."} '
        "about_scheme is true if the user asks about Ayushman Bharat / PM-JAY / "
        "the Ayushman card, or asks in general for a government health or "
        "healthcare scheme or health insurance for the poor (for example "
        "'koi healthcare scheme batao'). "
        "language must be one of: hi (Hindi/Hinglish), en (English), "
        "mr (Marathi), gu (Gujarati), bn (Bengali). "
        "query_en is the user's question rewritten as a short, clear English search query."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _guess_lang(self, transcript: str) -> str:
        """Fast script-based language guess — no API call needed."""
        if re.search(r"[\u0980-\u09FF]", transcript):  # Bengali
            return "bn"
        if re.search(r"[\u0A80-\u0AFF]", transcript):  # Gujarati
            return "gu"
        if re.search(r"[\u0900-\u097F]", transcript):  # Devanagari (Hindi/Marathi)
            return "hi"  # treat as Hindi; Marathi shares script
        return "en"

    def analyse(self, transcript: str, locked_lang: str | None = None) -> Analysis:
        """Detect the language of this message, using a button choice as fallback."""
        guess_lang = self._guess_lang(transcript)
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": self.PROMPT},
                 {"role": "user", "content": transcript}],
                json_mode=True,
            )
            d = json.loads(raw)
            detected = d.get("language")
            # Prefer the language detected from this message so every reply
            # stays in the same language as the current input. A button
            # selection is only a fallback when detection is unavailable.
            lang = (detected if detected in self.VALID_LANGS
                    else (guess_lang or locked_lang or "en"))
            return Analysis(bool(d.get("about_scheme")), lang,
                            str(d.get("query_en") or transcript))
        except Exception as e:
            print(f"[Analyzer ERROR] {e!r}")
            # fail open: a hiccup in the AI service must not tell a real
            # question that it is off-topic
            return Analysis(True, guess_lang, transcript)


class Answerer:
    NOT_FOUND = "I_DONT_KNOW"
    LANG_NAMES = {
        "hi": "Hindi (Devanagari script)",
        "en": "simple English",
        "mr": "Marathi (Devanagari script)",
        "gu": "Gujarati (Gujarati script)",
        "bn": "Bengali (Bengali script)",
    }
    SYSTEM = (
        "You are a voice assistant for the Ayushman Bharat PM-JAY scheme. "
        "Your answer will be read aloud to the user. "
        "Use ONLY the CONTEXT. Never use outside knowledge, never guess, and "
        "never add examples, groups or numbers that are not in the CONTEXT. "
        f"If the CONTEXT does not answer the question, reply with exactly: {NOT_FOUND}. "
        "Be SPECIFIC: answer exactly what the user asked and nothing more. "
        "Do not list every case or category (for example do not explain rural "
        "and urban and every group) unless the user asked for all of them. "
        "If the answer truly depends on a detail the user did not give, give "
        "the most direct answer you can and then ask ONE short follow-up "
        "question. "
        "Style: 2 to 3 short spoken sentences, under 50 words, no bullet "
        "points, no markdown, no emojis, no symbols. Write numbers the way "
        "they are spoken. Reply in the requested language."
    )

    def __init__(self, llm: LLMClient, retriever: Retriever):
        self.llm = llm
        self.retriever = retriever

    def answer(self, transcript: str, query_en: str, previous_query: str,
               lang: str) -> str | None:
        # previous query is included so follow-ups ("and for old people?") work
        search_query = f"{previous_query} {query_en}".strip()
        context = "\n\n".join(self.retriever.search(search_query, k=3))
        reply = self.llm.chat(
            [{"role": "system", "content": self.SYSTEM},
             {"role": "user",
              "content": (f"CONTEXT:\n{context}\n\n"
                          f"QUESTION (user's own words): {transcript}\n\n"
                          f"Reply language: {self.LANG_NAMES.get(lang, 'simple English')}")}],
            temperature=0.0,
        )
        if self.NOT_FOUND in reply:
            return None
        return re.sub(r"[*_#`>]", "", reply).strip()  # clean symbols for TTS


# --------------------------------------------------------------------------
# Channels (WhatsApp or terminal)
# --------------------------------------------------------------------------
class Channel(ABC):
    @abstractmethod
    def send_text(self, to: str, body: str) -> None: ...

    @abstractmethod
    def send_audio(self, to: str, path: Path, spoken_text: str = "") -> None: ...

    @abstractmethod
    def send_video(self, to: str, path: Path, caption: str = "") -> None: ...

    @abstractmethod
    def send_buttons(self, to: str, body: str,
                     buttons: list[tuple[str, str]]) -> None: ...

    @abstractmethod
    def send_link_button(self, to: str, text: str, label: str, url: str) -> None: ...

    @abstractmethod
    def download_media(self, media_id: str, dest: Path) -> Path: ...


class ConsoleChannel(Channel):
    def __init__(self):
        OUT_DIR.mkdir(exist_ok=True)
        self.n = 0

    def send_text(self, to, body):
        print(f"\nBot (text): {body}")

    def send_audio(self, to, path, spoken_text=""):
        self.n += 1
        saved = OUT_DIR / f"reply_{self.n}.ogg"
        shutil.copy(path, saved)
        print(f"\nBot (voice, saved {saved}): {spoken_text}")

    def send_video(self, to, path, caption=""):
        print(f"\nBot: [VIDEO would be sent: {path.name}] {caption}")

    def send_buttons(self, to, body, buttons):
        labels = "  ".join(f"[{title}]" for _, title in buttons)
        print(f"\nBot (buttons): {body}  {labels}")

    def send_link_button(self, to, text, label, url):
        print(f"\nBot (link button): {text}  [{label} -> {url}]")

    def download_media(self, media_id, dest):
        raise NotImplementedError("No media download in CLI mode")


class WhatsAppChannel(Channel):
    GRAPH = "https://graph.facebook.com/v25.0"

    def __init__(self, token: str, phone_number_id: str):
        self.phone_number_id = phone_number_id
        self.http = httpx.Client(
            headers={"Authorization": f"Bearer {token}"},
            timeout=120, follow_redirects=True)
        self._media_cache: dict[str, str] = {}

    def _post_message(self, payload: dict) -> None:
        payload["messaging_product"] = "whatsapp"
        r = self.http.post(f"{self.GRAPH}/{self.phone_number_id}/messages",
                           json=payload)
        if r.status_code >= 400:
            print(f"[WhatsApp ERROR] {r.status_code}: {r.text}")

    def _upload(self, path: Path, mime: str, cache: bool = False) -> str:
        key = str(path)
        if cache and key in self._media_cache:
            return self._media_cache[key]
        with open(path, "rb") as f:
            r = self.http.post(
                f"{self.GRAPH}/{self.phone_number_id}/media",
                data={"messaging_product": "whatsapp", "type": mime},
                files={"file": (path.name, f, mime)},
            )
        if r.status_code >= 400:
            print(f"[Upload ERROR] {r.status_code}: {r.text}")
            r.raise_for_status()
        media_id = r.json()["id"]
        if cache:
            self._media_cache[key] = media_id
        return media_id

    def download_media(self, media_id: str, dest: Path) -> Path:
        meta = self.http.get(f"{self.GRAPH}/{media_id}")
        meta.raise_for_status()
        r = self.http.get(meta.json()["url"])
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest

    def send_text(self, to, body):
        self._post_message({"to": to, "type": "text", "text": {"body": body}})

    def send_audio(self, to, path, spoken_text=""):
        media_id = self._upload(path, "audio/ogg")
        self._post_message({"to": to, "type": "audio", "audio": {"id": media_id}})

    def send_buttons(self, to, body, buttons):
        # WhatsApp allows at most 3 reply buttons, titles up to 20 characters
        self._post_message({
            "to": to, "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": body},
                "action": {"buttons": [
                    {"type": "reply", "reply": {"id": bid, "title": title[:20]}}
                    for bid, title in buttons[:3]
                ]},
            },
        })

    def send_link_button(self, to, text, label, url):
        # one button that opens a website
        self._post_message({
            "to": to, "type": "interactive",
            "interactive": {
                "type": "cta_url",
                "body": {"text": text},
                "action": {"name": "cta_url",
                           "parameters": {"display_text": label[:20], "url": url}},
            },
        })

    def send_video(self, to, path, caption=""):
        if not path.exists():
            self.send_text(to, f"(Intro video not found on server: {path.name})")
            return
        try:
            media_id = self._upload(path, "video/mp4", cache=True)
        except Exception as e:
            print(f"[Video ERROR] {e!r}")
            self.send_text(to, "(Sorry, the video couldn't be sent right now.)")
            return
        self._post_message({"to": to, "type": "video",
                            "video": {"id": media_id, "caption": caption}})


# --------------------------------------------------------------------------
# Bot logic
# --------------------------------------------------------------------------
@dataclass
class Session:
    # Stage 0: greeting not yet sent
    # Stage 1: greeting sent, waiting for language choice or first question
    # Stage 2: language chosen, video not yet sent
    # Stage 3: video sent, in Q&A mode
    stage: int = 0
    lang: str = "hi"          # current language (may be overridden by button)
    lang_locked: bool = False  # True once the user explicitly picked a language
    last_query: str = ""


class VoiceBot:
    # ------------------------------------------------------------------ texts
    # All fixed spoken lines keyed by (message_key, language_code).
    # Add a key for every language the bot supports.
    TEXTS = {
        # ---- greeting (sent as text alongside the pre-recorded audio) ------
        "greeting_text": {
            "hi": ("नमस्ते! मैं Sahayak हूँ 🙏\n"
                   "आयुष्मान भारत PM-JAY के बारे में आपकी सहायता कर सकती हूँ।\n"
                   "आप वॉइस मैसेज या टेक्स्ट में अपना सवाल पूछ सकते हैं।\n\n"
                   "अपनी भाषा चुनिए 👇"),
            "en": ("Hello! I am Sahayak 🙏\n"
                   "I can help you with Ayushman Bharat PM-JAY.\n"
                   "You can ask your question by voice or text.\n\n"
                   "Choose your language 👇"),
            "mr": ("नमस्कार! मी Sahayak आहे 🙏\n"
                   "मी तुम्हाला आयुष्मान भारत PM-JAY बद्दल मदत करू शकते.\n"
                   "तुम्ही आवाज किंवा मजकुरातून प्रश्न विचारू शकता.\n\n"
                   "तुमची भाषा निवडा 👇"),
            "gu": ("નમસ્તે! હું Sahayak છું 🙏\n"
                   "હું તમને આયુષ્માન ભારત PM-JAY વિશે મદદ કરી શકું છું.\n"
                   "તમે અવાજ અથવા ટેક્સ્ટમાં તમારો પ્રશ્ન પૂછી શકો છો.\n\n"
                   "તમારી ભાષા પસંદ કરો 👇"),
            "bn": ("নমস্কার! আমি Sahayak 🙏\n"
                   "আমি আপনাকে আয়ুষ্মান ভারত PM-JAY সম্পর্কে সাহায্য করতে পারি।\n"
                   "আপনি ভয়েস বা টেক্সটে আপনার প্রশ্ন করতে পারেন।\n\n"
                   "আপনার ভাষা বেছে নিন 👇"),
        },
        # ---- after language chosen ----------------------------------------
        "lang_ack": {
            "hi": "बढ़िया! आप हिंदी में पूछ सकते हैं। बताइए, मैं क्या मदद कर सकती हूँ?",
            "en": "Great! You can ask me in English. How can I help you?",
            "mr": "छान! तुम्ही मराठीत विचारू शकता. मी कशी मदत करू?",
            "gu": "સરસ! તમે ગુજરાતીમાં પૂછી શકો છો. હું કેવી રીતે મદદ કરી શકું?",
            "bn": "চমৎকার! আপনি বাংলায় জিজ্ঞেস করতে পারেন। আমি কীভাবে সাহায্য করতে পারি?",
        },
        # ---- first info about Ayushman Bharat ----------------------------
        "confirm": {
            "hi": "आप आयुष्मान भारत PM-JAY के बारे में पूछ रहे हैं। यह भारत की सबसे बड़ी स्वास्थ्य बीमा योजना है। इसे समझाने के लिए यह छोटा वीडियो देखिए।",
            "en": "You are asking about Ayushman Bharat PM-JAY — India's largest health insurance scheme. Please watch this short video that explains it.",
            "mr": "तुम्ही आयुष्मान भारत PM-JAY बद्दल विचारत आहात — भारतातील सर्वात मोठी आरोग्य विमा योजना. हा छोटा व्हिडिओ पहा.",
            "gu": "તમે આયુષ્માન ભારત PM-JAY વિશે પૂछી રہा છો — ભારતની સૌથી મોટી આરોગ્ય વીમા યોજना. આ ટૂંકી વिडियो जुओ.",
            "bn": "আপনি আয়ুষ্মান ভারত PM-JAY সম্পর্কে জিজ্ঞেস করছেন — ভারতের সবচেয়ে বড় স্বাস্থ্য বিমা প্রকল্প। এই ছোট ভিডিওটি দেখুন।",
        },
        # ---- after video: invite questions --------------------------------
        "ask_doubts": {
            "hi": "वीडियो देखने के बाद अगर कोई सवाल हो, तो वॉइस मैसेज या टेक्स्ट में पूछिए। पात्रता जाँचने के लिए नीचे दिया बटन दबाएं।",
            "en": "After the video, ask me anything by voice or text. Tap the button below to check your eligibility on our website.",
            "mr": "व्हिडिओ नंतर कोणताही प्रश्न आवाज संदेश किंवा मजकुराने विचारा. पात्रता तपासण्यासाठी खालील बटण दाबा.",
            "gu": "વीडियो પછी, અવाज संदेश या ટेক्स्ट माँ कंईपण पूछो. पात्रता तपासवा माटे नीचे बटण दबावो.",
            "bn": "ভিডিওর পরে, ভয়েস বার্তা বা টেক্সটে যেকোনো প্রশ্ন জিজ্ঞেস করুন। যোগ্যতা পরীক্ষা করতে নিচের বোতাম টিপুন।",
        },
        # ---- off-topic ---------------------------------------------------
        "off_topic": {
            "hi": "मैं सिर्फ़ आयुष्मान भारत PM-JAY योजना की जानकारी दे सकती हूँ। कृपया इसी के बारे में पूछिए।",
            "en": "I can only help with the Ayushman Bharat PM-JAY scheme. Please ask about it.",
            "mr": "मी फक्त आयुष्मान भारत PM-JAY योजनेची माहिती देऊ शकतो. कृपया याबद्दलच विचारा.",
            "gu": "હું ফقત आयुष्मान ভারত PM-JAY যোজনার তথ্য দিতে পারি। অনুগ্রহ করে এটি সম্পর্কে জিজ্ঞেস করুন।",
            "bn": "আমি শুধুমাত্র আয়ুষ্মান ভারত PM-JAY প্রকল্পের তথ্য দিতে পারি। অনুগ্রহ করে এটি সম্পর্কে জিজ্ঞেস করুন।",
        },
        # ---- RAG fallback -----------------------------------------------
        "fallback": {
            "hi": "माफ़ कीजिए, यह जानकारी मेरे पास नहीं है। आधिकारिक वेबसाइट देखिए या हेल्पलाइन एक, चार, पाँच, पाँच, पाँच पर कॉल कीजिए।",
            "en": "Sorry, I do not have this detail. Please check the official website or call helpline one four five five five.",
            "mr": "माफ करा, ही माहिती माझ्याकडे नाही. अधिकृत वेबसाइट पहा किंवा हेल्पलाइन एक चार पाच पाच पाच वर कॉल करा.",
            "gu": "માફ કरો, мारী пास आ माहिती नथी. अधिकृत वेबसाइट जुओ या हेल्पलाइन एक चार पांच पांच पांच पर कॉल करो.",
            "bn": "দুঃখিত, এই তথ্য আমার কাছে নেই। অফিসিয়াল ওয়েবসাইট দেখুন বা হেল্পলাইন এক চার পাঁচ পাঁচ পাঁচ-এ ফোন করুন।",
        },
        # ---- unclear audio ----------------------------------------------
        "unclear": {
            "hi": "माफ़ कीजिए, मैं आपकी बात समझ नहीं पाई। कृपया फिर से बोलकर भेजिए।",
            "en": "Sorry, I could not understand that. Please send your message again.",
            "mr": "माफ करा, मला तुमचे म्हणणे समजले नाही. कृपया पुन्हा पाठवा.",
            "gu": "माफ करशो, मने तमारी वात समजाई नहि. कृपया फरी मोकलो.",
            "bn": "দুঃখিত, আমি বুঝতে পারিনি। অনুগ্রহ করে আবার পাঠান।",
        },
    }

    # WhatsApp button label (max 20 chars) + hidden question per topic
    TOPICS = {
        "topic_eligibility": {
            "title": {"hi": "पात्रता", "en": "Eligibility",
                      "mr": "पात्रता", "gu": "પात્રতা", "bn": "যোগ্যতা"},
            "question": {
                "hi": "आयुष्मान भारत के लिए कौन पात्र है?",
                "en": "Who is eligible for Ayushman Bharat?",
                "mr": "आयुष्मान भारतसाठी कोण पात्र आहे?",
                "gu": "આยुष्मान ভারতের জন্য কে যোগ্য?",
                "bn": "আয়ুষ্মান ভারতের জন্য কে যোগ্য?",
            },
        },
        "topic_coverage": {
            "title": {"hi": "कवरेज", "en": "Coverage",
                      "mr": "कव्हरेज", "gu": "કवरेज", "bn": "কভারেজ"},
            "question": {
                "hi": "आयुष्मान भारत में कितना कवर मिलता है?",
                "en": "How much cover does Ayushman Bharat give?",
                "mr": "आयुष्मान भारतमध्ये किती कव्हर मिळते?",
                "gu": "आयुष्मान भारतमाँ केटलुं कवर मळे छे?",
                "bn": "আয়ুষ্মান ভারতে কত কভার পাওয়া যায়?",
            },
        },
        "topic_card": {
            "title": {"hi": "कार्ड बनवाएं", "en": "Get the card",
                      "mr": "कार्ड बनवा", "gu": "કার्ड बनावो", "bn": "কার্ড বানান"},
            "question": {
                "hi": "आयुष्मान कार्ड कैसे बनवाएं?",
                "en": "How do I get the Ayushman card?",
                "mr": "आयुष्मान कार्ड कसे बनवायचे?",
                "gu": "आयुष्मान कार्ड केवी रीते बनावशो?",
                "bn": "আয়ুষ্মান কার্ড কীভাবে বানাবেন?",
            },
        },
    }

    # Language selection buttons shown in the greeting
    LANGUAGES = {
        "lang_hi": "हिंदी",
        "lang_mr": "मराठी",
        "lang_gu": "ગુજરાતી",
        "lang_bn": "বাংলা",
        "lang_en": "English",
    }
    LANG_BTN_TO_CODE = {
        "lang_hi": "hi",
        "lang_mr": "mr",
        "lang_gu": "gu",
        "lang_bn": "bn",
        "lang_en": "en",
    }

    BUTTON_BODY = {
        "hi": "या नीचे से कोई विषय चुनिए 👇",
        "en": "Or tap a topic below 👇",
        "mr": "किंवा खालून एक विषय निवडा 👇",
        "gu": "या नीचेथी एक विषय पसंद करो 👇",
        "bn": "অথবা নীচে থেকে একটি বিষয় বেছে নিন 👇",
    }

    # Eligibility website URL (override via env var ELIGIBILITY_WEBSITE_URL)
    ELIGIBILITY_URL = os.environ.get(
        "ELIGIBILITY_WEBSITE_URL", "https://sambhavjain-js.github.io/sahayak-elg/"
    )
    ELIGIBILITY_TEXT = {
        "hi": "अपनी पात्रता जाँचें — आयुष्मान भारत",
        "en": "Check your eligibility — Ayushman Bharat",
        "mr": "आपली पात्रता तपासा — आयुष्मान भारत",
        "gu": "तमारी पात्रता चकासो — आयुष्मान भारत",
        "bn": "আপনার যোগ্যতা পরীক্ষা করুন — আয়ুষ্মান ভারত",
    }
    ELIGIBILITY_LABEL = {
        "hi": "पात्रता जाँचें",
        "en": "Check eligibility",
        "mr": "पात्रता तपासा",
        "gu": "পাত্রতা তপাসো",
        "bn": "যোগ্যতা পরীক্ষা করুন",
    }

    def __init__(self, channel: Channel, stt: STT, tts: TTS,
                 analyzer: QueryAnalyzer, answerer: Answerer,
                 video_path: Path = VIDEO_FILE):
        self.channel = channel
        self.stt = stt
        self.tts = tts
        self.analyzer = analyzer
        self.answerer = answerer
        self.video_path = video_path
        self.sessions: dict[str, Session] = {}
        # Fixed TTS lines are synthesised once and cached in memory
        self._audio_cache: dict[tuple[str, str], bytes] = {}

    # ---- helpers -------------------------------------------------------
    def _speak(self, user_id: str, text: str, lang: str, tmp: Path,
               cache: bool = False) -> None:
        """Synthesise text to OGG and send as a voice note."""
        try:
            key = (text, lang)
            ogg = tmp / f"{uuid.uuid4().hex}.ogg"
            if cache and key in self._audio_cache:
                ogg.write_bytes(self._audio_cache[key])
            else:
                ogg = self.tts.speak(text, lang, ogg)
                if cache:
                    self._audio_cache[key] = ogg.read_bytes()
            self.channel.send_audio(user_id, ogg, text)
        except Exception as e:  # never leave the user without an answer
            print(f"[TTS ERROR] {e!r}")
            self.channel.send_text(user_id, text)

    def _say(self, user_id: str, key: str, lang: str, tmp: Path) -> None:
        """Send a fixed spoken line (cached after first synthesis)."""
        text = self.TEXTS[key].get(lang) or self.TEXTS[key].get("hi") or ""
        self._speak(user_id, text, lang, tmp, cache=True)

    def _reply(self, user_id: str, text: str, lang: str, tmp: Path,
               voice: bool = False, cache: bool = False) -> None:
        """Send reply as voice note (if voice) or plain text."""
        if voice:
            self._speak(user_id, text, lang, tmp, cache=cache)
        else:
            self.channel.send_text(user_id, text)

    def _reply_fixed(self, user_id: str, key: str, lang: str, tmp: Path,
                     voice: bool = False) -> None:
        """Send a fixed line as voice or text based on input mode."""
        text = self.TEXTS[key].get(lang) or self.TEXTS[key].get("hi") or ""
        self._reply(user_id, text, lang, tmp, voice=voice, cache=voice)

    def _send_topic_buttons(self, user_id: str, lang: str) -> None:
        buttons = [(tid, t["title"].get(lang, t["title"]["en"]))
                   for tid, t in self.TOPICS.items()]
        self.channel.send_buttons(user_id, self.BUTTON_BODY.get(lang, self.BUTTON_BODY["en"]),
                                  buttons)

    def _send_eligibility_button(self, user_id: str, lang: str) -> None:
        self.channel.send_link_button(
            user_id,
            self.ELIGIBILITY_TEXT.get(lang, self.ELIGIBILITY_TEXT["en"]),
            self.ELIGIBILITY_LABEL.get(lang, self.ELIGIBILITY_LABEL["en"])[:20],
            self.ELIGIBILITY_URL,
        )

    def _send_fallback_website(self, user_id: str, lang: str) -> None:
        """Called when RAG has no answer: fallback voice + website button."""
        with tempfile.TemporaryDirectory() as td:
            self._say(user_id, "fallback", lang, Path(td))
        self._send_eligibility_button(user_id, lang)

    # ---- greeting (Stage 0 -> 1) ---------------------------------------
    # Supported input formats for the greeting audio file
    _GREETING_EXTS = (".ogg", ".mp3", ".mp4", ".m4a", ".wav", ".aac")
    _greeting_ogg_cache: Path | None = None   # converted OGG cached after first run

    def _resolve_greeting_audio(self) -> Path | None:
        """Find greeting.<ext> in the audio/ folder and return an OGG/Opus Path.
        Converts mp3/mp4/wav etc. to OGG on first call and caches the result.
        Returns None if no greeting file is found.
        """
        # Return cached result if already resolved
        if self._greeting_ogg_cache is not None:
            return self._greeting_ogg_cache

        folder = GREETING_AUDIO_DIR
        if not folder.exists():
            return None

        # Find the first greeting.* file in the audio/ folder
        src: Path | None = None
        for ext in self._GREETING_EXTS:
            candidate = folder / f"greeting{ext}"
            if candidate.exists():
                src = candidate
                break

        if src is None:
            return None

        # Already OGG — use as-is
        if src.suffix.lower() == ".ogg":
            self._greeting_ogg_cache = src
            print(f"[Greeting] Using audio: {src.name}")
            return src

        # Convert to OGG/Opus so WhatsApp shows it as a voice note
        ogg = folder / "greeting_converted.ogg"
        try:
            AudioConverter.to_ogg_opus(src, ogg)
            self._greeting_ogg_cache = ogg
            print(f"[Greeting] Converted {src.name} -> {ogg.name}")
            return ogg
        except Exception as e:
            print(f"[Greeting] Conversion failed ({e!r}); sending text only.")
            return None

    def _video_for_lang(self, lang: str) -> Path:
        """Return the language-specific intro video, falling back to the default."""
        candidate = VIDEO_FILES.get(lang)
        if candidate and candidate.exists():
            print(f"[Video] Using language-specific video: {candidate.name}")
            return candidate
        # Fallback: use the default video
        print(f"[Video] No video for lang='{lang}', using default: {VIDEO_FILE.name}")
        return VIDEO_FILE

    def _send_greeting(self, user_id: str, reply_voice: bool = False) -> None:
        """Send greeting text + language-choice buttons.
        If reply_voice, also send the pre-recorded greeting audio."""
        # 1. Pre-recorded audio (any format — auto-converted to OGG/Opus)
        if reply_voice:
            ogg = self._resolve_greeting_audio()
            if ogg:
                self.channel.send_audio(user_id, ogg,
                                        "नमस्ते! मैं आपकी क्या सहायता कर सकती हूँ?")
            else:
                print("[Greeting] No greeting audio file found in audio/ folder; "
                      "sending text only.")

        # 2. Keep the greeting text in the current input language.
        greeting = self.TEXTS["greeting_text"].get(
            self.sessions[user_id].lang,
            self.TEXTS["greeting_text"]["hi"],
        )
        if not reply_voice:
            self.channel.send_text(user_id, greeting)
 
        # 3. Language-choice buttons (WhatsApp supports max 3 reply buttons;
        #    we send two rounds of 3 and 2 — or use a list message if needed)
        lang_buttons = list(self.LANGUAGES.items())  # [(id, label), ...]
        # Send first 3, then remaining 2 as a second message
        self.channel.send_buttons(user_id, "भाषा / Language 👇", lang_buttons[:3])
        if len(lang_buttons) > 3:
            self.channel.send_buttons(user_id, "\u200b", lang_buttons[3:])

    # ---- button handler (language + topic buttons) ---------------------
    def handle_button(self, user_id: str, button_id: str) -> None:
        s = self.sessions.setdefault(user_id, Session())

        # Language-choice button
        if button_id in self.LANG_BTN_TO_CODE:
            lang = self.LANG_BTN_TO_CODE[button_id]
            s.lang = lang
            s.lang_locked = True
            s.stage = max(s.stage, 1)  # at least past greeting
            text = self.TEXTS["lang_ack"].get(lang) or self.TEXTS["lang_ack"].get("hi") or ""
            self.channel.send_text(user_id, text)
            return

        # Topic button (shown after the video)
        topic = self.TOPICS.get(button_id)
        if topic:
            question = topic["question"].get(s.lang, topic["question"]["en"])
            self.handle_transcript(user_id, question)

    # ---- entry points --------------------------------------------------
    def handle_voice(self, user_id: str, audio_path: Path) -> None:
        with tempfile.TemporaryDirectory() as td:
            wav = AudioConverter.to_wav(audio_path, Path(td) / "in.wav")
            transcript = self.stt.transcribe(wav)
        print(f"[STT] {transcript!r}")
        self.handle_transcript(user_id, transcript, reply_voice=True)

    # Words that are pure greetings with no question intent — these get only the greeting
    _GREETING_ONLY = {
        "hi", "hello", "hey", "hii", "helo", "hllo", "helo",
        "नमस्ते", "नमस्कार", "हैलो", "हेलो", "हेल्लो",
        "namaskar", "namasthe", "namaste",
    }

    def handle_transcript(self, user_id: str, transcript: str,
                          reply_voice: bool = False) -> None:
        s = self.sessions.setdefault(user_id, Session())
        lang = s.lang

        # ---- Stage 0: very first message -> send greeting ----------------
        if s.stage == 0:
            # Use the input script for the initial greeting. The analyzer
            # below still performs the complete language detection for the
            # actual answer.
            if not s.lang_locked and transcript.strip():
                s.lang = self.analyzer._guess_lang(transcript)
            s.stage = 1
            self._send_greeting(user_id, reply_voice=reply_voice)
            # If the first message is just "Hi" / "नमस्ते", stop here.
            # But if it's a real question, fall through and process it so
            # the user doesn't have to repeat themselves.
            if (not transcript.strip()
                    or transcript.strip().lower() in self._GREETING_ONLY):
                return
            # else: fall through to Stage 1 processing below

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            if not transcript.strip():
                self._reply_fixed(user_id, "unclear", lang, tmp, voice=reply_voice)
                return

            # Detect language (honour the locked language if chosen)
            a = self.analyzer.analyse(
                transcript,
                locked_lang=s.lang if s.lang_locked else None,
            )
            s.lang = a.language
            lang = a.language
            print(f"[Analysis] about_scheme={a.about_scheme} lang={lang} "
                  f"query={a.query_en!r}")

            # ---- Stage 1/2: first real question --------------------------
            if s.stage < 3:
                if not a.about_scheme:
                    self._reply_fixed(user_id, "off_topic", lang, tmp, voice=reply_voice)
                    return
                # Send intro
                s.stage = 3
                self._reply_fixed(user_id, "confirm", lang, tmp, voice=reply_voice)
                self.channel.send_video(user_id, self._video_for_lang(lang),
                                        "Ayushman Bharat PM-JAY")
                self._reply_fixed(user_id, "ask_doubts", lang, tmp, voice=reply_voice)
                # Eligibility website button
                self._send_eligibility_button(user_id, lang)
                # Quick-topic buttons
                self._send_topic_buttons(user_id, lang)
                return

            # ---- Stage 3: Q&A via RAG ------------------------------------
            reply = self.answerer.answer(transcript, a.query_en,
                                         s.last_query, lang)
            if reply is None:
                self._reply_fixed(user_id, "fallback", lang, tmp, voice=reply_voice)
                self._send_eligibility_button(user_id, lang)
            else:
                self._reply(user_id, reply, lang, tmp, voice=reply_voice)
            s.last_query = a.query_en


def build_bot(channel: Channel) -> VoiceBot:
    key = os.environ["MISTRAL_API_KEY"]
    llm = MistralClient(key)
    retriever = EmbeddingRetriever(KNOWLEDGE_FILE)

    stt: STT = VoxtralSTT(key)
    tts: TTS = EdgeTTS()
    sarvam_key = os.environ.get("SARVAM_API_KEY")
    if sarvam_key:  # Sarvam first, the free options stay as backup
        stt = FallbackSTT(SarvamSTT(sarvam_key), stt)
        tts_model = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
        default_speaker = "anand" if tts_model == "bulbul:v3" else "anushka"
        tts = FallbackTTS(
            SarvamTTS(sarvam_key, model=tts_model,
                      speaker=os.environ.get("SARVAM_SPEAKER", default_speaker)),
            tts)
        print("[Audio] Sarvam (backup: Voxtral + edge-tts)")
    else:
        print("[Audio] Voxtral + edge-tts")
    return VoiceBot(channel, stt, tts, QueryAnalyzer(llm), Answerer(llm, retriever))


# --------------------------------------------------------------------------
# FastAPI webhook (WhatsApp Cloud API)
# --------------------------------------------------------------------------
_bot: VoiceBot | None = None
_seen_ids: set[str] = set()  # Meta retries webhooks; avoid double replies


def get_bot() -> VoiceBot:
    global _bot
    if _bot is None:
        channel = WhatsAppChannel(os.environ["WHATSAPP_TOKEN"],
                                  os.environ["PHONE_NUMBER_ID"])
        _bot = build_bot(channel)
    return _bot


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_bot()  # build embeddings once at startup
    yield


app = FastAPI(lifespan=lifespan)


def process_message(msg: dict) -> None:
    bot = get_bot()
    user = msg["from"]
    try:
        if msg.get("type") == "audio":
            with tempfile.TemporaryDirectory() as td:
                src = bot.channel.download_media(msg["audio"]["id"],
                                                 Path(td) / "in.ogg")
                bot.handle_voice(user, src)
        elif msg.get("type") == "interactive":
            reply = msg["interactive"].get("button_reply") or {}
            bot.handle_button(user, reply.get("id", ""))
        else:
            # "Hii" / any text should run the same flow as a transcribed voice note
            text = (msg.get("text") or {}).get("body") or ""
            bot.handle_transcript(user, text)
    except Exception as e:  # keep the demo alive on any error
        print(f"[ERROR] {e!r}")
        bot.channel.send_text(
            user, "Sorry, something went wrong. Please send your message again.")


@app.get("/webhook")
async def verify(request: Request):
    q = request.query_params
    if (q.get("hub.mode") == "subscribe"
            and q.get("hub.verify_token") == os.environ.get("VERIFY_TOKEN")):
        return PlainTextResponse(q.get("hub.challenge", ""))
    return PlainTextResponse("Forbidden", status_code=403)


@app.post("/webhook")
async def incoming(request: Request, background: BackgroundTasks):
    data = await request.json()
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            for msg in change.get("value", {}).get("messages", []):
                if msg["id"] in _seen_ids:
                    continue
                _seen_ids.add(msg["id"])
                background.add_task(process_message, msg)
    return {"ok": True}  # reply 200 fast; real work happens in background


# --------------------------------------------------------------------------
# Terminal test mode (no WhatsApp needed)
# --------------------------------------------------------------------------
def run_cli() -> None:
    bot = build_bot(ConsoleChannel())
    print("CLI test mode. Type what the user says (as text), or @path/to/voice.ogg "
          "to test a real audio file. Ctrl+C to quit.")
    while True:
        text = input("\nYou: ").strip()
        if not text:
            continue
        if text.startswith("@"):
            bot.handle_voice("cli-user", Path(text[1:].strip().strip('"')))
        else:
            bot.handle_transcript("cli-user", text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="test in terminal")
    if parser.parse_args().cli:
        run_cli()
    else:
        print("Run the WhatsApp server with: uvicorn bot:app --port 8000")