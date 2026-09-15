---
title: Voice Mode
description: Talk to OpenJarvis hands-free from the browser or desktop app — wake word in, spoken reply out
---

# Voice Mode

The chat UI can run hands-free: leave the microphone open, get its attention
with a wake word, and hear the reply read back. This is separate from the
push-to-talk mic button, which records one message at a time.

## Turn it on

1. Make sure a speech-to-text backend is available — **Settings → Speech →
   Backend status** should say *Available*. The `speech` extra installs
   [faster-whisper](https://github.com/SYSTRAN/faster-whisper), which runs
   locally:

   ```bash
   uv sync --extra speech
   ```

2. Click the waveform button next to the microphone in the chat input, or
   toggle **Settings → Speech → Voice mode (hands-free)**. The browser will ask
   for microphone permission the first time.

The status line under the input shows what it is doing (`Listening — say
"jarvis"`, `Transcribing…`, `Thinking…`, `Speaking…`) and the last thing it
heard. The setting persists across reloads.

## Talking to it

Speech is split into utterances on silence and each one is transcribed.
Nothing is sent to the model unless the utterance contains the wake word
(default `jarvis`, change it under **Settings → Speech → Wake word**):

| You say | What happens |
|---------|--------------|
| "yo jarvis, what's the weather in Denver" | Sends *what's the weather in Denver* |
| "jarvis" | It answers "Yeah?" and treats your next sentence (within 8 seconds) as the request |
| anything without the wake word | Ignored — including Whisper's habit of hearing "Thank you." in silence |

Common mis-hearings of the default wake word (*Travis*, *Jarvas*) are accepted.
While the assistant is generating or speaking, the microphone input is ignored
so it does not transcribe its own voice.

## Hearing the reply

When a response finishes streaming it is read aloud. Markdown formatting, code
blocks, links, and `[1]`-style citations are stripped before synthesis, and
very long replies are trimmed to the first couple of paragraphs.

Which voice you hear depends on what the server has installed:

| Backend | How to get it | Notes |
|---------|---------------|-------|
| **Kokoro** (default) | `uv sync --extra voice` | Local, open-source, no key. Voice is `speech.voice_id` — British male `bm_george` by default. |
| **OpenAI TTS** | `OPENAI_API_KEY` + `speech.tts_backend = "openai_tts"` | Cloud. Audio leaves your machine. |
| **Cartesia** | `CARTESIA_API_KEY` + `speech.tts_backend = "cartesia"` | Cloud. |
| **Browser voice** | nothing | Used automatically when the server reports no TTS backend (`Settings → Speech → Text-to-speech` shows *Browser fallback*). Quality depends on the OS; Firefox on Linux needs `speech-dispatcher` installed to produce any sound. |

Voice settings live in the `[speech]` section of `config.toml` — see
[Configuration](../getting-started/configuration.md#speech-speech-to-text-and-voice).

## API

Voice mode is built on three routes of the local server, which you can also
call directly:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/speech/transcribe` | `POST` (multipart `file`) | Audio → `{ "text", "language", "confidence", "duration_seconds" }` |
| `/v1/speech/synthesize` | `POST` `{ "text", "voice_id"?, "speed"? }` | Text → audio (`audio/wav` from Kokoro). Returns **501** when no TTS backend is installed, which is the UI's cue to use the browser voice. |
| `/v1/speech/health` | `GET` | `{ "available", "backend", "tts": { "available", "backend" } }` |

```bash
curl -X POST http://localhost:8000/v1/speech/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Good morning."}' --output reply.wav
```

## Troubleshooting

- **Status stays on "Starting microphone…"** — the browser denied mic access.
  Check the site permission (padlock icon) and reload.
- **It never reacts** — check the *heard:* text in the status line. If words
  come through but not the wake word, try a different one, or say it a beat
  before the request.
- **It reacts but says nothing** — the server has no TTS backend and the
  browser voice is silent. Install Kokoro (`uv sync --extra voice`) and
  restart `jarvis serve`.
- **It transcribes its own reply** — enable echo cancellation on your input
  device, or use headphones. The UI already mutes input while speaking, but
  audio that arrives after the speech ends can still be picked up.
