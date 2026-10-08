# P14 — Local-First Voice & Production Hardening

## Overview
P14 introduces an offline, local-first voice architecture for P-MAI (Multipurpose Agentic AI Assistant). It enables complete speech-to-text (STT) and text-to-speech (TTS) execution locally without requiring paid cloud API keys or external network connections, while preserving OpenAI and Groq cloud adapters as optional production providers.

Voice remains strictly an **I/O presentation layer**. The MAI brain (Qwen/Ollama/PostgreSQL) remains the sole authoritative authority for routing, planning, tool execution, memory, task state, clarification, recovery, and grounding.

---

## Architecture

```text
                                MAI Voice Layer
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                       STT                         TTS
                        │                           │
               ┌────────┼────────┐          ┌───────┴────────┐
               │        │        │          │                │
             mock     local    cloud       mock            local
                                 │                           │
                           ┌─────┴─────┐                   pyttsx3
                           │           │                  (SAPI5/
                         OpenAI       Groq                 eSpeak)
                                       │
                                (whisper-large)
```

In offline mode:
```text
Microphone
    ↓
Local Whisper STT (faster-whisper)
    ↓
Authoritative MAI Brain (Ollama/Qwen)
    ↓
Grounded Response Text
    ↓
Voice Presentation Formatter
    ↓
Local TTS (pyttsx3 / SAPI5)
    ↓
Speaker
```

---

## 1. Local STT Provider (`LocalWhisperSTTAdapter`)
- **Backend**: `faster-whisper` (CTranslate2-based) with automatic fallback to `openai-whisper`.
- **Default Model**: `"tiny"` (lightweight, ~75MB disk space, runs effortlessly on standard CPUs).
- **Audio Ingestion**: Canonical 16kHz, mono, 16-bit signed PCM or WAV containers.
- **Lazy Initialization**: Model weights are downloaded and cached only upon the first transcription request. Module imports and server startups remain instant.
- **Worker Offloading**: Executes CPU inference in thread pools via `asyncio.to_thread` to ensure the FastAPI / WebSocket event loop remains unblocked.
- **Streaming Frame Ingestion**: Accumulates continuous PCM chunks via `StreamingAudioAssembler` and produces partial and finalized transcripts.
- **Invariants**: Partial transcripts NEVER trigger planner execution, tool use, memory mutations, or task state updates.

---

## 2. Local TTS Provider (`LocalTTSAdapter`)
- **Backend**: `pyttsx3` wrapping platform-native offline speech engines:
  - Windows: SAPI5
  - macOS: NSSpeechSynthesizer
  - Linux: eSpeak
- **Audio Output**: Canonical 16kHz, mono, 16-bit signed PCM / WAV via `AudioNormalizer`.
- **Continuous PCM Streaming**: Synthesizes speech text and slices continuous raw PCM frames into bounded chunks (default: 0.5s chunks, 16,000 bytes per chunk) without corrupting stream headers.
- **Interruption / Barge-In**: Immediately terminates generator loop upon receiving `INTERRUPT` events, freeing memory and speaker buffers.

---

## 3. Canonical Audio Contract
P-MAI strictly enforces canonical audio across all providers (mock, local, and cloud):
- **Sample Rate**: 16,000 Hz
- **Channels**: 1 (Mono)
- **Sample Width**: 2 bytes (16-bit signed Little-Endian PCM)
- **Bitrate**: 32,000 bytes/second (256 kbps)
- **Normalization**: `AudioNormalizer` provides frame boundary alignment, arithmetic stereo-to-mono downmixing (`(L + R) // 2`), and linear interpolation resampling.

### Continuous Stream vs. Standalone File Contract
- **Continuous Realtime Stream**: WebSockets transmit raw PCM frames across sequential `RESPONSE_CHUNK` events. Downstream playback treats consecutive chunks as a single unbroken stream.
- **Standalone Audio Files**: REST `/voice/chat` and storage endpoints wrap raw PCM into valid RIFF WAV containers via `pcm_to_wav()`.
- **Assembler**: `StreamingAudioAssembler` strips WAV headers from incoming chunks if present, ensuring uninterrupted PCM continuity.

---

## 4. Environment Configuration & Switching

### A. Completely Offline Setup (Local-First)
```env
# Local Voice Configuration
VOICE_ENABLED=true
VOICE_REALTIME_ENABLED=true
VOICE_STT_PROVIDER=local
VOICE_TTS_PROVIDER=local
VOICE_STT_LOCAL_MODEL=tiny
VOICE_TTS_LOCAL_ENGINE=pyttsx3
VOICE_SAMPLE_RATE=16000
VOICE_AUDIO_FORMAT=audio/wav
```
*No cloud API keys required. Entire system operates with local Ollama, local PostgreSQL, and local voice engines.*

### B. Cloud Setup (Optional)
```env
# Cloud Voice Configuration
VOICE_STT_PROVIDER=groq
VOICE_TTS_PROVIDER=openai
VOICE_STT_API_KEY=gsk_...
VOICE_TTS_API_KEY=sk-...
VOICE_STT_BASE_URL=https://api.groq.com/openai/v1
VOICE_TTS_BASE_URL=https://api.openai.com/v1
```

### C. Testing Setup (Mock)
```env
VOICE_STT_PROVIDER=mock
VOICE_TTS_PROVIDER=mock
```

---

## 5. Hardware Requirements & Performance

| Component | Provider | Compute Type | RAM / VRAM | Avg Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Local STT** | `faster-whisper-tiny` | CPU (int8) | ~150 MB RAM | ~800–1200 ms |
| **Local TTS** | `pyttsx3 (SAPI5)` | CPU | ~30 MB RAM | ~350–450 ms |
| **Barge-In** | Event Bus | In-Memory | Minimal | **0.14 ms** |

---

## 6. Voice UX State Machine & Invariants

```text
 IDLE
  │
  ▼
LISTENING (Mic capture / VAD energy active)
  │
  ▼
TRANSCRIBING (Audio finalized / Local Whisper transcription)
  │
  ▼
PROCESSING (Authoritative MAI Brain reasoning / Tools)
  │
  ▼
SPEAKING (VoicePresentationFormatter -> Streaming Local TTS -> Playback)
  │
  ├── User Speaks (Barge-In) ──► Flush TTS Buffer ──► LISTENING
  │
  ▼
IDLE / LISTENING
```

### Stop vs Cancel Semantics
- **"Stop" / "Stop talking"**: Halts speech playback, flushes audio buffers, and sets `action: stop_speaking_only`. Does NOT cancel background tasks or P8 task states.
- **"Cancel this task"**: Forwarded directly to the MAI brain, invoking authoritative P8 task cancellation.

---

## 7. Troubleshooting

1. **`STTUnavailableError: Local Whisper STT requires 'faster-whisper'`**:
   - Install local Whisper: `pip install faster-whisper`
2. **`TTSUnavailableError: Local TTS requires 'pyttsx3'`**:
   - Install local TTS: `pip install pyttsx3 comtypes pywin32`
3. **Symlink Warnings on Windows HuggingFace Cache**:
   - Enable Windows Developer Mode or set environment variable: `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
