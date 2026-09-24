# Voice (English, development mode)

Voice is a **channel** on top of `ElaraCore`, like the CLI and HTTP API. It turns one spoken
utterance into text and submits it as `CoreRequest(channel="voice")`. Routing, tools, memory,
confirmations, security and LLM use all stay in the core.

```
microphone (PipeWire/PortAudio, 16 kHz mono int16, in memory)
  → energy VAD: one utterance, pre-roll, end on silence, hard cap
  → faster-whisper small, CPU, int8, 8 threads, 1 worker, beam 5, language forced "en"
  → quality gate (deterministic; accepts "stop", "cancel", "what time is it")
  → ElaraCore.process(CoreRequest(text, channel="voice"))  → CoreResult
  → optional speech output (espeak-ng fallback, English)
```

Code lives in `src/elara/voice/`:
- `config.py`: `VoiceConfig`, environment prefix `ELARA_VOICE_`.
- `microphone.py`: capture of one bounded utterance.
- `stt.py`: `FasterWhisperSTT`; the model is loaded once and reused.
- `quality.py`: `TranscriptQualityGate`.
- `tts.py`: `Speaker` interface plus `EspeakSpeaker`.
- `session.py`: `VoiceSession`, which connects voice to the core.
- `cli/voice_cmd.py`: `elara voice`.

## Decisions

- **Production voice input is English only.** On this laptop's CPU, `small` scored 15.3% WER
  on English and 59.6% on Azerbaijani, which was also never detected as Azerbaijani (0/3; see
  `bench/voice`). `VoiceConfig.language` only accepts `en`, and the STT adapter refuses any
  other language. Azerbaijani STT stays experimental in `bench/voice/`. Typed Azerbaijani in
  the text interface and core is unchanged.
- **Local only.** Audio never leaves the machine. It is held in memory for one utterance,
  dropped after transcription, and never logged or written to disk. Only the accepted
  transcript reaches `ElaraCore`, and Claude only sees it if the core decides an LLM is needed.
- **Transcripts are untrusted user input.** They get the same injection defences, tool
  permissions, path jail and confirmation flow as typed text.
- **Not implemented yet:**
  - wake word ("Hey ELARA")
  - an always-listening daemon
  - Azerbaijani/Turkish voice
  - neural TTS (Piper/MMS)

## Hardware assumptions

The target is a CPU-only laptop: Ryzen 7 5700U (8 cores / 16 threads), 16 GB RAM, PipeWire.
Measured: `small` int8 runs at about RTF 0.3–0.5, using roughly 1 GB RAM. No CUDA/ROCm.

## Install

```bash
pip install -e ".[voice]"                  # sounddevice, numpy, faster-whisper
sudo apt install libportaudio2 espeak-ng   # PortAudio for capture; espeak-ng is optional
```

The first `elara voice` run downloads the Whisper `small` model (~0.5 GB) once. To forbid
downloads, set `ELARA_VOICE_LOCAL_FILES_ONLY=true`.

## Use

```bash
elara voice --list-devices          # * marks the default input
elara voice                         # one utterance: speak after "listening ..."
elara voice --turns 3 --no-speak    # up to 3 turns in one conversation, text output only
elara voice --device 4 --json       # a specific microphone, machine-readable output
```

Each turn prints the transcript, the gate decision (with a reason code), the `CoreResult`
(intent, resolver, tier, `used_llm`, tools) and timings: `capture_s`, `audio_s`, `stt_s`,
`rtf`, `core_s`, `speak_s`, `total_s`. If an action needs confirmation, say "yes"/"no" on the
next turn (`--turns 2`).

## Configuration (`ELARA_VOICE_*`)

| Variable | Default | Meaning |
|---|---|---|
| `INPUT_DEVICE` | system default | sounddevice index or name |
| `MODEL` / `COMPUTE_TYPE` | `small` / `int8` | Whisper model and quantization (CPU only) |
| `CPU_THREADS` / `BEAM_SIZE` | `8` / `5` | decoding settings |
| `MODEL_DIR`, `LOCAL_FILES_ONLY` | HF cache, `false` | model location / forbid download |
| `START_TIMEOUT_S` | `8` | how long to wait for speech to start |
| `MAX_UTTERANCE_S` | `12` | hard cap per utterance |
| `END_SILENCE_MS` | `800` | silence that ends an utterance |
| `MIN_SPEECH_MS` | `250` | shorter bursts are treated as noise |
| `VAD_MIN_RMS`, `VAD_RATIO` | `300`, `3.0` | energy-VAD sensitivity |
| `TTS_ENABLED`, `TTS_VOICE`, `TTS_RATE_WPM` | `true`, `en-us`, `170` | spoken answers |

## Known limitations

- **Azerbaijani voice input is not supported in production** (experimental, benchmark only).
- **Spoken arithmetic goes to the LLM.** "17 times 42" (words) doesn't match the core's local
  calculator patterns; symbols (`17 * 42`) do.
- **Energy VAD is basic.** Steady loud background noise can delay the end-of-speech detection
  (the hard cap still applies). Silero VAD is a later option.
- **The espeak-ng voice is robotic**; `Speaker` is replaceable.
- **Legacy `Settings` fields.** `ELARA_TTS_*` / `ELARA_STT_*` in the general settings belong to
  the older generic adapters; the voice channel uses `ELARA_VOICE_*` only.
