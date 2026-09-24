# Voice

Implemented as separate, swappable components (`src/elara/voice/`). The wake word, the VAD
and speech recognition are **separate**; Whisper is never used as a wake-word detector.

```
microphone* → [WakeWordDetector "Hey ELARA"] → EnergyVAD + UtteranceSegmenter → SpeechToText
           → TranscriptQualityGate → Assistant → TextToSpeech → speaker*
```
\* Microphone capture and audio playback are **not implemented yet**. The pipeline accepts
16 kHz mono 16-bit PCM frames of 30 ms from any source (`VoicePipeline.feed(frame)`).

| Component | Implementation | Status |
|-----------|----------------|--------|
| VAD | `EnergyVAD`: adaptive noise floor, no dependencies | working, tested |
| Segmentation | `UtteranceSegmenter`: start on speech, end after 700 ms silence, drops clicks, 30 s cap | working, tested |
| Quality gate | `TranscriptQualityGate`: empty, too short, `no_speech_prob`, low confidence, known Whisper hallucinations ("Thank you."), repetition | working, tested |
| STT | `FasterWhisperSTT` (lazy import, `pip install faster-whisper`) | adapter; not exercised in CI |
| Wake word | `OpenWakeWordDetector` (openWakeWord + a user-provided model file) | adapter; **no "hey elara" model exists yet**; one must be trained |
| TTS | `EspeakTTS` (espeak-ng voices `az`, `en-us`, `tr`); registry `TTS_PROVIDERS` | working when espeak-ng is installed |

Configuration:
- `ELARA_TTS_PROVIDER` (default `espeak`)
- `ELARA_TTS_VOICES` (JSON, default `{"az":"az","en":"en-us","tr":"tr"}`)
- `ELARA_STT_PROVIDER`, `ELARA_STT_MODEL`

Priority for better voices: Azerbaijani first, then English, then Turkish.
espeak-ng sounds robotic, but it is the only fully offline engine with Azerbaijani out of the box.
Neural engines (e.g. Piper, or cloud TTS) can be added as `TTS_PROVIDERS` entries.

Next steps: a `sounddevice` capture/playback loop, a trained "hey elara" openWakeWord model,
and an `elara voice` command.
