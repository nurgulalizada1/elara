# Voice benchmark harness (isolated, not production)

An engineering tool for measuring microphone capture and CPU speech-to-text on this machine.
The `elara` package does not import it, and it is not part of the main test suite.

- Recordings are saved to `bench/voice/recordings/` (git-ignored). They are never uploaded
  or played back automatically.
- The only network access is the one-time Whisper model download from Hugging Face on first
  use of a model. No audio is sent anywhere.
- Speech-to-text runs on the CPU with `compute_type="int8"`, 8 threads by default, and the
  language forced to `az`.

Run everything from `elara-v2/` with the venv active (`. .venv/bin/activate`).

```bash
# 0. install (once)
pip install sounddevice numpy faster-whisper
ldconfig -p | grep -q libportaudio.so.2 || sudo apt install libportaudio2

# 1. devices + microphone sanity test (no model download)
python bench/voice/voice_bench.py devices
python bench/voice/voice_bench.py mic-check --seconds 5            # default input
python bench/voice/voice_bench.py mic-check --seconds 5 --device 3 # a specific input index

# 2. record ONE benchmark clip, reuse it for every model
python bench/voice/voice_bench.py record --seconds 8 --out bench/voice/recordings/az-01.wav

# 3. transcribe (first run of each model downloads it: small ~0.5 GB, medium ~1.5 GB)
python bench/voice/voice_bench.py transcribe bench/voice/recordings/az-01.wav --model small  --threads 8
python bench/voice/voice_bench.py transcribe bench/voice/recordings/az-01.wav --model medium --threads 8
# add --json for machine-readable output, --local-files-only to forbid downloads

# 4. az vs en: identical model/config on both clips (model loaded once)
python bench/voice/voice_bench.py record --seconds 8 --out bench/voice/recordings/en-01.wav
python bench/voice/voice_bench.py compare \
    --az bench/voice/recordings/az-01.wav --az-ref "<exactly what you said in Azerbaijani>" \
    --en bench/voice/recordings/en-01.wav --en-ref "<exactly what you said in English>" \
    --model small --threads 8 --beam-size 5

# 5. fixed corpus: 3 az + 3 en clips, same model/settings, pooled WER/CER per language
#    read the sentences in corpus.example.tsv exactly; record each for 10 s
python bench/voice/voice_bench.py corpus bench/voice/corpus.example.tsv \
    --model small --threads 8 --beam-size 5
# TSV format: language<TAB>wav (relative to the TSV)<TAB>exact reference text

# tests for this harness only
python -m pytest -q bench/voice
```
