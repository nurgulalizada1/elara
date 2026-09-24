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

## Wake-word spike: "Hey ELARA" (`wakeword_bench.py`)

This is an experiment and is not wired into `elara voice`. It detects the phrase with
openWakeWord's frozen feature front end (`melspectrogram.onnx` + `embedding_model.onnx`, about
2.4 MB), run directly with `onnxruntime`. faster-whisper already installs onnxruntime, so there
are **no new Python packages**. There is no trained "Hey ELARA" model yet, so the spike uses
**few-shot enrollment**: you record the phrase about 5 times, and the live audio is compared
with those templates every 80 ms using DTW on the 96-dimensional embeddings. Whisper is never
used, and no audio leaves the machine. Enrollment stores embeddings only, not audio.

```bash
python bench/voice/wakeword_bench.py fetch-models        # once; sha256-pinned, into bench/voice/models/
python bench/voice/wakeword_bench.py selftest            # optional synthetic check (needs espeak-ng)

# 1. enroll: 5 x "Hey ELARA", then 15 s of normal talking WITHOUT the phrase (for the threshold)
python bench/voice/wakeword_bench.py enroll --count 5 --neg-seconds 15 [--device N]

# 2. live test: say "Hey ELARA" a few times, talk normally, stay quiet; Ctrl+C to stop early
python bench/voice/wakeword_bench.py listen --seconds 60 [--device N]
```

`listen` prints each detection with its DTW distance and how long after the end of the speech
it fired. At the end it prints the compute time per 80 ms chunk and the process CPU as a
percentage of one core. To score recorded WAVs offline (for example clips from `voice_bench.py
record`), use `eval --pos a.wav ... --neg b.wav`.
