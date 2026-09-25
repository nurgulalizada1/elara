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

This is an experiment and is not wired into `elara voice`. It uses openWakeWord's frozen
feature front end (`melspectrogram.onnx` + `embedding_model.onnx`, about 2.4 MB), run directly
with `onnxruntime`. faster-whisper already installs onnxruntime, so there are **no new Python
packages**. It works by **few-shot enrollment**: recordings of the phrase become templates, and
the live audio is compared with them every 80 ms using subsequence DTW on the 96-dimensional
embeddings. Whisper is never used, and no audio leaves the machine.

Measurements come from labelled recordings, not from a single live run:

```bash
python bench/voice/wakeword_bench.py fetch-models                 # once, sha256-pinned

# 1. real, labelled data on the laptop mic (~10 min, guided prompts):
#    5 enrollment takes, 30 s calibration talk + one take of each confusable phrase,
#    20 test positives (normal/fast/slow/quiet/loud), 2 takes of each hard negative
#    (Hey Alexa / Hey Sarah / Hey Laura / Elara / What time is it?), 90 s conversation,
#    180 s TV/background speech, 60 s silence
python bench/voice/wakeword_bench.py collect --out bench/voice/recordings/ww_real [--device N]

# 2. score every decision rule: TP/FP/FN, precision, recall (95% CI), false activations/h
#    (with its 95% upper bound), latency, CPU; also writes templates for `listen`
python bench/voice/wakeword_bench.py evaluate --set bench/voice/recordings/ww_real \
    --save-templates bench/voice/recordings/hey_elara_templates.npz

# 3. live: every candidate is printed with distance, threshold, best template, top-3
#    template distances, run length, nearby matches, matched duration, latency, compute;
#    repeats of one event are marked. --save-wav keeps the session for offline `scan`.
python bench/voice/wakeword_bench.py listen --rule robust --seconds 120 --save-wav /tmp/ww.wav
python bench/voice/wakeword_bench.py scan --rule robust /tmp/ww.wav

# synthetic sanity set (espeak-ng voices; NOT a substitute for real recordings)
python bench/voice/wakeword_bench.py synth --out bench/voice/recordings/ww_synth
python bench/voice/wakeword_bench.py evaluate --set bench/voice/recordings/ww_synth
```

Decision rules (`--rule`):

| Rule | What it does |
|---|---|
| `legacy` | The original spike: minimum over templates and 3 fixed window lengths; fires on one chunk |
| `nearest` | Subsequence DTW against the nearest template; fires on one chunk |
| `k2` | Mean distance to the 2 nearest templates |
| `robust` | `k2`, plus 2 consecutive qualifying chunks and a matched duration of 0.6–1.6× the template |

The threshold is calibrated only from enrollment data: leave-one-out scores of the
enrollment takes against the rule's firing level on the calibration talk and the
confusable phrases. The test clips never influence it. Recordings stay in
`recordings/` (git-ignored); delete them when you are done.
