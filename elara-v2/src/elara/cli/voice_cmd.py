"""`elara voice`: development-only, bounded voice round trip (not an always-on daemon).

Listens for at most --turns utterances (default 1), transcribes locally in English,
applies the quality gate, sends accepted text to ElaraCore (channel "voice"), prints the
CoreResult with timings, and optionally speaks the answer. No wake word yet.
"""

from __future__ import annotations

import json
import sys

from elara.config.settings import Settings


def _print_turn(n: int, turn, as_json: bool) -> None:
    tr, gate, res = turn.transcript, turn.gate, turn.result
    if as_json:
        print(json.dumps({
            "turn": n, "status": turn.status,
            "capture": None if turn.utterance is None else {
                k: getattr(turn.utterance, k) for k in (
                    "reason", "overflows", "noise_floor_dbfs", "start_threshold_dbfs",
                    "stop_threshold_dbfs", "speech_ms", "trailing_silence_ms")},
            "transcript": None if tr is None else {
                "text": tr.text, "language": tr.language,
                "language_probability": tr.language_probability,
                "avg_logprob": tr.avg_logprob, "no_speech_prob": tr.no_speech_prob,
                "compression_ratio": tr.compression_ratio, "error": tr.error},
            "gate": None if gate is None else {"accepted": gate.accepted, "code": gate.code,
                                               "reason": gate.reason},
            "result": None if res is None else res.model_dump(mode="json"),
            "spoken": turn.spoken, "speak_error": turn.speak_error, "timings": turn.timings,
        }, indent=2, ensure_ascii=False))
        return
    print(f"\n── turn {n}: {turn.status}")
    if turn.utterance is not None and not turn.utterance.pcm and tr is None:
        print(f"   no speech detected ({turn.utterance.reason})")
    if tr is not None:
        print(f"   heard:   {tr.text!r}  [{tr.language}, avg_logprob={tr.avg_logprob}, "
              f"no_speech={tr.no_speech_prob}]")
    if gate is not None:
        print(f"   gate:    {'accepted' if gate.accepted else 'REJECTED'} ({gate.code}: "
              f"{gate.reason})")
    if res is not None:
        print(f"   ELARA:   {res.text}")
        print(f"   core:    intent={res.intent} resolver={res.resolver} tier={res.tier} "
              f"used_llm={res.used_llm} tools={[t.tool for t in res.tool_calls]}")
        if res.pending_action:
            print(f"   pending: {res.pending_action.description} (answer by voice or "
                  f"`elara ask -c {res.conversation_id} yes`)")
    if turn.speak_error:
        print(f"   speech output failed: {turn.speak_error}")
    u = turn.utterance
    if u is not None:
        print(f"   capture: {u.reason}, speech={u.speech_ms}ms, trailing_silence="
              f"{u.trailing_silence_ms}ms, floor={u.noise_floor_dbfs} dBFS, start/stop="
              f"{u.start_threshold_dbfs}/{u.stop_threshold_dbfs} dBFS")
    t = turn.timings
    print("   timing:  " + "  ".join(f"{k}={v}" for k, v in t.items() if v is not None))


def cmd_voice(args, settings: Settings) -> int:
    from elara.voice.config import VoiceConfig
    from elara.voice.microphone import Microphone, MicrophoneError, list_input_devices
    from elara.voice.stt import FasterWhisperSTT, STTUnavailable
    from elara.voice.tts import EspeakSpeaker

    config = VoiceConfig()
    if args.device is not None:
        config = config.model_copy(update={"input_device": args.device})
    if args.list_devices:
        try:
            for d in list_input_devices():
                mark = "*" if d["is_default"] else " "
                print(f"{mark} [{d['index']}] {d['name']} ({d['channels']} ch, "
                      f"{d['default_samplerate']:.0f} Hz)")
        except MicrophoneError as e:
            print(f"microphone unavailable: {e}", file=sys.stderr)
            return 2
        return 0
    if not config.enabled:
        print("voice is disabled (ELARA_VOICE_ENABLED=false)", file=sys.stderr)
        return 2

    speaker = None
    if config.tts_enabled and not args.no_speak:
        speaker = EspeakSpeaker(config.tts_voice, config.tts_rate_wpm)
        if not speaker.available():
            print("note: espeak-ng not found; answers will be shown, not spoken",
                  file=sys.stderr)
            speaker = None

    stt = FasterWhisperSTT(config)
    try:
        print(f"loading Whisper '{config.model}' (cpu, {config.compute_type}, "
              f"{config.cpu_threads} threads, language={config.language}) ...", file=sys.stderr)
        load_s = stt.load()
        print(f"model loaded in {load_s}s", file=sys.stderr)
    except STTUnavailable as e:
        print(f"speech recognition unavailable: {e}", file=sys.stderr)
        return 2

    import asyncio

    from elara.core.service import ElaraCore
    from elara.voice.session import VoiceSession

    async def run() -> int:
        async with ElaraCore.open(settings) as core:
            session = VoiceSession(core, Microphone(config), stt, speaker=speaker)
            for n in range(1, args.turns + 1):
                print(f"\n🎙  listening (English, up to {config.max_utterance_s:.0f}s; "
                      f"waits {config.start_timeout_s:.0f}s for speech) ...", file=sys.stderr)
                try:
                    turn = await session.listen_once()
                except MicrophoneError as e:
                    print(f"microphone error: {e}", file=sys.stderr)
                    return 2
                _print_turn(n, turn, args.json)
        return 0

    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 130
