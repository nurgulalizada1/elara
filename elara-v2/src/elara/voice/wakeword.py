"""Wake-word detection, separate from STT.

`OpenWakeWordDetector` wraps openWakeWord with a user-supplied model file for the
"Hey ELARA" phrase. No pretrained "hey elara" model ships with ELARA; one must be
trained (openWakeWord provides a training pipeline) and configured.
"""

from __future__ import annotations

from array import array
from pathlib import Path

from elara.core.errors import ElaraError


class WakeWordUnavailable(ElaraError):
    pass


class OpenWakeWordDetector:
    phrase = "hey elara"

    def __init__(self, model_path: Path, threshold: float = 0.5):
        if not model_path.exists():
            raise WakeWordUnavailable(f"wake-word model not found: {model_path}")
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise WakeWordUnavailable("openwakeword is not installed") from e
        self._model = Model(wakeword_models=[str(model_path)])
        self.threshold = threshold

    def detect(self, frame: bytes) -> bool:
        import numpy as np
        samples = array("h")
        samples.frombytes(frame)
        scores = self._model.predict(np.asarray(samples, dtype=np.int16))
        return any(score >= self.threshold for score in scores.values())
