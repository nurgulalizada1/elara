from elara.core.engine import ElaraEngine


def test_engine_accepts_command():
    engine = ElaraEngine()

    result = engine.run("salam")

    assert result.success is True
    assert result.message == "ELARA əmri qəbul etdi: salam"


def test_engine_rejects_empty_command():
    engine = ElaraEngine()

    result = engine.run("")

    assert result.success is False
    assert result.message == "Komanda boş ola bilməz."
