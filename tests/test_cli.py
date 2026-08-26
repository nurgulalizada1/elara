from typer.testing import CliRunner

from elara.cli import app, build_greeting


runner = CliRunner()


def test_build_greeting_without_name() -> None:
    assert build_greeting() == "Salam! Mən ELARA."


def test_build_greeting_with_name() -> None:
    assert build_greeting("Nurgul") == "Salam, Nurgul! Mən ELARA."


def test_salam_command() -> None:
    result = runner.invoke(app, ["salam"])

    assert result.exit_code == 0
    assert "Salam! Mən ELARA." in result.stdout


def test_salam_command_with_name() -> None:
    result = runner.invoke(app, ["salam", "--ad", "Nurgul"])

    assert result.exit_code == 0
    assert "Salam, Nurgul! Mən ELARA." in result.stdout


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "ELARA v0.1.0" in result.stdout

