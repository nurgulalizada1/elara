import json

import pytest

from elara.cli import main as cli
from elara.cli.doctor import Check, render, run_checks
from elara.cli.repl import Repl
from elara.core.container import build_container


@pytest.fixture
def env(monkeypatch, settings):
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    return settings


def test_ask_one_shot(env, capsys):
    assert cli.main(["ask", "What's", "17", "*", "42?"]) == 0
    assert "714" in capsys.readouterr().out


def test_ask_json(env, capsys):
    cli.main(["ask", "--json", "Salam"])
    assert json.loads(capsys.readouterr().out)["text"] == "Salam! Necəsən?"


def test_memory_commands(env, capsys):
    assert cli.main(["memory", "add", "I", "prefer", "tea"]) == 0
    cli.main(["memory", "list"])
    assert "I prefer tea" in capsys.readouterr().out
    assert cli.main(["memory", "add", "my", "password", "is", "x1"]) == 1


def test_tools_commands(env, capsys):
    cli.main(["tools"])
    assert "calculator" in capsys.readouterr().out
    assert cli.main(["tools", "calculator", "--args", '{"expression": "6*7"}']) == 0
    assert '"result": 42' in capsys.readouterr().out
    (env.write_dirs[0] / "z.txt").write_text("z")
    assert cli.main(["tools", "delete_file", "--args", '{"path": "z.txt"}']) == 3
    assert (env.write_dirs[0] / "z.txt").exists()


def test_serve_refuses_public_bind_without_token(env):
    assert cli.main(["serve", "--host", "0.0.0.0"]) == 2


async def test_repl_commands(settings, capsys, monkeypatch):
    c = build_container(settings, use_env_provider=False)
    repl = Repl(c, color=False)
    await repl.say("Remember that I like jazz")
    await repl.command("/memory")
    await repl.command("/status")
    await repl.command("/tools")
    out = capsys.readouterr().out
    assert "I like jazz" in out and "provider: none" in out and "calculator" in out
    assert await repl.command("/exit") is False
    inputs = iter(["Salam", "/help", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    await repl.run()
    out = capsys.readouterr().out
    assert "Salam! Necəsən?" in out and "/memory" in out
    await c.aclose()


async def test_doctor_offline(settings):
    checks = await run_checks(settings, online=False)
    by = {c.name: c for c in checks}
    assert by["Database"].status == "ok"
    assert by["LLM provider"].status == "warn" and "ANTHROPIC_API_KEY" in by["LLM provider"].fix
    assert by["Tools"].status == "ok"
    text = render(checks, color=False)
    assert text.startswith("ELARA Doctor") and "✓ Database" in text


def test_render_symbols():
    out = render([Check("A", "ok", "x"), Check("B", "fail", "y", "do z")], color=False)
    assert "✓ A: x" in out and "✗ B: y" in out and "→ do z" in out
