"""`elara` command-line entry point.

An adapter only: it turns arguments into ElaraCore calls and prints the results.
Conversational requests go through `ElaraCore.process`; management commands (memory,
tools, research) use the same core's services. No business logic lives here.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import sys

from elara.config.settings import Settings
from elara.core.buildinfo import build_label
from elara.core.logging import configure_logging


def _settings() -> Settings:
    return Settings()


def _setup_logging(settings: Settings, *, console_level: str = "ERROR") -> None:
    configure_logging(settings.log_level, json_format=False, log_file=settings.log_path,
                      console=True, console_level=console_level,
                      secrets=settings.secret_values())


async def _with_core(settings: Settings, fn):
    from elara.core.service import ElaraCore
    async with ElaraCore.open(settings) as core:
        return await fn(core)


def cmd_chat(args, settings: Settings) -> int:
    from elara.cli.repl import Repl

    async def go(core):
        await Repl(core).run()
    asyncio.run(_with_core(settings, go))
    return 0


def cmd_ask(args, settings: Settings) -> int:
    from elara.core.service import CoreRequest

    async def go(core):
        result = await core.process(CoreRequest(text=" ".join(args.message),
                                                conversation_id=args.conversation,
                                                channel="cli"))
        if args.json:
            print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        else:
            print(result.text)
        return 0
    return asyncio.run(_with_core(settings, go))


def cmd_doctor(args, settings: Settings) -> int:
    from elara.cli.doctor import render, run_checks
    checks = asyncio.run(run_checks(settings, online=not args.offline, run_tests=args.tests))
    print(render(checks, color=sys.stdout.isatty()))
    return 1 if any(c.status == "fail" for c in checks) else 0


def cmd_serve(args, settings: Settings) -> int:
    import uvicorn

    from elara.api.app import create_app
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback and not settings.api_token:
        print("Refusing to listen on a non-loopback address without ELARA_API_TOKEN.",
              file=sys.stderr)
        return 2
    configure_logging(settings.log_level, json_format=settings.log_json,
                      log_file=settings.log_path, secrets=settings.secret_values())
    uvicorn.run(create_app(settings), host=host, port=port, log_level="warning")
    return 0


def cmd_memory(args, settings: Settings) -> int:
    from elara.memory import MemorySource
    from elara.security.untrusted import Trust

    async def go(c):
        if args.action == "list":
            for m in c.memory.store.list(limit=1000):
                print(f"#{m.id}\t{m.kind}\t{m.content}\t({m.source}, {m.created_at[:10]})")
        elif args.action == "search":
            for m in c.memory.relevant(" ".join(args.text), 20):
                print(f"#{m.id}\t{m.kind}\t{m.content}")
        elif args.action == "add":
            out = c.memory.remember(" ".join(args.text), source=MemorySource.API,
                                    trust=Trust.USER)
            print(f"{out.result.action}: #{out.result.memory.id}" if out.saved and out.result
                  else f"not saved: {out.reason}")
            return 0 if out.saved else 1
        elif args.action == "delete":
            ok = c.memory.store.delete(int(args.text[0]))
            print("deleted" if ok else "not found")
            return 0 if ok else 1
        return 0
    return asyncio.run(_with_core(settings, lambda core: go(core.container)))


def cmd_tools(args, settings: Settings) -> int:
    from elara.tools.base import Origin, ToolContext

    async def go(c):
        if not args.name:
            for t in c.registry.all():
                print(f"{t.name:<24} {t.permission.value:<16} {t.description}")
            return 0
        try:
            tool_args = json.loads(args.args or "{}")
        except json.JSONDecodeError as e:
            print(f"--args must be JSON: {e}", file=sys.stderr)
            return 2
        outcome = await c.executor.execute(args.name, tool_args, ToolContext(
            origin=Origin.API, confirmed=args.yes))
        if outcome.pending:
            c.confirmations.resolve(outcome.pending.id, "cancelled")
            print(f"needs confirmation ({outcome.error}); re-run with --yes to approve: "
                  f"{outcome.pending.description}")
            return 3
        print(json.dumps({"status": outcome.status.value, "output": outcome.output,
                          "error": outcome.error}, indent=2, ensure_ascii=False))
        return 0 if outcome.ok else 1
    return asyncio.run(_with_core(settings, lambda core: go(core.container)))


def cmd_research(args, settings: Settings) -> int:
    async def go(c):
        result = await c.research.research(" ".join(args.query), language=args.language)
        syn = await c.assistant.synthesizer.synthesize(" ".join(args.query), result,
                                                       args.language)
        print(syn.text)
        return 0
    return asyncio.run(_with_core(settings, lambda core: go(core.container)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="elara", description="ELARA personal AI assistant")
    p.add_argument("--version", action="version", version=build_label())
    p.add_argument("-v", "--verbose", action="store_true", help="show info logs on stderr")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("chat", help="interactive chat (default)")
    a = sub.add_parser("ask", help="one-shot message")
    a.add_argument("message", nargs="+")
    a.add_argument("--conversation", "-c")
    a.add_argument("--json", action="store_true")
    d = sub.add_parser("doctor", help="self-diagnostics")
    d.add_argument("--offline", action="store_true", help="skip network checks")
    d.add_argument("--tests", action="store_true", help="also run the test suite")
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    m = sub.add_parser("memory", help="inspect/manage memories")
    m.add_argument("action", choices=["list", "search", "add", "delete"])
    m.add_argument("text", nargs="*")
    t = sub.add_parser("tools", help="list tools or run one")
    t.add_argument("name", nargs="?")
    t.add_argument("--args", help="JSON arguments")
    t.add_argument("--yes", action="store_true", help="approve actions needing confirmation")
    r = sub.add_parser("research", help="search scientific sources")
    r.add_argument("query", nargs="+")
    r.add_argument("--language", default="en", choices=["az", "en", "tr"])
    v = sub.add_parser("voice", help="(dev) one bounded English voice round trip")
    v.add_argument("--turns", type=int, default=1, choices=range(1, 21), metavar="N",
                   help="utterances to handle before exiting (1-20, default 1)")
    v.add_argument("--device", help="input device index or name (see --list-devices)")
    v.add_argument("--list-devices", action="store_true", help="list microphones and exit")
    v.add_argument("--no-speak", action="store_true", help="do not speak the answer")
    v.add_argument("--json", action="store_true")
    return p


def cmd_voice(args, settings: Settings) -> int:
    from elara.cli.voice_cmd import cmd_voice as run
    return run(args, settings)


COMMANDS = {"chat": cmd_chat, "ask": cmd_ask, "doctor": cmd_doctor, "serve": cmd_serve,
            "memory": cmd_memory, "tools": cmd_tools, "research": cmd_research,
            "voice": cmd_voice}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "chat"
    try:
        settings = _settings()
    except Exception as e:  # invalid env configuration
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    if command != "serve":
        _setup_logging(settings, console_level="INFO" if args.verbose else "ERROR")
    return COMMANDS[command](args, settings)


if __name__ == "__main__":
    sys.exit(main())
