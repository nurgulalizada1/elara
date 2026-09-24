"""Interactive chat loop. Thin layer over Assistant; no business logic here."""

from __future__ import annotations

import sys

from elara.agent.assistant import AssistantReply
from elara.core.container import Container

HELP = """\
Commands:
  /help            this help
  /memory [query]  list memories (or search them)
  /forget <id>     delete a memory by id
  /tools           list tools and their permission levels
  /status          provider, models, usage and database stats
  /clear           clear this conversation's history (memories are kept)
  /new             start a new conversation
  /yes, /no        answer a pending confirmation
  /exit            quit"""

DIM, BOLD, YELLOW, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[0m"


class Repl:
    def __init__(self, container: Container, color: bool | None = None):
        self.c = container
        self.cid: str | None = None
        self.color = sys.stdout.isatty() if color is None else color

    def _p(self, text: str, style: str = "") -> None:
        print(f"{style}{text}{RESET}" if self.color and style else text, flush=True)

    async def run(self) -> None:
        self._p("ELARA — /help for commands, /exit to quit.", DIM)
        while True:
            try:
                line = input("Sən: " if self.c.settings.default_language == "az" else "You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if await self.command(line) is False:
                    break
                continue
            await self.say(line)

    async def say(self, text: str) -> AssistantReply:
        reply = await self.c.assistant.handle(text, self.cid)
        self.cid = reply.conversation_id
        self._p(f"ELARA: {reply.text}", BOLD)
        if reply.memory_events:
            self._p(f"  (memory: {', '.join(reply.memory_events)})", DIM)
        tools = [f"{t.tool}:{t.status}" for t in reply.tool_calls]
        if tools:
            self._p(f"  (tools: {', '.join(tools)})", DIM)
        if reply.pending_action:
            self._p("  (answer yes/no, bəli/xeyr, evet/hayır)", YELLOW)
        return reply

    async def command(self, line: str) -> bool | None:
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        match cmd:
            case "/exit" | "/quit":
                return False
            case "/help":
                self._p(HELP)
            case "/memory":
                items = (self.c.memory.relevant(arg, 20) if arg
                         else self.c.memory.store.list(limit=100))
                if not items:
                    self._p("(no memories)", DIM)
                for m in items:
                    exp = f", expires {m.expires_at[:10]}" if m.expires_at else ""
                    self._p(f"  #{m.id} [{m.kind}] {m.content}  {DIM if self.color else ''}"
                            f"({m.source}, {m.created_at[:10]}{exp}){RESET if self.color else ''}")
            case "/forget":
                if not arg.isdigit():
                    self._p("usage: /forget <id>")
                elif self.c.memory.store.delete(int(arg)):
                    self._p(f"deleted memory #{arg}")
                else:
                    self._p(f"no memory #{arg}")
            case "/tools":
                for t in self.c.registry.all():
                    llm = "" if t.exposed_to_llm else " (API/CLI only)"
                    self._p(f"  {t.name:<24} {t.permission.value:<16} {t.description[:60]}{llm}")
            case "/status":
                s = self.c.settings
                self._p(f"  provider: {s.resolved_provider()} (available: {self.c.llm.available})")
                self._p(f"  models:   fast={s.model_for('fast')}  strong={s.model_for('strong')}")
                self._p(f"  usage:    {self.c.audit.usage_totals()}")
                self._p(f"  database: {s.db_path} (schema v{self.c.db.schema_version()})")
                self._p(f"  memories: {len(self.c.memory.store.list(limit=10_000))}, "
                        f"tools: {len(self.c.registry)}, conversation: {self.cid}")
            case "/clear":
                if self.cid:
                    self.c.conversations.clear(self.cid)
                self._p("(conversation cleared; memories kept)", DIM)
            case "/new":
                self.cid = None
                self._p("(new conversation)", DIM)
            case "/yes" | "/no":
                await self.say("yes" if cmd == "/yes" else "no")
            case _:
                self._p(f"unknown command {cmd}; /help lists commands")
        return None
