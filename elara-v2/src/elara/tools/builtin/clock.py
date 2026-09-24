"""Current local date/time."""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel

from elara.tools.base import Permission, Tool, ToolContext


class NoInput(BaseModel):
    pass


class ClockOutput(BaseModel):
    iso: str
    date: str
    time: str
    weekday: str
    timezone: str


class CurrentTimeTool(Tool):
    name: ClassVar[str] = "current_time"
    description: ClassVar[str] = "Get the current local date, time, weekday and timezone."
    Input = NoInput
    Output = ClockOutput
    permission = Permission.READ_ONLY
    category = "utility"

    async def run(self, args: NoInput, ctx: ToolContext) -> ClockOutput:
        now = datetime.now().astimezone()
        return ClockOutput(iso=now.isoformat(timespec="seconds"), date=now.strftime("%Y-%m-%d"),
                           time=now.strftime("%H:%M"), weekday=now.strftime("%A"),
                           timezone=str(now.tzinfo))
