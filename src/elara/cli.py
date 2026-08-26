"""ELARA-nın komanda sətri interfeysi."""

from __future__ import annotations

import typer

from elara import __version__
from elara.core.orchestrator import ElaraOrchestrator
from elara.llm.mock import MockLLMProvider
from elara.memory.persistent import PersistentMemory


app = typer.Typer(
    help="ELARA — şəxsi AI sistemi.",
    no_args_is_help=True,
)


def build_greeting(ad: str | None = None) -> str:
    """Standart ELARA salamlamasını qurur."""

    if ad:
        return f"Salam, {ad}! Mən ELARA."

    return "Salam! Mən ELARA."


def build_orchestrator() -> ElaraOrchestrator:
    """Chat üçün persistent orchestrator yaradır."""

    return ElaraOrchestrator(
        provider=MockLLMProvider(),
        persistent_memory=PersistentMemory(),
    )


@app.command()
def salam(
    ad: str | None = typer.Option(
        None,
        "--ad",
        "-a",
        help="Sənin adın.",
    ),
) -> None:
    """ELARA ilə salamlaş."""

    typer.echo(build_greeting(ad))


@app.command()
def version() -> None:
    """ELARA versiyasını göstər."""

    typer.echo(f"ELARA v{__version__}")


@app.command()
def chat() -> None:
    """ELARA ilə interaktiv söhbət et."""

    orchestrator = build_orchestrator()

    typer.echo("ELARA chat başladı. Çıxmaq üçün 'exit' yaz.")

    while True:
        try:
            user_text = typer.prompt("Sən")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nELARA chat bağlandı.")
            break

        if user_text.strip().lower() in {"exit", "quit", "çıx"}:
            typer.echo("ELARA chat bağlandı.")
            break

        response = orchestrator.handle(user_text)

        typer.echo(f"ELARA: {response}")
