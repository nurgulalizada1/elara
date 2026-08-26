import typer

from elara import __version__
from elara.core.engine import ElaraEngine
from elara.core.orchestrator import ElaraOrchestrator
from elara.llm.mock import MockLLMProvider


app = typer.Typer()
engine = ElaraEngine()

chat_orchestrator = ElaraOrchestrator(
    provider=MockLLMProvider()
)


def build_greeting(ad: str | None = None) -> str:
    """Build the standard ELARA greeting."""

    if ad:
        return f"Salam, {ad}! Mən ELARA."

    return "Salam! Mən ELARA."


@app.command()
def salam(ad: str = typer.Option(None, "--ad")) -> None:
    """ELARA ilə salamlaş."""

    command = "salam"

    if ad:
        command = f"salam {ad}"

    result = engine.run(command)

    if result.success:
        typer.echo(build_greeting(ad))
    else:
        typer.echo(result.message)


@app.command()
def version() -> None:
    """ELARA versiyasını göstər."""

    typer.echo(f"ELARA v{__version__}")


@app.command()
def chat() -> None:
    """ELARA ilə interaktiv söhbət et."""

    typer.echo("ELARA chat başladı. Çıxmaq üçün 'exit' yaz.")

    while True:
        user_text = typer.prompt("Sən")

        if user_text.strip().lower() == "exit":
            typer.echo("ELARA chat bağlandı.")
            break

        response = chat_orchestrator.handle(user_text)
        typer.echo(f"ELARA: {response}")
