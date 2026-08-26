import typer

from elara import __version__
from elara.core.engine import ElaraEngine


app = typer.Typer()
engine = ElaraEngine()


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
