import typer
from rich.console import Console

app = typer.Typer()
console = Console()


def build_greeting(name: str | None = None) -> str:
    """Build a greeting message."""
    if name:
        return f"Salam, {name}! Mən ELARA."
    return "Salam! Mən ELARA."


@app.command()
def salam(
    ad: str | None = typer.Option(
        None,
        "--ad",
        help="Salamlanacaq şəxsin adı.",
    ),
) -> None:
    """ELARA-nın salamlamasını göstər."""
    console.print(build_greeting(ad))


@app.command()
def version() -> None:
    """ELARA versiyasını göstər."""
    console.print("ELARA v0.1.0")


if __name__ == "__main__":
    app()
