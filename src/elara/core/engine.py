from elara.core.result import ElaraResult


class ElaraEngine:
    """Central execution engine for ELARA."""

    def run(self, command: str) -> ElaraResult:
        """Execute a command and return a standardized result."""

        if not command.strip():
            return ElaraResult(
                success=False,
                message="Komanda boş ola bilməz.",
            )

        return ElaraResult(
            success=True,
            message=f"ELARA əmri qəbul etdi: {command}",
        )
