"""Registry of available research sources. Add new sources here."""

from __future__ import annotations

from elara.config.settings import Settings
from elara.research.http import ResearchHttp
from elara.research.sources.base import ResearchSource
from elara.research.sources.genomics import EnsemblSource, GnomADSource
from elara.research.sources.literature import CrossrefSource, EuropePMCSource, SemanticScholarSource
from elara.research.sources.ncbi import ClinVarSource, NCBIGeneSource, PubMedSource


def build_sources(http: ResearchHttp, settings: Settings) -> list[ResearchSource]:
    ncbi_key = settings.ncbi_api_key.get_secret_value() if settings.ncbi_api_key else None
    s2_key = (settings.semantic_scholar_api_key.get_secret_value()
              if settings.semantic_scholar_api_key else None)
    email = settings.contact_email
    return [
        PubMedSource(http, ncbi_key, email), ClinVarSource(http, ncbi_key, email),
        NCBIGeneSource(http, ncbi_key, email), EuropePMCSource(http),
        CrossrefSource(http, email), SemanticScholarSource(http, s2_key),
        EnsemblSource(http), GnomADSource(http),
    ]


__all__ = ["ResearchSource", "build_sources"]
