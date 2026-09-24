import httpx
import pytest

from elara.core.container import build_container
from tests import research_fixtures as fx
from tests.fakes import ScriptedProvider


@pytest.fixture
def provider():
    return ScriptedProvider()


@pytest.fixture
async def app(settings, provider):
    settings = settings.model_copy(update={"model_fast": "fast-model",
                                           "model_strong": "strong-model",
                                           "named_paths": {"project": settings.write_dirs[0]}})
    research_client = httpx.AsyncClient(transport=httpx.MockTransport(fx.handler()))
    c = build_container(settings, provider=provider, research_client=research_client)
    yield c
    await c.aclose()
    await research_client.aclose()
