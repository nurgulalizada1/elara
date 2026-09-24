"""Composition root: builds every service once, with explicit dependencies.

Both the CLI and the API call `build_container()`; tests inject fakes via arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from elara.agent.assistant import Assistant
from elara.agent.intent import IntentClassifier
from elara.agent.loop import AgentLoop
from elara.agent.router import Router
from elara.config.settings import Settings
from elara.conversation.store import ConversationStore
from elara.core.logging import get_logger
from elara.database.audit import AuditLog
from elara.database.db import Database
from elara.memory import MemoryPolicy, MemoryService, MemoryStore
from elara.providers.base import LLMProvider
from elara.providers.service import LLMService, build_provider
from elara.research.engine import ResearchEngine
from elara.research.http import ResearchHttp
from elara.research.planner import QueryPlanner
from elara.research.sources import build_sources
from elara.research.synthesis import Synthesizer
from elara.security.injection import InjectionDetector
from elara.security.paths import PathGuard
from elara.tools.builtin.calculator import CalculatorTool
from elara.tools.builtin.clock import CurrentTimeTool
from elara.tools.builtin.files import (
    DeleteFileTool,
    FileInfoTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from elara.tools.builtin.memory_tools import MemoryDeleteTool, MemorySearchTool, MemoryStoreTool
from elara.tools.builtin.research_tools import ResearchSearchTool, SourceSearchTool
from elara.tools.builtin.system import OpenPathTool, RunPythonTool
from elara.tools.builtin.web import WebFetchTool
from elara.tools.confirmations import ConfirmationStore
from elara.tools.executor import ToolExecutor
from elara.tools.permissions import PermissionPolicy
from elara.tools.registry import ToolRegistry

log = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    db: Database
    audit: AuditLog
    http: httpx.AsyncClient
    llm: LLMService
    memory: MemoryService
    conversations: ConversationStore
    registry: ToolRegistry
    executor: ToolExecutor
    confirmations: ConfirmationStore
    research: ResearchEngine
    assistant: Assistant

    async def aclose(self) -> None:
        await self.http.aclose()


def build_container(settings: Settings, *, provider: LLMProvider | None = None,
                    http_client: httpx.AsyncClient | None = None,
                    research_client: httpx.AsyncClient | None = None,
                    use_env_provider: bool = True) -> Container:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_path)
    db.migrate()
    audit = AuditLog(db)
    http = http_client or httpx.AsyncClient(timeout=30, follow_redirects=False)

    if provider is None and use_env_provider:
        provider = build_provider(settings, http)
    llm = LLMService(provider, settings, audit)

    detector = InjectionDetector()
    memory = MemoryService(MemoryStore(db), MemoryPolicy(detector), audit)
    memory.store.purge_expired()
    conversations = ConversationStore(db)
    confirmations = ConfirmationStore(db, settings.confirmation_ttl_s)

    min_intervals = {"eutils.ncbi.nlm.nih.gov": 0.11 if settings.ncbi_api_key else 0.34,
                     "api.semanticscholar.org": 1.0 if not settings.semantic_scholar_api_key
                     else 0.1}
    rhttp = ResearchHttp(research_client or http, db, cache_ttl_s=settings.http_cache_ttl_s,
                         timeout_s=settings.research_timeout_s, min_intervals=min_intervals,
                         contact_email=settings.contact_email)
    sources = build_sources(rhttp, settings)
    research = ResearchEngine(sources, QueryPlanner(llm), db,
                              per_source_limit=settings.research_max_results,
                              source_timeout_s=settings.research_timeout_s + 5,
                              min_primary_results=settings.research_min_primary_results)

    guard = PathGuard(settings.read_dirs, settings.write_dirs, deny_dirs=[settings.data_dir],
                      named_paths=settings.named_paths)
    registry = ToolRegistry()
    tools = [
        CalculatorTool(), CurrentTimeTool(),
        ListFilesTool(guard, settings.max_file_read_bytes),
        ReadFileTool(guard, settings.max_file_read_bytes), WriteFileTool(guard),
        SearchFilesTool(guard), FileInfoTool(guard), DeleteFileTool(guard),
        OpenPathTool(guard, settings.named_paths), RunPythonTool(),
        WebFetchTool(http, settings.allow_private_network_fetch),
        MemorySearchTool(memory), MemoryStoreTool(memory), MemoryDeleteTool(memory),
        ResearchSearchTool(research),
        *[SourceSearchTool(research, s) for s in sources],
    ]
    for tool in tools:
        registry.register(tool)
    policy = PermissionPolicy(settings.disabled_tools,
                              {"run_python": settings.enable_code_execution})
    executor = ToolExecutor(registry, policy, confirmations, audit, detector)

    assistant = Assistant(
        settings=settings, conversations=conversations, memory=memory, llm=llm,
        executor=executor, confirmations=confirmations, research=research,
        synthesizer=Synthesizer(llm), classifier=IntentClassifier(memory.policy),
        router=Router(), loop=AgentLoop(llm, registry, executor, settings.agent_max_steps,
                                        settings.max_output_tokens))
    log.info("elara.ready", extra={"provider": settings.resolved_provider(),
                                   "tools": len(registry), "db": str(settings.db_path)})
    return Container(settings=settings, db=db, audit=audit, http=http, llm=llm, memory=memory,
                     conversations=conversations, registry=registry, executor=executor,
                     confirmations=confirmations, research=research, assistant=assistant)
