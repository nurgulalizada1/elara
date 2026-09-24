"""URL validation for outbound fetches (SSRF protection)."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

from elara.core.errors import PermissionDenied


async def validate_public_url(url: str, *, allow_private: bool = False) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise PermissionDenied(f"only http(s) URLs are allowed, got '{parts.scheme}'")
    if not parts.hostname:
        raise PermissionDenied("URL has no host")
    if parts.username or parts.password:
        raise PermissionDenied("URLs with embedded credentials are not allowed")
    if allow_private:
        return url
    host = parts.hostname
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, parts.port or 443)
    except socket.gaierror as e:
        raise PermissionDenied(f"cannot resolve host {host}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise PermissionDenied(f"{host} resolves to a non-public address ({ip})")
    return url
