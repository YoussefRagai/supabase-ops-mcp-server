#!/usr/bin/env python3
"""Supabase Ops MCP server exposed over SSE transport."""

from __future__ import annotations

import os

import uvicorn
from starlette.responses import JSONResponse
from starlette.routing import Route

from supabase_ops_server import mcp


app = mcp.streamable_http_app()


async def health(_request):
    return JSONResponse({"status": "ok", "service": "supabase_ops"})


app.router.routes.append(Route("/health", endpoint=health))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
