"""
Kapruka MCP Server

Exposes the Kapruka.com REST API as MCP tools for LLMs and third-party clients.
Transport: streamable HTTP, fronted by Caddy at https://mcp.kapruka.com.
"""

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from src.activity_log import ActivityLogger, ActivityLogMiddleware
from src.cache import cache
from src.config.settings import settings
from src.middleware import RateLimitMiddleware
from src.order_rate_limit import OrderRateLimitMiddleware
from src.well_known import well_known_mcp, well_known_mcp_options

_STATIC_DIR = Path(__file__).parent / "static"
_LANDING_HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "kapruka_mcp",
    instructions=(
        "You are connected to the Kapruka MCP server, which provides read-only access "
        "to Kapruka.com — Sri Lanka's largest e-commerce platform. Use the available "
        "tools to search products, browse categories, and look up product details. "
        "This is a free public tier; treat results as cached for up to 30 minutes."
    ),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=settings.enable_dns_rebinding_protection,
        allowed_hosts=settings.public_hosts,
        allowed_origins=settings.public_origins,
    ),
)

# ── Tool modules: importing them registers their @mcp.tool decorators.
from src.tools import categories, delivery, orders, products  # noqa: F401, E402


async def _landing(_request: Request) -> HTMLResponse:
    html_content = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html_content)


async def _health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def _stats(_request: Request) -> JSONResponse:
    return JSONResponse({"cache": cache.stats()})


async def _api_config(_request: Request) -> JSONResponse:
    import os
    has_gemini_key = bool(os.getenv("GEMINI_API_KEY"))
    return JSONResponse({"has_gemini_key": has_gemini_key})


async def _api_chat(request: Request) -> JSONResponse:
    import os
    import httpx
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return JSONResponse({"error": "Gemini API key not configured on server"}, status_code=500)
        
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        
    model = body.get("model", "gemini-3.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # We strip the model field from the body as it's part of the url in the Gemini API
            cleaned_body = dict(body)
            if "model" in cleaned_body:
                del cleaned_body["model"]
            response = await client.post(url, json=cleaned_body)
            response.raise_for_status()
            return JSONResponse(response.json())
    except httpx.HTTPStatusError as e:
        logger.error("Gemini API error: %s", e.response.text)
        try:
            err_json = e.response.json()
            return JSONResponse(err_json, status_code=e.response.status_code)
        except Exception:
            return JSONResponse({"error": f"Gemini API returned status {e.response.status_code}: {e.response.text}"}, status_code=e.response.status_code)
    except Exception as e:
        logger.exception("Error proxying to Gemini API")
        return JSONResponse({"error": str(e)}, status_code=500)

_public_mcp_session_id = None


async def _call_public_mcp_server(tool_name: str, arguments: dict, attempt: int = 1) -> str:
    global _public_mcp_session_id
    import httpx
    import json
    import asyncio
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Establish session if not cached
        if _public_mcp_session_id is None:
            # Step 1: Handshake
            r = await client.get("https://mcp.kapruka.com/mcp", headers={"User-Agent": headers["User-Agent"]})
            if r.status_code == 429:
                if attempt < 3:
                    await asyncio.sleep(5.0)
                    return await _call_public_mcp_server(tool_name, arguments, attempt + 1)
                raise Exception("Public MCP server rate limited during handshake")
                
            session_id = r.headers.get("mcp-session-id")
            if not session_id:
                raise Exception(f"Failed to acquire session ID from public MCP server: {r.status_code} {r.text[:200]}")
            
            # Step 2: Initialize
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "kapruka-python-client", "version": "1.0.0"}
                }
            }
            headers["Mcp-Session-Id"] = session_id
            await client.post(f"https://mcp.kapruka.com/mcp?session_id={session_id}", json=init_payload, headers=headers)
            
            # Step 3: Initialized notification
            init_notify = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            await client.post(f"https://mcp.kapruka.com/mcp?session_id={session_id}", json=init_notify, headers=headers)
            
            _public_mcp_session_id = session_id
            
        session_id = _public_mcp_session_id
        headers["Mcp-Session-Id"] = session_id
        
        # Step 4: Call tool
        call_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": {"params": arguments}
            }
        }
        r_call = await client.post(f"https://mcp.kapruka.com/mcp?session_id={session_id}", json=call_payload, headers=headers)
        
        # Check for session expiration or error
        body_text = r_call.text
        if r_call.status_code in (400, 404) or "Session not found" in body_text or "Missing session ID" in body_text:
            if attempt < 3:
                _public_mcp_session_id = None
                return await _call_public_mcp_server(tool_name, arguments, attempt + 1)
            raise Exception("Public MCP server session expired and retry failed")
            
        if r_call.status_code == 429:
            if attempt < 3:
                await asyncio.sleep(attempt * 3.0)
                return await _call_public_mcp_server(tool_name, arguments, attempt + 1)
            raise Exception("Public MCP server rate limited (429) on tool call")
            
        lines = body_text.split("\n")
        data = None
        for line in lines:
            if line.strip().startswith("data:"):
                try:
                    data = json.loads(line.strip()[5:].strip())
                    break
                except Exception:
                    pass
        if not data:
            try:
                data = r_call.json()
            except Exception:
                raise Exception(f"Unparseable response from public MCP server: {body_text[:200]}")
                
        if "error" in data:
            raise Exception(data["error"].get("message", "Tool call failed"))
            
        result = data.get("result", {})
        content = result.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "")
        return ""


async def _api_execute_tool(request: Request) -> JSONResponse:
    tool_name = request.path_params["tool_name"]
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    # Check if settings has the placeholder key. If so, fallback to public MCP server
    from src.config.settings import settings
    if settings.api_key == "your-internal-kapruka-api-key" or not settings.api_key:
        try:
            import json
            # Default response_format to 'json' for programmatical rendering, unless explicitly overridden
            if "response_format" not in body:
                body["response_format"] = "json"
                
            result_str = await _call_public_mcp_server(tool_name, body)
            
            if body.get("response_format") == "json":
                try:
                    parsed_json = json.loads(result_str)
                    return JSONResponse(parsed_json)
                except Exception:
                    return JSONResponse({"result": result_str})
            else:
                return JSONResponse({"result": result_str})
        except Exception as e:
            logger.exception("Error executing tool %s on public MCP server fallback", tool_name)
            return JSONResponse({"error": str(e)}, status_code=400)
            
    # Standard local code execution with real API key
    from src.tools.products import kapruka_search_products, kapruka_get_product, SearchProductsInput, GetProductInput
    from src.tools.categories import kapruka_list_categories, ListCategoriesInput
    from src.tools.delivery import kapruka_list_delivery_cities, kapruka_check_delivery, ListDeliveryCitiesInput, CheckDeliveryInput
    from src.tools.orders import kapruka_create_order, kapruka_track_order, CreateOrderInput, TrackOrderInput
    
    tool_map = {
        "kapruka_search_products": (kapruka_search_products, SearchProductsInput),
        "kapruka_get_product": (kapruka_get_product, GetProductInput),
        "kapruka_list_categories": (kapruka_list_categories, ListCategoriesInput),
        "kapruka_list_delivery_cities": (kapruka_list_delivery_cities, ListDeliveryCitiesInput),
        "kapruka_check_delivery": (kapruka_check_delivery, CheckDeliveryInput),
        "kapruka_create_order": (kapruka_create_order, CreateOrderInput),
        "kapruka_track_order": (kapruka_track_order, TrackOrderInput),
    }
    
    if tool_name not in tool_map:
        return JSONResponse({"error": f"Tool '{tool_name}' not found"}, status_code=404)
        
    func, input_model = tool_map[tool_name]
    
    try:
        # Default response_format to 'json' for programmatical rendering, unless explicitly overridden
        if "response_format" not in body:
            body["response_format"] = "json"
            
        params = input_model(**body)
        result_str = await func(params)
        
        if body.get("response_format") == "json":
            try:
                import json
                parsed_json = json.loads(result_str)
                return JSONResponse(parsed_json)
            except Exception:
                return JSONResponse({"result": result_str})
        else:
            return JSONResponse({"result": result_str})
    except Exception as e:
        logger.exception("Error executing tool %s", tool_name)
        return JSONResponse({"error": str(e)}, status_code=400)



def build_app() -> Starlette:
    """Compose the MCP Starlette app with our health routes + middleware."""
    app: Starlette = mcp.streamable_http_app()

    app.router.routes.insert(0, Route("/", _landing, methods=["GET"]))
    app.router.routes.insert(1, Route("/health", _health, methods=["GET"]))
    app.router.routes.insert(2, Route("/stats", _stats, methods=["GET"]))
    app.router.routes.insert(3, Route("/api/config", _api_config, methods=["GET"]))
    app.router.routes.insert(4, Route("/api/chat", _api_chat, methods=["POST"]))
    app.router.routes.insert(5, Route("/api/tools/{tool_name}", _api_execute_tool, methods=["POST"]))
    app.router.routes.insert(6, Route("/.well-known/mcp.json", well_known_mcp, methods=["GET"]))
    app.router.routes.insert(7, Route("/.well-known/mcp.json", well_known_mcp_options, methods=["OPTIONS"]))


    # ── Activity logging (optional — disabled when ACTIVITY_DB_URL unset).
    # The middleware lazy-inits the pool on first request, so no lifespan plumbing.
    if settings.activity_db_url:
        activity_log = ActivityLogger(settings.activity_db_url)
        app.add_middleware(
            ActivityLogMiddleware,
            log=activity_log,
            trusted_proxies=settings.trusted_proxies,
        )
        logger.info("Activity logging: enabled (Postgres, lazy init)")
    else:
        logger.info("Activity logging: disabled (ACTIVITY_DB_URL not set)")

    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            limit_per_minute=settings.rate_limit_per_minute,
            trusted_proxies=settings.trusted_proxies,
        )
        logger.info(
            "Rate limit: %d req/min per IP (trusted proxies: %s)",
            settings.rate_limit_per_minute,
            settings.trusted_proxies,
        )
        # add_middleware wraps outermost-last, so this sits in front of the
        # per-minute limiter and runs first on every /mcp request.
        app.add_middleware(
            OrderRateLimitMiddleware,
            limit_per_hour=settings.order_rate_limit_per_hour,
        )
        logger.info(
            "Order rate limit: %d/hour per IP for kapruka_create_order",
            settings.order_rate_limit_per_hour,
        )
    else:
        logger.warning("Rate limit DISABLED")

    return app


def main() -> None:
    import uvicorn

    logger.info(
        "Starting Kapruka MCP server on %s:%s", settings.mcp_host, settings.mcp_port
    )
    uvicorn.run(
        build_app(),
        host=settings.mcp_host,
        port=settings.mcp_port,
        log_level=settings.log_level.lower(),
        access_log=False,
        proxy_headers=True,
        forwarded_allow_ips=",".join(settings.trusted_proxies),
    )


if __name__ == "__main__":
    main()
