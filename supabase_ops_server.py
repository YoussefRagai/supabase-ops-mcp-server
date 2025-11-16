#!/usr/bin/env python3
"""Simple Supabase Ops MCP Server - Manage Supabase Postgres via REST and SQL."""

import os
import sys
import json
import logging
import functools
from datetime import datetime, timezone
from urllib.parse import parse_qsl
import textwrap

import httpx
from rapidfuzz import fuzz
from mcp.server.fastmcp import FastMCP


LOG_LEVEL = os.environ.get("MCP_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("supabase_ops-server")

mcp = FastMCP("supabase_ops", stateless_http=True)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://kfeunworthagyqqkfxef.supabase.co").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_SCHEMA = os.environ.get("SUPABASE_SCHEMA", "public")
_timeout_env = os.environ.get("SUPABASE_TIMEOUT_SECONDS", "20")
try:
    DEFAULT_TIMEOUT_SECONDS = float(_timeout_env)
except ValueError:
    logger.warning("Invalid SUPABASE_TIMEOUT_SECONDS=%s, falling back to 20 seconds", _timeout_env)
    DEFAULT_TIMEOUT_SECONDS = 20.0

OUTPUT_DIR = os.environ.get("SUPABASE_OUTPUT_DIR", "/app/output")
FALLBACK_OUTPUT_DIR = os.environ.get("SUPABASE_FALLBACK_OUTPUT_DIR", "/tmp/supabase_ops")
MAX_INLINE_CHARS = int(os.environ.get("SUPABASE_INLINE_LIMIT", "6000"))

_EVENT_TYPE_CACHE = []
_EVENT_TYPE_CACHE_READY = False


_SCHEMA_LAST_REFRESHED = None

def _require_schema_ready_message():
    return ("⚠️ Please call supabase-ops:introspect_schema first to refresh the schema cache before "
            "running other Supabase tools.")

def _requires_schema_ready(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if not _SCHEMA_LAST_REFRESHED:
            return _require_schema_ready_message()
        return await func(*args, **kwargs)
    return wrapper


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _headers(prefer: str = "") -> dict:
    key = SUPABASE_SERVICE_ROLE_KEY.strip()
    if not key:
        return {}
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    if prefer.strip():
        headers["Prefer"] = prefer
    return headers


def _ensure_output_dir() -> str:
    for path in (OUTPUT_DIR, FALLBACK_OUTPUT_DIR):
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except Exception as exc:
            logger.error("Failed to ensure output directory %s: %s", path, exc)
    return FALLBACK_OUTPUT_DIR


def _generate_filename(prefix: str, extension: str) -> str:
    safe_prefix = prefix.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_prefix}_{stamp}.{extension}"


def _format_records(records, prefix: str = "supabase") -> str:
    try:
        formatted = json.dumps(records, indent=2)
    except TypeError:
        formatted = json.dumps(str(records))

    if len(formatted) <= MAX_INLINE_CHARS:
        return formatted

    directory = _ensure_output_dir()
    filename = _generate_filename(prefix, "json")
    path = os.path.join(directory, filename)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(formatted)
    except Exception as exc:
        logger.error("Failed to write output file: %s", exc)
        return formatted[:MAX_INLINE_CHARS] + f"\n... (failed to write full payload: {exc})"

    preview = formatted[:MAX_INLINE_CHARS]
    return f"{preview}\n... (truncated, full payload saved to {path})"


async def _request(method: str, path: str, params=None, json_body=None, prefer: str = "") -> tuple:
    if not SUPABASE_SERVICE_ROLE_KEY.strip():
        return False, "❌ Error: Missing service role key. Set Docker secret SUPABASE_SERVICE_ROLE_KEY."

    url = f"{SUPABASE_URL}{path}"
    headers = _headers(prefer)
    if not headers:
        return False, "❌ Error: Missing service role key. Set Docker secret SUPABASE_SERVICE_ROLE_KEY."

    timeout = httpx.Timeout(DEFAULT_TIMEOUT_SECONDS)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                headers=headers,
            )
        except Exception as exc:
            logger.error("Supabase request failed (%s %s): %s", method, path, exc)
            return False, f"❌ Error: {str(exc)}"

    if response.status_code >= 400:
        message = response.text
        try:
            payload = response.json()
            message = payload.get("message", payload)
        except ValueError:
            pass
        logger.error("Supabase error (%s %s): %s", method, path, message)
        return False, f"❌ Error: {message}"

    if not response.content:
        return True, {}

    try:
        return True, response.json()
    except ValueError:
        return True, response.text


def _sanitize_sql(query: str) -> str:
    cleaned = query.strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


async def _execute_sql(query: str) -> tuple:
    body = {"query": _sanitize_sql(query)}
    success, payload = await _request(
        method="POST",
        path="/rest/v1/rpc/pg_execute_sql",
        json_body=body,
    )

    if not success and "Could not find the function" in str(payload):
        hint = (
            "❌ Error: pg_execute_sql is unavailable. Enable the Database API in Supabase "
            "or create the pg_execute_sql function manually."
        )
        return False, hint

    return success, payload


def _parse_filters(filters: str) -> dict:
    if not filters.strip():
        return {}
    pairs = parse_qsl(filters, keep_blank_values=True)
    params = {}
    for key, value in pairs:
        params[key] = value
    return params


def _ensure_json_payload(raw: str) -> tuple:
    if not raw.strip():
        return False, "❌ Error: json_rows is required."
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, f"❌ Error: Invalid JSON payload ({exc})."
    return True, data


def _escape_sql(value: str) -> str:
    return value.replace("'", "''")


def _validate_int(value: str, label: str) -> tuple:
    if not value.strip():
        return False, f"❌ Error: {label} is required."
    try:
        parsed = int(value.strip())
        return True, parsed
    except ValueError:
        return False, f"❌ Error: {label} must be an integer."


def _team_cte_clause(team_id: str, team_name: str) -> tuple:
    team_id_value = team_id.strip()
    team_name_value = team_name.strip()
    if team_id_value:
        ok, parsed = _validate_int(team_id_value, "team_id")
        if not ok:
            return False, parsed
        return True, f"WITH target AS (SELECT {parsed}::bigint AS id)"
    if team_name_value:
        safe = _escape_sql(team_name_value)
        return True, f"WITH target AS (SELECT id FROM teams WHERE lower(name) = lower('{safe}') ORDER BY id LIMIT 1)"
    return False, "❌ Error: Provide team_id or team_name."


def _count_rows(payload) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return len(payload["rows"])
        if isinstance(payload.get("data"), list):
            return len(payload["data"])
    return 0


def _fuzzy_rank(rows: list, query: str, fields: list, limit_value: int = None) -> list:
    if not rows or not query.strip():
        return rows
    decorated = []
    lowered_query = query.strip().lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        parts = []
        for field in fields:
            value = row.get(field)
            if value:
                parts.append(str(value))
        if not parts:
            continue
        text = " ".join(parts)
        score = float(fuzz.WRatio(lowered_query, text.lower()))
        row_copy = dict(row)
        row_copy["_fuzzy_score"] = round(score, 2)
        decorated.append(row_copy)
    decorated.sort(key=lambda item: item.get("_fuzzy_score", 0), reverse=True)
    if limit_value:
        decorated = decorated[:limit_value]
    return decorated


async def _load_event_type_cache(force: bool = False) -> tuple:
    global _EVENT_TYPE_CACHE_READY, _EVENT_TYPE_CACHE
    if _EVENT_TYPE_CACHE_READY and not force:
        return True, ""
    success, payload = await _execute_sql(
        """
        select id, name, category_id
        from event_types
        order by name
        """
    )
    if not success:
        return False, payload
    if not isinstance(payload, list):
        return False, "❌ Error: Unexpected response loading event types."
    _EVENT_TYPE_CACHE = payload
    _EVENT_TYPE_CACHE_READY = True
    return True, ""


def _filter_event_types(query: str, limit_value: int, exact: bool) -> list:
    normalized = query.lower()
    results = []
    if exact:
        for row in _EVENT_TYPE_CACHE:
            row_name = str(row.get("name", "")).lower()
            if row_name == normalized:
                copy_row = dict(row)
                copy_row["_fuzzy_score"] = 100.0
                results.append(copy_row)
                if len(results) >= limit_value:
                    break
        return results

    scored = _fuzzy_rank(_EVENT_TYPE_CACHE, query, ["name"], limit_value)
    if scored:
        return scored
    return []


async def _find_team_id_by_name(name: str) -> tuple:
    safe = _escape_sql(name)
    sql = (
        "select id, name "
        "from teams "
        f"where lower(name) = lower('{safe}') "
        "order by id "
        "limit 1"
    )
    success, payload = await _execute_sql(sql)
    if not success:
        return False, payload, []
    if isinstance(payload, list) and payload:
        row = payload[0]
        return True, int(row.get("id")), payload

    fuzzy_sql = (
        "select id, name "
        "from teams "
        f"where lower(name) like lower('%{safe}%') "
        "order by name "
        "limit 20"
    )
    success_fb, payload_fb = await _execute_sql(fuzzy_sql)
    if success_fb and isinstance(payload_fb, list) and payload_fb:
        ranked = _fuzzy_rank(payload_fb, name, ["name"], limit_value=10)
        message = f"""⚠️ Team '{name}' not found. Suggestions:
{_format_records(ranked, prefix='team_suggestions')}

Summary: No exact team match for '{name}' at {_iso_timestamp()}."""
        return False, message, payload_fb

    return False, f"⚠️ Team '{name}' not found.", []


async def _resolve_team_filters(team_ids: str, team_names: str) -> tuple:
    ids = []
    suggestions = []
    tokens = []
    if team_ids.strip():
        tokens.extend(token.strip() for token in team_ids.split(",") if token.strip())
    if team_names.strip():
        tokens.extend(token.strip() for token in team_names.split(",") if token.strip())
    for token in tokens:
        if not token:
            continue
        try:
            parsed = int(token)
            if parsed not in ids:
                ids.append(parsed)
            continue
        except ValueError:
            ok, result, _ = await _find_team_id_by_name(token)
            if ok:
                if result not in ids:
                    ids.append(result)
            else:
                suggestions.append(result)
    if suggestions:
        return False, "\n".join(suggestions), []
    clause = ""
    if ids:
        joined = ", ".join(str(val) for val in ids)
        clause = f" and me.team_id in ({joined})"
    return True, clause, ids


@mcp.tool()
@_requires_schema_ready
async def list_tables(schema: str = "") -> str:
    """List tables visible in the Supabase schema."""
    target_schema = schema.strip() or SUPABASE_SCHEMA
    sql = (
        "select table_name, table_type "
        "from information_schema.tables "
        f"where table_schema = '{target_schema}' "
        "order by table_name;"
    )
    success, payload = await _execute_sql(sql)
    if not success:
        return payload

    return f"""📊 Tables:
{_format_records(payload)}

Summary: Retrieved table metadata for schema '{target_schema}' at {_iso_timestamp()}."""


@mcp.tool()
async def introspect_schema(tables: str = "", include_foreign_keys: str = "true", schema: str = "") -> str:
    """Inspect tables, columns, and foreign keys for the schema."""
    target_schema = schema.strip() or SUPABASE_SCHEMA

    table_names = []
    if tables.strip():
        for item in tables.split(","):
            name = item.strip()
            if name:
                table_names.append(name)

    schema_literal = _escape_sql(target_schema)

    table_filter = ""
    if table_names:
        escaped_names = ", ".join(f"'{_escape_sql(name)}'" for name in table_names)
        table_filter = f" and table_name in ({escaped_names})"

    columns_sql = textwrap.dedent(
        f"""
        select table_name, column_name, data_type, is_nullable, column_default
        from information_schema.columns
        where table_schema = '{schema_literal}'{table_filter}
        order by table_name, ordinal_position
        """
    ).strip()

    success, columns_payload = await _execute_sql(columns_sql)
    if not success:
        return columns_payload

    foreign_keys_payload = []
    include_fk = include_foreign_keys.strip().lower() != "false"
    if include_fk:
        fk_filter = ""
        if table_names:
            escaped_names = ", ".join(f"'{_escape_sql(name)}'" for name in table_names)
            fk_filter = f" and tc.table_name in ({escaped_names})"
        fk_sql = textwrap.dedent(
            f"""
            select
              tc.table_name,
              kcu.column_name,
              ccu.table_name as foreign_table,
              ccu.column_name as foreign_column,
              rc.update_rule,
              rc.delete_rule
            from information_schema.table_constraints tc
            join information_schema.key_column_usage kcu
              on tc.constraint_name = kcu.constraint_name
             and tc.table_schema = kcu.table_schema
            join information_schema.constraint_column_usage ccu
              on ccu.constraint_name = tc.constraint_name
             and ccu.table_schema = tc.table_schema
            join information_schema.referential_constraints rc
              on rc.constraint_name = tc.constraint_name
             and rc.constraint_schema = tc.constraint_schema
            where tc.constraint_type = 'FOREIGN KEY'
              and tc.table_schema = '{schema_literal}'{fk_filter}
            order by tc.table_name, kcu.ordinal_position
            """
        ).strip()
        success_fk, fk_payload = await _execute_sql(fk_sql)
        if not success_fk:
            return fk_payload
        foreign_keys_payload = fk_payload

    summary = {
        "schema": target_schema,
        "tables_requested": table_names if table_names else "all",
        "columns": columns_payload,
        "foreign_keys": foreign_keys_payload,
    }

    global _SCHEMA_LAST_REFRESHED
    _SCHEMA_LAST_REFRESHED = _iso_timestamp()

    return f"""📊 Schema Details:
{_format_records(summary, prefix="schema")}

Summary: Schema introspection completed at {_SCHEMA_LAST_REFRESHED}."""


@mcp.tool()
@_requires_schema_ready
async def describe_table(table: str = "", schema: str = "") -> str:
    """Describe columns for a Supabase table."""
    if not table.strip():
        return "❌ Error: table is required."

    target_schema = schema.strip() or SUPABASE_SCHEMA
    sql = (
        "select column_name, data_type, is_nullable, column_default "
        "from information_schema.columns "
        f"where table_schema = '{target_schema}' "
        f"and table_name = '{table.strip()}' "
        "order by ordinal_position;"
    )
    success, payload = await _execute_sql(sql)
    if not success:
        return payload

    return f"""📊 Columns:
{_format_records(payload)}

Summary: Column metadata fetched for {target_schema}.{table.strip()} at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def run_sql(query: str = "") -> str:
    """Execute SQL against the Supabase project."""
    if not query.strip():
        return "❌ Error: query is required."

    success, payload = await _execute_sql(query)
    if not success:
        return payload

    return f"""⚡ SQL Result:
{_format_records(payload)}

Summary: SQL executed at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def get_top_shooters(limit: str = "5") -> str:
    """List the players with the most Shoot events (event_id = 21)."""
    ok, parsed = _validate_int(limit, "limit")
    if not ok:
        return parsed

    query = f"""
SELECT p.name AS player_name,
       COUNT(*) AS shot_count
FROM match_events me
JOIN players p ON me.player_id = p.id
WHERE me.event_id = 21
GROUP BY p.name
ORDER BY shot_count DESC
LIMIT {parsed}
"""
    success, payload = await _execute_sql(query)
    if not success:
        return payload

    return f"""📊 Top Shooters:
{_format_records(payload, prefix="top_shooters")}

Summary: Calculated top {parsed} shooters at {_SCHEMA_LAST_REFRESHED or _iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def get_top_goal_scorers(limit: str = "5") -> str:
    """List players with the most goal outcomes in Shoot events."""
    ok, parsed = _validate_int(limit, "limit")
    if not ok:
        return parsed

    query = f"""
SELECT p.name AS player_name,
       COUNT(*) AS goal_count
FROM match_events me
JOIN players p ON me.player_id = p.id
JOIN event_results er ON me.result_id = er.id
WHERE me.event_id = 21
  AND lower(er.name) = 'goal'
GROUP BY p.name
ORDER BY goal_count DESC
LIMIT {parsed}
"""
    success, payload = await _execute_sql(query)
    if not success:
        return payload

    return f"""📊 Top Goal Scorers:
{_format_records(payload, prefix="top_goal_scorers")}

Summary: Calculated top {parsed} goal scorers at {_SCHEMA_LAST_REFRESHED or _iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def list_recommended_indexes() -> str:
    """Show recommended indexes to improve common Supabase queries."""
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_match_events_match_id ON match_events(match_id)",
        "CREATE INDEX IF NOT EXISTS idx_match_events_team_id ON match_events(team_id)",
        "CREATE INDEX IF NOT EXISTS idx_match_events_player_id ON match_events(player_id)",
        "CREATE INDEX IF NOT EXISTS idx_match_events_event_id ON match_events(event_id)",
        "CREATE INDEX IF NOT EXISTS idx_match_events_team_event ON match_events(team_id, event_id)",
        "CREATE INDEX IF NOT EXISTS idx_match_events_match_player ON match_events(match_id, player_id)",
    ]
    joined = "\n".join(f"{idx + 1}. {sql}" for idx, sql in enumerate(statements))
    return f"""🧱 Recommended Indexes:
{joined}

⚠️ Supabase REST API cannot execute DDL. Run these statements once in the Supabase SQL editor or via psql using your service-role key.

Summary: Index recommendations generated at {_SCHEMA_LAST_REFRESHED or _iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def select_rows(table: str = "", filters: str = "", limit: str = "", order: str = "") -> str:
    """Fetch rows from a Supabase table with optional filters."""
    if not table.strip():
        return "❌ Error: table is required."

    params = _parse_filters(filters)

    if limit.strip():
        try:
            limit_value = max(1, int(limit.strip()))
            params["limit"] = str(limit_value)
        except ValueError:
            return f"❌ Error: Invalid limit value: {limit}"

    if order.strip():
        params["order"] = order.strip()

    path = f"/rest/v1/{table.strip()}"
    success, payload = await _request("GET", path, params=params)
    if not success:
        return payload

    if isinstance(payload, list) and not payload:
        return f"⚠️ No rows found for {table.strip()}."

    return f"""📊 Rows:
{_format_records(payload)}

Summary: Retrieved data from {table.strip()} at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def insert_rows(table: str = "", json_rows: str = "") -> str:
    """Insert rows into a Supabase table."""
    if not table.strip():
        return "❌ Error: table is required."

    ok, payload = _ensure_json_payload(json_rows)
    if not ok:
        return payload

    prefer = "return=representation"
    path = f"/rest/v1/{table.strip()}"
    success, data = await _request("POST", path, json_body=payload, prefer=prefer)
    if not success:
        return data

    return f"""✅ Inserted:
{_format_records(data)}

Summary: Insert completed for {table.strip()} at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def upsert_rows(table: str = "", json_rows: str = "", on_conflict: str = "") -> str:
    """Upsert rows into a Supabase table using merge semantics."""
    if not table.strip():
        return "❌ Error: table is required."

    ok, payload = _ensure_json_payload(json_rows)
    if not ok:
        return payload

    params = {}
    if on_conflict.strip():
        params["on_conflict"] = on_conflict.strip()

    prefer = "return=representation,resolution=merge-duplicates"
    path = f"/rest/v1/{table.strip()}"
    success, data = await _request("POST", path, params=params, json_body=payload, prefer=prefer)
    if not success:
        return data

    return f"""✅ Upserted:
{_format_records(data)}

Summary: Upsert completed for {table.strip()} at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def delete_rows(table: str = "", filters: str = "") -> str:
    """Delete rows from a Supabase table with filters."""
    if not table.strip():
        return "❌ Error: table is required."

    params = _parse_filters(filters)
    if not params:
        return "❌ Error: filters are required to prevent full-table deletion."

    path = f"/rest/v1/{table.strip()}"
    success, payload = await _request("DELETE", path, params=params, prefer="return=representation")
    if not success:
        return payload

    if isinstance(payload, list) and not payload:
        return f"⚠️ No rows matched the delete filters for {table.strip()}."

    return f"""✅ Deleted:
{_format_records(payload)}

Summary: Delete completed for {table.strip()} at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def search_event_type(name: str = "", limit: str = "10", refresh: str = "") -> str:
    """Search event types by name substring with optional cache refresh."""
    if not name.strip():
        return "❌ Error: name is required."
    limit_value = 10
    if limit.strip():
        try:
            limit_value = max(1, int(limit.strip()))
        except ValueError:
            return f"❌ Error: Invalid limit value: {limit}"
    force_refresh = refresh.strip().lower() == "true"
    ok, message = await _load_event_type_cache(force_refresh)
    if not ok:
        return message
    matches = _filter_event_types(name.strip(), limit_value, exact=False)
    if not matches:
        return f"⚠️ No event types found containing '{name.strip()}'."
    return f"""📊 Event Types:
{_format_records(matches, prefix="event_types")}

Summary: Located {len(matches)} event types at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def get_event_type_id(name: str = "", refresh: str = "") -> str:
    """Lookup an event type ID by exact name (case-insensitive)."""
    if not name.strip():
        return "❌ Error: name is required."
    force_refresh = refresh.strip().lower() == "true"
    ok, message = await _load_event_type_cache(force_refresh)
    if not ok:
        return message
    matches = _filter_event_types(name.strip(), limit_value=5, exact=True)
    if matches:
        return f"""📊 Event Type Match:
{_format_records(matches, prefix="event_type_exact")}

Summary: Matched event type '{name.strip()}' at {_iso_timestamp()}."""
    suggestions = _filter_event_types(name.strip(), limit_value=5, exact=False)
    if suggestions:
        return f"""⚠️ Exact event type not found. Did you mean:
{_format_records(suggestions, prefix="event_type_suggestions")}

Summary: No exact match for '{name.strip()}' at {_iso_timestamp()}."""
    return f"⚠️ Event type '{name.strip()}' not found. Try search_event_type with a broader query."


@mcp.tool()
@_requires_schema_ready
async def get_match_shots(match_id: str = "", team_ids: str = "", team_names: str = "", event_name: str = "Shoot", limit: str = "") -> str:
    """Fetch shot-level data for a match optionally filtered to specific teams."""
    ok_match, match_val = _validate_int(match_id, "match_id")
    if not ok_match:
        return match_val
    event_label = event_name.strip() or "Shoot"
    ok_cache, message = await _load_event_type_cache(False)
    if not ok_cache:
        return message
    exact_matches = _filter_event_types(event_label, 1, exact=True)
    event_id = exact_matches[0]["id"] if exact_matches else None
    if not event_id:
        fuzzy_matches = _filter_event_types(event_label, 5, exact=False)
        if fuzzy_matches:
            return f"""⚠️ Event type '{event_label}' not found exactly. Suggestions:
{_format_records(fuzzy_matches, prefix='event_type_fuzzy')}

Summary: Please pick one of the suggested event types and retry."""
        return f"❌ Error: No event type matches '{event_label}'."

    ok_teams, clause, resolved_ids = await _resolve_team_filters(team_ids, team_names)
    if not ok_teams:
        return clause

    limit_clause = ""
    if limit.strip():
        ok_limit, lim_value = _validate_int(limit, "limit")
        if not ok_limit:
            return lim_value
        limit_clause = f" limit {lim_value}"

    sql = textwrap.dedent(
        f"""
        select
          me.id as event_id,
          me.match_id,
          me.team_id,
          t.name as team_name,
          me.player_id,
          coalesce(p.name, '') as player_name,
          me.x::float as x,
          me.y::float as y,
          coalesce(er.name, '') as shot_outcome,
          me.half,
          me.minute,
          me.second
        from match_events me
        join teams t on me.team_id = t.id
        left join players p on me.player_id = p.id
        left join event_results er on me.result_id = er.id
        where me.match_id = {match_val}
          and me.event_id = {int(event_id)}
          {clause}
        order by me.minute, me.second nulls last, me.id
        {limit_clause}
        """
    ).strip()

    success, payload = await _execute_sql(sql)
    if not success:
        return payload
    rows = payload if isinstance(payload, list) else []
    count = len(rows)
    summary = {
        "match_id": match_val,
        "event_type": event_label,
        "team_ids": resolved_ids,
        "row_count": count,
    }
    return f"""📊 Match Shots:
{_format_records(rows, prefix="match_shots")}

Summary: {_format_records(summary, prefix="match_shots_summary")}"""


@mcp.tool()
@_requires_schema_ready
async def get_player_id_from_name(name: str = "") -> str:
    """Find player IDs by exact name match."""
    if not name.strip():
        return "❌ Error: name is required."
    safe = _escape_sql(name.strip())
    sql = (
        "select id, name, nickname "
        "from players "
        f"where lower(name) = lower('{safe}') "
        "order by id"
    )
    success, payload = await _execute_sql(sql)
    if not success:
        return payload
    rows = payload if isinstance(payload, list) else []
    if rows:
        ranked = _fuzzy_rank(rows, name.strip(), ["name", "nickname"])
        count = len(ranked)
        return f"""📊 Player Lookup:
{_format_records(ranked, prefix="player_lookup")}

Summary: Found {count} record(s) for '{name.strip()}' at {_iso_timestamp()}."""

    fallback_sql = (
        "select id, name, nickname "
        "from players "
        f"where lower(name) like lower('%{safe}%') "
        "order by name "
        "limit 20"
    )
    success_fb, payload_fb = await _execute_sql(fallback_sql)
    if success_fb and isinstance(payload_fb, list) and payload_fb:
        ranked_fb = _fuzzy_rank(payload_fb, name.strip(), ["name", "nickname"], limit_value=10)
        return f"""⚠️ Exact player not found. Closest candidates:
{_format_records(ranked_fb, prefix="player_lookup_fuzzy")}

Summary: No exact match for '{name.strip()}' at {_iso_timestamp()}."""

    return f"⚠️ No player found matching '{name.strip()}'."


@mcp.tool()
@_requires_schema_ready
async def search_player_id_from_nickname(nickname: str = "", limit: str = "") -> str:
    """Search player IDs by nickname fragment."""
    if not nickname.strip():
        return "❌ Error: nickname is required."
    safe = _escape_sql(nickname.strip())
    lim_clause = ""
    if limit.strip():
        ok, lim = _validate_int(limit, "limit")
        if not ok:
            return lim
        lim_clause = f" limit {lim}"
    sql = (
        "select id, name, nickname "
        "from players "
        f"where nickname is not null and lower(nickname) like lower('%{safe}%') "
        "order by name"
        f"{lim_clause}"
    )
    success, payload = await _execute_sql(sql)
    if not success:
        return payload
    rows = payload if isinstance(payload, list) else []
    ranked = _fuzzy_rank(rows, nickname.strip(), ["nickname", "name"], limit_value=len(rows) or None)
    count = len(ranked)
    return f"""📊 Nickname Search:
{_format_records(ranked, prefix="player_nickname")}

Summary: Found {count} player(s) matching '{nickname.strip()}' at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def search_player_id_from_age(age: str = "") -> str:
    """Search player IDs by age in whole years."""
    ok, age_value = _validate_int(age, "age")
    if not ok:
        return age_value
    sql = textwrap.dedent(
        f"""
        select id, name, dob,
               cast(extract(year from age(current_date, dob)) as int) as age_years
        from players
        where dob is not null
          and cast(extract(year from age(current_date, dob)) as int) = {age_value}
        order by name
        """
    ).strip()
    success, payload = await _execute_sql(sql)
    if not success:
        return payload
    count = _count_rows(payload)
    return f"""📊 Players By Age:
{_format_records(payload, prefix="player_age")}

Summary: Found {count} player(s) aged {age_value} at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def search_primary_position_id(position_name: str = "") -> str:
    """Lookup position IDs by name."""
    if not position_name.strip():
        return "❌ Error: position_name is required."
    safe = _escape_sql(position_name.strip())
    sql = (
        "select id, name "
        "from positions "
        f"where lower(name) = lower('{safe}') "
        "order by id"
    )
    success, payload = await _execute_sql(sql)
    if not success:
        return payload
    rows = payload if isinstance(payload, list) else []
    if rows:
        ranked = _fuzzy_rank(rows, position_name.strip(), ["name"])
        count = len(ranked)
        return f"""📊 Position Lookup:
{_format_records(ranked, prefix="position_lookup")}

Summary: Located {count} record(s) for position '{position_name.strip()}' at {_iso_timestamp()}."""

    fuzzy_sql = (
        "select id, name "
        "from positions "
        f"where lower(name) like lower('%{safe}%') "
        "order by name "
        "limit 20"
    )
    success_fb, payload_fb = await _execute_sql(fuzzy_sql)
    if success_fb and isinstance(payload_fb, list) and payload_fb:
        ranked_fb = _fuzzy_rank(payload_fb, position_name.strip(), ["name"], limit_value=10)
        return f"""⚠️ Position not found. Closest matches:
{_format_records(ranked_fb, prefix="position_lookup_fuzzy")}

Summary: No exact position match for '{position_name.strip()}' at {_iso_timestamp()}."""

    return f"⚠️ No position found matching '{position_name.strip()}'."
    return f"""📊 Position Lookup:
{_format_records(payload, prefix="position_lookup")}

Summary: Located {count} record(s) for position '{position_name.strip()}' at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def search_secondary_position_id(position_name: str = "") -> str:
    """Alias for searching position IDs by name."""
    return await search_primary_position_id(position_name)


@mcp.tool()
@_requires_schema_ready
async def search_player_age_by_id(player_id: str = "") -> str:
    """Fetch player age, name, and DOB by player ID."""
    ok, pid = _validate_int(player_id, "player_id")
    if not ok:
        return pid
    sql = textwrap.dedent(
        f"""
        select id, name, nickname, dob,
               cast(extract(year from age(current_date, dob)) as int) as age_years
        from players
        where id = {pid}
        """
    ).strip()
    success, payload = await _execute_sql(sql)
    if not success:
        return payload
    if isinstance(payload, list) and not payload:
        return f"⚠️ No player found with id {pid}."
    return f"""📊 Player Age:
{_format_records(payload, prefix="player_age")}

Summary: Retrieved age info for player {pid} at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def search_team_id_from_name(name: str = "", limit: str = "") -> str:
    """Search team IDs by name fragment."""
    if not name.strip():
        return "❌ Error: name is required."
    safe = _escape_sql(name.strip())
    lim_clause = ""
    if limit.strip():
        ok, lim = _validate_int(limit, "limit")
        if not ok:
            return lim
        lim_clause = f" limit {lim}"
    sql = (
        "select id, name "
        "from teams "
        f"where lower(name) like lower('%{safe}%') "
        "order by name"
        f"{lim_clause}"
    )
    success, payload = await _execute_sql(sql)
    if not success:
        return payload
    rows = payload if isinstance(payload, list) else []
    ranked = _fuzzy_rank(rows, name.strip(), ["name"], limit_value=len(rows) or None)
    count = len(ranked)
    return f"""📊 Team Search:
{_format_records(ranked, prefix="team_lookup")}

Summary: Found {count} team(s) matching '{name.strip()}' at {_iso_timestamp()}."""


async def _lineup_query(role: str, match_id: str, team_id: str, team_name: str) -> str:
    ok_match, match_val = _validate_int(match_id, "match_id")
    if not ok_match:
        return match_val
    ok_cte, clause = _team_cte_clause(team_id, team_name)
    if not ok_cte:
        return clause
    sql = textwrap.dedent(
        f"""
        {clause}
        select ml.player_id,
               coalesce(p.name, '') as player_name,
               ml.shirt_number,
               ml.primary_position_id,
               ml.primary_position_name,
               ml.secondary_position_id,
               ml.secondary_position_name
        from target t
        join match_lineups ml
          on ml.team_id = t.id
        left join players p
          on ml.player_id = p.id
        where ml.match_id = {match_val}
          and ml.role = '{role}'
        order by ml.shirt_number nulls last, player_name
        """
    ).strip()
    success, payload = await _execute_sql(sql)
    if not success:
        return payload
    if isinstance(payload, list) and not payload:
        label = f"match {match_val}"
        if team_name.strip():
            label += f" and team '{team_name.strip()}'"
        elif team_id.strip():
            label += f" and team id {team_id.strip()}"
        return f"⚠️ No {role} players found for {label}."
    return f"""📊 Lineup ({role}):
{_format_records(payload, prefix=f"lineup_{role}")}

Summary: Retrieved {role} list for match {match_val} at {_iso_timestamp()}."""


@mcp.tool()
@_requires_schema_ready
async def get_starting_id(match_id: str = "", team_id: str = "", team_name: str = "") -> str:
    """List starters (player IDs) for a given match and team."""
    return await _lineup_query("starter", match_id, team_id, team_name)


@mcp.tool()
@_requires_schema_ready
async def get_sub_id(match_id: str = "", team_id: str = "", team_name: str = "") -> str:
    """List substitutes (player IDs) for a given match and team."""
    return await _lineup_query("sub", match_id, team_id, team_name)


async def _team_record(team_id: str, team_name: str, metric: str) -> str:
    ok_cte, clause = _team_cte_clause(team_id, team_name)
    if not ok_cte:
        return clause
    outcome_case = {
        "wins": "(m.home_team_id = t.id and m.home_score > m.away_score) or (m.away_team_id = t.id and m.away_score > m.home_score)",
        "draws": "(m.home_team_id = t.id or m.away_team_id = t.id) and m.home_score = m.away_score",
        "loss": "(m.home_team_id = t.id and m.home_score < m.away_score) or (m.away_team_id = t.id and m.away_score < m.home_score)",
    }.get(metric)
    if not outcome_case:
        return "❌ Error: Unsupported metric requested."
    sql = textwrap.dedent(
        f"""
        {clause}
        SELECT
          (SELECT count(*) FROM target) AS team_count,
          COALESCE(SUM(CASE WHEN {outcome_case} THEN 1 ELSE 0 END),0) AS result_count
        FROM target t
        LEFT JOIN matches m
          ON m.home_team_id = t.id OR m.away_team_id = t.id;
        """
    ).strip()
    success, payload = await _execute_sql(sql)
    if not success:
        return payload
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        record = payload[0]
        team_count = record.get("team_count", 0)
        result = record.get("result_count", 0)
        if not team_count:
            return "⚠️ Team not found."
        label = {"wins": "wins", "draws": "draws", "loss": "losses"}[metric]
        content = {"count": int(result)}
        return f"""📊 Team {label.title()}:
{json.dumps(content, indent=2)}

Summary: Computed {label} at {_iso_timestamp()}."""
    return f"⚠️ Unexpected response: {_format_records(payload, prefix='team_record')}"


@mcp.tool()
@_requires_schema_ready
async def get_wins(team_id: str = "", team_name: str = "") -> str:
    """Count wins for a team across all matches."""
    return await _team_record(team_id, team_name, "wins")


@mcp.tool()
@_requires_schema_ready
async def get_draws(team_id: str = "", team_name: str = "") -> str:
    """Count draws for a team across all matches."""
    return await _team_record(team_id, team_name, "draws")


@mcp.tool()
@_requires_schema_ready
async def get_loss(team_id: str = "", team_name: str = "") -> str:
    """Count losses for a team across all matches."""
    return await _team_record(team_id, team_name, "loss")


if __name__ == "__main__":
    logger.info("Starting Supabase Ops MCP server...")
    try:
        mcp.run(transport="stdio")
    except Exception as exc:
        logger.error("Server error: %s", exc, exc_info=True)
        sys.exit(1)
