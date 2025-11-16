# Supabase Ops MCP Server Notes

## Overview
- **Service name:** Supabase Ops
- **Server entrypoint:** `supabase_ops_server.py`
- **Purpose:** Provide schema exploration, SQL execution, and CRUD helpers for a Supabase Postgres instance.

## Configuration
- Env vars:
  - `SUPABASE_URL` (defaults to `https://kfeunworthagyqqkfxef.supabase.co`)
  - `SUPABASE_SERVICE_ROLE_KEY` (required)
  - `SUPABASE_SCHEMA` (defaults to `public`)
  - `SUPABASE_TIMEOUT_SECONDS` (defaults to `20`)
- Store `SUPABASE_SERVICE_ROLE_KEY` as Docker Desktop secret and map it to the container environment.

## Tools
- `list_tables(schema="")`
  - Executes `information_schema.tables` query via `pg_execute_sql`.
- `describe_table(table="", schema="")`
  - Lists column metadata from `information_schema.columns`.
- `introspect_schema(tables="", include_foreign_keys="", schema="")`
  - Returns columns and FK relationships for the schema subset.
- `run_sql(query="")`
  - Pass-through SQL executor using `pg_execute_sql`.
- `select_rows(table="", filters="", limit="", order="")`
  - Wraps `/rest/v1/<table>` GET with PostgREST filters (`col=eq.value` syntax).
- `insert_rows(table="", json_rows="")`
  - POST insert with `return=representation`.
- `upsert_rows(table="", json_rows="", on_conflict="")`
  - POST merge upsert using `resolution=merge-duplicates`.
- `delete_rows(table="", filters="")`
  - DELETE scoped by filters; requires filter guard.

## Error Handling
- Missing secrets surfaced with friendly `❌` messages.
- HTTP errors return response body details.
- SQL execution catches missing `pg_execute_sql` function and guides enabling the Database API.

## Logging
- Configured at INFO level to stderr.
- Logs key actions (SQL execution, REST requests) without leaking payload contents.

## Docker Runtime
- Based on `python:3.11-slim`.
- Installs `mcp[cli]` and `httpx`.
- Runs as non-root user `mcpuser`.
- Entrypoint: `python supabase_ops_server.py`.

## Testing Checklist
```bash
export SUPABASE_SERVICE_ROLE_KEY="service-role-key"
python supabase_ops_server.py  # start stdio server
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python supabase_ops_server.py
```

## Maintenance
- Keep tool docstrings single-line.
- Preserve emoji status hints for user feedback.
- When adding new REST helpers, reuse `_request` and `_parse_filters`.
- Ensure new SQL helpers acknowledge the pg_execute_sql requirement.
