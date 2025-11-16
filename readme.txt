# Supabase Ops MCP Server

A Model Context Protocol (MCP) server that manages Supabase Postgres schemas, data, and SQL execution through the Supabase REST interface.

## Purpose

This MCP server provides a secure interface for AI assistants to inspect schemas, run queries, and perform data mutations against a Supabase-hosted Postgres instance.

## Features

### Current Implementation

- **`list_tables`** - Returns table names and types for the target schema.
- **`describe_table`** - Shows column metadata, data types, and default values.
- **`introspect_schema`** - Retrieves columns and foreign-key relationships for selected tables.
- **`run_sql`** - Executes arbitrary SQL using Supabase's `pg_execute_sql` endpoint.
- **`select_rows`** - Reads table data with optional filters, ordering, and limits.
- **`insert_rows`** - Inserts JSON payloads and echoes inserted rows.
- **`upsert_rows`** - Performs merge upserts with optional `on_conflict` keys.
- **`delete_rows`** - Deletes rows matching PostgREST-style filters.
- **Lookup helpers** – `get_player_id_from_name`, `search_player_id_from_nickname`, `search_player_id_from_age`, `search_player_age_by_id`.
- **Position and team helpers** – `search_primary_position_id`, `search_secondary_position_id`, `search_team_id_from_name`.
- **Lineup helpers** – `get_starting_id`, `get_sub_id`.
- **Record helpers** – `get_wins`, `get_draws`, `get_loss`.

## Prerequisites

- Docker Desktop with MCP Toolkit enabled
- Docker MCP CLI plugin (`docker mcp`)
- Supabase Database API (pg_execute_sql) enabled in the project
- Docker secrets configured for `SUPABASE_SERVICE_ROLE_KEY`

## Installation

See the step-by-step instructions provided with the files.

## Usage Examples

In Claude Desktop, you can ask:

- "Show me every table in Supabase schema public."
- "Describe the columns for the customers table."
- "Introspect Supabase schema to find foreign keys across players and match_events."
- "Insert a new customer row into Supabase with this JSON payload."
- "Upsert orders on order_id with this data."
- "Delete inactive users where status=eq.inactive."
- "Run SQL to create a reporting table in Supabase."

## Architecture

Claude Desktop → MCP Gateway → Supabase Ops MCP Server → Supabase REST (PostgREST / pg_execute_sql)
↓
Docker Desktop Secrets
(SUPABASE_SERVICE_ROLE_KEY)

## Development

### Local Testing

```bash
# Set environment variables for testing
export SUPABASE_SERVICE_ROLE_KEY="service-role-key"
export SUPABASE_URL="https://kfeunworthagyqqkfxef.supabase.co"

# Run directly
python supabase_ops_server.py

# Test MCP protocol
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python supabase_ops_server.py
```

### Adding New Tools

1. Add the function to `supabase_ops_server.py`
2. Decorate with `@mcp.tool()`
3. Update the catalog entry with the new tool name
4. Rebuild the Docker image

### Troubleshooting

**Tools Not Appearing**
- Verify Docker image built successfully
- Check catalog and registry files
- Ensure Claude Desktop config includes custom catalog
- Restart Claude Desktop

**Authentication Errors**
- Verify secrets with `docker mcp secret ls`
- Ensure the service role key matches the Supabase project

**SQL Execution Errors**
- Confirm the `pg_execute_sql` function is enabled in the Supabase Database API
- Ensure SQL statements end with semicolons when required

### Security Considerations

- Service role key stored in Docker Desktop secrets
- No credentials logged
- Runs as non-root user in container
- Errors returned without exposing sensitive payloads

### License

MIT License
