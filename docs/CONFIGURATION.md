# Silica Configuration System

Silica provides a PostgreSQL-compatible configuration system for runtime parameter management. Configuration can be set through:

1. **Configuration file** (`silica.conf`) — persistent defaults
2. **SQL commands** (`SET`, `SHOW`, `RESET`) — session overrides
3. **Command-line options** — startup overrides (future)

---

## Configuration File (silica.conf)

### File Locations

Silica searches for `silica.conf` in the following locations (in order):

1. `./silica.conf` — current directory
2. `~/.config/silica/silica.conf` — user config directory
3. `/etc/silica/silica.conf` — system config directory

The first file found is used. If no file is found, built-in defaults are used.

### File Format

The configuration file uses INI-style syntax:

```ini
# Comment lines start with # or ;
parameter_name = value
parameter_name: value  # Colon separator also supported
```

**Features:**
- Comments: `#` or `;` prefix
- Separators: `=` or `:` between parameter and value
- Whitespace: trimmed automatically
- Quoted strings: preserve internal whitespace using `"..."` or `'...'`
- Multiline values: use backslash `\` at end of line
- Inline comments: `parameter = value # comment`

**Example:**

```ini
# Memory configuration
work_mem = 8MB

# Connection settings
max_connections = 200

# Schema search path (multiline)
search_path = public, \
              admin, \
              test

# Application identification
application_name = "My Application"
```

### Example File

An example configuration file is provided: [`silica.conf.example`](../silica.conf.example)

Copy it to one of the standard locations:

```bash
cp silica.conf.example ./silica.conf
```

---

## SQL Configuration Commands

### SET — Change a parameter

Change a configuration parameter for the current session:

```sql
SET parameter_name = 'value';
SET parameter_name TO 'value';  -- PostgreSQL-compatible syntax
```

**Examples:**

```sql
SET work_mem = '16MB';
SET statement_timeout = '5000';  -- 5 seconds in milliseconds
SET search_path = 'public, admin';
SET application_name = 'MyApp';
```

**Scope:** Session-level only. The change persists until:
- The session ends
- The parameter is reset with `RESET`
- The parameter is changed again with another `SET`

**Note:** Runtime `SET` overrides the config file value. The config file value is not modified.

### SHOW — Display current value

Display the current value of a parameter:

```sql
SHOW parameter_name;
```

Display all parameters:

```sql
SHOW ALL;
```

**Examples:**

```sql
SHOW work_mem;
-- Output: work_mem = 8MB

SHOW statement_timeout;
-- Output: statement_timeout = 0

SHOW ALL;
-- Output: (table of all parameters)
```

### RESET — Restore default value

Reset a parameter to its default value from the config file (or built-in default if no config file):

```sql
RESET parameter_name;
```

Reset all parameters:

```sql
RESET ALL;
```

**Examples:**

```sql
SET work_mem = '32MB';
SHOW work_mem;
-- Output: work_mem = 32MB

RESET work_mem;
SHOW work_mem;
-- Output: work_mem = 8MB (config file value or built-in default)
```

---

## Supported Parameters

### Memory Configuration

| Parameter | Type | Default | Min | Max | Hot-reload | Description |
|-----------|------|---------|-----|-----|------------|-------------|
| `work_mem` | SIZE | 4MB | 64KB | 2GB | ✅ YES | Memory for query workspaces (sorts, hashes, window functions) |

**Size values** accept suffixes: `KB`, `MB`, `GB`, or plain bytes.

Examples: `8MB`, `8192KB`, `8388608`

### Connection Settings

| Parameter | Type | Default | Min | Max | Hot-reload | Description |
|-----------|------|---------|-----|-----|------------|-------------|
| `max_connections` | INTEGER | 100 | 1 | 10000 | ❌ NO (restart required) | Maximum concurrent connections |

### Query Execution

| Parameter | Type | Default | Min | Max | Hot-reload | Description |
|-----------|------|---------|-----|-----|------------|-------------|
| `statement_timeout` | INTEGER | 0 | 0 | 2147483647 | ✅ YES | Statement timeout in milliseconds (0 = no timeout) |

### Schema & Naming

| Parameter | Type | Default | Hot-reload | Description |
|-----------|------|---------|------------|-------------|
| `search_path` | TEXT | `public` | ✅ YES | Schema search order for unqualified names |

### Client Connection Defaults

| Parameter | Type | Default | Hot-reload | Description |
|-----------|------|---------|------------|-------------|
| `application_name` | TEXT | `""` | ✅ YES | Application name for statistics and logs |

---

## Hot-Reload Behavior

**Hot-reloadable parameters** (marked ✅ YES) can be changed without restarting the database:
- Changes in `silica.conf` are applied immediately when the file is saved
- Runtime `SET` commands take effect immediately for the current session
- `RESET` restores the current config file value

**Restart-required parameters** (marked ❌ NO) need a database restart:
- Changes in `silica.conf` are ignored until the database is restarted
- Runtime `SET` commands are rejected with an error
- These parameters affect global database state that cannot change at runtime (e.g., `max_connections`)

---

## Parameter Types

### INTEGER

Numeric values with optional range constraints.

**Format:** Plain integer string

**Examples:**
```sql
SET max_connections = '200';
SET statement_timeout = '5000';
```

### SIZE

Memory sizes with unit parsing.

**Format:** Number + optional unit suffix (`KB`, `MB`, `GB`)

**Examples:**
```sql
SET work_mem = '8MB';
SET work_mem = '8192KB';  -- equivalent to 8MB
SET work_mem = '8388608';  -- 8MB in bytes
```

**Units:**
- `KB` = 1024 bytes
- `MB` = 1024 KB = 1,048,576 bytes
- `GB` = 1024 MB = 1,073,741,824 bytes

### TEXT

Arbitrary string values.

**Format:** Any string (quotes optional, but preserve whitespace if used)

**Examples:**
```sql
SET search_path = 'public, admin';
SET application_name = 'My Application';
```

### BOOLEAN

True/false values (reserved for future use).

**Format:** `true`, `false`, `on`, `off`, `1`, `0` (case-insensitive)

---

## Configuration Lifecycle

1. **Database startup:**
   - Load built-in default values
   - Search for `silica.conf` in standard locations
   - If found, apply config file values (override defaults)

2. **Session start:**
   - Each new session inherits the current configuration state
   - Session-level `SET` commands override config for that session only

3. **Runtime changes:**
   - `SET` command: changes parameter for current session
   - File modification: hot-reloadable parameters are updated globally (future feature)

4. **Session end:**
   - Session-level overrides are discarded
   - Next session starts with current config state

---

## Advanced Usage

### Precedence Order

Parameter values are resolved in this order (highest to lowest precedence):

1. Session-level `SET` command
2. Config file value (`silica.conf`)
3. Built-in default value

**Example:**

```
# Built-in default: work_mem = 4MB
# Config file:      work_mem = 8MB
# Session SET:      work_mem = 16MB

SHOW work_mem;  → 16MB (session override active)
RESET work_mem;
SHOW work_mem;  → 8MB (config file value restored)
```

### Validation

Configuration values are validated when set:

- **Type checking:** Integer parameters reject non-numeric values
- **Range checking:** Parameters with min/max bounds reject out-of-range values
- **Format checking:** Size parameters require valid unit suffixes

**Error examples:**

```sql
SET max_connections = 'abc';
-- ERROR: InvalidType

SET max_connections = '50000';
-- ERROR: OutOfRange (max is 10000)

SET work_mem = '4XB';
-- ERROR: InvalidSizeFormat (unknown unit XB)
```

### Future Features

- **File watching:** Automatic hot-reload when `silica.conf` is modified
- **Command-line overrides:** `silica --work-mem=16MB`
- **Boolean parameters:** Feature flags and toggles
- **Contextual parameters:** Database-level, role-level, or query-level overrides

---

## Migration from PostgreSQL

Silica's configuration system is designed to be PostgreSQL-compatible where possible:

**Compatible:**
- `SET` / `SHOW` / `RESET` syntax
- Parameter names (e.g., `work_mem`, `statement_timeout`, `search_path`)
- Size unit suffixes (`KB`, `MB`, `GB`)

**Differences:**
- Smaller parameter set (Silica implements a subset of PostgreSQL's ~300+ parameters)
- No `pg_settings` view (use `SHOW ALL` instead)
- No `ALTER SYSTEM SET` command (modify `silica.conf` directly)
- No `postgresql.auto.conf` (runtime changes are session-level only)

---

## Troubleshooting

### Config file not found

If Silica doesn't find a config file, it uses built-in defaults. To verify:

```sql
SHOW work_mem;
-- If output is 4194304 (4MB in bytes), built-in default is used
```

To use a config file, create it in one of the standard locations:

```bash
mkdir -p ~/.config/silica
cp silica.conf.example ~/.config/silica/silica.conf
```

### Config file ignored

If changes to `silica.conf` are not taking effect:

1. **Restart-required parameter:** Restart the database
2. **Session override active:** Use `RESET parameter_name` to restore file value
3. **File location:** Ensure file is in one of the search paths (Silica reads the first match)
4. **Syntax error:** Check for invalid syntax (missing `=`, unquoted special characters, etc.)

To debug config file loading, check startup logs for warnings:

```
Warning: Failed to load config from /path/to/silica.conf: InvalidSyntax
```

### Parameter not recognized

If `SET` or `SHOW` returns "UnknownParameter":

1. Check spelling (parameter names are case-sensitive by default)
2. Verify the parameter exists in the supported list (above)
3. The parameter may not be implemented yet (Silica is under active development)

---

## See Also

- [`silica.conf.example`](../silica.conf.example) — Example configuration file
- [`src/config/manager.zig`](../src/config/manager.zig) — Configuration manager implementation
- [`src/config/file.zig`](../src/config/file.zig) — Config file parser
- [PostgreSQL Configuration Documentation](https://www.postgresql.org/docs/current/runtime-config.html)
