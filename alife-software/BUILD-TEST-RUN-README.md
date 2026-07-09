## FreeClaudeCode - Test, Linter, and Run

### 0. Prerequisites
The project uses uv for dependency management. Install it once if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Clone and set up
```bash
git clone https://github.com/alifesoftware/free-claude-code.git
cd free-claude-code

# Install all dependencies (creates .venv automatically )
uv sync
```

### 2. Run unit tests
```bash
# Full test suite (parallel, recommended)
uv run pytest tests/ -q

# Expected output: 1941 passed, 9 skipped

# Run a specific test file
uv run pytest tests/providers/test_xiaomimimo.py -v

# Run a specific test by name
uv run pytest tests/providers/test_xiaomimimo.py::test_default_base_url -v

# Run tests for a single provider
uv run pytest tests/providers/test_wandb_inference.py -v

# Run contract tests only (fast, catches wiring errors)
uv run pytest tests/contracts/ -v

# Run without parallelism (easier to read tracebacks)
uv run pytest tests/ --override-ini="addopts=" -v
```

### 3. Lint and format checks
```bash
# Check formatting (does not modify files)
uv run ruff format --check .

# Auto-format all files
uv run ruff format .

# Check for lint errors
uv run ruff check .

# Auto-fix fixable lint errors
uv run ruff check --fix .
```

### 4. Configure the project
Copy the example env file and fill in your API keys:
```bash
cp .env.example .env
```

Minimum required for DeepSeek:
```bash
# .env
MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=your-key-here
```

For MiMo:
```bash
MODEL=xiaomimimo/MiMo-7B-RL
XIAOMIMIMO_API_KEY=your-key-here
# Optional: Token Plan endpoint
# XIAOMIMIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
```

For W&B Inference:
```bash
MODEL=wandb_inference/deepseek-ai/DeepSeek-V3.1
WANDB_API_KEY=your-wandb-api-key
```

### 5. Run the server
```bash
# Start the FCC proxy server (default port 8080 )
uv run fcc-server

# Or with a custom port
uv run fcc-server --port 8080
```

Then point Claude Code at it:
```bash
# In your Claude Code config (~/.claude/settings.json or via env):
ANTHROPIC_BASE_URL=http://localhost:8080
ANTHROPIC_API_KEY=any-non-empty-string
```

### 6. Admin UI
Once the server is running, open http://localhost:8080/admin in your browser to configure providers, models, and routing tiers through the web interface.

### 7. Quick reference
| Task              | Command                                               |
| ----------------- | ----------------------------------------------------- |
| Install deps      | `uv sync`                                             |
| Run all tests     | `uv run pytest tests/ -q`                             |
| Run one test file | `uv run pytest tests/providers/test_xiaomimimo.py -v` |
| Format check      | `uv run ruff format --check .`                        |
| Auto-format       | `uv run ruff format .`                                |
| Lint check        | `uv run ruff check .`                                 |
| Start server      | `uv run fcc-server`                                   |
| Admin UI          | `http://localhost:8080/admin`                         |

