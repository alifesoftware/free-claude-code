#!/bin/bash
# DeepSeek instance — cheap tasks (scripts, tests, tool calling, automations)

export ANTHROPIC_BASE_URL="http://localhost:8082"
export ANTHROPIC_AUTH_TOKEN="freecc"
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY="1"
export CLAUDE_CODE_AUTO_COMPACT_WINDOW="190000"

claude "$@"
