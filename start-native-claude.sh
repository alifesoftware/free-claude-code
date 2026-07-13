#!/bin/bash
# Native Claude instance — expensive tasks (reasoning, web dev, architecture, reviews)
# Uses your real ANTHROPIC_API_KEY from the environment

unset ANTHROPIC_BASE_URL
unset ANTHROPIC_AUTH_TOKEN
unset CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY

claude "$@"
