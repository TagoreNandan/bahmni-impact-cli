#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAG="$SCRIPT_DIR/rag"

# Run the three demo questions with stable single-result output
"$RAG" ask --best "Why is PatientService risky?"
"$RAG" ask --best "What business rules must be preserved here?"
"$RAG" ask --best "Is this service safe to optimize?"
