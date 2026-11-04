#!/usr/bin/env bash
set -euo pipefail
# Historical filename only: the current recipe uses a permanently frozen 32B teacher.
exec bash "$(dirname "$0")/run_head_opd_8gpu.sh" "$@"
