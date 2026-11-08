#!/usr/bin/env bash
set -euo pipefail
# Keep the original entry point working; both names run the same NVIDIA pipeline.
exec bash "$(dirname "$0")/run_head_opd_8gpu.sh" "$@"
