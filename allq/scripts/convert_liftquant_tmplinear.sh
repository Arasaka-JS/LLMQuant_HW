#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec python "${PROJECT_ROOT}/allq/tools/convert_liftquant_tmplinear.py" "$@"
