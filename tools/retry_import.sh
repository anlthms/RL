#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Retry imports in fresh processes to tolerate transient filesystem misses.
set -eu

MODULE=${1:?"usage: bash tools/retry_import.sh module.name"}
ATTEMPTS=${NRL_IMPORT_PREFLIGHT_ATTEMPTS:-8}
DELAY_S=${NRL_IMPORT_PREFLIGHT_DELAY_S:-1.5}
UV_BIN=${NRL_IMPORT_PREFLIGHT_UV_BIN:-uv}

if [[ ! "${MODULE}" =~ ^[A-Za-z_][A-Za-z0-9_.]*$ ]]; then
  echo "invalid Python module name: ${MODULE}" >&2
  exit 2
fi
if [[ ! "${ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NRL_IMPORT_PREFLIGHT_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
  if "${UV_BIN}" run python -c "import ${MODULE}"; then
    echo "import preflight passed for ${MODULE} (${attempt}/${ATTEMPTS})"
    exit 0
  fi
  if ((attempt < ATTEMPTS)); then
    echo "import preflight failed for ${MODULE} (${attempt}/${ATTEMPTS}); retrying" >&2
    sleep "${DELAY_S}"
  fi
done

echo "import preflight failed for ${MODULE} after ${ATTEMPTS} attempts" >&2
exit 1
