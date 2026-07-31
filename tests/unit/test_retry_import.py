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

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "tools" / "retry_import.sh"


def test_retry_import_uses_fresh_processes(tmp_path: Path) -> None:
    state = tmp_path / "attempts"
    uv_stub = tmp_path / "uv"
    uv_stub.write_text(
        "#!/bin/bash\n"
        'count=$(cat "${RETRY_STATE}" 2>/dev/null || echo 0)\n'
        "count=$((count + 1))\n"
        'echo "${count}" > "${RETRY_STATE}"\n'
        '[ "${count}" -ge 3 ]\n'
    )
    uv_stub.chmod(0o755)
    env = {
        **os.environ,
        "NRL_IMPORT_PREFLIGHT_UV_BIN": str(uv_stub),
        "NRL_IMPORT_PREFLIGHT_ATTEMPTS": "4",
        "NRL_IMPORT_PREFLIGHT_DELAY_S": "0",
        "RETRY_STATE": str(state),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT), "nemo_rl.algorithms.grpo"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert state.read_text().strip() == "3"
    assert "import preflight passed" in result.stdout


def test_retry_import_rejects_invalid_module_name() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "module;false"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "invalid Python module name" in result.stderr
