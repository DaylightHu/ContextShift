#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -x .venv/bin/contextshift-gui ]; then
    exec .venv/bin/contextshift-gui
fi
if command -v uv >/dev/null 2>&1; then
    exec uv run --extra desktop --extra speech contextshift-gui
fi
if [ -x "$HOME/.local/bin/uv" ]; then
    exec "$HOME/.local/bin/uv" run --extra desktop --extra speech contextshift-gui
fi
printf '请先安装 uv，并在项目目录运行 uv sync --extra desktop --extra speech。\n'
read -r -p '按回车退出。'
