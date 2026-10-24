#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if command -v uv >/dev/null 2>&1; then
    UV_CMD="uv"
    echo "uv found"
else
    UV_CMD="pipx run uv"
    echo "uv not found, using pipx"
fi

$UV_CMD sync --link-mode=symlink --upgrade
$UV_CMD run --prerelease=if-necessary-or-explicit main.py