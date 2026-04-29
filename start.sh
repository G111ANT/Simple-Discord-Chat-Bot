#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if (uv self version) ; then
    echo "uv found"
else
    echo "uv not found"
    alias uv="pipx run uv"
fi

uv sync --link-mode=symlink --upgrade
uv run --prerelease=if-necessary-or-explicit main.py