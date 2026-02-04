#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

(uv run --prerelease=if-necessary-or-explicit main.py || pipx run uv run --prerelease=if-necessary-or-explicit main.py)