#!/bin/bash -e

set -x
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place peepomap tests scripts --exclude=__init__.py
black peepomap tests scripts
isort peepomap tests scripts
