#!/bin/bash -e

set -x
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place peepomap scripts --exclude=__init__.py
black peepomap scripts
isort peepomap scripts
