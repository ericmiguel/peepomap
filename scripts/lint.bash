#!/bin/bash

set -e
set -x

mypy peepomap --enable-incomplete-features
flake8 peepomap tests
black peepomap tests --check
pydocstyle peepomap
