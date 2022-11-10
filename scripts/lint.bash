#!/bin/bash

set -e
set -x

mypy peepomap
flake8 peepomap tests
black peepomap tests --check
pydocstyle peepomap
