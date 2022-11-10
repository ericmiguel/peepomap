#!/bin/bash

set -e
set -x

mypy peepomap
flake8 peepomap
black peepomap --check
pydocstyle peepomap
