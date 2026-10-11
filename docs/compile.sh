#!/usr/bin/env bash
# Build the paper (docs/paper.pdf). Run from anywhere:  bash docs/compile.sh
set -euo pipefail
cd "$(dirname "$0")"

latexmk -pdf -interaction=nonstopmode paper.tex
latexmk -c paper.tex          # remove the .aux/.log/... files, keep the PDF
echo "Built docs/paper.pdf"
