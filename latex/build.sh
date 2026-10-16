#!/bin/sh
# Build main.pdf (English) and main_ja.pdf (Japanese) with Docker in WSL (texlive/texlive image).
# From Windows: wsl sh latex/build.sh
set -e
cd "$(dirname "$0")"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/workdir -w /workdir \
  texlive/texlive:latest latexmk main.tex main_ja.tex
cp build/main.pdf main.pdf
cp build/main_ja.pdf main_ja.pdf
