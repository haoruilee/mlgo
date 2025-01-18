#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -f go.mod ]; then
  go mod init go-mips-inference
fi

go mod tidy

go build -o mips_inference mips_inference.go

echo "Build complete. The executable is named 'mips_inference'."
