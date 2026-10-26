#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -shared -fPIC -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -o libfast_graph.so fast_graph.cpp -lz
echo "built $(pwd)/libfast_graph.so"
