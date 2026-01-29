#!/bin/bash

mkdir -p results
for t in 64 128 256; do
  for n in 16 32 64 128 256 512 1024 2048 4096; do
    uv run python multiplebodyproblem/main.py --headless -n $n --tpb $t > results/raport_${n}_$t.txt
  done
done

