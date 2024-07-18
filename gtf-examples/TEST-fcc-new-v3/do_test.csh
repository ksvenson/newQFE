#!/bin/tcsh

ising_cubic \
  -X 32 -Y 32 -Z 32 -S 1009 -B 0.103 \
  -C 0. -D 0. -E 0. \
  -F 1.00 -G 1.00 -H 1.00 -I 0.94 -J 0.94 -K 0.94 \
  -L 0. -M 0. -N 0. -O 0. \
  -h 2000 -t 50000 -w 140 \
  -d data \
  >! OUT-32
