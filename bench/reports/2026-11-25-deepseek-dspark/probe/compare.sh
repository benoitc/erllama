#!/bin/bash
set -eu
cd /tmp/erllama-dspark-layers
for part in pre one; do
    /usr/bin/scp -q "10.100.168.1:$PWD/split-$part.f32" "follower-$part.f32"
done
exec python3 compare.py
