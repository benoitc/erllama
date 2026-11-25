#!/bin/bash
set -euo pipefail
root=$HOME/erllama-tp/erllama/apps/erllama
otp=$HOME/erllama-tp/otp-28.3.1
c=$root/c_src
l=$c/llama.cpp
gcc -O2 -fPIC -I"$otp/usr/include" -I"$c" -I"$l/include" -I"$l/ggml/include" -c ctx.c -o ctx.o
g++ -O2 -std=c++17 -shared -fPIC -I"$otp/usr/include" -I"$l/include" -I"$l/ggml/include" -I"$l/common" -I"$l/vendor" -I"$l/vendor/nlohmann" probe.cpp ctx.o -o probe.so -L"$root/priv" -l:erllama_nif.so -Wl,-rpath,"$PWD/fast/erllama/priv"
"$otp/bin/erlc" -Werror dsprobe.erl
