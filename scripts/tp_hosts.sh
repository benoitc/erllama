#!/usr/bin/env bash
# Prepare Linux CUDA hosts for the tensor-parallel tests, without root.
# Everything goes under ~/erllama-tp on each host (rm -rf removes it).
#
#   scripts/tp_hosts.sh setup HOST...   NCCL (pip wheel), OpenSSL, OTP 28.3.1, rebar3
#   scripts/tp_hosts.sh sync  HOST...   copy this checkout to ~/erllama-tp/erllama
#   scripts/tp_hosts.sh build HOST...   build it with CUDA and NCCL (test profile)
#   scripts/tp_hosts.sh env             print the environment a rank node needs
#
# `setup' runs in the background on each host and logs to
# ~/erllama-tp/setup.log; the other commands run in the foreground.
# CUDA_ARCH (default 121, for GB10) and JOBS (default 6) tune the build.
set -euo pipefail

CMD=${1:?usage: scripts/tp_hosts.sh setup|sync|build|env HOST...}
shift || true
CUDA_ARCH=${CUDA_ARCH:-121}
JOBS=${JOBS:-6}

remote_setup() {
    cat <<'EOF'
set -euo pipefail
T=$HOME/erllama-tp
mkdir -p "$T"
cd "$T"
echo "[nccl] $(date)"
[ -d venv ] || python3 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q nvidia-nccl-cu13
W=$(echo "$T"/venv/lib/python3*/site-packages/nvidia/nccl)
mkdir -p nccl/lib
ln -sfn "$W/include" nccl/include
ln -sfn "$W/lib/libnccl.so.2" nccl/lib/libnccl.so.2
ln -sfn libnccl.so.2 nccl/lib/libnccl.so
venv/bin/pip show nvidia-nccl-cu13 | grep -E '^(Name|Version)'
echo "[openssl] $(date)"
if [ ! -x openssl/bin/openssl ]; then
  curl -sSLO https://github.com/openssl/openssl/releases/download/openssl-3.0.15/openssl-3.0.15.tar.gz
  tar xzf openssl-3.0.15.tar.gz
  (cd openssl-3.0.15 && ./Configure --prefix="$T/openssl" --libdir=lib no-tests shared >/dev/null \
     && make -j8 >/dev/null && make install_sw >/dev/null)
fi
echo "[otp] $(date)"
if [ ! -x otp-28.3.1/bin/erl ]; then
  curl -sSLo kerl https://raw.githubusercontent.com/kerl/kerl/master/kerl && chmod +x kerl
  export KERL_BASE_DIR="$T/.kerl" MAKEFLAGS=-j8
  export KERL_CONFIGURE_OPTIONS="--with-ssl=$T/openssl --without-termcap --without-javac --without-wx --without-odbc --without-debugger --without-observer --without-et"
  export LDFLAGS="-Wl,-rpath,$T/openssl/lib"
  ./kerl build 28.3.1 28.3.1
  ./kerl install 28.3.1 "$T/otp-28.3.1"
fi
echo "[rebar3] $(date)"
[ -x rebar3 ] || { curl -sSLO https://s3.amazonaws.com/rebar3/rebar3 && chmod +x rebar3; }
"$T/otp-28.3.1/bin/erl" -noshell -eval 'io:format("otp ~s, crypto ~p~n", [erlang:system_info(otp_release), crypto:info_lib()]), halt().'
PATH="$T/otp-28.3.1/bin:$PATH" ./rebar3 version
echo "[done] $(date)"
EOF
}

case "$CMD" in
    setup)
        for H in "$@"; do
            remote_setup | ssh "$H" 'mkdir -p ~/erllama-tp && cat > ~/erllama-tp/setup.sh && (nohup bash ~/erllama-tp/setup.sh > ~/erllama-tp/setup.log 2>&1 &) && echo "setup started on $(hostname)"'
        done
        ;;
    sync)
        ROOT=$(cd "$(dirname "$0")/.." && pwd)
        for H in "$@"; do
            rsync -a --delete --exclude _build --exclude 'apps/*/_build' --exclude 'apps/erllama/priv/*.so' \
                --exclude '.git' --exclude 'examples/*/_build' "$ROOT/" "$H:erllama-tp/erllama/"
            echo "synced to $H"
        done
        ;;
    build)
        for H in "$@"; do
            ssh "$H" "T=\$HOME/erllama-tp; cd \$T/erllama && export PATH=\$T/otp-28.3.1/bin:/usr/local/cuda/bin:\$PATH \
                ERLLAMA_OPTS='-DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH -DNCCL_ROOT='\$T/nccl \
                ERLLAMA_BUILDOPTS=-j$JOBS && \$T/rebar3 as test compile 2>&1 | grep -E 'NCCL|GGML_CUDA|error|Error|===>' | tail -20"
        done
        ;;
    env)
        cat <<'EOF'
T=$HOME/erllama-tp
export PATH=$T/otp-28.3.1/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$T/nccl/lib:$T/openssl/lib:${LD_LIBRARY_PATH:-}
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
# Plain TCP instead of RDMA (the interface on the link between the hosts):
# export NCCL_NET=Socket NCCL_SOCKET_IFNAME==enp1s0f0np0
EOF
        ;;
    *)
        echo "unknown command: $CMD" >&2
        exit 2
        ;;
esac
