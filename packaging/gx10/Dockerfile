# Build with an ARM64 Docker builder and the release source as context.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl cmake g++ make git libncurses6 libssl3t64 && \
    rm -rf /var/lib/apt/lists/*
RUN curl -fsSL --retry 3 https://builds.hex.pm/builds/otp/arm64/ubuntu-24.04/OTP-28.3.1.tar.gz -o /tmp/otp.tgz && \
    echo 'b276a25ef9595676011140fd303cd439841cd629c43a4d44811bd8c948ab7cf6  /tmp/otp.tgz' | sha256sum -c - && \
    mkdir /opt/otp && tar -xzf /tmp/otp.tgz -C /opt/otp --strip-components=1 && \
    /opt/otp/Install -minimal /opt/otp && rm /tmp/otp.tgz && \
    curl -fsSL --retry 3 https://github.com/erlang/rebar3/releases/download/3.25.0/rebar3 -o /usr/local/bin/rebar3 && \
    echo '2d46024331792aa31c8d12242c0073042d24e2650b45fedba6c17156a8220fa3  /usr/local/bin/rebar3' | sha256sum -c - && \
    chmod 755 /usr/local/bin/rebar3
ENV PATH="/opt/otp/bin:${PATH}"
WORKDIR /src
COPY rebar.config rebar.lock ./
COPY apps/ apps/
COPY config/ config/
COPY rel/ rel/
COPY scripts/ scripts/
ARG BUILD_JOBS=4
RUN ERLLAMA_OPTS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DGGML_NATIVE=OFF" \
    ERLLAMA_BUILDOPTS="-j${BUILD_JOBS}" rebar3 as prod release && \
    ldd _build/prod/rel/erllama_server/lib/erllama-*/priv/erllama_nif.so | grep libcublas

FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04
LABEL org.opencontainers.image.source="https://github.com/benoitc/erllama"
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libncurses6 libssl3t64 libsctp1 openssl procps && rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 erllama && mkdir /data && chown erllama:erllama /data
COPY --from=build --chown=erllama:erllama /src/_build/prod/rel/erllama_server /opt/erllama
COPY --chmod=755 packaging/gx10/entrypoint.sh /usr/local/bin/erllama-entrypoint
ENV HOME=/home/erllama ERLLAMA_DATA_DIR=/data PATH="/opt/erllama/bin:${PATH}"
USER erllama
VOLUME /data
ENTRYPOINT ["erllama-entrypoint"]
CMD ["erllama_server", "foreground"]
