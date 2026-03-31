FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential pkg-config libssl-dev jq \
 && rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain stable \
 && rustup component add clippy rustfmt

CMD ["bash"]
