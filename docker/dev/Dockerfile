FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential pkg-config libssl-dev jq \
 && rm -rf /var/lib/apt/lists/*

# Install the Rust toolchain in a world-accessible location so the container
# can be run as any UID (--user $(id -u):$(id -g)) without needing root.
# /opt/cargo and /opt/rustup are readable and executable by all users.
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
ENV PATH=/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain stable \
 && rustup component add clippy rustfmt \
 && chmod -R a+rwX /opt/rustup /opt/cargo

# Provide a writable home directory for arbitrary UIDs (some tools require
# $HOME to be writable; invoked with -e HOME=/home/builder).
RUN mkdir -p /home/builder && chmod 1777 /home/builder

CMD ["bash"]
