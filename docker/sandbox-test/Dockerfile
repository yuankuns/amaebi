FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git curl wget coreutils findutils grep sed procps \
 && rm -rf /var/lib/apt/lists/*
CMD ["bash"]
