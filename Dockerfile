# Multi-stage build for ShrestiDB & Web Studio
FROM rust:latest AS builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Cache dependencies
RUN mkdir src && echo "pub fn dummy() {}" > src/lib.rs && \
    mkdir -p src/bin && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > src/bin/studio.rs && \
    echo "fn main() {}" > src/bin/db_server.rs && \
    cargo build --release || true && \
    rm -rf src target/release/deps/shrestidb*

# Copy real source code and build production binaries
COPY . .
RUN cargo build --release --bin shrestidb-studio --bin shrestidb

# Minimal runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binaries
COPY --from=builder /app/target/release/shrestidb-studio /usr/local/bin/shrestidb-studio
COPY --from=builder /app/target/release/shrestidb /usr/local/bin/shrestidb

# Standard Cloud Run / AWS ECS environment variables
ENV HOST=0.0.0.0
ENV PORT=8080

VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:8080/health || exit 1

CMD ["shrestidb-studio", "--wal=/data/shrestidb.wal"]
