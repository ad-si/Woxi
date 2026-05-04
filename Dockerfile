# Reproducible environment for Woxi's machine-specific tests.
#
# The tests in `tests/interpreter_tests/machine_specific.rs` pin the
# values of host-derived symbols (`$UserName`, `$MachineName`,
# `$HomeDirectory`, …) to anonymised strings. Outside a controlled
# environment those values are unpredictable, so this image fixes
# them:
#
#   * USER=woxi            → `$UserName`           = "woxi"
#   * HOME=/home/woxi      → `$HomeDirectory`      = "/home/woxi"
#                            Environment["HOME"]   = "/home/woxi"
#   * WORKDIR=/home/woxi/woxi
#                          → `$InitialDirectory`   = "/home/woxi/woxi"
#                            ParentDirectory[]     = "/home/woxi"
#   * --hostname=woxi-test → `$MachineName`        = "woxi-test"
#                            (set by `make test-docker`)
#   * Linux base image     → `$OperatingSystem`    = "Unix"
#
# Build & run from the repo root:
#   docker build -t woxi-test -f Dockerfile .
#   docker run --rm --hostname woxi-test woxi-test
#
# Or via the makefile:
#   make test-docker

# Pin to a specific Rust release so test runs are byte-for-byte
# reproducible across machines. Bump deliberately when upgrading the
# toolchain — Cargo edition 2024 needs Rust >= 1.85, and current
# `cargo-nextest` (0.9.133) requires Rust >= 1.91.
FROM rust:1.91-slim-bookworm

# - git: `cargo` clones the `astro-float` git dependency
# - curl: Woxi's `Import["…", "URL"]` shells out to `curl` at runtime
# - pkg-config / libssl-dev: cover transitive crates that probe for
#   system OpenSSL during the build
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    pkg-config \
    libssl-dev \
  && rm -rf /var/lib/apt/lists/*

RUN cargo install --locked cargo-nextest

# Pin a deterministic, anonymised capture environment so the values
# the host-derived symbols return (`$UserName`, `$HomeDirectory`,
# `$InitialDirectory`, …) are reproducible across hosts.
ENV USER=woxi \
    HOME=/home/woxi

WORKDIR /home/woxi/woxi

COPY . .

# Run only the `machine_specific` module; everything else belongs in
# the host's `make test`.
CMD ["cargo", "nextest", "run", \
     "--show-progress=none", \
     "--status-level=all", \
     "--failure-output=final", \
     "-E", "test(machine_specific)"]
