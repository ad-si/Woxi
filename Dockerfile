# Linux runner for Woxi's machine-specific tests.
#
# The tests in `tests/interpreter_tests/machine_specific.rs` derive
# their expected values from the host environment (`$USER`, `$HOME`,
# `gethostname`, `current_dir`, `cfg!(target_os = …)`), so they pass
# on any host. This image's purpose is to exercise the Linux branches
# of the platform-dependent symbols on hosts that aren't Linux —
# `$OperatingSystem` = "Unix", `$BaseDirectory` = "/usr/share/Wolfram",
# `$UserBaseDirectory` = "$HOME/.Wolfram".
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

# Fetch crates in their own layer, keyed only on the manifests +
# lockfile. Source-only edits no longer redownload from crates.io /
# the astro-float git remote on rebuild. Stub source files are needed
# because cargo validates that each workspace member's lib/bin paths
# exist before resolving.
COPY Cargo.toml Cargo.lock ./
COPY woxi-studio/Cargo.toml woxi-studio/Cargo.toml
RUN mkdir -p src woxi-studio/src \
  && : > src/lib.rs \
  && echo 'fn main() {}' > src/main.rs \
  && echo 'fn main() {}' > woxi-studio/src/main.rs \
  && cargo fetch --locked

# Bring in the real source (overwrites the stubs above).
COPY . .

# build.rs registers `.git/HEAD` and `.git/index` as rerun-if-changed
# triggers. .dockerignore excludes `.git`, so without these stubs
# cargo treats the fingerprint as perpetually stale and recompiles
# woxi on every `docker run`. Empty files are fine — `git describe`
# still fails and the build script falls back to CARGO_PKG_VERSION.
RUN mkdir -p .git && : > .git/HEAD && : > .git/index

# Pre-build the test binaries while online with the exact nextest
# invocation used at runtime so `docker run` reuses the artefacts
# instead of recompiling. --offline guarantees we'd notice if
# `cargo fetch` above missed anything; --package woxi avoids
# dragging in woxi-studio's GUI dependency tree (iced, resvg, …)
# which the machine-specific tests don't need.
RUN cargo nextest run --no-run \
      --locked --offline \
      --package woxi \
      -E "test(machine_specific)"

# Run only the `machine_specific` module from the woxi crate;
# everything else belongs in the host's `make test`.
CMD ["cargo", "nextest", "run", \
     "--package", "woxi", \
     "--offline", \
     "--show-progress=none", \
     "--status-level=pass", \
     "--failure-output=final", \
     "-E", "test(machine_specific)"]
