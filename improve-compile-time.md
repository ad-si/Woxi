# Improve Woxi compile times (config-only)

## Context

Compile times "became really bad." Root cause investigation found that the
workspace (`Cargo.toml`) declares two members — `.` (the `woxi` crate) and
`woxi-studio` — but sets **no `default-members`**. With Cargo, an unscoped
command at the workspace root (`cargo build`, `cargo test`, and the
`cargo nextest run` behind `make test`) operates on *all* members. So every
default build also compiles `woxi-studio`, which depends on `iced` 0.14 →
`wgpu`/`winit`/`naga`/`tiny-skia` (100+ extra crates). `woxi-studio` is only
needed for `make install` (which already targets it explicitly via
`-p woxi-studio`) and direct studio work — it has no reason to be in the
`make test` / day-to-day build path.

On top of that, the build leaves two macOS-specific costs on the table:
default builds emit **full** debug info (slow to generate and link; the macOS
linker also has to handle it), and linking uses the default `ld64` even though
`ld64.lld` 21 is already on PATH (via the nix toolchain).

Scope chosen by the user: **config-only, zero-risk changes**, and reduce
dev/test debug info to **line-tables-only** (keeps panic/backtrace line numbers,
drops local-variable debug info). No source refactors, no nightly toolchain.

Expected impact, roughly in order: excluding `woxi-studio` from default builds
is the dominant win for `make test`; reduced debug info + `lld` cut link time
on every incremental build.

## Changes

All changes are in two files. Both are easily reversible.

### 1. Exclude `woxi-studio` from default builds — `Cargo.toml`

In the existing `[workspace]` table (currently lines 1–3):

```toml
[workspace]
members = [".", "woxi-studio"]
default-members = ["."]
resolver = "3"
```

Effect: `cargo build`, `cargo test`, and `cargo nextest run` at the root now
build only the `woxi` crate (and its dev-deps), skipping `iced`/`wgpu`/etc.
`make install-macos-app` (`cargo build --release -p woxi-studio`) and any
explicit `-p woxi-studio` / building inside `woxi-studio/` are unaffected —
studio is still a full workspace member, just not a *default* one.

### 2. Reduce dev/test debug info — `Cargo.toml`

Add a `[profile.dev]` section (the existing `[profile.release]` at lines
102–104 stays as-is; release is intentionally `lto = true` / `codegen-units = 1`
for shipping). `[profile.dev]` settings are inherited by the `test` profile, so
this covers `make test`/nextest binaries too:

```toml
[profile.dev]
debug = "line-tables-only"
split-debuginfo = "unpacked"
```

- `debug = "line-tables-only"`: backtraces/panics keep file:line; drops
  local-variable debug info. Faster to generate and link.
- `split-debuginfo = "unpacked"`: on macOS, leaves debug info in the `.o`
  files and skips the slow `dsymutil` packaging step — a notable per-build
  win for incremental dev/test builds.

### 3. Use the `lld` linker for host builds — new `.cargo/config.toml`

There is currently no `.cargo/config.toml`. Create one scoped to the host
triple (`aarch64-apple-darwin`) so it does **not** touch wasm builds:

```toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-fuse-ld=lld"]
```

`ld64.lld` (LLVM lld 21) is already on PATH via nix, and the active linker
driver is the nix clang wrapper, which honors `-fuse-ld=lld`. Linking is a
large share of incremental-build wall-clock, and lld is substantially faster
than the default `ld64`.

Fallback / risk note: this is the only change with any (small) risk — if a
link ever fails with an lld-specific error, remove this file (or the
`rustflags` line) to revert to the system linker; everything else still
applies. Adding rustflags changes the build fingerprint, so the **first**
build after this change is a full rebuild (one-time cost).

## Verification

1. **Sanity-check the dep graph shrank** (studio no longer in default builds):
   - `cargo build` from the repo root should no longer mention `iced`, `wgpu`,
     `winit`, or `naga` in its compile output.
   - `cargo metadata --format-version 1 | jq '.workspace_default_members'`
     should list only the `woxi` package.
2. **Confirm lld is actually used:**
   - `cargo build -v 2>&1 | grep -- '-fuse-ld=lld'` shows the flag on the link
     invocation; the build links without errors.
3. **Time the main path before/after** (run twice to warm caches, compare the
   second runs):
   - `cargo clean && time make test` (full, includes the cold rebuild).
   - Then touch one source file (e.g. `src/lib.rs`) and `time cargo nextest run`
     to measure the incremental/link path that these changes most affect.
4. **Regression check:** `make test` still passes (no behavior change — these
   are build-config only). `make install-macos-app` still builds `woxi-studio`.

## Out of scope (deferred, per user)

Recorded for later if more speedup is wanted:
- Consolidate the 6 smaller top-level `tests/*.rs` into one harness so the full
  rlib links once instead of ~7×.
- Nightly Cranelift codegen backend for dev/test builds.
- Split the ~10k-line `src/evaluator/dispatch/mod.rs` and its ~100-arm match
  into smaller functions/modules to ease LLVM.
