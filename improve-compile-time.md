# Improve Woxi compile times

## Context

Compile times "became really bad." Root-cause investigation found five
independent problems; all of the fixes below are implemented.

1. **No `default-members`.** The workspace declares two members — `.` (the
   `woxi` crate) and `woxi-studio` — so every unscoped command at the root
   (`cargo build`, `cargo test`, the `cargo nextest run` behind `make test`,
   and CI) also compiled `woxi-studio`, which depends on `iced` 0.14 →
   `wgpu`/`winit`/`naga`/`tiny-skia` (100+ extra crates). Studio is only
   needed for `make install` / direct studio work.
2. **Two copies of the SVG stack in every build.** `woxi` pinned
   `resvg 0.47` while `svg2pdf 0.13` (a direct woxi dependency) and
   `iced 0.14` (studio) both pin `resvg 0.45` — so `resvg`, `usvg`, and
   `tiny-skia` were each compiled and linked **twice**, even in plain
   `cargo build` of the interpreter alone.
3. **`build.rs` re-ran on `.git/index`.** Any `git add`/commit re-stamped
   `WOXI_GIT_VERSION`, which invalidates the ~400k-line crate and forces a
   rebuild plus a relink of the CLI and all 9 test binaries.
4. **Full debug info in dev/test builds** — slow to generate and link
   (on macOS the default `ld64` also packages it via `dsymutil`).
5. **Fat LTO + `codegen-units = 1` applied to Studio release builds.**
   `make install-macos-app` used `[profile.release]`, serializing codegen of
   500+ crates for a GUI whose runtime doesn't need interpreter-grade
   optimization.

## Implemented changes

### 1. Exclude `woxi-studio` from default builds — `Cargo.toml`

```toml
[workspace]
members = [".", "woxi-studio"]
default-members = ["."]
resolver = "3"
```

`cargo build` / `make test` / CI now build only the `woxi` crate.
`make install-macos-app` and explicit `-p woxi-studio` / `cargo run -p
woxi-studio` are unaffected — studio is still a full workspace member,
just not a *default* one.

### 2. Single resvg stack — `Cargo.toml`

`woxi`'s direct `resvg` dependency moved 0.47 → **0.45** to match
`svg2pdf 0.13` and `iced 0.14`. One copy of `resvg`/`usvg`/`tiny-skia`
across the whole workspace (−125 lines in `Cargo.lock`; ~10 fewer big
crates per build). Bump resvg again only in lockstep with svg2pdf and iced.

### 3. Stop rebuilding on `git add` — `build.rs`

The build script no longer tracks `.git/index`; it tracks `.git/HEAD` plus
the branch ref file HEAD points at. New commits / branch switches still
refresh the version stamp, but staging files no longer forces a full
rebuild+relink. Trade-off: the `-dirty` suffix reflects the tree as of the
last source-triggered compile.

### 4. Reduce dev/test debug info — `Cargo.toml`

```toml
[profile.dev]
debug = "line-tables-only"
split-debuginfo = "unpacked"
```

Backtraces/panics keep file:line; local-variable debug info is dropped.
`split-debuginfo = "unpacked"` skips the slow `dsymutil` step on macOS.
`[profile.dev]` is inherited by the `test` profile, so `make test` benefits.

### 5. Dedicated Studio release profile — `Cargo.toml` + `makefile`

```toml
[profile.studio]
inherits = "release"
lto = "thin"
codegen-units = 16
```

`make install-macos-app` now builds with `--profile studio` (binary at
`target/studio/woxi-studio`). Thin LTO + parallel codegen builds several
times faster at near-identical runtime performance. The interpreter's own
`[profile.release]` (fat LTO, 1 CGU, used by `cargo install --path .`)
is unchanged — that's a deliberate runtime-performance choice.

### 6. Use the `lld` linker for macOS host builds — `.cargo/config.toml`

```toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-fuse-ld=lld"]
```

`lld` is provided by the nix dev shell (flake.nix) and the clang driver
honors `-fuse-ld=lld`. Linking is a large share of incremental wall-clock
and lld is substantially faster than the default `ld64`. This is the only
change with any (small) risk — if a link ever fails with an lld-specific
error, delete `.cargo/config.toml` to revert; everything else still
applies. Note: adding rustflags changes the build fingerprint, so the
first build after pulling this change is a full rebuild (one-time cost).

## Verification

- `cargo metadata | jq .workspace_default_members` lists only `woxi`.
- `Cargo.lock` contains exactly one `resvg`/`usvg`/`tiny-skia` entry.
- `cargo build` output no longer mentions `iced`, `wgpu`, `winit`, `naga`,
  or a second resvg.
- `make test` passes (build-config-only changes, plus the resvg 0.45
  downgrade which is covered by the SVG rendering snapshot tests).
- On macOS: `cargo build -v 2>&1 | grep -- '-fuse-ld=lld'` shows the flag
  on the link invocation.

## Out of scope (deferred)

Recorded for later if more speedup is wanted:

- Consolidate the 9 top-level `tests/*.rs` harnesses into fewer binaries so
  the full rlib links once instead of ~9× (biggest remaining lever for
  `make test` incremental time).
- Nightly Cranelift codegen backend for dev/test builds.
- Split the `woxi` crate itself into workspace sub-crates (parser →
  evaluator → function areas) so edits recompile a slice instead of all
  ~400k lines, and crates compile in parallel. This is the only structural
  fix for cold-build time; everything above trims fat around it.
- Split the ~12k-line `src/evaluator/dispatch/mod.rs` and its ~100-arm
  match into smaller functions/modules to ease LLVM.
- Gate heavyweight, rarely-exercised dependencies behind cargo features
  (e.g. `keshvar` with its embedded gazetteer data, `calamine` /
  `rust_xlsxwriter`) for a leaner default dev loop.
