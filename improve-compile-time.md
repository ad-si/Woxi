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

### 6. ~~Use the `lld` linker for macOS host builds~~ — not present

This was planned as a `.cargo/config.toml` with
`rustflags = ["-C", "link-arg=-fuse-ld=lld"]` for `aarch64-apple-darwin`,
but no such file is in the repository. Linux x86_64 hosts already link with
`rust-lld` by default (Rust ≥ 1.90), and Xcode ≥ 15's `ld-prime` is much
faster than the old `ld64`, so the remaining gain is small.

## Round 2

Measured on a 4-core Linux container (`cargo build --tests`, which is what
`make test` compiles). An "incremental" build is the rebuild after appending
a comment to one file in `src/`.

| | cold | incremental |
| --- | ---: | ---: |
| before | 11m27s | 43s |
| after | 7m55s | 23s |

Plus: `cargo build` / `cargo run` right after `make test` (or vice versa)
no longer recompiles the woxi crate at all — previously a second full
compile (≈ 4½ min cold).

### 7. Share artifacts between `cargo build` and `cargo test` — `Cargo.toml`

The dev-dependencies (criterion, insta, proptest) enable extra features of
crates woxi also uses (`serde/alloc`, `aho-corasick/default`,
`either/use_std`, `itertools/default`, `bit-set`/`bit-vec` `default`).
Resolver ≥ 2 keeps those features out of non-test builds, so
`cargo build` and `cargo test` resolved *different* versions of these
crates, which changed the hash of everything above them — including the
woxi crate, which was therefore compiled twice and kept twice on disk.
Declaring the same features as (otherwise unused) native-only normal
dependencies makes both builds resolve identical units (the
`cargo-hakari` "workspace hack" technique). Verify with

```sh
RUSTC_BOOTSTRAP=1 cargo build -Zunstable-options --unit-graph [--tests]
```

and comparing the `features` of each unit. Re-check when adding or
upgrading dev-dependencies.

### 8. Turn off implicit local ThinLTO in dev/test — `Cargo.toml`

```toml
[profile.dev]
lto = "off"
```

With `opt-level > 0` and multiple codegen units, rustc runs a ThinLTO pass
across the CGUs of each crate by default (`lto = false` means "thin-local",
not "off"). `-Z time-passes` showed 34s of `LLVM_thinlto` in the
`interpreter_tests` binary alone. With all other changes in place, a cold
build takes 7m55s instead of 10m08s (CPU time 27m vs 35m), while the full
`make test` run gets ≈ 5% slower (152s vs 146s) from less inlining across
codegen units — a good trade for the dev loop. Release builds are
unaffected.

### 9. Move the pest parser into `crates/woxi-parser`

`#[derive(Parser)]` re-generates the parser from the 1,100-line
`wolfram.pest` on *every* compilation of the crate containing it: 3.2s of
the 4.7s of macro expansion in each incremental woxi rebuild, paid twice
(lib + lib unit tests). In its own crate it only reruns when the grammar
changes. `woxi` re-exports `Rule` and `WolframParser`, so all paths stay
the same. Incremental lib rebuild: 19s → 14s.

Note: publishing `woxi` to crates.io now requires publishing
`woxi-parser` first (as already with `woxi-reduce`).

### 10. Split `interpreter_tests` into parallel test binaries

Every change to the library forces each test binary to be recompiled, and
rustc processes a crate largely on one core. The ~300k-line
`interpreter_tests` binary took ~20s after every edit, *after* the library
finished, with the other cores idle. Its modules are now spread over six
binaries (`interpreter_tests`, `interpreter_tests_{algebra,data,domains,
language,math}`) that build in parallel in ~4s each. All of them nest their
modules under `mod interpreter_tests`, so test paths (and nextest filters)
are unchanged, and the insta snapshots stay in the binary that keeps the
`interpreter_tests` name. `make test-reduce` now names
`--test interpreter_tests_algebra`.

## Verification

- `cargo metadata | jq .workspace_default_members` lists only `woxi`.
- `Cargo.lock` contains exactly one `resvg`/`usvg`/`tiny-skia` entry.
- `cargo build` output no longer mentions `iced`, `wgpu`, `winit`, `naga`,
  or a second resvg.
- `make test` passes (build-config-only changes, plus the resvg 0.45
  downgrade which is covered by the SVG rendering snapshot tests).

## Out of scope (deferred)

Recorded for later if more speedup is wanted:

- The library's own `#[cfg(test)]` unit tests (~400, mostly in
  `notebook.rs` and `graphics.rs`) make cargo compile the whole woxi crate a
  second time in test mode. It runs in parallel with the normal lib build,
  so it costs little wall time incrementally on a multi-core machine, but
  it is ~4 CPU-minutes of every cold build. Moving those tests to
  `tests/` (exposing what they need) would remove it.
- `keshvar` (~40s) sits on the critical path of cold builds: the woxi crate
  cannot start until it is done.
- The `cdylib` crate type (needed only by `wasm-pack`) makes every native
  build link an extra shared library (~1s per incremental build). A small
  dedicated wasm wrapper crate would avoid it.
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
