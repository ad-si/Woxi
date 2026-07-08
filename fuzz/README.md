# Fuzzing Woxi

Two complementary fuzzing setups live in this repo:

1. **Crash fuzzing** (this directory) — coverage-guided libFuzzer targets
   that assert the parser and interpreter never panic, hang, or overflow
   the stack, whatever the input.
2. **Differential fuzzing** (`src/bin/diff_fuzz.rs`) — generates random
   valid programs and checks that `woxi eval` and `wolframscript -code`
   produce identical output, which is the project's core contract.


## Crash fuzzing (`cargo fuzz`)

Requires a nightly toolchain and [cargo-fuzz](https://rust-fuzz.github.io/book/cargo-fuzz.html)
(`cargo install cargo-fuzz`). The make targets handle corpus seeding and
the cargo-fuzz install:

```sh
make fuzz-parse      # fuzz woxi::parse (syntax level)
make fuzz-interpret  # fuzz woxi::interpret (full pipeline)
```

Both run until interrupted (Ctrl-C). Findings land in `fuzz/artifacts/`;
reproduce one with:

```sh
cargo +nightly fuzz run interpret fuzz/artifacts/interpret/crash-<hash>
```

Once a crash is fixed, keep the input as a regression test (a unit test in
`tests/`, per the repo testing rules) — the corpus and artifacts
directories are gitignored.

Details:

- The corpora are seeded from `tests/scripts/*.wls` plus generated
  expressions (`make fuzz-corpus`), so the fuzzer starts from valid
  programs instead of random bytes. The `slow_script_test!` scripts are
  excluded from the `interpret` corpus: they legitimately run for tens of
  seconds, which the `-timeout` hang detector would misreport as crashes.
- The `interpret` target skips inputs containing filesystem/network heads
  (`Export`, `Import`, `Run`, …) so fuzzing has no side effects; see
  `SIDE_EFFECT_DENYLIST` in `fuzz_targets/interpret.rs`.
- Hangs count as findings: libFuzzer's `-timeout` flag (set in the make
  targets) turns evaluation loops into reported crashes.
- The nightly CI workflow (`.github/workflows/nightly.yml`) runs each
  target for 5 minutes per night and uploads crashing inputs as
  artifacts.


## Differential fuzzing (`woxi-diff-fuzz`)

Generates random, terminating expressions built only from functions that
`functions.csv` marks implemented (curated argument shapes keep programs
meaningful — see `FN_SPECS` in `src/bin/diff_fuzz.rs`), then compares
woxi against a wolframscript oracle and greedily shrinks any divergence
to a minimal reproducer:

```sh
make fuzz-diff
# or directly:
cargo run --bin woxi-diff-fuzz -- --cases 500 --seed 42
```

The oracle is auto-detected: a local `wolframscript` binary if one is on
the `PATH`, otherwise the `cmd-server.js` HTTP bridge at
`http://host.docker.internal:3456/exec` (for Docker dev environments).
Override with `--oracle wolframscript|bridge|woxi`, `--wolframscript
<path>`, or `--bridge-url <url>`. `--oracle woxi` is a self-check mode
that uses woxi as its own oracle — useful for validating the harness
itself (it must report zero divergences).

How it works:

- Cases are pre-filtered in batches (`Print[InputForm[…]]` statements
  separated by marker prints), so one slow wolframscript start-up covers
  `--batch-size` cases. Any differing case is re-confirmed individually
  with the bare expression — the exact `woxi eval` vs `wolframscript
  -code` pairing the CLI doc tests use — so the batch scaffolding can
  never cause a false positive.
- Outputs are compared as a sorted line-bag over stdout + stderr by
  default, because the two tools route messages to different streams
  (the CLI doc tests need `output_stream: combined` for the same
  reason). Use `--compare combined|stdout` for stricter modes.
- Runs are fully deterministic: every finding prints the master seed and
  its per-case seed, and `--seed <n>` replays a run exactly.
  `--print-cases` shows the generated programs without evaluating them.
- A woxi crash, non-zero exit, or hang on generated input is always
  reported, independent of the oracle's answer.
- Exit code: 0 = no divergences, 1 = divergences found (with shrunk
  reproducers in the report), 2 = setup error.

When a divergence is found, fix it and add a regression test with the
shrunk reproducer, following the repo testing rules in `CLAUDE.md`.
