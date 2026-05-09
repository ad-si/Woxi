//! Benchmarks targeting Woxi interpreter primitives that show up as hot
//! paths in real-world scripts. Each bench isolates one primitive so that
//! a regression or speedup is easy to attribute. The final
//! `script_build_summary` bench ties the micro-benchmarks back to a real
//! end-to-end workload.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::path::PathBuf;
use std::time::Duration;

/// Run a Wolfram snippet through `woxi::interpret`, clearing interpreter
/// state first so iterations don't accumulate symbol bindings.
fn run(prog: &str) {
  woxi::clear_state();
  woxi::interpret(prog).expect("interpret failed");
}

// --- #1: regex compile cost -----------------------------------------------

fn bench_string_cases_repeat_pattern(c: &mut Criterion) {
  // Mirrors `parseEntry` in scripts/build_summary.wls: the same complex
  // pattern is matched against many lines.
  let prog = r#"
    lines = Table["  - [Foo](bar/baz.md)", {1500}];
    Do[
      StringCases[line,
        StartOfString ~~ ind:((" " | "\t")...) ~~ "- [" ~~
          lbl:(Except["]"]..) ~~ "](" ~~ tgt:(Except[")"]..) ~~ ")" ~~
          ((" " | "\t")...) ~~ EndOfString :> {ind, lbl, tgt}, 1],
      {line, lines}]
  "#;
  c.bench_function("string_cases_repeat_pattern", |b| {
    b.iter(|| run(black_box(prog)))
  });
}

fn bench_string_split_repeat_pattern(c: &mut Criterion) {
  let prog = r#"
    lines = Table["alpha,beta,gamma,delta", {1500}];
    Do[StringSplit[line, ","], {line, lines}]
  "#;
  c.bench_function("string_split_repeat_pattern", |b| {
    b.iter(|| run(black_box(prog)))
  });
}

fn bench_string_replace_repeat_pattern(c: &mut Criterion) {
  let prog = r#"
    lines = Table["foo bar baz quux", {1500}];
    Do[StringReplace[line, "bar" -> "BAR"], {line, lines}]
  "#;
  c.bench_function("string_replace_repeat_pattern", |b| {
    b.iter(|| run(black_box(prog)))
  });
}

// --- #2: in-place string append ------------------------------------------

fn bench_string_concat_grow(c: &mut Criterion) {
  // The `s = s <> c` accumulator pattern used by splitLines.
  let prog = r#"s = ""; Do[s = s <> "x", {50000}]"#;
  c.bench_function("string_concat_grow", |b| b.iter(|| run(black_box(prog))));
}

fn bench_string_join_many_args(c: &mut Criterion) {
  // Sanity check — the single-shot StringJoin path must not regress when
  // we add a fast path for `var = var <> rhs`.
  let prog = r#"StringJoin @@ Table["x", {50000}]"#;
  c.bench_function("string_join_many_args", |b| {
    b.iter(|| run(black_box(prog)))
  });
}

// --- #3: Characters allocation -------------------------------------------

fn bench_characters_large(c: &mut Criterion) {
  // 75 KB string (matches tests/SUMMARY.md size).
  let prog = r#"
    s = StringJoin @@ Table["abcdefghij", {7500}];
    Characters[s]
  "#;
  c.bench_function("characters_large", |b| b.iter(|| run(black_box(prog))));
}

// --- #4/#5/#6: Sort with user-defined comparator -------------------------

fn bench_sort_with_user_comparator(c: &mut Criterion) {
  // Mirrors `asciiLess` from scripts/build_summary.wls: a Module-wrapped
  // comparator that uses ToCharacterCode for byte-wise comparison.
  let prog = r#"
    asciiLess[s1_String, s2_String] := Catch[Module[
        {a = ToCharacterCode[s1], b = ToCharacterCode[s2], n},
        n = Min[Length[a], Length[b]];
        Do[
          Which[
            a[[i]] < b[[i]], Throw[True],
            a[[i]] > b[[i]], Throw[False]],
          {i, n}];
        Length[a] < Length[b]]];
    names = Table["filename-" <> ToString[i] <> ".md", {i, 50}];
    Sort[names, asciiLess]
  "#;
  c.bench_function("sort_with_user_comparator", |b| {
    b.iter(|| run(black_box(prog)))
  });
}

fn bench_to_character_code_loop(c: &mut Criterion) {
  let prog = r#"Do[ToCharacterCode["some-filename.md"], {20000}]"#;
  c.bench_function("to_character_code_loop", |b| {
    b.iter(|| run(black_box(prog)))
  });
}

fn bench_module_call_overhead(c: &mut Criterion) {
  let prog = r#"
    f[x_] := Module[{a, b, c}, a = x; b = x + 1; c = x + 2; a + b + c];
    Do[f[i], {i, 50000}]
  "#;
  c.bench_function("module_call_overhead", |b| b.iter(|| run(black_box(prog))));
}

// --- end-to-end canary ---------------------------------------------------

/// Drive `scripts/build_summary.wls` against a tempdir copy of the data
/// files so the bench doesn't churn the repo.
fn bench_script_build_summary(c: &mut Criterion) {
  let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let script_src = manifest_dir.join("scripts/build_summary.wls");
  let summary_src = manifest_dir.join("tests/SUMMARY.md");
  let zensical_src = manifest_dir.join("tests/zensical.toml");
  let cli_src = manifest_dir.join("tests/cli");

  // Create a tempdir with `scripts/`, `tests/SUMMARY.md`, `tests/zensical.toml`,
  // and a symlink at `tests/cli` pointing at the real cli docs (the docs
  // are read-only inputs, so sharing them is fine).
  let tmp =
    std::env::temp_dir().join(format!("woxi-bench-{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&tmp);
  std::fs::create_dir_all(tmp.join("scripts")).unwrap();
  std::fs::create_dir_all(tmp.join("tests")).unwrap();
  std::fs::copy(&script_src, tmp.join("scripts/build_summary.wls")).unwrap();
  std::fs::copy(&summary_src, tmp.join("tests/SUMMARY.md")).unwrap();
  std::fs::copy(&zensical_src, tmp.join("tests/zensical.toml")).unwrap();
  #[cfg(unix)]
  std::os::unix::fs::symlink(&cli_src, tmp.join("tests/cli")).unwrap_or_else(
    |_| {
      // Fall back to an empty dir if symlink fails — the script will
      // still run, just with no children to enumerate.
      std::fs::create_dir_all(tmp.join("tests/cli")).unwrap();
    },
  );
  #[cfg(not(unix))]
  std::fs::create_dir_all(tmp.join("tests/cli")).unwrap();

  let script_path = tmp.join("scripts/build_summary.wls");
  let script_path_abs = script_path.canonicalize().unwrap();
  let script_str = std::fs::read_to_string(&script_path_abs).unwrap();
  let script_str = woxi::without_shebang(&script_str);
  let abs_str = script_path_abs.to_string_lossy().to_string();

  c.bench_function("script_build_summary", |b| {
    b.iter(|| {
      woxi::clear_state();
      woxi::set_system_variable("$InputFileName", &format!("\"{}\"", abs_str));
      // Refresh the per-iteration mutable inputs so each iteration starts
      // from a clean baseline.
      std::fs::copy(&summary_src, tmp.join("tests/SUMMARY.md")).unwrap();
      std::fs::copy(&zensical_src, tmp.join("tests/zensical.toml")).unwrap();
      // Capture stdout so the script's `Print["wrote …"]` calls don't
      // contaminate criterion's output.
      let _ = woxi::interpret_with_stdout(black_box(&script_str));
    })
  });
}

criterion_group! {
  // Sample size kept at the criterion minimum (10) so the
  // `script_build_summary` end-to-end bench finishes in reasonable time.
  // Each iteration of that bench currently takes ~25–30 seconds; 10
  // samples is enough to flag regressions without making the suite
  // unworkable to iterate on.
  name = interpreter;
  config = Criterion::default()
    .measurement_time(Duration::from_secs(8))
    .sample_size(10);
  targets =
    bench_string_cases_repeat_pattern,
    bench_string_split_repeat_pattern,
    bench_string_replace_repeat_pattern,
    bench_string_concat_grow,
    bench_string_join_many_args,
    bench_characters_large,
    bench_sort_with_user_comparator,
    bench_to_character_code_loop,
    bench_module_call_overhead,
    bench_script_build_summary
}
criterion_main!(interpreter);
