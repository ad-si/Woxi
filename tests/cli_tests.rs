//! Integration tests that exercise the `woxi` binary's CLI parsing.
//!
//! These spawn the compiled `woxi` binary as a subprocess and assert on its
//! stdout. They complement the in-process `interpret()` tests by validating
//! command-line argument handling (e.g. `woxi eval -12` should accept a
//! leading-hyphen value rather than treating it as a flag).

use std::path::PathBuf;
use std::process::Command;

fn woxi_bin() -> PathBuf {
  // Build path: target/<profile>/woxi alongside the test binary.
  let mut path = std::env::current_exe().unwrap();
  path.pop(); // remove the test executable name
  if path.ends_with("deps") {
    path.pop();
  }
  path.push("woxi");
  path
}

fn run_eval(args: &[&str]) -> (String, String, bool) {
  let output = Command::new(woxi_bin())
    .arg("eval")
    .args(args)
    .output()
    .expect("failed to spawn woxi");
  (
    String::from_utf8_lossy(&output.stdout).into_owned(),
    String::from_utf8_lossy(&output.stderr).into_owned(),
    output.status.success(),
  )
}

#[test]
fn eval_negative_integer_value() {
  // The audit's `### Integer` case: passing `-12` as the expression argument
  // should be accepted even though it starts with `-`.
  let (stdout, stderr, ok) = run_eval(&["-12"]);
  assert!(ok, "woxi eval -12 failed: stderr={}", stderr);
  assert_eq!(stdout.trim(), "-12");
}

#[test]
fn eval_negative_expression() {
  let (stdout, stderr, ok) = run_eval(&["-3 + 5"]);
  assert!(ok, "woxi eval '-3 + 5' failed: stderr={}", stderr);
  assert_eq!(stdout.trim(), "2");
}

#[test]
fn eval_positive_integer_still_works() {
  let (stdout, stderr, ok) = run_eval(&["42"]);
  assert!(ok, "woxi eval 42 failed: stderr={}", stderr);
  assert_eq!(stdout.trim(), "42");
}
