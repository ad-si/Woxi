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

fn run_eval_stdin(expression: &str) -> (String, String, bool) {
  use std::io::Write;
  use std::process::Stdio;
  let mut child = Command::new(woxi_bin())
    .arg("eval")
    .arg("-")
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .expect("failed to spawn woxi");
  child
    .stdin
    .as_mut()
    .expect("no stdin")
    .write_all(expression.as_bytes())
    .expect("failed to write stdin");
  let output = child.wait_with_output().expect("failed to wait");
  (
    String::from_utf8_lossy(&output.stdout).into_owned(),
    String::from_utf8_lossy(&output.stderr).into_owned(),
    output.status.success(),
  )
}

#[test]
fn eval_reads_expression_from_stdin_when_arg_is_dash() {
  // `woxi eval -` reads the expression from stdin. Useful for inputs
  // that exceed the shell's ARG_MAX (the audit harness's huge-image
  // cases would otherwise fail with `Argument list too long`).
  let (stdout, stderr, ok) = run_eval_stdin("1 + 2 * 3");
  assert!(ok, "woxi eval - failed: stderr={}", stderr);
  assert_eq!(stdout.trim(), "7");
}

#[test]
fn eval_stdin_handles_large_image_input() {
  // Roughly approximate the audit harness's pain point: generate a big
  // Image literal (several KB) and pipe it via stdin. It shouldn't hit
  // ARG_MAX limits and should evaluate to `Image`.
  let mut row = String::from("{");
  for i in 0..50 {
    if i > 0 {
      row.push_str(", ");
    }
    row.push_str("0.5");
  }
  row.push('}');
  let mut matrix = String::from("{");
  for i in 0..50 {
    if i > 0 {
      matrix.push_str(", ");
    }
    matrix.push_str(&row);
  }
  matrix.push('}');
  let expression = format!("Head[Image[{}]]", matrix);
  let (stdout, stderr, ok) = run_eval_stdin(&expression);
  assert!(ok, "woxi eval - on large image failed: stderr={}", stderr);
  assert_eq!(stdout.trim(), "Image");
}

/// Run `woxi run <file>` and return (stdout, stderr, success).
fn run_file(path: &std::path::Path) -> (String, String, bool) {
  let output = Command::new(woxi_bin())
    .arg("run")
    .arg(path)
    .output()
    .expect("failed to spawn woxi");
  (
    String::from_utf8_lossy(&output.stdout).into_owned(),
    String::from_utf8_lossy(&output.stderr).into_owned(),
    output.status.success(),
  )
}

#[test]
fn run_notebook_hello_world() {
  // `woxi run` should accept a real `.nb` notebook file, evaluate its
  // Input cells, and print their results (skipping Output cells).
  let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/notebooks/hello_world.nb");
  let (stdout, stderr, ok) = run_file(&path);
  assert!(ok, "woxi run hello_world.nb failed: stderr={}", stderr);
  assert_eq!(stdout.trim(), "Hello World!");
}

#[test]
fn run_notebook_evaluates_only_input_cells_in_order() {
  // Multiple Input/Code cells evaluate top-to-bottom; Output, Text and
  // heading cells are skipped. Print side-effects appear inline.
  let nb = r#"Notebook[{
Cell["A title", "Title"],
Cell[CellGroupData[{
Cell[BoxData["1 + 2"], "Input"],
Cell[BoxData["3"], "Output"]
}, Open]],
Cell["prose to ignore", "Text"],
Cell[CellGroupData[{
Cell[BoxData["Range[3]"], "Input"],
Cell[BoxData["{1, 2, 3}"], "Output"]
}, Open]],
Cell[BoxData[RowBox[{"Print[\"hi\"]", "\n", "x = 5"}]], "Code"]
}]
"#;
  let dir = std::env::temp_dir();
  let path = dir.join("woxi_cli_test_notebook.nb");
  std::fs::write(&path, nb).expect("write temp notebook");
  let (stdout, stderr, ok) = run_file(&path);
  let _ = std::fs::remove_file(&path);
  assert!(ok, "woxi run notebook failed: stderr={}", stderr);
  assert_eq!(stdout, "3\n{1, 2, 3}\nhi\n5\n");
}

#[test]
fn run_notebook_notebook_directory_resolves_to_file_dir() {
  // Regression: `NotebookDirectory[]` must resolve to the `.nb` file's
  // own directory when run via `woxi run` (so Export paths etc. work),
  // instead of emitting the `nosv` "not available outside a front-end"
  // message.
  let dir = std::env::temp_dir();
  let path = dir.join("woxi_cli_test_nbdir.nb");
  let nb =
    "Notebook[{\nCell[BoxData[\"NotebookDirectory[]\"], \"Input\"]\n}]\n";
  std::fs::write(&path, nb).expect("write temp notebook");
  let (stdout, stderr, ok) = run_file(&path);
  let _ = std::fs::remove_file(&path);
  assert!(ok, "woxi run notebook failed: stderr={}", stderr);
  // The canonical temp dir, with a trailing separator (WL convention).
  let expected = format!("{}/", dir.to_string_lossy().trim_end_matches('/'));
  assert!(
    !stderr.contains("nosv"),
    "NotebookDirectory emitted nosv message: stderr={}",
    stderr
  );
  assert_eq!(stdout.trim(), expected.trim_end_matches('/'));
}
