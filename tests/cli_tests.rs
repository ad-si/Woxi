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
  assert!(ok, "woxi eval -12 failed: stderr={stderr}");
  assert_eq!(stdout.trim(), "-12");
}

#[test]
fn eval_negative_expression() {
  let (stdout, stderr, ok) = run_eval(&["-3 + 5"]);
  assert!(ok, "woxi eval '-3 + 5' failed: stderr={stderr}");
  assert_eq!(stdout.trim(), "2");
}

#[test]
fn eval_positive_integer_still_works() {
  let (stdout, stderr, ok) = run_eval(&["42"]);
  assert!(ok, "woxi eval 42 failed: stderr={stderr}");
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
  assert!(ok, "woxi eval - failed: stderr={stderr}");
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
  let expression = format!("Head[Image[{matrix}]]");
  let (stdout, stderr, ok) = run_eval_stdin(&expression);
  assert!(ok, "woxi eval - on large image failed: stderr={stderr}");
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
fn run_routes_messages_to_stdout_like_wolframscript() {
  // `wolframscript -file` writes evaluation messages (e.g. `Get::noopen`,
  // with a leading blank line) to stdout, not stderr. `woxi run` must match
  // so its captured output is byte-for-byte identical.
  let script = "Get[\"missing_file.m\"]\nPrint[\"done\"]\n";
  let path = std::env::temp_dir().join("woxi_cli_run_message.wls");
  std::fs::write(&path, script).expect("write temp script");
  let (stdout, stderr, ok) = run_file(&path);
  let _ = std::fs::remove_file(&path);
  assert!(ok, "woxi run failed: stderr={stderr}");
  assert_eq!(
    stdout, "\nGet::noopen: Cannot open missing_file.m.\ndone\n",
    "message must go to stdout (matching wolframscript -file)"
  );
  assert_eq!(stderr, "", "no message should leak to stderr");
}

#[test]
fn run_notebook_hello_world() {
  // `woxi run` should accept a real `.nb` notebook file, evaluate its
  // Input cells, and print their results (skipping Output cells).
  let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/notebooks/hello_world.nb");
  let (stdout, stderr, ok) = run_file(&path);
  assert!(ok, "woxi run hello_world.nb failed: stderr={stderr}");
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
  assert!(ok, "woxi run notebook failed: stderr={stderr}");
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
  assert!(ok, "woxi run notebook failed: stderr={stderr}");
  // The canonical temp dir, with a trailing separator (WL convention).
  let sep = std::path::MAIN_SEPARATOR;
  let expected =
    format!("{}{}", dir.to_string_lossy().trim_end_matches(sep), sep);
  assert!(
    !stderr.contains("nosv"),
    "NotebookDirectory emitted nosv message: stderr={stderr}"
  );
  assert_eq!(stdout.trim(), expected.trim_end_matches(sep));
}

/// Pipe a REPL session's input via stdin and capture (stdout, stderr, ok).
fn run_repl(input: &str) -> (String, String, bool) {
  use std::io::Write;
  use std::process::Stdio;
  let mut child = Command::new(woxi_bin())
    .arg("repl")
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .expect("failed to spawn woxi");
  child
    .stdin
    .as_mut()
    .expect("no stdin")
    .write_all(input.as_bytes())
    .expect("failed to write stdin");
  let output = child.wait_with_output().expect("failed to wait");
  (
    String::from_utf8_lossy(&output.stdout).into_owned(),
    String::from_utf8_lossy(&output.stderr).into_owned(),
    output.status.success(),
  )
}

#[test]
fn repl_evaluates_and_numbers_output() {
  let (stdout, stderr, ok) = run_repl("1 + 2\n3 * 4\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(stdout, "Out[1]= 3\n\nOut[2]= 12\n\n");
}

#[test]
fn repl_persists_state_across_lines() {
  // Variable bindings and function definitions must survive across inputs
  // in a single REPL process (unlike `woxi eval`, a fresh process each call).
  let (stdout, stderr, ok) = run_repl("x = 5\nx^2\nf[n_] := n!\nf[4]\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  // x=5 -> Out[1], x^2 -> Out[2]=25, the := definition is suppressed (Out[3]
  // is skipped), f[4] -> Out[4]=24.
  assert_eq!(stdout, "Out[1]= 5\n\nOut[2]= 25\n\nOut[4]= 24\n\n");
}

#[test]
fn repl_percent_references_previous_output() {
  let (stdout, stderr, ok) = run_repl("10 + 5\n% + 1\n% * 2\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(stdout, "Out[1]= 15\n\nOut[2]= 16\n\nOut[3]= 32\n\n");
}

#[test]
fn repl_suppresses_output_on_trailing_semicolon() {
  let (stdout, stderr, ok) = run_repl("a = 7;\na + 1\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  // The trailing semicolon suppresses Out[1]; the line counter still advances.
  assert_eq!(stdout, "Out[2]= 8\n\n");
}

#[test]
fn repl_joins_multiline_bracketed_input() {
  // An input with unbalanced brackets continues onto the next line until the
  // brackets close, then evaluates as a single expression.
  let (stdout, stderr, ok) = run_repl("Sum[i^2,\n  {i, 1, 10}]\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(stdout, "Out[1]= 385\n\n");
}

#[test]
fn repl_joins_line_ending_in_assignment_operator() {
  // A line ending in `=` is only the start of an input: the value follows on
  // the next line, and both together are a single `In[]` (issue #354).
  let (stdout, stderr, ok) = run_repl("c =\n5\nc\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(stdout, "Out[1]= 5\n\nOut[2]= 5\n\n");
}

#[test]
fn repl_joins_definition_ending_in_set_delayed() {
  // Same for `:=`, the form reported in issue #354: the body of the
  // definition is typed on the following line.
  let (stdout, stderr, ok) = run_repl(concat!(
    "CF[x_Real, n_Integer?Positive] :=\n",
    "Block[{xi, xp = x, r = {}}, Do[xi = Floor[xp]; AppendTo[r, xi];",
    " xp = 1 / (xp - xi), {n}]; Return[r]]\n",
    "CF[N[Pi], 10]\n",
  ));
  assert!(ok, "woxi repl failed: stderr={stderr}");
  // The definition is suppressed (Out[1] skipped), the call is Out[2].
  assert_eq!(stdout, "Out[2]= {3, 7, 15, 1, 292, 1, 1, 1, 2, 1}\n\n");
}

#[test]
fn repl_joins_line_ending_in_infix_operator() {
  // Any dangling operator continues, not just the assignment ones.
  let (stdout, stderr, ok) = run_repl("1 +\n2\nx /.\nx -> 7\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(stdout, "Out[1]= 3\n\nOut[2]= 7\n\n");
}

#[test]
fn repl_syntax_error_does_not_swallow_the_next_line() {
  // Input that cannot be completed by anything is a syntax error, reported
  // immediately — the following line stays a separate input.
  let (stdout, stderr, ok) = run_repl("1 +* 2\n7\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert!(stderr.contains("Parse error"), "stderr={stderr}");
  assert_eq!(stdout, "Out[2]= 7\n\n");
}

#[test]
fn repl_print_writes_before_suppressed_output() {
  // Print emits to stdout during evaluation; it returns Null so no Out[] line
  // is shown for that input.
  let (stdout, stderr, ok) = run_repl("Print[\"hi\"]\n2 + 2\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(stdout, "hi\nOut[2]= 4\n\n");
}

#[test]
fn repl_quit_command_exits() {
  // Lines after `Quit` are not evaluated.
  let (stdout, stderr, ok) = run_repl("1 + 1\nQuit\n99\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(stdout, "Out[1]= 2\n\n");
}

#[test]
fn repl_reports_errors_without_aborting_session() {
  // A bad expression prints an error to stderr but the session continues.
  let (stdout, stderr, ok) = run_repl("1/0\n6 * 7\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert!(
    stdout.contains("Out[1]= ComplexInfinity"),
    "stdout={stdout}"
  );
  assert!(stdout.contains("Out[2]= 42"), "stdout={stdout}");
}

/// `wolframscript`'s terminal REPL prints results in OutputForm, which shows a
/// machine-precision real at 6 significant figures — `3203.60 - 2711.16` is
/// `492.44` there, while `wolframscript -code` prints the full round-trip
/// `492.44000000000005`. `woxi repl` follows the REPL; `woxi eval` follows
/// `-code`.
#[test]
fn repl_shows_machine_reals_at_six_significant_figures() {
  let (stdout, stderr, ok) = run_repl(concat!(
    "3203.60 - 2711.16\n",
    "1/3.\n",
    "0.1 + 0.2\n",
    "Range[3]*1.111111111\n",
  ));
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(
    stdout,
    concat!(
      "Out[1]= 492.44\n\n",
      "Out[2]= 0.333333\n\n",
      "Out[3]= 0.3\n\n",
      "Out[4]= {1.11111, 2.22222, 3.33333}\n\n",
    )
  );
}

/// The scientific-notation thresholds (|x| < 1e-5 or >= 1e6) apply to the
/// *rounded* value, so `999999.6` displays as `1.*^6` and `0.000012345678`
/// stays decimal — matching the REPL's `1. 10^6` / `0.0000123457`.
#[test]
fn repl_applies_scientific_thresholds_after_rounding() {
  let (stdout, stderr, ok) = run_repl(concat!(
    "999999.6\n",
    "0.000012345678\n",
    "-0.000001234\n",
    "1234567.89\n",
    "2^100 + 0.5\n",
  ));
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(
    stdout,
    concat!(
      "Out[1]= 1.*^6\n\n",
      "Out[2]= 0.0000123457\n\n",
      "Out[3]= -1.234*^-6\n\n",
      "Out[4]= 1.23457*^6\n\n",
      "Out[5]= 1.26765*^30\n\n",
    )
  );
}

/// An arbitrary-precision real drops its backtick precision marker and shows
/// exactly its stored precision in significant figures (`N[Pi, 20]` →
/// `3.1415926535897932385`), unlike the `-code` InputForm echo.
#[test]
fn repl_shows_arbitrary_precision_reals_without_marker() {
  let (stdout, stderr, ok) =
    run_repl("N[Pi, 20]\nN[Pi, 3]\nSetAccuracy[0, 5]\n");
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(
    stdout,
    concat!(
      "Out[1]= 3.1415926535897932385\n\n",
      "Out[2]= 3.14\n\n",
      "Out[3]= 0.\n\n",
    )
  );
}

/// Display rounding is a rendering step only: `%` still holds the full
/// machine value, and a bare literal or variable echo is rounded the same way
/// a computed result is.
#[test]
fn repl_rounds_display_only_not_stored_values() {
  let (stdout, stderr, ok) = run_repl(concat!(
    "492.44000000000005\n",
    "x = 3203.60 - 2711.16\n",
    "x\n",
    "(x - 492.44)*10^16\n",
  ));
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(
    stdout,
    concat!(
      "Out[1]= 492.44\n\n",
      "Out[2]= 492.44\n\n",
      "Out[3]= 492.44\n\n",
      "Out[4]= 568.434\n\n",
    )
  );
}

/// Digits inside a string value are text, not a number: the REPL prints them
/// verbatim (`"3.14159265358979"` stays 15 digits).
#[test]
fn repl_does_not_round_digits_inside_strings() {
  let (stdout, stderr, ok) = run_repl(concat!(
    "\"3.14159265358979\"\n",
    "s = \"9.87654321\"\n",
    "{1.23456789, s}\n",
  ));
  assert!(ok, "woxi repl failed: stderr={stderr}");
  assert_eq!(
    stdout,
    concat!(
      "Out[1]= 3.14159265358979\n\n",
      "Out[2]= 9.87654321\n\n",
      "Out[3]= {1.23457, 9.87654321}\n\n",
    )
  );
}
