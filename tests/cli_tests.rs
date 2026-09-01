//! Integration tests that exercise the `woxi` binary's CLI parsing.
//!
//! These spawn the compiled `woxi` binary as a subprocess and assert on its
//! stdout. They complement the in-process `interpret()` tests by validating
//! command-line argument handling (e.g. `woxi eval -12` should accept a
//! leading-hyphen value rather than treating it as a flag).

use std::path::{Path, PathBuf};
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

/// Run `woxi eval <expression>` with `input` piped to stdin, so `Input[]` and
/// `InputString[]` have something to read.
fn run_eval_with_input(
  expression: &str,
  input: &str,
) -> (String, String, bool) {
  use std::io::Write;
  use std::process::Stdio;
  let mut child = Command::new(woxi_bin())
    .arg("eval")
    .arg(expression)
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
fn input_string_reads_a_line_from_stdin() {
  // Regression for #462: `InputString[prompt]` must consume a line of stdin
  // instead of immediately returning `EndOfFile`, so scripts that prompt for
  // a value actually get one.
  let (stdout, stderr, ok) =
    run_eval_with_input(r#"InputString["Please enter your name:"]"#, "Alice\n");
  assert!(ok, "woxi eval failed: stderr={stderr}");
  assert_eq!(stdout, "Please enter your name:Alice\n");
}

#[test]
fn input_string_reads_successive_lines() {
  // Each call advances through stdin rather than re-reading the first line.
  let (stdout, stderr, ok) =
    run_eval_with_input("Table[InputString[], {3}]", "a\nb\nc\n");
  assert!(ok, "woxi eval failed: stderr={stderr}");
  assert_eq!(stdout.trim(), "{a, b, c}");
}

#[test]
fn input_string_returns_end_of_file_when_stdin_is_exhausted() {
  // Past the last line — and with a closed stdin — the result is `EndOfFile`,
  // the behaviour wolframscript shows when there is nothing left to read.
  let (stdout, stderr, ok) =
    run_eval_with_input("{InputString[], InputString[]}", "only\n");
  assert!(ok, "woxi eval failed: stderr={stderr}");
  assert_eq!(stdout.trim(), "{only, EndOfFile}");
}

#[test]
fn input_string_keeps_a_blank_line_as_empty_string() {
  // A bare newline is an empty string, not `EndOfFile`.
  let (stdout, stderr, ok) =
    run_eval_with_input("StringLength[InputString[]]", "\n");
  assert!(ok, "woxi eval failed: stderr={stderr}");
  assert_eq!(stdout.trim(), "0");
}

#[test]
fn input_parses_the_line_as_an_expression() {
  // `Input[]` evaluates what it reads, unlike `InputString[]`.
  let (stdout, stderr, ok) = run_eval_with_input(r#"Input["n: "]"#, "2 + 3\n");
  assert!(ok, "woxi eval failed: stderr={stderr}");
  assert_eq!(stdout, "n: 5\n");
}

#[test]
fn input_reports_syntax_errors_as_failed() {
  // An unparsable line yields `$Failed` with a Syntax message — the message
  // is owned by `Syntax`, not by the shared `ToExpression` implementation.
  let (stdout, stderr, ok) = run_eval_with_input("Input[]", "1 +\n");
  assert!(ok, "woxi eval failed: stderr={stderr}");
  assert!(
    stdout.contains("Syntax::sntxi:"),
    "expected a Syntax message, got: {stdout}"
  );
  assert!(
    !stdout.contains("ToExpression::"),
    "message should not leak ToExpression: {stdout}"
  );
  assert_eq!(stdout.trim_end().lines().last().unwrap(), "$Failed");
}

#[test]
fn input_returns_null_for_a_blank_line() {
  let (stdout, stderr, ok) = run_eval_with_input("Input[]", "\n");
  assert!(ok, "woxi eval failed: stderr={stderr}");
  assert_eq!(stdout.trim(), "Null");
}

#[test]
fn eval_from_stdin_still_returns_end_of_file_for_input() {
  // `woxi eval -` consumes stdin for the expression itself, so there is
  // nothing left for `InputString[]` to read.
  let (stdout, stderr, ok) = run_eval_stdin("InputString[]");
  assert!(ok, "woxi eval - failed: stderr={stderr}");
  assert_eq!(stdout.trim(), "EndOfFile");
}

/// Run `woxi run <file>` with `input` piped to stdin.
fn run_file_with_input(
  path: &std::path::Path,
  input: &str,
) -> (String, String, bool) {
  use std::io::Write;
  use std::process::Stdio;
  let mut child = Command::new(woxi_bin())
    .arg("run")
    .arg(path)
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
fn run_script_prompts_and_reads_stdin() {
  // The exact script from #462: it must greet the name it was given.
  let script = "userName = InputString[\"Please enter your name:\"];\n\
                Echo[StringJoin[\"Hello \", userName]];\n";
  let path = std::env::temp_dir().join("woxi_cli_input_string.wls");
  std::fs::write(&path, script).expect("write script");
  let (stdout, stderr, ok) = run_file_with_input(&path, "Alice\n");
  let _ = std::fs::remove_file(&path);
  assert!(ok, "woxi run failed: stderr={stderr}");
  assert_eq!(stdout, "Please enter your name:>> Hello Alice\n");
}

#[test]
fn run_script_falls_back_to_end_of_file_without_stdin() {
  // With stdin closed immediately the script still terminates rather than
  // hanging, and `InputString[]` reports `EndOfFile`.
  let script = "Print[InputString[]];\n";
  let path = std::env::temp_dir().join("woxi_cli_input_string_eof.wls");
  std::fs::write(&path, script).expect("write script");
  let (stdout, stderr, ok) = run_file_with_input(&path, "");
  let _ = std::fs::remove_file(&path);
  assert!(ok, "woxi run failed: stderr={stderr}");
  assert_eq!(stdout, "EndOfFile\n");
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

// Regression test for #422: a file pulled in with `Get` sees its own path in
// `$InputFileName`, and the including script sees its own again afterwards.
#[test]
fn get_scopes_input_file_name_to_the_included_file() {
  let dir = std::env::temp_dir().join("woxi_cli_get_input_file_name");
  std::fs::create_dir_all(&dir).expect("create temp dir");
  let included = dir.join("included_file.wl");
  let main = dir.join("test.wl");
  // Use forward slashes inside the script: on Windows a backslash in a
  // string literal reads as an escape sequence.
  let included_arg = included.to_string_lossy().replace('\\', "/");
  std::fs::write(&included, "Echo[$InputFileName];\n").expect("write include");
  std::fs::write(
    &main,
    format!("Get[\"{included_arg}\"];\nEcho[$InputFileName];\n"),
  )
  .expect("write main");

  let (stdout, stderr, ok) = run_file(&main);
  let _ = std::fs::remove_dir_all(&dir);
  assert!(ok, "woxi run failed: stderr={stderr}");
  // `$InputFileName` is spelled with forward slashes on every platform (see
  // `woxi::utils::wolfram_path_string`), so the Windows path of the script
  // has to be spelled the same way to be compared against.
  let main_arg = main.to_string_lossy().replace('\\', "/");
  assert_eq!(
    stdout,
    format!(">> {included_arg}\n>> {main_arg}\n"),
    "the included file must report its own path, the script its own"
  );
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
fn run_notebook_manipulate_epilog_label_has_no_precision_marker() {
  // Regression for a Wolfram Demonstrations Project notebook pattern: a
  // Manipulate body computes a value with `NIntegrate`, then builds a
  // `Plot` `Epilog` label via `ToString[SetPrecision[expr, n]]`. The label
  // text must be the plain rounded decimal (matching wolframscript), not
  // the raw InputForm digits with a `` `n `` precision marker still
  // attached — and `SetPrecision` must have re-evaluated the expression its
  // numeric leaves unlocked, so this is `0.762` and not `Tanh[1.00]`.
  let nb = concat!(
    "Notebook[{\n",
    "Cell[BoxData[\"Print[ToString[SetPrecision[Tanh[1], 3]]]\"], \"Input\"]\n",
    "}]\n"
  );
  let dir = std::env::temp_dir();
  let path = dir.join("woxi_cli_test_manipulate_epilog_label.nb");
  std::fs::write(&path, nb).expect("write temp notebook");
  let (stdout, stderr, ok) = run_file(&path);
  let _ = std::fs::remove_file(&path);
  assert!(ok, "woxi run notebook failed: stderr={stderr}");
  assert_eq!(stdout.trim(), "0.762");
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

// Regression test for #444: the paclet scenario from the issue — a script
// registers its own directory with `PacletDirectoryLoad`, then `Needs` finds
// the paclet's `PacletInfo.wl`, loads the declared kernel file and the
// package's function becomes callable.
#[test]
fn paclet_directory_load_makes_a_paclet_context_available() {
  let dir = std::env::temp_dir().join("woxi_cli_paclet");
  std::fs::remove_dir_all(&dir).ok();
  std::fs::create_dir_all(dir.join("Kernel")).expect("create paclet dir");
  std::fs::write(
    dir.join("PacletInfo.wl"),
    r#"PacletObject[
  <|
    "Name" -> "MyPaclet",
    "Version" -> "1.0.0",
    "WolframVersion" -> "15+",
    "Extensions" -> {
      {
        "Kernel",
        "Root" -> "Kernel",
        "Context" -> {"MyPaclet`"}
      }
    }
  |>
]
"#,
  )
  .expect("write PacletInfo.wl");
  std::fs::write(
    dir.join("Kernel/MyPaclet.wl"),
    "BeginPackage[\"MyPaclet`\"]\n\
     MyFunction::usage = \"MyFunction[] greets.\";\n\
     Begin[\"`Private`\"]\n\
     MyFunction[] := Echo[\"Hello from Paclet Package\"]\n\
     End[]\n\
     EndPackage[]\n",
  )
  .expect("write package file");
  let script = dir.join("testpaclet.wl");
  std::fs::write(
    &script,
    "PacletDirectoryLoad[Directory[]]\nNeeds[\"MyPaclet`\"]\nMyFunction[]\n",
  )
  .expect("write script");

  // `Directory[]` is the process working directory, so run from the paclet.
  let output = Command::new(woxi_bin())
    .arg("run")
    .arg(&script)
    .current_dir(&dir)
    .output()
    .expect("failed to spawn woxi");
  let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
  let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
  std::fs::remove_dir_all(&dir).ok();

  assert!(output.status.success(), "woxi run failed: stderr={stderr}");
  assert_eq!(stdout, ">> Hello from Paclet Package\n");
}

// Two paclets that each keep a private `helper` do not collide: symbols are
// resolved per context, so each package's API calls its own helper. Under a
// flat namespace the second paclet's helper would have replaced the first's.
#[test]
fn paclets_keep_their_private_helpers_to_themselves() {
  let dir = std::env::temp_dir().join("woxi_cli_paclet_contexts");
  std::fs::remove_dir_all(&dir).ok();
  for n in ["1", "2"] {
    let paclet = dir.join(format!("Pac{n}"));
    std::fs::create_dir_all(paclet.join("Kernel")).expect("create paclet dir");
    std::fs::write(
      paclet.join("PacletInfo.wl"),
      format!(
        "PacletObject[<|\"Name\" -> \"Pac{n}\", \"Version\" -> \"1.0.0\", \
         \"Extensions\" -> {{{{\"Kernel\", \"Root\" -> \"Kernel\", \
         \"Context\" -> {{\"Pac{n}`\"}}}}}}|>]\n"
      ),
    )
    .expect("write PacletInfo.wl");
    std::fs::write(
      paclet.join(format!("Kernel/Pac{n}.wl")),
      format!(
        "BeginPackage[\"Pac{n}`\"]\n\
         api{n}::usage = \"api{n}[x] uses a private helper\";\n\
         Begin[\"`Private`\"]\n\
         helper[x_] := x * {n}\n\
         api{n}[x_] := helper[x] + 100\n\
         End[]\n\
         EndPackage[]\n"
      ),
    )
    .expect("write package file");
  }
  let script = dir.join("main.wl");
  std::fs::write(
    &script,
    "PacletDirectoryLoad[Directory[]]\n\
     Needs[\"Pac1`\"]\n\
     Needs[\"Pac2`\"]\n\
     Print[{api1[10], api2[10]}]\n\
     Print[{Context[api1], Context[helper]}]\n\
     Print[ToString[Names[\"*`helper\"], InputForm]]\n",
  )
  .expect("write script");

  let output = Command::new(woxi_bin())
    .arg("run")
    .arg(&script)
    .current_dir(&dir)
    .output()
    .expect("failed to spawn woxi");
  let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
  let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
  std::fs::remove_dir_all(&dir).ok();

  assert!(output.status.success(), "woxi run failed: stderr={stderr}");
  assert_eq!(
    stdout,
    "{110, 120}\n\
     {Pac1`, Global`}\n\
     {\"helper\", \"Pac1`Private`helper\", \"Pac2`Private`helper\"}\n"
  );
}

// --- `woxi install-kernel` ------------------------------------------------

/// Scratch directory for an `install-kernel` test, laid out as
/// `bin/` (the process `PATH`), `data/` (`JUPYTER_DATA_DIR`) and `tmp/`.
fn kernel_test_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("woxi_cli_{name}"));
  std::fs::remove_dir_all(&dir).ok();
  for sub in ["bin", "tmp"] {
    std::fs::create_dir_all(dir.join(sub)).expect("create test dir");
  }
  dir
}

/// Run `woxi install-kernel` with a `PATH` holding only `dir/bin`, a private
/// Jupyter data directory and a working directory without a `kernelspec/`
/// subdirectory — i.e. every invocation outside a source checkout.
fn run_install_kernel(
  dir: &Path,
  extra_args: &[&str],
) -> (String, String, bool) {
  let output = Command::new(woxi_bin())
    .arg("install-kernel")
    .args(extra_args)
    .current_dir(dir)
    .env("PATH", dir.join("bin"))
    .env("JUPYTER_DATA_DIR", dir.join("data"))
    .env("TMPDIR", dir.join("tmp"))
    .output()
    .expect("failed to spawn woxi");
  (
    String::from_utf8_lossy(&output.stdout).into_owned(),
    String::from_utf8_lossy(&output.stderr).into_owned(),
    output.status.success(),
  )
}

/// Regression test for issue #547: the kernelspec used to be read from
/// `./kernelspec/woxi`, so installing from anywhere but a source checkout
/// crashed with `No such file or directory`.
#[test]
fn install_kernel_works_outside_a_source_checkout() {
  let dir = kernel_test_dir("install_kernel_no_checkout");
  let (stdout, stderr, ok) = run_install_kernel(&dir, &[]);
  assert!(ok, "woxi install-kernel failed: stderr={stderr}");
  assert!(
    !stderr.contains("No such file or directory"),
    "install-kernel read the kernelspec from disk: stderr={stderr}"
  );

  let installed = dir.join("data").join("kernels").join("woxi");
  let spec = std::fs::read_to_string(installed.join("kernel.json"))
    .expect("kernel.json installed");
  assert!(
    installed.join("logo-32x32.png").is_file(),
    "32px logo missing"
  );
  assert!(
    installed.join("logo-64x64.png").is_file(),
    "64px logo missing"
  );
  assert!(
    stdout.contains(&installed.display().to_string()),
    "install location not reported: stdout={stdout}"
  );

  // `argv[0]` must be an absolute path to the running binary so the kernel
  // also starts when the Jupyter server's PATH has no `woxi` in it. Read it
  // as JSON rather than by splitting on quotes: a Windows path is stored
  // with escaped separators (`C:\\dir\\woxi.exe`) and only survives the
  // round trip when it is unescaped.
  let parsed: serde_json::Value =
    serde_json::from_str(&spec).expect("kernel.json is valid JSON");
  let argv0 = parsed["argv"][0]
    .as_str()
    .expect("argv[0] in kernel.json")
    .to_string();
  assert!(
    Path::new(&argv0).is_absolute() && Path::new(&argv0).is_file(),
    "argv[0] is not an existing absolute path: {argv0}"
  );
  assert!(spec.contains("\"jupyter\""), "kernel.json: {spec}");
  assert!(
    spec.contains("\"{connection_file}\""),
    "kernel.json: {spec}"
  );
  assert!(
    spec.contains("Woxi (Wolfram Language)"),
    "kernel.json: {spec}"
  );
  assert!(
    spec.contains("\"language\": \"wolfram\""),
    "kernel.json: {spec}"
  );

  // The staged copy is a temporary and must not be left behind.
  let leftovers: Vec<_> = std::fs::read_dir(dir.join("tmp"))
    .map(|entries| entries.filter_map(Result::ok).map(|e| e.path()).collect())
    .unwrap_or_default();
  assert!(
    leftovers.is_empty(),
    "staging directories left behind: {leftovers:?}"
  );

  std::fs::remove_dir_all(&dir).ok();
}

/// A second run replaces the previously installed spec instead of failing.
#[test]
fn install_kernel_replaces_an_existing_installation() {
  let dir = kernel_test_dir("install_kernel_replace");
  let installed = dir.join("data").join("kernels").join("woxi");
  std::fs::create_dir_all(&installed).expect("create stale spec");
  std::fs::write(installed.join("stale.json"), "{}").expect("write stale file");

  let (_stdout, stderr, ok) = run_install_kernel(&dir, &[]);
  assert!(ok, "woxi install-kernel failed: stderr={stderr}");
  assert!(
    installed.join("kernel.json").is_file(),
    "kernel.json missing"
  );
  assert!(
    !installed.join("stale.json").exists(),
    "stale file survived the reinstall"
  );

  std::fs::remove_dir_all(&dir).ok();
}

/// Write an executable `jupyter` stub into `dir/bin` that logs its arguments
/// to `dir/args.txt`, copies the kernelspec it was pointed at to `dir/staged`
/// and exits with `exit_code`.
#[cfg(unix)]
fn write_jupyter_stub(dir: &Path, exit_code: i32) {
  use std::os::unix::fs::PermissionsExt;
  let stub = dir.join("bin").join("jupyter");
  // `woxi` is run with a PATH that only contains the stub, so the stub
  // restores the test process' PATH to reach `cp`.
  let path = std::env::var("PATH").unwrap_or_default();
  std::fs::write(
    &stub,
    format!(
      "#!/bin/sh\n\
       PATH='{path}'\n\
       for arg in \"$@\"; do\n\
       printf '%s\\n' \"$arg\" >> '{dir}/args.txt'\n\
       last=\"$arg\"\n\
       done\n\
       cp -r \"$last\" '{dir}/staged'\n\
       exit {exit_code}\n",
      dir = dir.display()
    ),
  )
  .expect("write jupyter stub");
  std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755))
    .expect("chmod jupyter stub");
}

/// When the `jupyter` CLI is available it performs the installation, and it
/// is handed a complete kernelspec directory.
#[cfg(unix)]
#[test]
fn install_kernel_hands_a_complete_spec_to_jupyter() {
  let dir = kernel_test_dir("install_kernel_via_jupyter");
  write_jupyter_stub(&dir, 0);

  let (stdout, stderr, ok) = run_install_kernel(&dir, &[]);
  assert!(ok, "woxi install-kernel failed: stderr={stderr}");
  assert!(
    stdout.contains("installed successfully"),
    "stdout={stdout} stderr={stderr}"
  );

  let logged = std::fs::read_to_string(dir.join("args.txt")).expect("ran");
  let args: Vec<&str> = logged.lines().collect();
  assert_eq!(
    args[..args.len() - 1],
    [
      "kernelspec",
      "install",
      "--replace",
      "--name",
      "woxi",
      "--user"
    ]
  );
  let source_dir = Path::new(args.last().expect("source directory argument"));
  assert_eq!(
    source_dir.file_name().and_then(|name| name.to_str()),
    Some("woxi"),
    "jupyter derives the kernel name from the directory name: {source_dir:?}"
  );

  // The staged directory jupyter was pointed at held the whole spec.
  let staged = dir.join("staged");
  assert!(
    staged.join("kernel.json").is_file(),
    "staged kernel.json missing"
  );
  assert!(
    staged.join("logo-32x32.png").is_file(),
    "staged 32px logo missing"
  );
  assert!(
    staged.join("logo-64x64.png").is_file(),
    "staged 64px logo missing"
  );

  // Nothing is written directly when jupyter handles the installation.
  assert!(
    !dir.join("data").exists(),
    "the spec was also installed directly"
  );

  std::fs::remove_dir_all(&dir).ok();
}

/// `--system` is forwarded to `jupyter kernelspec install`.
#[cfg(unix)]
#[test]
fn install_kernel_forwards_the_system_scope() {
  let dir = kernel_test_dir("install_kernel_system_scope");
  write_jupyter_stub(&dir, 0);

  let (_stdout, stderr, ok) = run_install_kernel(&dir, &["--system"]);
  assert!(ok, "woxi install-kernel --system failed: stderr={stderr}");
  let logged = std::fs::read_to_string(dir.join("args.txt")).expect("ran");
  assert!(
    logged.lines().any(|arg| arg == "--system"),
    "--system not forwarded: {logged}"
  );
  assert!(
    !logged.lines().any(|arg| arg == "--user"),
    "--user forwarded alongside --system: {logged}"
  );

  std::fs::remove_dir_all(&dir).ok();
}

/// A failing `jupyter kernelspec install` is reported and exits non-zero.
#[cfg(unix)]
#[test]
fn install_kernel_fails_when_jupyter_fails() {
  let dir = kernel_test_dir("install_kernel_jupyter_error");
  write_jupyter_stub(&dir, 1);

  let (stdout, stderr, ok) = run_install_kernel(&dir, &[]);
  assert!(!ok, "expected a non-zero exit: stdout={stdout}");
  assert!(
    stderr.contains("Error installing kernel"),
    "stderr={stderr}"
  );

  std::fs::remove_dir_all(&dir).ok();
}
