//! Python bindings for the Woxi Wolfram Language interpreter.
//!
//! Compiled by maturin into the extension module `woxi._woxi`; the
//! pure-Python package in `python/woxi` re-exports its public API.
//!
//! Note on state: the interpreter keeps its session state (variable
//! definitions, RNG seed, `%` history, …) in thread-locals, so every
//! Python thread gets an independent Wolfram session.

mod convert;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use convert::{expr_to_py, py_to_expr};

create_exception!(
  _woxi,
  WolframError,
  PyException,
  "Raised when Woxi fails to parse or evaluate Wolfram Language code."
);

fn to_py_err(err: woxi::InterpreterError) -> PyErr {
  WolframError::new_err(err.to_string())
}

/// The interpreter returns "\0" as a sentinel for suppressed output
/// (Null, a trailing semicolon, or output that was already printed).
/// Map it to "Null" like the CLI does, matching wolframscript.
fn normalize_result(result: String) -> String {
  if result == "\0" {
    "Null".to_string()
  } else {
    result
  }
}

/// Audio captured during an evaluation (from `Play`, `Sound`, or an
/// `Audio` object).
#[pyclass(frozen, get_all, module = "woxi")]
#[derive(Clone)]
struct Sound {
  /// Base64-encoded audio data (may be empty when unavailable).
  base64: String,
  /// MIME type of the encoded data, e.g. "audio/wav".
  mime: String,
  /// Display label, e.g. the file name for file-backed audio.
  label: Option<String>,
}

#[pymethods]
impl Sound {
  fn __repr__(&self) -> String {
    format!(
      "Sound(mime={:?}, label={:?}, base64=<{} bytes>)",
      self.mime,
      self.label,
      self.base64.len()
    )
  }
}

/// Full result of evaluating Wolfram Language code with `evaluate()`.
#[pyclass(frozen, get_all, module = "woxi")]
struct EvaluationResult {
  /// The final expression value formatted as Wolfram Language text.
  result: String,
  /// The final value as a structured expression tree (Symbol/Expr/int/…).
  ///
  /// None when the evaluation produced no value at all — a definition, or
  /// a statement whose display a trailing `;` suppressed. An evaluation
  /// whose value *is* Null (`Print["hi"]`) reports `Symbol("Null")`.
  expr: Option<Py<PyAny>>,
  /// Text printed during evaluation (Print, WriteString["stdout", …], …).
  stdout: String,
  /// SVG markup of captured graphics (Plot, Graphics, …), if any.
  graphics: Option<String>,
  /// SVG rendering of the result for display forms (TableForm, …), if any.
  svg: Option<String>,
  /// Playable audio produced by the evaluation, if any.
  sound: Option<Sound>,
  /// Warnings and messages emitted during evaluation.
  warnings: Vec<String>,
}

#[pymethods]
impl EvaluationResult {
  fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
    Ok(format!(
      "EvaluationResult(result={:?}, expr={}, stdout={:?}, graphics={}, svg={}, sound={}, warnings={:?})",
      self.result,
      match &self.expr {
        Some(e) => e.bind(py).repr()?.to_string(),
        None => "None".to_string(),
      },
      self.stdout,
      if self.graphics.is_some() {
        "<svg>"
      } else {
        "None"
      },
      if self.svg.is_some() { "<svg>" } else { "None" },
      if self.sound.is_some() {
        "<sound>"
      } else {
        "None"
      },
      self.warnings,
    ))
  }
}

/// Convert an interpreter outcome into the Python-facing result object.
/// Shared by `evaluate` and `evaluate_expr` so the two report identically.
fn to_evaluation_result(
  py: Python<'_>,
  r: woxi::InterpretResult,
) -> PyResult<EvaluationResult> {
  Ok(EvaluationResult {
    result: normalize_result(r.result),
    expr: r.expr.as_ref().map(|e| expr_to_py(py, e)).transpose()?,
    stdout: r.stdout,
    graphics: r.graphics,
    svg: r.output_svg,
    sound: r.sound.map(|s| Sound {
      base64: s.base64,
      mime: s.mime,
      label: s.label,
    }),
    warnings: r.warnings,
  })
}

/// Evaluate Wolfram Language code and return the result as a string.
///
/// Print output goes directly to the process stdout, exactly like
/// `woxi eval` on the command line. Raises WolframError on failure.
#[pyfunction]
fn interpret(py: Python<'_>, code: &str) -> PyResult<String> {
  py.detach(|| woxi::interpret(code))
    .map(normalize_result)
    .map_err(to_py_err)
}

/// Evaluate Wolfram Language code and capture everything it produces.
///
/// Returns an EvaluationResult with the result string, captured stdout,
/// graphics/SVG output, sound, and warnings. Unless `print_to_stdout` is
/// true, Print output is only captured, not echoed to the process stdout.
#[pyfunction]
#[pyo3(signature = (code, *, print_to_stdout = false))]
fn evaluate(
  py: Python<'_>,
  code: &str,
  print_to_stdout: bool,
) -> PyResult<EvaluationResult> {
  let outcome = py
    .detach(|| {
      let was_quiet = woxi::is_quiet_print();
      woxi::set_quiet_print(!print_to_stdout);
      let outcome = woxi::interpret_with_stdout(code);
      woxi::set_quiet_print(was_quiet);
      outcome
    })
    .map_err(to_py_err)?;
  to_evaluation_result(py, outcome)
}

/// Evaluate a structured expression tree and capture everything it produces.
///
/// The tree is built from Symbol/Expr nodes and plain Python values —
/// `evaluate_expr(wl.Integrate(wl.Power(wl.x, 2), wl.x))` — so no string
/// concatenation is needed to pass Python data into an evaluation.
/// Returns the same EvaluationResult as `evaluate()`.
#[pyfunction]
#[pyo3(signature = (expr, *, print_to_stdout = false))]
fn evaluate_expr(
  py: Python<'_>,
  expr: &Bound<'_, PyAny>,
  print_to_stdout: bool,
) -> PyResult<EvaluationResult> {
  let tree = py_to_expr(expr)?;
  let outcome = py
    .detach(|| {
      let was_quiet = woxi::is_quiet_print();
      woxi::set_quiet_print(!print_to_stdout);
      let outcome = woxi::interpret_expr_with_stdout(&tree);
      woxi::set_quiet_print(was_quiet);
      outcome
    })
    .map_err(to_py_err)?;
  to_evaluation_result(py, outcome)
}

/// Parse Wolfram Language code into a structured expression tree *without*
/// evaluating it, so held and unevaluated forms stay intact:
/// `parse_expr("1 + 1")` is `Expr(Symbol("Plus"), [1, 1])`, not `2`.
///
/// Raises WolframError if the code does not parse.
#[pyfunction]
fn parse_expr(py: Python<'_>, code: &str) -> PyResult<Py<PyAny>> {
  let tree = py.detach(|| woxi::parse_to_expr(code)).map_err(to_py_err)?;
  expr_to_py(py, &tree)
}

/// Reset the interpreter session of the current thread: clears all
/// variable and function definitions, contexts, and output history.
#[pyfunction]
fn clear_state() {
  woxi::clear_state();
}

/// Seed the interpreter's random number generator for reproducible
/// RandomInteger/RandomReal/… results.
#[pyfunction]
fn seed_rng(seed: u64) {
  woxi::seed_rng(seed);
}

/// Remove a previously set RNG seed and return to entropy-based
/// random numbers.
#[pyfunction]
fn unseed_rng() {
  woxi::unseed_rng();
}

/// Enable or disable REPL session mode: persistent `%` / `Out[n]` output
/// history across interpret()/evaluate() calls, plus the OutputForm number
/// display wolframscript's terminal REPL uses (a machine real shows 6
/// significant figures, so `3203.60 - 2711.16` prints as `492.44`).
#[pyfunction]
fn set_repl_mode(enabled: bool) {
  woxi::set_repl_mode(enabled);
}

/// Route Wolfram diagnostic messages (e.g. `Power::infy`) to stdout
/// instead of the internal capture buffer, like `woxi run` /
/// `wolframscript -file` do.
#[pyfunction]
fn set_messages_to_stdout(enabled: bool) {
  woxi::set_messages_to_stdout(enabled);
}

/// Set a Wolfram system variable (e.g. "$InputFileName") to a raw
/// Wolfram Language value, given as source text.
#[pyfunction]
fn set_system_variable(name: &str, value: &str) {
  woxi::set_system_variable(name, value);
}

/// Set `$ScriptCommandLine` for script execution: the first element is
/// the script path, the rest are its arguments.
#[pyfunction]
fn set_script_command_line(args: Vec<String>) {
  woxi::set_script_command_line(&args);
}

/// Return and clear the stack trace of the most recent evaluation
/// error, if one was recorded.
#[pyfunction]
fn take_error_trace() -> Option<String> {
  woxi::take_error_trace()
}

#[pymodule]
fn _woxi(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(interpret, m)?)?;
  m.add_function(wrap_pyfunction!(evaluate, m)?)?;
  m.add_function(wrap_pyfunction!(evaluate_expr, m)?)?;
  m.add_function(wrap_pyfunction!(parse_expr, m)?)?;
  m.add_function(wrap_pyfunction!(clear_state, m)?)?;
  m.add_function(wrap_pyfunction!(seed_rng, m)?)?;
  m.add_function(wrap_pyfunction!(unseed_rng, m)?)?;
  m.add_function(wrap_pyfunction!(set_repl_mode, m)?)?;
  m.add_function(wrap_pyfunction!(set_messages_to_stdout, m)?)?;
  m.add_function(wrap_pyfunction!(set_system_variable, m)?)?;
  m.add_function(wrap_pyfunction!(set_script_command_line, m)?)?;
  m.add_function(wrap_pyfunction!(take_error_trace, m)?)?;
  m.add_class::<EvaluationResult>()?;
  m.add_class::<Sound>()?;
  m.add_class::<convert::Symbol>()?;
  m.add_class::<convert::Expr>()?;
  m.add_class::<convert::BigReal>()?;
  m.add_class::<convert::Graphics>()?;
  m.add_class::<convert::Image>()?;
  m.add("WolframError", m.py().get_type::<WolframError>())?;
  m.add("__version__", env!("CARGO_PKG_VERSION"))?;
  Ok(())
}
