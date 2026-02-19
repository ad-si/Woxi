use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cell::RefCell;
use std::collections::HashMap;
use thiserror::Error;

pub mod evaluator;
pub mod functions;
pub mod syntax;
pub mod utils;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[derive(Parser)]
#[grammar = "wolfram.pest"]
pub struct WolframParser;

#[derive(Clone)]
enum StoredValue {
  Association(Vec<(String, String)>),
  Raw(String),           // keep evaluated textual value
  ExprVal(syntax::Expr), // keep as structured AST for fast Part access
}
thread_local! {
    static ENV: RefCell<HashMap<String, StoredValue>> = RefCell::new(HashMap::new());
    //            name         Vec of (param_names, conditions, defaults, head_constraints, body_AST) for multi-arity + condition + optional support
    static FUNC_DEFS: RefCell<HashMap<String, Vec<(Vec<String>, Vec<Option<syntax::Expr>>, Vec<Option<syntax::Expr>>, Vec<Option<String>>, syntax::Expr)>>> = RefCell::new(HashMap::new());
    // Function attributes (e.g., Listable, Flat, etc.)
    static FUNC_ATTRS: RefCell<HashMap<String, Vec<String>>> = RefCell::new(HashMap::new());
    // Track Part evaluation nesting depth for Part::partd warnings
    static PART_DEPTH: RefCell<usize> = const { RefCell::new(0) };
    // Reap/Sow stack: each Reap call pushes a Vec to collect (value, tag) pairs
    pub static SOW_STACK: RefCell<Vec<Vec<(syntax::Expr, syntax::Expr)>>> = const { RefCell::new(Vec::new()) };
}

#[derive(Error, Debug)]
pub enum InterpreterError {
  #[error("Parse error: {0}")]
  ParseError(#[from] Box<pest::error::Error<Rule>>),
  #[error("Empty input")]
  EmptyInput,
  #[error("Evaluation error: {0}")]
  EvaluationError(String),
  #[error("Return")]
  ReturnValue(Box<syntax::Expr>),
  #[error("Break")]
  BreakSignal,
  #[error("Continue")]
  ContinueSignal,
  #[error("Throw")]
  ThrowValue(Box<syntax::Expr>, Option<Box<syntax::Expr>>),
  #[error("$Aborted")]
  Abort,
}

/// Extended result type that includes both stdout and the result
#[derive(Debug, Clone)]
pub struct InterpretResult {
  pub stdout: String,
  pub result: String,
  pub graphics: Option<String>,
  pub output_svg: Option<String>,
  pub warnings: Vec<String>,
}

impl WolframParser {
  pub fn parse_wolfram(
    input: &str,
  ) -> Result<pest::iterators::Pairs<'_, Rule>, Box<pest::error::Error<Rule>>>
  {
    Self::parse(Rule::Program, input).map_err(Box::new)
  }
}

pub fn parse(
  input: &str,
) -> Result<pest::iterators::Pairs<'_, Rule>, Box<pest::error::Error<Rule>>> {
  WolframParser::parse_wolfram(input)
}

// Global RNG state: None = use thread_rng(), Some = use seeded ChaCha8Rng
thread_local! {
    static SEEDED_RNG: RefCell<Option<ChaCha8Rng>> = const { RefCell::new(None) };
}

/// Seed the global RNG with a specific seed value (SeedRandom[n]).
pub fn seed_rng(seed: u64) {
  SEEDED_RNG.with(|rng| {
    *rng.borrow_mut() = Some(ChaCha8Rng::seed_from_u64(seed));
  });
}

/// Reset the global RNG to non-deterministic mode (SeedRandom[]).
pub fn unseed_rng() {
  SEEDED_RNG.with(|rng| {
    *rng.borrow_mut() = None;
  });
}

/// Execute a closure with a mutable reference to the current RNG.
/// Uses the seeded RNG if set, otherwise falls back to thread_rng().
pub fn with_rng<F, R>(f: F) -> R
where
  F: FnOnce(&mut dyn rand::RngCore) -> R,
{
  SEEDED_RNG.with(|cell| {
    let mut borrow = cell.borrow_mut();
    if let Some(ref mut seeded) = *borrow {
      f(seeded)
    } else {
      f(&mut rand::thread_rng())
    }
  })
}

// Captured output from Print statements
thread_local! {
    static CAPTURED_STDOUT: RefCell<String> = const { RefCell::new(String::new()) };
}

// Visual display mode flag — set by interpret_with_stdout to enable
// rendering of display wrappers like TableForm as SVG grids
thread_local! {
    static VISUAL_MODE: RefCell<bool> = const { RefCell::new(false) };
}

// Captured graphical output (SVG) from Plot and related functions
thread_local! {
    static CAPTURED_GRAPHICS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

// Captured GraphicsBox expression (Mathematica .nb format) from Graphics/Plot
thread_local! {
    static CAPTURED_GRAPHICSBOX: RefCell<Option<String>> = const { RefCell::new(None) };
}

// Captured SVG rendering of the text output (always generated, with superscripts etc.)
thread_local! {
    static CAPTURED_OUTPUT_SVG: RefCell<Option<String>> = const { RefCell::new(None) };
}

// Captured unimplemented function calls (e.g. "Quantity[13.77, \"BillionYears\"]")
thread_local! {
    static UNIMPLEMENTED_CALLS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

// Captured warnings (general-purpose, e.g. deprecation notices)
thread_local! {
    static CAPTURED_WARNINGS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Clears the captured stdout buffer
fn clear_captured_stdout() {
  CAPTURED_STDOUT.with(|buffer| {
    buffer.borrow_mut().clear();
  });
}

/// Appends to the captured stdout buffer
fn capture_stdout(text: &str) {
  CAPTURED_STDOUT.with(|buffer| {
    buffer.borrow_mut().push_str(text);
    buffer.borrow_mut().push('\n');
  });
}

/// Gets the captured stdout content
fn get_captured_stdout() -> String {
  CAPTURED_STDOUT.with(|buffer| buffer.borrow().clone())
}

/// Clears the captured warnings and unimplemented calls buffers
fn clear_captured_warnings() {
  UNIMPLEMENTED_CALLS.with(|buffer| {
    buffer.borrow_mut().clear();
  });
  CAPTURED_WARNINGS.with(|buffer| {
    buffer.borrow_mut().clear();
  });
}

/// Records a call to an unimplemented built-in function (e.g. "Quantity[13.77, \"BillionYears\"]")
pub fn capture_unimplemented_call(call_str: &str) {
  UNIMPLEMENTED_CALLS.with(|buffer| {
    buffer.borrow_mut().push(call_str.to_string());
  });
}

/// Appends a warning message
pub fn capture_warning(text: &str) {
  CAPTURED_WARNINGS.with(|buffer| {
    buffer.borrow_mut().push(text.to_string());
  });
}

/// Gets the captured warnings, consolidating unimplemented function calls into a single message
pub fn get_captured_warnings() -> Vec<String> {
  let mut warnings = Vec::new();

  let calls = UNIMPLEMENTED_CALLS.with(|buffer| buffer.borrow().clone());
  if !calls.is_empty() {
    let joined = calls.join(", ");
    let verb = if calls.len() == 1 {
      "is a built-in Wolfram Language function"
    } else {
      "are built-in Wolfram Language functions"
    };
    warnings.push(format!("{} {} not yet implemented in Woxi.", joined, verb));
  }

  CAPTURED_WARNINGS.with(|buffer| {
    warnings.extend(buffer.borrow().clone());
  });

  warnings
}

/// Clears the captured graphics buffer
fn clear_captured_graphics() {
  CAPTURED_GRAPHICS.with(|buffer| {
    buffer.borrow_mut().clear();
  });
}

/// Stores SVG graphics for capture by the Jupyter kernel
pub fn capture_graphics(svg: &str) {
  CAPTURED_GRAPHICS.with(|buffer| {
    buffer.borrow_mut().push(svg.to_string());
  });
}

/// Gets the last captured graphics content (backward compatible)
pub fn get_captured_graphics() -> Option<String> {
  CAPTURED_GRAPHICS.with(|buffer| buffer.borrow().last().cloned())
}

/// Gets all captured graphics SVGs
pub fn get_all_captured_graphics() -> Vec<String> {
  CAPTURED_GRAPHICS.with(|buffer| buffer.borrow().clone())
}

/// Clears the captured GraphicsBox buffer
fn clear_captured_graphicsbox() {
  CAPTURED_GRAPHICSBOX.with(|buffer| {
    *buffer.borrow_mut() = None;
  });
}

/// Stores a GraphicsBox expression string for .nb export
pub fn capture_graphicsbox(expr: &str) {
  CAPTURED_GRAPHICSBOX.with(|buffer| {
    *buffer.borrow_mut() = Some(expr.to_string());
  });
}

/// Gets the captured GraphicsBox expression
pub fn get_captured_graphicsbox() -> Option<String> {
  CAPTURED_GRAPHICSBOX.with(|buffer| buffer.borrow().clone())
}

/// Clears the captured output SVG buffer
fn clear_captured_output_svg() {
  CAPTURED_OUTPUT_SVG.with(|buffer| {
    *buffer.borrow_mut() = None;
  });
}

/// Stores an SVG rendering of the text output
fn capture_output_svg(svg: &str) {
  CAPTURED_OUTPUT_SVG.with(|buffer| {
    *buffer.borrow_mut() = Some(svg.to_string());
  });
}

/// Gets the captured output SVG
pub fn get_captured_output_svg() -> Option<String> {
  CAPTURED_OUTPUT_SVG.with(|buffer| buffer.borrow().clone())
}

// Re-export evaluate_expr from evaluator module
pub use evaluator::evaluate_expr;

/// Set a system variable (like $ScriptCommandLine) in the environment
pub fn set_system_variable(name: &str, value: &str) {
  ENV.with(|e| {
    e.borrow_mut()
      .insert(name.to_string(), StoredValue::Raw(value.to_string()));
  });
}

/// Remove a first line that starts with "#!" (shebang);
/// returns the remainder as a new `String`.
pub fn without_shebang(src: &str) -> String {
  if src.starts_with("#!") {
    src.lines().skip(1).collect::<Vec<_>>().join("\n")
  } else {
    src.to_owned()
  }
}

/// Clear all thread-local interpreter state (environment variables
/// and user-defined functions).  Useful for isolating test runs.
pub fn clear_state() {
  ENV.with(|e| e.borrow_mut().clear());
  FUNC_DEFS.with(|m| m.borrow_mut().clear());
  FUNC_ATTRS.with(|m| m.borrow_mut().clear());
  SOW_STACK.with(|s| s.borrow_mut().clear());
  unseed_rng();
  clear_captured_stdout();
  clear_captured_graphics();
  clear_captured_graphicsbox();
}

/// Set the $ScriptCommandLine variable from command-line arguments
pub fn set_script_command_line(args: &[String]) {
  // Format as a Wolfram list: {"script.wls", "arg1", "arg2", ...}
  let list_str = format!(
    "{{{}}}",
    args
      .iter()
      .map(|s| format!("\"{}\"", s))
      .collect::<Vec<_>>()
      .join(", ")
  );
  set_system_variable("$ScriptCommandLine", &list_str);
}

// Track recursion depth to avoid clearing stdout in nested calls
thread_local! {
    static INTERPRET_DEPTH: std::cell::RefCell<usize> = const { std::cell::RefCell::new(0) };
}

pub fn interpret(input: &str) -> Result<String, InterpreterError> {
  let trimmed = input.trim();

  // Fast path for simple literals that don't need parsing
  // Check for integer
  if let Ok(n) = trimmed.parse::<i64>() {
    return Ok(n.to_string());
  }
  // Check for float (must contain '.' to distinguish from integer)
  if trimmed.contains('.')
    && let Ok(n) = trimmed.parse::<f64>()
  {
    return Ok(format_real_result(n));
  }
  // Check for quoted string - return content without quotes (like wolframscript)
  if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
    // Make sure there are no unescaped quotes inside
    let inner = &trimmed[1..trimmed.len() - 1];
    if !inner.contains('"') {
      return Ok(inner.to_string());
    }
  }

  // Fast path for simple list literals like {a, b, c}
  // This handles many cases where we're just passing data around
  if trimmed.starts_with('{') && trimmed.ends_with('}') {
    // Check if it's a simple list (no operators that need evaluation)
    if !trimmed.contains("->")
      && !trimmed.contains(":>")
      && !trimmed.contains("/.")
      && !trimmed.contains("//")
      && !trimmed.contains("/@")
      && !trimmed.contains("@@")
      && !trimmed.contains('+')
      && !trimmed.contains('-')
      && !trimmed.contains('*')
      && !trimmed.contains('/')
      && !trimmed.contains('[')
      && !trimmed.contains('"')
      && !trimmed.contains('#')
      && !trimmed.contains("Nothing")
      && !trimmed.contains(" . ")
      && !trimmed.contains(".{")
      && !trimmed.contains('^')
      && !trimmed.contains('.')
      && !trimmed.contains('=')
    // Reals may need scientific notation formatting
    {
      // Simple list with no function calls or operators - return as-is
      return Ok(trimmed.to_string());
    }
  }

  // Fast path for simple identifiers (variable lookup)
  if trimmed
    .chars()
    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
    && !trimmed.is_empty()
    && trimmed.chars().next().unwrap().is_ascii_alphabetic()
  {
    // This is a simple identifier
    if let Some(stored) = ENV.with(|e| e.borrow().get(trimmed).cloned()) {
      return Ok(match stored {
        StoredValue::ExprVal(e) => syntax::expr_to_output(&e),
        StoredValue::Raw(val) => val,
        StoredValue::Association(items) => {
          let items_expr: Vec<(syntax::Expr, syntax::Expr)> = items
            .iter()
            .map(|(k, v)| {
              let key_expr = syntax::string_to_expr(k)
                .unwrap_or(syntax::Expr::Identifier(k.clone()));
              let val_expr = syntax::string_to_expr(v)
                .unwrap_or(syntax::Expr::Raw(v.clone()));
              (key_expr, val_expr)
            })
            .collect();
          syntax::expr_to_output(&syntax::Expr::Association(items_expr))
        }
      });
    }
    // Handle built-in symbols that evaluate to values
    #[cfg(not(target_arch = "wasm32"))]
    if trimmed == "Now" {
      use chrono::Local;
      let now = Local::now();
      let seconds = now
        .format("%S%.f")
        .to_string()
        .parse::<f64>()
        .unwrap_or(0.0);
      let tz_offset_hours = now.offset().local_minus_utc() as f64 / 3600.0;
      let expr = syntax::Expr::FunctionCall {
        name: "DateObject".to_string(),
        args: vec![
          syntax::Expr::List(vec![
            syntax::Expr::Integer(
              now.format("%Y").to_string().parse::<i128>().unwrap(),
            ),
            syntax::Expr::Integer(
              now.format("%m").to_string().parse::<i128>().unwrap(),
            ),
            syntax::Expr::Integer(
              now.format("%d").to_string().parse::<i128>().unwrap(),
            ),
            syntax::Expr::Integer(
              now.format("%H").to_string().parse::<i128>().unwrap(),
            ),
            syntax::Expr::Integer(
              now.format("%M").to_string().parse::<i128>().unwrap(),
            ),
            syntax::Expr::Real(seconds),
          ]),
          syntax::Expr::Identifier("Instant".to_string()),
          syntax::Expr::Identifier("Gregorian".to_string()),
          syntax::Expr::Real(tz_offset_hours),
        ],
      };
      return Ok(syntax::expr_to_output(&expr));
    }
    // Return identifier as-is if not found
    return Ok(trimmed.to_string());
  }

  // Fast path for simple function calls like MemberQ[{a, b}, x]
  // This handles the common case in Select predicates
  if let Some(result) = try_fast_function_call(trimmed) {
    return result;
  }

  let depth = INTERPRET_DEPTH.with(|d| {
    let mut depth = d.borrow_mut();
    let current = *depth;
    *depth += 1;
    current
  });

  // Only clear buffers at top level
  if depth == 0 {
    clear_captured_stdout();
    clear_captured_warnings();
  }

  // Decrement depth on scope exit
  struct DepthGuard;
  impl Drop for DepthGuard {
    fn drop(&mut self) {
      INTERPRET_DEPTH.with(|d| *d.borrow_mut() -= 1);
    }
  }
  let _guard = DepthGuard;

  // Insert semicolons at top-level newline boundaries so the PEG grammar
  // correctly separates statements like "fib[0] = 0\nfib[1] = 1" instead
  // of treating them as implicit multiplication.
  let preprocessed = insert_statement_separators(trimmed);

  // Regular interpretation - use AST-based evaluation
  let pairs = parse(&preprocessed)?;
  let mut pairs = pairs.into_iter();
  let program = pairs.next().ok_or(InterpreterError::EmptyInput)?;

  if program.as_rule() != Rule::Program {
    return Err(InterpreterError::EvaluationError(format!(
      "Expected Program, got {:?}",
      program.as_rule()
    )));
  }

  let mut last_result = None;
  let mut any_nonempty = false;
  let mut trailing_semicolon = false;
  for node in program.into_inner() {
    match node.as_rule() {
      Rule::Expression => {
        // Convert Pair to Expr AST
        let expr = syntax::pair_to_expr(node);
        // Evaluate using AST-based evaluation
        // At top level, uncaught Return[] becomes symbolic Return[val] (like wolframscript)
        let result_expr = match evaluator::evaluate_expr_to_expr(&expr) {
          Err(InterpreterError::ReturnValue(val)) => {
            syntax::Expr::FunctionCall {
              name: "Return".to_string(),
              args: vec![*val],
            }
          }
          Err(InterpreterError::Abort) => {
            return Ok("$Aborted".to_string());
          }
          other => other?,
        };
        // If the result is a Grid expression, render it as SVG
        let result_expr = render_grid_if_needed(result_expr);
        // If the result is a Dataset expression, render it as an SVG table
        let result_expr = render_dataset_if_needed(result_expr);
        // In visual mode, render TableForm[list] as a Grid SVG
        let result_expr = if VISUAL_MODE.with(|v| *v.borrow()) {
          render_tableform_if_needed(result_expr)
        } else {
          result_expr
        };
        // If the result is a list of Graphics objects, combine their SVGs
        let result_expr = render_graphics_list_if_needed(result_expr);
        // Generate SVG rendering of the result for playground display
        generate_output_svg(&result_expr);
        // Convert to output string (strips quotes from strings for display)
        last_result = Some(syntax::expr_to_output(&result_expr));
        any_nonempty = true;
      }
      Rule::FunctionDefinition => {
        store_function_definition(node)?;
        any_nonempty = true;
      }
      Rule::TrailingSemicolon => {
        trailing_semicolon = true;
      }
      _ => {} // ignore EOI, etc.
    }
  }

  // Print consolidated unimplemented-function warning to stderr (top-level only)
  if depth == 0 {
    for w in get_captured_warnings() {
      eprintln!("{}", w);
    }
  }

  if any_nonempty {
    if trailing_semicolon {
      Ok("Null".to_string())
    } else {
      last_result.ok_or(InterpreterError::EmptyInput)
    }
  } else {
    Err(InterpreterError::EmptyInput)
  }
}

/// If `expr` is a Grid[…] or TableForm[…] call (possibly nested in a list),
/// render it as SVG and return `-Graphics-`. Grid/TableForm stay symbolic
/// during evaluation so that part-assignment works; rendering only happens
/// at the output stage.
fn render_grid_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "Grid" && !args.is_empty() =>
    {
      match functions::graphics::grid_ast(args) {
        Ok(result) => result,
        Err(_) => expr,
      }
    }
    syntax::Expr::List(items) => {
      let new_items: Vec<syntax::Expr> =
        items.iter().cloned().map(render_grid_if_needed).collect();
      syntax::Expr::List(new_items)
    }
    _ => expr,
  }
}

/// If `expr` is a Dataset[data, …] call, render it as an SVG table
/// and return `-Graphics-`.
fn render_dataset_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "Dataset" && !args.is_empty() =>
    {
      let data = &args[0];
      if let Some(svg) = functions::graphics::dataset_to_svg(data) {
        capture_graphics(&svg);
        syntax::Expr::Identifier("-Graphics-".to_string())
      } else {
        expr
      }
    }
    _ => expr,
  }
}

/// Check if an expression represents a Graphics placeholder
/// (either `-Graphics-` directly or `Style[-Graphics-, ...]`)
fn is_graphics_placeholder(expr: &syntax::Expr) -> bool {
  match expr {
    syntax::Expr::Identifier(s) if s == "-Graphics-" => true,
    syntax::Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
    {
      is_graphics_placeholder(&args[0])
    }
    _ => false,
  }
}

/// Check if an expression tree contains any Graphics placeholder
fn contains_graphics_placeholder(expr: &syntax::Expr) -> bool {
  if is_graphics_placeholder(expr) {
    return true;
  }
  match expr {
    syntax::Expr::List(items) => {
      items.iter().any(contains_graphics_placeholder)
    }
    syntax::Expr::FunctionCall { args, .. } => {
      args.iter().any(contains_graphics_placeholder)
    }
    _ => false,
  }
}

/// Check if a list's items form a 3D structure (list of lists of lists)
fn is_3d_list(items: &[syntax::Expr]) -> bool {
  !items.is_empty()
    && items.iter().all(|item| {
      if let syntax::Expr::List(sub) = item {
        !sub.is_empty()
          && sub.iter().all(|s| matches!(s, syntax::Expr::List(_)))
      } else {
        false
      }
    })
}

/// If `expr` is a TableForm[list] with non-graphics data, render as a Grid SVG.
/// This is only called from `interpret_with_stdout` (visual contexts),
/// not from plain `interpret` (where TableForm stays symbolic).
fn render_tableform_if_needed(expr: syntax::Expr) -> syntax::Expr {
  match &expr {
    syntax::Expr::FunctionCall { name, args }
      if name == "TableForm" && args.len() == 1 =>
    {
      let data = &args[0];
      // Skip if content contains Graphics placeholders (handled by render_graphics_list_if_needed)
      if contains_graphics_placeholder(data) {
        return expr;
      }
      // Build grid data and optional group gap indices
      let (grid_data, group_gaps) = match data {
        syntax::Expr::List(items) if is_3d_list(items) => {
          // 3D list M[dim1][dim2][dim3]:
          // Each block M[i] is transposed (sub-lists become columns),
          // then blocks are stacked vertically.
          let mut rows: Vec<syntax::Expr> = Vec::new();
          let mut gaps: Vec<usize> = Vec::new();
          for (bi, block) in items.iter().enumerate() {
            if let syntax::Expr::List(sub_lists) = block {
              if bi > 0 {
                gaps.push(rows.len());
              }
              let dim3 = sub_lists
                .iter()
                .map(|sl| {
                  if let syntax::Expr::List(v) = sl {
                    v.len()
                  } else {
                    0
                  }
                })
                .max()
                .unwrap_or(0);
              for k in 0..dim3 {
                let row: Vec<syntax::Expr> = sub_lists
                  .iter()
                  .map(|sl| {
                    if let syntax::Expr::List(v) = sl {
                      v.get(k)
                        .cloned()
                        .unwrap_or(syntax::Expr::Identifier(String::new()))
                    } else {
                      sl.clone()
                    }
                  })
                  .collect();
                rows.push(syntax::Expr::List(row));
              }
            }
          }
          (syntax::Expr::List(rows), gaps)
        }
        syntax::Expr::List(items)
          if items
            .iter()
            .all(|item| matches!(item, syntax::Expr::List(_))) =>
        {
          (data.clone(), vec![])
        }
        syntax::Expr::List(items) if !items.is_empty() => {
          // 1D list — wrap each element in a single-element list (column)
          (
            syntax::Expr::List(
              items
                .iter()
                .map(|e| syntax::Expr::List(vec![e.clone()]))
                .collect(),
            ),
            vec![],
          )
        }
        _ => return expr,
      };
      let result = if group_gaps.is_empty() {
        functions::graphics::grid_ast(&[grid_data])
      } else {
        functions::graphics::grid_ast_with_gaps(&[grid_data], &group_gaps)
      };
      match result {
        Ok(result) => result,
        Err(_) => expr,
      }
    }
    _ => expr,
  }
}

/// If the result is a list (1D, 2D, or 3D) of `-Graphics-` items,
/// or a `TableForm` wrapping such a list, combine captured SVGs into a grid.
fn render_graphics_list_if_needed(expr: syntax::Expr) -> syntax::Expr {
  // Unwrap TableForm[list] or MathMLForm[TableForm[list]] etc.
  let has_tableform = has_form_wrapper(&expr, "TableForm");
  let inner = unwrap_form_wrappers(&expr);

  let all_svgs = get_all_captured_graphics();
  if all_svgs.is_empty() {
    return expr;
  }

  // 1D list of Graphics
  if let syntax::Expr::List(items) = inner {
    if items.iter().all(is_graphics_placeholder)
      && items.len() > 1
      && items.len() <= all_svgs.len()
    {
      // Take the last N SVGs (they correspond to the list items)
      let start = all_svgs.len() - items.len();
      let row: Vec<String> = all_svgs[start..].to_vec();
      if let Some(combined) = functions::graphics::combine_graphics_svgs(&[row])
      {
        // Clear and re-capture with the combined SVG
        clear_captured_graphics();
        capture_graphics(&combined);
        return syntax::Expr::Identifier("-Graphics-".to_string());
      }
    }

    // 2D list: list of lists of Graphics
    if items.iter().all(|e| {
      if let syntax::Expr::List(inner) = e {
        inner.iter().all(is_graphics_placeholder) && !inner.is_empty()
      } else {
        false
      }
    }) && !items.is_empty()
    {
      let total_cells: usize = items
        .iter()
        .map(|e| {
          if let syntax::Expr::List(inner) = e {
            inner.len()
          } else {
            0
          }
        })
        .sum();
      if total_cells <= all_svgs.len() {
        let start = all_svgs.len() - total_cells;
        let mut offset = start;
        let mut rows: Vec<Vec<String>> = Vec::new();
        for item in items {
          if let syntax::Expr::List(inner) = item {
            let row: Vec<String> =
              all_svgs[offset..offset + inner.len()].to_vec();
            offset += inner.len();
            rows.push(row);
          }
        }
        if let Some(combined) =
          functions::graphics::combine_graphics_svgs(&rows)
        {
          clear_captured_graphics();
          capture_graphics(&combined);
          return syntax::Expr::Identifier("-Graphics-".to_string());
        }
      }
    }

    // 3D list: list of lists of lists of Graphics
    // Structure: items[dim1][dim2][dim3]
    if items.iter().all(|e| {
      if let syntax::Expr::List(rows) = e {
        rows.iter().all(|r| {
          if let syntax::Expr::List(cols) = r {
            cols.iter().all(is_graphics_placeholder) && !cols.is_empty()
          } else {
            false
          }
        }) && !rows.is_empty()
      } else {
        false
      }
    }) && !items.is_empty()
    {
      let total_cells: usize = items
        .iter()
        .map(|e| {
          if let syntax::Expr::List(rows) = e {
            rows
              .iter()
              .map(|r| {
                if let syntax::Expr::List(cols) = r {
                  cols.len()
                } else {
                  0
                }
              })
              .sum()
          } else {
            0
          }
        })
        .sum();
      if total_cells <= all_svgs.len() {
        let start = all_svgs.len() - total_cells;

        // Collect SVGs into 3D structure [dim1][dim2][dim3]
        let mut offset = start;
        let mut svg_3d: Vec<Vec<Vec<String>>> = Vec::new();
        for item in items {
          if let syntax::Expr::List(inner_rows) = item {
            let mut block: Vec<Vec<String>> = Vec::new();
            for r in inner_rows {
              if let syntax::Expr::List(cols) = r {
                let mut row_svgs: Vec<String> = Vec::new();
                for _ in cols {
                  if offset < all_svgs.len() {
                    row_svgs.push(all_svgs[offset].clone());
                    offset += 1;
                  }
                }
                block.push(row_svgs);
              }
            }
            svg_3d.push(block);
          }
        }

        let rows: Vec<Vec<String>> = if has_tableform {
          // TableForm: transpose each block (dim3→rows, dim2→cols),
          // stack blocks vertically
          let mut rows = Vec::new();
          for block in &svg_3d {
            let dim3 = block.iter().map(|r| r.len()).max().unwrap_or(0);
            for k in 0..dim3 {
              let row: Vec<String> = block
                .iter()
                .map(|sub| sub.get(k).cloned().unwrap_or_default())
                .collect();
              rows.push(row);
            }
          }
          rows
        } else {
          // No TableForm: one row per dim1, flatten dim2×dim3 as columns
          svg_3d
            .into_iter()
            .map(|block| block.into_iter().flatten().collect())
            .collect()
        };

        if let Some(combined) =
          functions::graphics::combine_graphics_svgs(&rows)
        {
          clear_captured_graphics();
          capture_graphics(&combined);
          return syntax::Expr::Identifier("-Graphics-".to_string());
        }
      }
    }
  }

  expr
}

/// Check if an expression has a specific form wrapper (e.g. "TableForm")
fn has_form_wrapper(expr: &syntax::Expr, target: &str) -> bool {
  match expr {
    syntax::Expr::FunctionCall { name, args }
      if args.len() == 1
        && matches!(
          name.as_str(),
          "TableForm"
            | "MathMLForm"
            | "StandardForm"
            | "InputForm"
            | "OutputForm"
        ) =>
    {
      name == target || has_form_wrapper(&args[0], target)
    }
    _ => false,
  }
}

/// Unwrap form wrappers like TableForm, MathMLForm, StandardForm, etc.
fn unwrap_form_wrappers(expr: &syntax::Expr) -> &syntax::Expr {
  match expr {
    syntax::Expr::FunctionCall { name, args }
      if args.len() == 1
        && matches!(
          name.as_str(),
          "TableForm"
            | "MathMLForm"
            | "StandardForm"
            | "InputForm"
            | "OutputForm"
        ) =>
    {
      unwrap_form_wrappers(&args[0])
    }
    _ => expr,
  }
}

/// Generate an SVG rendering of the result expression and capture it.
/// This is used by the playground to display all results with proper formatting.
fn generate_output_svg(expr: &syntax::Expr) {
  // Skip for Graphics results (they already have captured SVG)
  if matches!(expr, syntax::Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-")
  {
    return;
  }
  let markup = functions::graphics::expr_to_svg_markup(expr);
  let char_width = 8.4_f64;
  let font_size = 14_usize;
  let display_width = functions::graphics::estimate_display_width(expr);
  let width = (display_width * char_width).ceil().max(1.0) as usize;
  let (height, text_y) = if functions::graphics::has_fraction(expr) {
    (32_usize, 18_usize) // taller SVG with adjusted baseline for stacked fractions
  } else {
    (font_size + 4, font_size)
  };
  let svg = format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\
     <text x=\"0\" y=\"{text_y}\" font-family=\"monospace\" font-size=\"{font_size}\">{markup}</text>\
     </svg>",
    width, height
  );
  capture_output_svg(&svg);
}

/// Insert semicolons at top-level newline boundaries so the parser treats
/// each logical line as a separate statement.  Newlines inside brackets,
/// parentheses or braces are left alone (they're part of a multiline expression).
/// Lines that already end with `;` or `:=` are also left alone.
fn insert_statement_separators(input: &str) -> String {
  // Fast path: no newlines means nothing to do
  if !input.contains('\n') {
    return input.to_string();
  }

  let mut result = String::with_capacity(input.len() + 32);
  let mut depth: i32 = 0; // nesting depth of [], (), {}
  let mut in_string = false;
  let mut in_comment = false;
  let mut line_has_code = false; // whether the current line has non-whitespace, non-comment content
  let mut last_code_char: Option<char> = None; // last meaningful (non-comment) character
  let mut prev_code_char: Option<char> = None; // second-to-last meaningful character
  let chars: Vec<char> = input.chars().collect();
  let len = chars.len();
  let mut i = 0;

  while i < len {
    let ch = chars[i];

    // Track comment state: (* ... *)
    if !in_string && i + 1 < len && ch == '(' && chars[i + 1] == '*' {
      in_comment = true;
      result.push(ch);
      i += 1;
      continue;
    }
    if in_comment && i + 1 < len && ch == '*' && chars[i + 1] == ')' {
      in_comment = false;
      result.push(ch);
      result.push(chars[i + 1]);
      i += 2;
      continue;
    }
    if in_comment {
      result.push(ch);
      i += 1;
      continue;
    }

    // Track string state
    if ch == '"' {
      in_string = !in_string;
      result.push(ch);
      line_has_code = true;
      prev_code_char = last_code_char;
      last_code_char = Some(ch);
      i += 1;
      continue;
    }
    if in_string {
      result.push(ch);
      prev_code_char = last_code_char;
      last_code_char = Some(ch);
      i += 1;
      continue;
    }

    // Track nesting depth
    match ch {
      '[' | '(' | '{' => depth += 1,
      ']' | ')' | '}' => depth -= 1,
      _ => {}
    }

    if ch == '\n' && depth == 0 {
      // Only add `;` if the current line had actual code (not just comments/whitespace)
      // and doesn't already end with `;` or `:=`
      let ends_with_set_delayed =
        last_code_char == Some('=') && prev_code_char == Some(':');
      let needs_semi =
        line_has_code && last_code_char != Some(';') && !ends_with_set_delayed;

      if needs_semi {
        result.push(';');
      }
      result.push('\n');

      // Reset line tracking
      line_has_code = false;
      last_code_char = None;
      prev_code_char = None;
    } else if ch == '\n' {
      // Newline inside nesting — just pass through
      result.push(ch);
    } else {
      if !ch.is_whitespace() {
        line_has_code = true;
        prev_code_char = last_code_char;
        last_code_char = Some(ch);
      }
      result.push(ch);
    }

    i += 1;
  }

  result
}

/// Split input into top-level statements at newline boundaries.
/// Respects bracket nesting (newlines inside `[]`, `()`, `{}` are kept),
/// strings, comments, and `:=` continuations.
pub fn split_into_statements(input: &str) -> Vec<String> {
  let trimmed = input.trim();
  if trimmed.is_empty() {
    return vec![String::new()];
  }
  if !trimmed.contains('\n') {
    return vec![trimmed.to_string()];
  }

  let mut statements = Vec::new();
  let mut current = String::with_capacity(trimmed.len());
  let mut depth: i32 = 0;
  let mut in_string = false;
  let mut in_comment = false;
  let mut line_has_code = false;
  let mut last_code_char: Option<char> = None;
  let mut prev_code_char: Option<char> = None;
  let chars: Vec<char> = trimmed.chars().collect();
  let len = chars.len();
  let mut i = 0;

  while i < len {
    let ch = chars[i];

    // Track comment state: (* ... *)
    if !in_string && i + 1 < len && ch == '(' && chars[i + 1] == '*' {
      in_comment = true;
      current.push(ch);
      i += 1;
      continue;
    }
    if in_comment && i + 1 < len && ch == '*' && chars[i + 1] == ')' {
      in_comment = false;
      current.push(ch);
      current.push(chars[i + 1]);
      i += 2;
      continue;
    }
    if in_comment {
      current.push(ch);
      i += 1;
      continue;
    }

    // Track string state
    if ch == '"' {
      in_string = !in_string;
      current.push(ch);
      line_has_code = true;
      prev_code_char = last_code_char;
      last_code_char = Some(ch);
      i += 1;
      continue;
    }
    if in_string {
      current.push(ch);
      prev_code_char = last_code_char;
      last_code_char = Some(ch);
      i += 1;
      continue;
    }

    // Track nesting depth
    match ch {
      '[' | '(' | '{' => depth += 1,
      ']' | ')' | '}' => depth -= 1,
      _ => {}
    }

    if ch == '\n' && depth == 0 {
      let ends_with_set_delayed =
        last_code_char == Some('=') && prev_code_char == Some(':');

      if line_has_code && !ends_with_set_delayed {
        let stmt = current.trim().to_string();
        if !stmt.is_empty() {
          statements.push(stmt);
        }
        current.clear();
      } else {
        current.push(ch);
      }

      line_has_code = false;
      last_code_char = None;
      prev_code_char = None;
    } else if ch == '\n' {
      // Newline inside nesting — just pass through
      current.push(ch);
    } else {
      if !ch.is_whitespace() {
        line_has_code = true;
        prev_code_char = last_code_char;
        last_code_char = Some(ch);
      }
      current.push(ch);
    }

    i += 1;
  }

  let stmt = current.trim().to_string();
  if !stmt.is_empty() {
    statements.push(stmt);
  }

  if statements.is_empty() {
    statements.push(String::new());
  }

  statements
}

/// Try to evaluate a simple function call without full parsing.
/// Returns Some(result) if successfully handled, None if needs full parsing.
fn try_fast_function_call(
  input: &str,
) -> Option<Result<String, InterpreterError>> {
  // Pattern: FunctionName[arg1, arg2, ...]
  // Must start with a letter and have exactly one balanced [...] pair at the end

  let open_bracket = input.find('[')?;
  if !input.ends_with(']') {
    return None;
  }

  let func_name = &input[..open_bracket];
  // Validate function name is a simple identifier
  if func_name.is_empty()
    || !func_name
      .chars()
      .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
    || !func_name.chars().next().unwrap().is_ascii_alphabetic()
  {
    return None;
  }

  let args_str = &input[open_bracket + 1..input.len() - 1];

  // Check for nested brackets (indicates complex expression)
  let mut depth = 0;
  for c in args_str.chars() {
    match c {
      '[' => depth += 1,
      ']' => depth -= 1,
      _ => {}
    }
    if depth < 0 {
      return None; // Unbalanced
    }
  }
  // After processing, we should be at depth 0 for well-formed args

  // Split arguments by comma (respecting nested structures)
  let args = split_args(args_str);

  // Handle specific functions that are commonly called
  match func_name {
    "MemberQ" => {
      if args.len() != 2 {
        return None;
      }
      // First arg should be a list, second is the element to find
      let list_str = args[0].trim();
      let elem_expr = args[1].trim();

      // Evaluate the element expression
      let target = match interpret(elem_expr) {
        Ok(v) => v,
        Err(_) => return None,
      };

      // Parse the list
      if !list_str.starts_with('{') || !list_str.ends_with('}') {
        return None; // Not a literal list, need full parsing
      }

      let inner = &list_str[1..list_str.len() - 1];
      let list_elems = split_args(inner);

      // Check if target is in the list
      for elem in list_elems {
        let elem = elem.trim();
        // Try to evaluate the list element if needed
        let elem_val = if elem.contains('[') {
          match interpret(elem) {
            Ok(v) => v,
            Err(_) => elem.to_string(),
          }
        } else {
          elem.to_string()
        };
        if elem_val == target {
          return Some(Ok("True".to_string()));
        }
      }
      Some(Ok("False".to_string()))
    }
    "First" => {
      if args.len() != 1 {
        return None;
      }
      let arg = args[0].trim();
      // If the argument is a variable, look it up
      let list_str = if arg
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
        && !arg.is_empty()
        && arg.chars().next().unwrap().is_ascii_alphabetic()
      {
        // It's an identifier, look it up
        match ENV.with(|e| e.borrow().get(arg).cloned()) {
          Some(StoredValue::Raw(val)) => val,
          Some(StoredValue::ExprVal(e)) => syntax::expr_to_string(&e),
          _ => return None,
        }
      } else if arg.starts_with('{') && arg.ends_with('}') {
        arg.to_string()
      } else {
        return None;
      };

      // Parse the list
      if !list_str.starts_with('{') || !list_str.ends_with('}') {
        return None;
      }
      let inner = &list_str[1..list_str.len() - 1];
      let elems = split_args(inner);
      if elems.is_empty() {
        eprintln!();
        eprintln!("{} has zero length and no first element.", list_str);
        return Some(Ok("First[{}]".to_string()));
      }
      Some(Ok(elems[0].trim().to_string()))
    }
    "Rest" => {
      if args.len() != 1 {
        return None;
      }
      let arg = args[0].trim();
      // If the argument is a variable, look it up
      let list_str = if arg
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
        && !arg.is_empty()
        && arg.chars().next().unwrap().is_ascii_alphabetic()
      {
        match ENV.with(|e| e.borrow().get(arg).cloned()) {
          Some(StoredValue::Raw(val)) => val,
          Some(StoredValue::ExprVal(e)) => syntax::expr_to_string(&e),
          _ => return None,
        }
      } else if arg.starts_with('{') && arg.ends_with('}') {
        arg.to_string()
      } else {
        return None;
      };

      // Parse the list
      if !list_str.starts_with('{') || !list_str.ends_with('}') {
        return None;
      }
      let inner = &list_str[1..list_str.len() - 1];
      let elems = split_args(inner);
      if elems.is_empty() {
        eprintln!();
        eprintln!("Cannot take Rest of expression {{}} with length zero.");
        return Some(Ok("Rest[{}]".to_string()));
      }
      let rest: Vec<_> = elems.iter().skip(1).map(|s| s.trim()).collect();
      Some(Ok(format!("{{{}}}", rest.join(", "))))
    }
    _ => None,
  }
}

/// Split a comma-separated argument list, respecting nested structures
fn split_args(s: &str) -> Vec<String> {
  let mut args = Vec::new();
  let mut current = String::new();
  let mut depth = 0;

  for c in s.chars() {
    match c {
      '{' | '[' | '(' => {
        depth += 1;
        current.push(c);
      }
      '}' | ']' | ')' => {
        depth -= 1;
        current.push(c);
      }
      ',' if depth == 0 => {
        args.push(current.trim().to_string());
        current.clear();
      }
      _ => {
        current.push(c);
      }
    }
  }
  if !current.is_empty() {
    args.push(current.trim().to_string());
  }
  args
}

/// New interpret function that returns both stdout and the result
pub fn interpret_with_stdout(
  input: &str,
) -> Result<InterpretResult, InterpreterError> {
  // Clear the capture buffers
  clear_captured_stdout();
  clear_captured_graphics();
  clear_captured_graphicsbox();
  clear_captured_warnings();
  clear_captured_output_svg();

  // Enable visual mode for display wrapper rendering (e.g. TableForm → Grid SVG)
  VISUAL_MODE.with(|v| *v.borrow_mut() = true);

  // Perform the standard interpretation
  let result = interpret(input);

  // Reset visual mode
  VISUAL_MODE.with(|v| *v.borrow_mut() = false);

  let result = result?;

  // Get the captured output
  let stdout = get_captured_stdout();
  let graphics = get_captured_graphics();
  let output_svg = get_captured_output_svg();
  let warnings = get_captured_warnings();

  // Return stdout, result, and any graphical output
  Ok(InterpretResult {
    stdout,
    result,
    graphics,
    output_svg,
    warnings,
  })
}

fn format_result(result: f64) -> String {
  if result.fract() == 0.0 && result.abs() < 1e15 {
    // Integer result - format without trailing dot
    format!("{}", result as i64)
  } else if result.abs() >= 1e6 || (result != 0.0 && result.abs() < 1e-5) {
    syntax::format_real(result)
  } else {
    format!("{}", result)
  }
}

/// Format a result as a real number (with trailing dot for whole numbers)
pub fn format_real_result(result: f64) -> String {
  syntax::format_real(result)
}

/// GCD helper function for fraction simplification
fn gcd_i64(a: i64, b: i64) -> i64 {
  let mut a = a.abs();
  let mut b = b.abs();
  while b != 0 {
    let temp = b;
    b = a % b;
    a = temp;
  }
  a
}

/// Format a rational number as a fraction (numerator/denominator)
pub fn format_fraction(numerator: i64, denominator: i64) -> String {
  if denominator == 0 {
    return "ComplexInfinity".to_string();
  }
  let g = gcd_i64(numerator, denominator);
  let num = numerator / g;
  let den = denominator / g;

  // Handle sign
  let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };

  if den == 1 {
    num.to_string()
  } else {
    format!("{}/{}", num, den)
  }
}

// Parse display-form strings like "{1, 2, 3}" into top-level comma-separated
// element strings.  Returns None if `s` is not a braced list.
pub fn parse_list_string(s: &str) -> Option<Vec<String>> {
  if !(s.starts_with('{') && s.ends_with('}')) {
    return None;
  }
  let inner = &s[1..s.len() - 1];
  let mut parts = Vec::new();
  let mut depth = 0usize;
  let mut start = 0usize;
  for (i, c) in inner.char_indices() {
    match c {
      '{' | '[' | '(' | '<' => depth += 1,
      '}' | ']' | ')' | '>' => {
        depth = depth.saturating_sub(1);
      }
      ',' if depth == 0 => {
        parts.push(inner[start..i].trim().to_string());
        start = i + 1;
      }
      _ => {}
    }
  }
  if start < inner.len() {
    parts.push(inner[start..].trim().to_string());
  }
  Some(parts)
}

fn store_function_definition(pair: Pair<Rule>) -> Result<(), InterpreterError> {
  // FunctionDefinition  :=  Identifier "[" (Pattern ("," Pattern)*)? "]" ":=" Expression
  let mut inner = pair.into_inner();
  let func_name = inner.next().unwrap().as_str().to_owned(); // Identifier

  // Collect all pattern parameters with their optional conditions, defaults, and head constraints
  let mut params = Vec::new();
  let mut conditions: Vec<Option<syntax::Expr>> = Vec::new();
  let mut defaults: Vec<Option<syntax::Expr>> = Vec::new();
  let mut heads: Vec<Option<String>> = Vec::new();
  let mut body_pair = None;
  let mut has_any_condition = false;

  for item in inner {
    match item.as_rule() {
      Rule::PatternCondition => {
        // PatternCondition = { PatternName ~ "_" ~ "/;" ~ ConditionExpr }
        let mut pat_inner = item.into_inner();
        let param_name = pat_inner.next().unwrap().as_str().to_owned();
        // The second child is the ConditionExpr
        let cond_pair = pat_inner.next().unwrap();
        let cond_expr = syntax::pair_to_expr(cond_pair);
        params.push(param_name);
        conditions.push(Some(cond_expr));
        defaults.push(None);
        heads.push(None);
        has_any_condition = true;
      }
      Rule::PatternOptionalSimple => {
        // PatternOptionalSimple = { PatternName ~ "_" ~ ":" ~ Term }
        let mut pat_inner = item.into_inner();
        let param_name = pat_inner.next().unwrap().as_str().to_owned();
        let default_pair = pat_inner.next().unwrap();
        let default_expr = syntax::pair_to_expr(default_pair);
        params.push(param_name);
        conditions.push(None);
        defaults.push(Some(default_expr));
        heads.push(None);
      }
      Rule::PatternOptionalWithHead => {
        // PatternOptionalWithHead = { PatternName ~ "_" ~ Identifier ~ ":" ~ Term }
        let mut pat_inner = item.into_inner();
        let param_name = pat_inner.next().unwrap().as_str().to_owned();
        let head_name = pat_inner.next().unwrap().as_str().to_owned();
        let default_pair = pat_inner.next().unwrap();
        let default_expr = syntax::pair_to_expr(default_pair);
        params.push(param_name);
        conditions.push(None);
        defaults.push(Some(default_expr));
        heads.push(Some(head_name));
      }
      Rule::PatternWithHead => {
        // Extract parameter name and head (e.g., "x_List" -> name="x", head="List")
        let mut pat_inner = item.into_inner();
        let param_name = pat_inner.next().unwrap().as_str().to_owned();
        let head_name = pat_inner.next().unwrap().as_str().to_owned();
        params.push(param_name);
        conditions.push(None);
        defaults.push(None);
        heads.push(Some(head_name));
      }
      Rule::PatternTest => {
        // PatternTest: x_?test or _?test — extract param name and test condition
        let mut pat_inner = item.into_inner();
        let first = pat_inner.next().unwrap();
        let (param_name, test_pair) = if first.as_rule() == Rule::PatternName {
          (first.as_str().to_owned(), pat_inner.next().unwrap())
        } else {
          // Anonymous blank _?test — generate a placeholder param name
          (format!("__pt{}", params.len()), first)
        };
        let test_expr = syntax::pair_to_expr(test_pair);
        // Build condition: testFunc[paramName]
        let cond_expr = syntax::Expr::FunctionCall {
          name: syntax::expr_to_string(&test_expr),
          args: vec![syntax::Expr::Identifier(param_name.clone())],
        };
        params.push(param_name);
        conditions.push(Some(cond_expr));
        defaults.push(None);
        heads.push(None);
        has_any_condition = true;
      }
      Rule::PatternSimple => {
        // Extract parameter name from pattern (e.g., "x_" -> "x")
        let param = item.as_str().trim_end_matches('_').to_owned();
        let param = param.split('_').next().unwrap_or(&param).to_owned();
        params.push(param);
        conditions.push(None);
        defaults.push(None);
        heads.push(None);
      }
      Rule::Expression
      | Rule::ExpressionNoImplicit
      | Rule::CompoundExpression => {
        body_pair = Some(item);
      }
      _ => {}
    }
  }

  // Convert body to AST instead of storing as string
  let body_expr = syntax::pair_to_expr(body_pair.ok_or_else(|| {
    InterpreterError::EvaluationError("Missing function body".into())
  })?);

  FUNC_DEFS.with(|m| {
    let mut defs = m.borrow_mut();
    let entry = defs.entry(func_name).or_insert_with(Vec::new);
    let arity = params.len();
    if has_any_condition {
      // Conditional definition: only remove existing definitions with same arity
      // that have the exact same conditions (re-definition of same pattern)
      // For simplicity, just append - Wolfram keeps all conditional defs
    } else {
      // Unconditional definition: remove only other unconditional defs with same arity
      // (keep conditional definitions - they are more specific)
      entry.retain(|(p, conds, _, _, _)| {
        p.len() != arity || conds.iter().any(|c| c.is_some())
      });
    }
    // Add the new definition with parsed AST, conditions, defaults, and head constraints
    entry.push((params, conditions, defaults, heads, body_expr));
  });
  Ok(())
}

fn nth_prime(n: usize) -> usize {
  if n == 0 {
    return 0; // Return 0 for invalid input
  }
  let mut count = 0;
  let mut num = 1;
  while count < n {
    num += 1;
    if is_prime(num) {
      count += 1;
    }
  }
  num
}

pub fn is_prime(n: usize) -> bool {
  if n <= 1 {
    return false;
  }
  if n == 2 {
    return true;
  }
  if n.is_multiple_of(2) {
    return false;
  }
  let sqrt_n = (n as f64).sqrt() as usize;
  for i in (3..=sqrt_n).step_by(2) {
    if n.is_multiple_of(i) {
      return false;
    }
  }
  true
}
