use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::cell::RefCell;
use std::collections::HashMap;
use thiserror::Error;

pub mod evaluator;
pub mod functions;
pub mod syntax;
pub mod utils;

#[derive(Parser)]
#[grammar = "wolfram.pest"]
pub struct WolframParser;

#[derive(Clone)]
enum StoredValue {
  Association(Vec<(String, String)>),
  Raw(String), // keep evaluated textual value
}
thread_local! {
    static ENV: RefCell<HashMap<String, StoredValue>> = RefCell::new(HashMap::new());
    //            name         Vec of (parameter names, body-AST) for multi-arity support
    static FUNC_DEFS: RefCell<HashMap<String, Vec<(Vec<String>, syntax::Expr)>>> = RefCell::new(HashMap::new());
}

#[derive(Error, Debug)]
pub enum InterpreterError {
  #[error("Parse error: {0}")]
  ParseError(#[from] Box<pest::error::Error<Rule>>),
  #[error("Empty input")]
  EmptyInput,
  #[error("Evaluation error: {0}")]
  EvaluationError(String),
}

/// Extended result type that includes both stdout and the result
#[derive(Debug, Clone)]
pub struct InterpretResult {
  pub stdout: String,
  pub result: String,
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

// Captured output from Print statements
thread_local! {
    static CAPTURED_STDOUT: RefCell<String> = const { RefCell::new(String::new()) };
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

// Re-export evaluate_expr from evaluator module
pub use evaluator::evaluate_expr;

/// Set a system variable (like $ScriptCommandLine) in the environment
pub fn set_system_variable(name: &str, value: &str) {
  ENV.with(|e| {
    e.borrow_mut()
      .insert(name.to_string(), StoredValue::Raw(value.to_string()));
  });
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
  // Check for float
  if let Ok(n) = trimmed.parse::<f64>() {
    return Ok(format_result(n));
  }
  // Check for quoted string - return as-is
  if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
    // Make sure there are no unescaped quotes inside
    let inner = &trimmed[1..trimmed.len() - 1];
    if !inner.contains('"') {
      return Ok(trimmed.to_string());
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
      && !trimmed.contains(" + ")
      && !trimmed.contains(" - ")
      && !trimmed.contains(" * ")
      && !trimmed.contains(" / ")
      && !trimmed.contains('[')
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
    if let Some(StoredValue::Raw(val)) =
      ENV.with(|e| e.borrow().get(trimmed).cloned())
    {
      return Ok(val);
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

  // Only clear stdout at top level
  if depth == 0 {
    clear_captured_stdout();
  }

  // Decrement depth on scope exit
  struct DepthGuard;
  impl Drop for DepthGuard {
    fn drop(&mut self) {
      INTERPRET_DEPTH.with(|d| *d.borrow_mut() -= 1);
    }
  }
  let _guard = DepthGuard;

  // Regular interpretation - use AST-based evaluation
  let pairs = parse(input)?;
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
  for node in program.into_inner() {
    match node.as_rule() {
      Rule::Expression => {
        // Convert Pair to Expr AST
        let expr = syntax::pair_to_expr(node);
        // Evaluate using AST-based evaluation
        let result_expr = evaluator::evaluate_expr_to_expr(&expr)?;
        // Convert to output string (strips quotes from strings for display)
        last_result = Some(syntax::expr_to_output(&result_expr));
        any_nonempty = true;
      }
      Rule::FunctionDefinition => {
        store_function_definition(node)?;
        any_nonempty = true;
      }
      _ => {} // ignore semicolons, etc.
    }
  }

  if any_nonempty {
    last_result.ok_or(InterpreterError::EmptyInput)
  } else {
    Err(InterpreterError::EmptyInput)
  }
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
        return Some(Err(InterpreterError::EvaluationError(
          "First called on empty list".into(),
        )));
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
        return Some(Err(InterpreterError::EvaluationError(
          "Rest called on empty list".into(),
        )));
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
  // Clear the stdout capture buffer
  clear_captured_stdout();

  // Perform the standard interpretation
  let result = interpret(input)?;

  // Get the captured stdout
  let stdout = get_captured_stdout();

  // Return both stdout and the result
  Ok(InterpretResult { stdout, result })
}

fn format_result(result: f64) -> String {
  if result.fract() == 0.0 {
    let int_result = result as i64;
    int_result.to_string()
  } else {
    format!("{:.10}", result)
      .trim_end_matches('0')
      .trim_end_matches('.')
      .to_string()
  }
}

/// Format a result as a real number (with trailing dot for whole numbers)
pub fn format_real_result(result: f64) -> String {
  if result.fract() == 0.0 {
    format!("{}.", result as i64)
  } else {
    format!("{:.10}", result).trim_end_matches('0').to_string()
  }
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

  // Collect all pattern parameters
  let mut params = Vec::new();
  let mut body_pair = None;

  for item in inner {
    match item.as_rule() {
      Rule::PatternSimple
      | Rule::PatternTest
      | Rule::PatternCondition
      | Rule::PatternWithHead => {
        // Extract parameter name from pattern (e.g., "x_" -> "x", "x_List" -> "x")
        let param = item.as_str().trim_end_matches('_').to_owned();
        // Handle patterns by taking just the name part before the _
        let param = param.split('_').next().unwrap_or(&param).to_owned();
        params.push(param);
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
    // Remove any existing definition with the same arity
    let arity = params.len();
    entry.retain(|(p, _)| p.len() != arity);
    // Add the new definition with parsed AST
    entry.push((params, body_expr));
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

fn is_prime(n: usize) -> bool {
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
