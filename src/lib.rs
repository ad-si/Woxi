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
    //            name         (parameter names)      body-text
    static FUNC_DEFS: RefCell<HashMap<String, (Vec<String>, String)>> = RefCell::new(HashMap::new());
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

// Re-export evaluate_expression and evaluate_term from evaluator module
pub use evaluator::{evaluate_expression, evaluate_term};

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

pub fn interpret(input: &str) -> Result<String, InterpreterError> {
  // Clear the stdout capture buffer
  clear_captured_stdout();

  // Regular interpretation
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
        last_result = Some(evaluate_expression(node)?);
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
fn parse_list_string(s: &str) -> Option<Vec<String>> {
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
  // FunctionDefinition  :=  Identifier "[" Pattern "]" ":=" Expression
  let mut inner = pair.into_inner();
  let func_name = inner.next().unwrap().as_str().to_owned(); // Identifier
  let pattern = inner.next().unwrap(); // Pattern
  let param = pattern.as_str().trim_end_matches('_').to_owned();
  let body_pair = inner.next().unwrap(); // Expression
  let body_txt = body_pair.as_str().to_owned();

  FUNC_DEFS.with(|m| {
    m.borrow_mut().insert(func_name, (vec![param], body_txt));
  });
  Ok(())
}

fn eval_association(
  pair: Pair<Rule>,
) -> Result<(Vec<(String, String)>, String), InterpreterError> {
  // Navigate to the actual Association node if needed
  let assoc_pair = match pair.as_rule() {
    Rule::Association => pair,
    Rule::Expression | Rule::Term => {
      // Find the Association inside the Expression/Term
      let mut found = None;
      for inner in pair.clone().into_inner() {
        if inner.as_rule() == Rule::Association {
          found = Some(inner);
          break;
        } else if inner.as_rule() == Rule::Term
          || inner.as_rule() == Rule::Expression
        {
          // Recurse into nested Term/Expression
          for deeper in inner.clone().into_inner() {
            if deeper.as_rule() == Rule::Association {
              found = Some(deeper);
              break;
            }
          }
        }
      }
      found.ok_or_else(|| {
        InterpreterError::EvaluationError("Expected association".into())
      })?
    }
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "Expected Association, got {:?}",
        pair.as_rule()
      )));
    }
  };

  let mut pairs = Vec::new();
  let mut disp_parts = Vec::new();
  for item in assoc_pair
    .into_inner()
    .filter(|p| p.as_rule() == Rule::AssociationItem)
  {
    let mut inner = item.into_inner();
    let key_pair = inner.next().unwrap();
    let val_pair = inner.next().unwrap();
    let key = extract_string(key_pair)?;
    let raw = val_pair.as_str().trim();
    let val = if raw.starts_with('"') && raw.ends_with('"') {
      raw.trim_matches('"').to_string()
    } else if raw.starts_with("<|") && raw.ends_with("|>") {
      let (_nested, nested_disp) = eval_association(val_pair.clone())?;
      nested_disp
    } else {
      raw.to_string()
    };
    disp_parts.push(format!("{} -> {}", key, val));
    pairs.push((key, val));
  }
  let disp = format!("<|{}|>", disp_parts.join(", "));
  Ok((pairs, disp))
}

fn extract_string(pair: Pair<Rule>) -> Result<String, InterpreterError> {
  match pair.as_rule() {
    Rule::String => Ok(pair.as_str().trim_matches('"').to_string()),
    Rule::Expression | Rule::Term => {
      let mut inner = pair.clone().into_inner();
      if let Some(first) = inner.next() {
        // If the inner part is a string, extract it.
        // Otherwise, evaluate the expression and hope it's a string.
        if first.as_rule() == Rule::String {
          return Ok(first.as_str().trim_matches('"').to_string());
        }
        // Fallback to evaluate_expression if not directly a string.
        // This handles cases like `DateString[Now, "ISO" <> "DateTime"]` if StringJoin was more general
        // or if the argument is a variable that holds a string.
        let result = evaluate_expression(pair)?;
        // Strip quotes if the result is a quoted string
        return Ok(result.trim_matches('"').to_string());
      }
      Err(InterpreterError::EvaluationError(
        "Expected string argument".into(),
      ))
    }
    _ => {
      // fall-back, keeps behaviour for exotic cases
      let result = evaluate_expression(pair)?;
      Ok(result.trim_matches('"').to_string())
    }
  }
}

fn apply_map_operator(
  func: pest::iterators::Pair<Rule>,
  list: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  // right after the function's opening brace
  let func_core = if func.as_rule() == Rule::Term {
    func.clone().into_inner().next().unwrap()
  } else {
    func.clone()
  };

  // ----- obtain list items (same extraction logic used in Map) -----
  // First, try to get the list directly from the parse tree
  let list_rule = list.as_rule();
  let direct_elements: Option<Vec<_>> = if list_rule == Rule::List {
    Some(
      list
        .clone()
        .into_inner()
        .filter(|p| p.as_str() != ",")
        .collect(),
    )
  } else if list_rule == Rule::Expression {
    let mut inner = list.clone().into_inner();
    if let Some(first) = inner.next() {
      if first.as_rule() == Rule::List {
        Some(first.into_inner().filter(|p| p.as_str() != ",").collect())
      } else {
        None
      }
    } else {
      None
    }
  } else {
    None
  };

  // If we couldn't get elements directly, evaluate the expression and parse the result
  let elements_strings: Vec<String> = if let Some(elems) = direct_elements {
    elems.iter().map(|p| p.as_str().to_string()).collect()
  } else {
    // Evaluate the right operand (e.g., Range[99,1,-1]) to get a list string
    let list_str = evaluate_expression(list)?;
    if list_str.starts_with('{') && list_str.ends_with('}') {
      parse_list_string(&list_str).ok_or_else(|| {
        InterpreterError::EvaluationError("Failed to parse list".into())
      })?
    } else {
      return Err(InterpreterError::EvaluationError(
        "Second operand of /@ must be a list".into(),
      ));
    }
  };

  // ----- identify mapped function ----------------------------------
  let func_src = func_core.as_str();
  let mut out = Vec::new();

  match func_core.as_rule() {
    Rule::Identifier => {
      // Named function: apply f[elem] for each element
      let name = func_src;
      for el in &elements_strings {
        let expr = format!("{}[{}]", name, el);
        out.push(interpret(&expr)?);
      }
      Ok(format!("{{{}}}", out.join(", ")))
    }
    Rule::AnonymousFunction => {
      // Anonymous function like #^2& or #+1&
      // Replace # with each element and evaluate
      let body = func_src.trim_end_matches('&');
      for el in &elements_strings {
        let expr = body.replace('#', el);
        out.push(interpret(&expr)?);
      }
      Ok(format!("{{{}}}", out.join(", ")))
    }
    Rule::FunctionCall => {
      // Function call being used as a mapping function
      // Apply f[elem] for each element
      for el in &elements_strings {
        let expr = format!("{}[{}]", func_src, el);
        out.push(interpret(&expr)?);
      }
      Ok(format!("{{{}}}", out.join(", ")))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Left operand of /@ must be a function".into(),
    )),
  }
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
