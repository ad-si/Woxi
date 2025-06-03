use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;
use std::cell::RefCell;
use std::collections::HashMap;
use thiserror::Error;

pub mod evaluator;
pub mod functions;
pub mod syntax;

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
  ParseError(#[from] pest::error::Error<Rule>),
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
  ) -> Result<pest::iterators::Pairs<Rule>, pest::error::Error<Rule>> {
    Self::parse(Rule::Program, input)
  }
}

pub fn parse(
  input: &str,
) -> Result<pest::iterators::Pairs<Rule>, pest::error::Error<Rule>> {
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
  let mut pairs = Vec::new();
  let mut disp_parts = Vec::new();
  for item in pair
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
        return evaluate_expression(pair);
      }
      Err(InterpreterError::EvaluationError(
        "Expected string argument".into(),
      ))
    }
    _ => evaluate_expression(pair), // fall-back, keeps behaviour for exotic cases
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
  let list_rule = list.as_rule();
  let elements: Vec<_> = if list_rule == Rule::List {
    list.into_inner().filter(|p| p.as_str() != ",").collect()
  } else if list_rule == Rule::Expression {
    let mut inner = list.into_inner();
    if let Some(first) = inner.next() {
      if first.as_rule() == Rule::List {
        first.into_inner().filter(|p| p.as_str() != ",").collect()
      } else {
        return Err(InterpreterError::EvaluationError(
          "Second operand of /@ must be a list".into(),
        ));
      }
    } else {
      return Err(InterpreterError::EvaluationError(
        "Second operand of /@ must be a list".into(),
      ));
    }
  } else {
    return Err(InterpreterError::EvaluationError(
      "Second operand of /@ must be a list".into(),
    ));
  };

  // ----- identify mapped function ----------------------------------
  match func_core.as_rule() {
    Rule::Identifier => {
      let name = func_core.as_str();
      match name {
        "Sign" => {
          let mut mapped = Vec::new();
          for el in elements {
            let v = evaluate_term(el.clone())?;
            let s = if v > 0.0 {
              1.0
            } else if v < 0.0 {
              -1.0
            } else {
              0.0
            };
            mapped.push(format_result(s));
          }
          Ok(format!("{{{}}}", mapped.join(", ")))
        }
        _ => Err(InterpreterError::EvaluationError(format!(
          "Unknown mapping function: {}",
          name
        ))),
      }
    }
    Rule::AnonymousFunction => {
      let parts: Vec<_> = func_core.clone().into_inner().collect();

      // identity function  (#&)
      if parts.len() == 1 {
        let mut out = Vec::new();
        for el in &elements {
          out.push(evaluate_expression(el.clone())?);
        }
        return Ok(format!("{{{}}}", out.join(", ")));
      }

      let operator = parts[1].as_str();
      let operand = parts[2].clone();
      let mut out = Vec::new();

      for el in elements {
        let v = evaluate_term(el.clone())?;
        let res = match operator {
          "+" => v + evaluate_term(operand.clone())?,
          "-" => v - evaluate_term(operand.clone())?,
          "*" => v * evaluate_term(operand.clone())?,
          "/" => {
            let d = evaluate_term(operand.clone())?;
            if d == 0.0 {
              return Err(InterpreterError::EvaluationError(
                "Division by zero".into(),
              ));
            }
            v / d
          }
          "^" => v.powf(evaluate_term(operand.clone())?),
          _ => {
            return Err(InterpreterError::EvaluationError(format!(
              "Unsupported operator in anonymous function: {}",
              operator
            )))
          }
        };
        out.push(format_result(res));
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
  if n % 2 == 0 {
    return false;
  }
  let sqrt_n = (n as f64).sqrt() as usize;
  for i in (3..=sqrt_n).step_by(2) {
    if n % i == 0 {
      return false;
    }
  }
  true
}
