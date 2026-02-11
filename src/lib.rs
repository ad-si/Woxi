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
  ReturnValue(syntax::Expr),
  #[error("Break")]
  BreakSignal,
  #[error("Continue")]
  ContinueSignal,
  #[error("Throw")]
  ThrowValue(syntax::Expr, Option<syntax::Expr>),
}

/// Extended result type that includes both stdout and the result
#[derive(Debug, Clone)]
pub struct InterpretResult {
  pub stdout: String,
  pub result: String,
  pub graphics: Option<String>,
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

// Captured graphical output (SVG) from Plot and related functions
thread_local! {
    static CAPTURED_GRAPHICS: RefCell<Option<String>> = const { RefCell::new(None) };
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

/// Clears the captured graphics buffer
fn clear_captured_graphics() {
  CAPTURED_GRAPHICS.with(|buffer| {
    *buffer.borrow_mut() = None;
  });
}

/// Stores SVG graphics for capture by the Jupyter kernel
pub fn capture_graphics(svg: &str) {
  CAPTURED_GRAPHICS.with(|buffer| {
    *buffer.borrow_mut() = Some(svg.to_string());
  });
}

/// Gets the captured graphics content
pub fn get_captured_graphics() -> Option<String> {
  CAPTURED_GRAPHICS.with(|buffer| buffer.borrow().clone())
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
  clear_captured_stdout();
  clear_captured_graphics();
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
              args: vec![val],
            }
          }
          other => other?,
        };
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
  // Clear the capture buffers
  clear_captured_stdout();
  clear_captured_graphics();

  // Perform the standard interpretation
  let result = interpret(input)?;

  // Get the captured output
  let stdout = get_captured_stdout();
  let graphics = get_captured_graphics();

  // Return stdout, result, and any graphical output
  Ok(InterpretResult {
    stdout,
    result,
    graphics,
  })
}

fn format_result(result: f64) -> String {
  if result.fract() == 0.0 {
    let int_result = result as i64;
    int_result.to_string()
  } else {
    format!("{}", result)
  }
}

/// Format a result as a real number (with trailing dot for whole numbers)
pub fn format_real_result(result: f64) -> String {
  if result.fract() == 0.0 {
    format!("{}.", result as i64)
  } else {
    format!("{}", result)
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
      Rule::PatternSimple | Rule::PatternTest => {
        // Extract parameter name from pattern (e.g., "x_" -> "x")
        let param = item.as_str().trim_end_matches('_').to_owned();
        // Handle patterns by taking just the name part before the _
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
