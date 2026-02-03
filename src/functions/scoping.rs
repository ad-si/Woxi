use pest::iterators::Pair;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{ENV, InterpreterError, Rule, StoredValue, interpret};

/// Global counter for generating unique symbol names
static SYMBOL_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique symbol name like "x$1", "x$2", etc.
fn unique_symbol(base_name: &str) -> String {
  let n = SYMBOL_COUNTER.fetch_add(1, Ordering::SeqCst);
  format!("{}${}", base_name, n)
}

/// Check if an expression string contains unbound symbolic variables
/// Returns true if there are identifiers that are not bound in the environment
fn has_unbound_symbols(expr: &str) -> bool {
  // Parse the expression and check for unbound identifiers
  // Simple approach: look for identifiers and check if they're bound

  // List of known function names that should not be treated as variables
  let known_functions = [
    "Sin",
    "Cos",
    "Tan",
    "Sec",
    "Log",
    "Exp",
    "Sqrt",
    "D",
    "Integrate",
    "Module",
    "If",
    "And",
    "Or",
    "Not",
    "List",
    "Table",
    "Range",
    "Map",
    "Apply",
    "Fold",
    "Nest",
    "Power",
    "Plus",
    "Times",
    "Pi",
  ];

  // Extract potential identifiers from the expression
  // This is a simplified approach - look for sequences of letters
  let mut chars = expr.chars().peekable();
  while let Some(ch) = chars.next() {
    if ch.is_ascii_alphabetic() {
      // Found start of identifier
      let mut ident = String::new();
      ident.push(ch);
      while let Some(&next) = chars.peek() {
        if next.is_ascii_alphanumeric() || next == '_' {
          ident.push(chars.next().unwrap());
        } else {
          break;
        }
      }

      // Skip known function names
      if known_functions.contains(&ident.as_str()) {
        continue;
      }

      // Check if this identifier is bound in the environment
      let is_bound = ENV.with(|e| e.borrow().contains_key(&ident));
      if !is_bound {
        return true;
      }
    }
  }
  false
}

/// Check if a string represents a purely numeric value
fn is_purely_numeric(s: &str) -> bool {
  s.trim().parse::<f64>().is_ok()
}

/// Handle Module[{vars}, body] - local variable scoping
///
/// Module creates a local scope with variable initializations.
/// Syntax: Module[{x = 1, y = 2, ...}, body]
///
/// Variables can be initialized with expressions, and the body is
/// evaluated with those local bindings. After evaluation, the
/// previous environment is restored.
///
/// Uninitialized variables get unique symbol names like x$1, y$2, etc.
pub fn module(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "Module expects 2 arguments; {} given",
      args_pairs.len()
    )));
  }

  // First argument should be a list of variable declarations: {x = 1, y = 2, ...}
  let vars_pair = &args_pairs[0];
  let body_pair = &args_pairs[1];

  // Parse the variable declarations from the first argument
  let local_vars = parse_module_vars(vars_pair)?;

  // Save previous bindings and set up new ones
  // We also track the mapping from original var name to the symbol we'll use
  let mut prev: Vec<(String, Option<StoredValue>)> = Vec::new();
  let mut var_mappings: Vec<(String, String)> = Vec::new();

  for (var_name, init_expr) in &local_vars {
    // Evaluate the initialization expression (if any)
    let (symbol_name, val) = if let Some(expr) = init_expr {
      // Initialized variable - use the original name
      let evaluated = interpret(expr)?;

      // Check if the expression contains unbound symbols but evaluated to a numeric result
      // This happens when e.g. x^2 evaluates to 0 because x is unbound
      // In that case, preserve the original expression for symbolic operations like D[]
      let val = if has_unbound_symbols(expr) && is_purely_numeric(&evaluated) {
        // Store the original expression text for symbolic computation
        expr.clone()
      } else {
        evaluated
      };

      (var_name.clone(), val)
    } else {
      // Uninitialized variable - generate a unique symbol
      let unique = unique_symbol(var_name);
      (var_name.clone(), unique.clone())
    };

    var_mappings.push((var_name.clone(), symbol_name.clone()));

    // Save current binding and set new one
    let pv = ENV.with(|e| {
      e.borrow_mut()
        .insert(var_name.clone(), StoredValue::Raw(val))
    });
    prev.push((var_name.clone(), pv));
  }

  // Evaluate the body expression
  let body_str = body_pair.as_str();
  let result = interpret(body_str)?;

  // Restore previous bindings
  for (var_name, old) in prev {
    ENV.with(|e| {
      let mut env = e.borrow_mut();
      if let Some(v) = old {
        env.insert(var_name, v);
      } else {
        env.remove(&var_name);
      }
    });
  }

  Ok(result)
}

/// Parse variable declarations from a Module's first argument
/// Returns a list of (variable_name, optional_init_expression)
fn parse_module_vars(
  pair: &Pair<Rule>,
) -> Result<Vec<(String, Option<String>)>, InterpreterError> {
  let mut vars = Vec::new();

  // The pair could be a List {x = 1, y} or an Expression wrapping a List
  let text = pair.as_str().trim();

  // Check if it looks like a list: {contents}
  if !text.starts_with('{') || !text.ends_with('}') {
    return Err(InterpreterError::EvaluationError(
      "Module expects a list of variable declarations as first argument".into(),
    ));
  }

  // Extract the contents between { and }
  let contents = &text[1..text.len() - 1];

  // Split by commas (but be careful about nested expressions)
  let parts = split_by_comma(contents);

  for part in parts {
    let part = part.trim();
    if part.is_empty() {
      continue;
    }

    // Check if it's an assignment: x = expr
    if let Some(eq_pos) = find_assignment_eq(part) {
      let var_name = part[..eq_pos].trim().to_string();
      let init_expr = part[eq_pos + 1..].trim().to_string();
      vars.push((var_name, Some(init_expr)));
    } else {
      // Just a variable name (uninitialized)
      vars.push((part.to_string(), None));
    }
  }

  Ok(vars)
}

/// Find the position of the assignment '=' in a string,
/// being careful to not confuse it with '==' or ':='
fn find_assignment_eq(s: &str) -> Option<usize> {
  let bytes = s.as_bytes();
  let mut i = 0;
  let mut bracket_depth = 0;
  let mut paren_depth = 0;

  while i < bytes.len() {
    match bytes[i] {
      b'[' => bracket_depth += 1,
      b']' => bracket_depth -= 1,
      b'(' => paren_depth += 1,
      b')' => paren_depth -= 1,
      b'=' if bracket_depth == 0 && paren_depth == 0 => {
        // Check it's not == or :=
        let prev_is_colon = i > 0 && bytes[i - 1] == b':';
        let next_is_eq = i + 1 < bytes.len() && bytes[i + 1] == b'=';
        if !prev_is_colon && !next_is_eq {
          return Some(i);
        }
      }
      _ => {}
    }
    i += 1;
  }
  None
}

/// Split a string by commas, respecting nested brackets and parentheses
fn split_by_comma(s: &str) -> Vec<String> {
  let mut parts = Vec::new();
  let mut current = String::new();
  let mut bracket_depth = 0;
  let mut paren_depth = 0;
  let mut brace_depth = 0;

  for ch in s.chars() {
    match ch {
      '[' => {
        bracket_depth += 1;
        current.push(ch);
      }
      ']' => {
        bracket_depth -= 1;
        current.push(ch);
      }
      '(' => {
        paren_depth += 1;
        current.push(ch);
      }
      ')' => {
        paren_depth -= 1;
        current.push(ch);
      }
      '{' => {
        brace_depth += 1;
        current.push(ch);
      }
      '}' => {
        brace_depth -= 1;
        current.push(ch);
      }
      ',' if bracket_depth == 0 && paren_depth == 0 && brace_depth == 0 => {
        parts.push(current.trim().to_string());
        current = String::new();
      }
      _ => current.push(ch),
    }
  }

  if !current.trim().is_empty() {
    parts.push(current.trim().to_string());
  }

  parts
}
