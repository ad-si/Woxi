#[allow(unused_imports)]
use super::*;

/// AST-based Module implementation to avoid interpret() recursion
pub fn module_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "Module expects 2 arguments; {} given",
      args.len()
    )));
  }

  let vars_expr = &args[0];
  let body_expr = &args[1];

  // Parse variable declarations from the first argument (should be a List)
  let local_vars = match vars_expr {
    Expr::List(items) => {
      let mut vars = Vec::new();
      for item in items {
        match item {
          // x = value (assignment via Rule syntax internally or via Set)
          Expr::FunctionCall {
            name,
            args: set_args,
          } if name == "Set" && set_args.len() == 2 => {
            if let Expr::Identifier(var_name) = &set_args[0] {
              vars.push((var_name.clone(), Some(set_args[1].clone())));
            }
          }
          // x = value (parsed as Rule or identifier = expr)
          Expr::Rule {
            pattern,
            replacement,
          } => {
            if let Expr::Identifier(var_name) = pattern.as_ref() {
              vars.push((var_name.clone(), Some(replacement.as_ref().clone())));
            }
          }
          // Just a variable name without initialization
          Expr::Identifier(var_name) => {
            vars.push((var_name.clone(), None));
          }
          // Try to extract from raw text (for cases like "x = 5")
          Expr::Raw(s) => {
            if let Some((name, init)) = s.split_once('=') {
              let name = name.trim();
              let init = init.trim();
              if !name.is_empty() {
                let init_expr = string_to_expr(init)?;
                vars.push((name.to_string(), Some(init_expr)));
              }
            } else {
              vars.push((s.trim().to_string(), None));
            }
          }
          // BinaryOp with some assignment-like pattern (less common)
          _ => {
            // Try to convert to string and parse
            let s = expr_to_string(item);
            if let Some((name, init)) = s.split_once('=') {
              let name = name.trim();
              let init = init.trim();
              if !name.is_empty() && !name.contains(' ') {
                let init_expr = string_to_expr(init)?;
                vars.push((name.to_string(), Some(init_expr)));
              }
            }
          }
        }
      }
      vars
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Module expects a list of variable declarations as first argument"
          .into(),
      ));
    }
  };

  // Save previous bindings and set up new ones
  let mut prev: Vec<(String, Option<StoredValue>)> = Vec::new();

  for (var_name, init_expr) in &local_vars {
    let val = if let Some(expr) = init_expr {
      // Evaluate the initialization expression
      let evaluated = evaluate_expr_to_expr(expr)?;
      expr_to_string(&evaluated)
    } else {
      // Uninitialized variable - generate a unique symbol
      crate::functions::scoping::unique_symbol(var_name)
    };

    // Save current binding and set new one
    let pv = ENV.with(|e| {
      e.borrow_mut()
        .insert(var_name.clone(), StoredValue::Raw(val))
    });
    prev.push((var_name.clone(), pv));
  }

  // Evaluate the body expression (Return propagates through Module, not caught here)
  let result = evaluate_expr_to_expr(body_expr);

  // Restore previous bindings (even on error/Return)
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

  result
}

/// AST-based Block implementation - dynamic scoping (like Module but without unique symbols).
/// Block[{x = 1, y}, body] - variables are localized but use their original names.
pub fn block_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "Block expects 2 arguments; {} given",
      args.len()
    )));
  }

  let vars_expr = &args[0];
  let body_expr = &args[1];

  // Parse variable declarations (same as Module)
  let local_vars = match vars_expr {
    Expr::List(items) => {
      let mut vars = Vec::new();
      for item in items {
        match item {
          Expr::FunctionCall {
            name,
            args: set_args,
          } if name == "Set" && set_args.len() == 2 => {
            if let Expr::Identifier(var_name) = &set_args[0] {
              vars.push((var_name.clone(), Some(set_args[1].clone())));
            }
          }
          Expr::Rule {
            pattern,
            replacement,
          } => {
            if let Expr::Identifier(var_name) = pattern.as_ref() {
              vars.push((var_name.clone(), Some(replacement.as_ref().clone())));
            }
          }
          Expr::Identifier(var_name) => {
            vars.push((var_name.clone(), None));
          }
          _ => {
            let s = expr_to_string(item);
            if let Some((name, init)) = s.split_once('=') {
              let name = name.trim();
              let init = init.trim();
              if !name.is_empty() && !name.contains(' ') {
                let init_expr = string_to_expr(init)?;
                vars.push((name.to_string(), Some(init_expr)));
              }
            }
          }
        }
      }
      vars
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Block expects a list of variable declarations as first argument"
          .into(),
      ));
    }
  };

  // Save previous bindings and set up new ones
  let mut prev: Vec<(String, Option<StoredValue>)> = Vec::new();

  for (var_name, init_expr) in &local_vars {
    let pv = if let Some(expr) = init_expr {
      let evaluated = evaluate_expr_to_expr(expr)?;
      let val = expr_to_string(&evaluated);
      ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name.clone(), StoredValue::Raw(val))
      })
    } else {
      // Block with no initializer removes the variable binding so it evaluates as a symbol
      ENV.with(|e| e.borrow_mut().remove(var_name))
    };
    prev.push((var_name.clone(), pv));
  }

  // Evaluate the body expression (Return propagates through Block, not caught here)
  let result = evaluate_expr_to_expr(body_expr);

  // Restore previous bindings (even if body returned an error)
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

  result
}

/// Valid domain names for Element[]
const VALID_DOMAINS: &[&str] = &[
  "Primes",
  "Integers",
  "Rationals",
  "Algebraics",
  "Reals",
  "Complexes",
  "Booleans",
];

/// Known real-valued constants (parsed as Constant or Identifier)
const REAL_CONSTANTS: &[&str] = &[
  "Pi",
  "E",
  "Degree",
  "EulerGamma",
  "GoldenRatio",
  "Catalan",
  "Khinchin",
  "Glaisher",
];

/// Check if an expression is a member of a given domain
pub fn is_member_of_domain(expr: &Expr, domain: &str) -> Option<bool> {
  match domain {
    "Integers" => match expr {
      Expr::Integer(_) | Expr::BigInteger(_) => Some(true),
      Expr::Real(f) => Some(*f == f.floor() && f.is_finite()),
      // Rational[n, d] with d != 1 is not an integer
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        match (&args[0], &args[1]) {
          (Expr::Integer(_), Expr::Integer(1)) => Some(true),
          (Expr::Integer(_), Expr::Integer(_)) => Some(false),
          _ => None,
        }
      }
      // Known constants like Pi, E are not integers
      Expr::Constant(c) if REAL_CONSTANTS.contains(&c.as_str()) => Some(false),
      Expr::Identifier(name) if name == "I" => Some(false),
      _ => None,
    },
    "Primes" => match expr {
      Expr::Integer(n) => Some(*n >= 2 && is_prime_simple(*n)),
      Expr::Real(_) => Some(false),
      Expr::Constant(_) => Some(false),
      _ => None,
    },
    "Rationals" => match expr {
      Expr::Integer(_) | Expr::BigInteger(_) => Some(true),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        Some(true)
      }
      // Known irrational constants
      Expr::Constant(c) if REAL_CONSTANTS.contains(&c.as_str()) => Some(false),
      _ => None,
    },
    "Reals" => match expr {
      Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => Some(true),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        Some(true)
      }
      // Known real constants
      Expr::Constant(c) if REAL_CONSTANTS.contains(&c.as_str()) => Some(true),
      Expr::Identifier(name) if REAL_CONSTANTS.contains(&name.as_str()) => {
        Some(true)
      }
      Expr::Identifier(name) if name == "I" => Some(false),
      _ => {
        // Check for complex numbers with nonzero imaginary part
        if let Some((_, (im, _))) =
          crate::functions::math_ast::try_extract_complex_exact(expr)
          && im != 0
        {
          return Some(false);
        }
        None
      }
    },
    "Complexes" => match expr {
      Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(_, _) => Some(true),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        Some(true)
      }
      // Known constants are complex numbers too
      Expr::Constant(c) if REAL_CONSTANTS.contains(&c.as_str()) => Some(true),
      Expr::Identifier(name) if REAL_CONSTANTS.contains(&name.as_str()) => {
        Some(true)
      }
      Expr::Identifier(name) if name == "I" => Some(true),
      _ => {
        if crate::functions::math_ast::try_extract_complex_exact(expr).is_some()
        {
          Some(true)
        } else {
          None
        }
      }
    },
    "Booleans" => match expr {
      Expr::Identifier(name) if name == "True" || name == "False" => Some(true),
      _ => Some(false),
    },
    "Algebraics" => match expr {
      Expr::Integer(_) | Expr::BigInteger(_) => Some(true),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        Some(true)
      }
      _ => None,
    },
    _ => None,
  }
}

/// Simple primality check for small numbers
pub fn is_prime_simple(n: i128) -> bool {
  if n < 2 {
    return false;
  }
  if n < 4 {
    return true;
  }
  if n % 2 == 0 || n % 3 == 0 {
    return false;
  }
  let mut i = 5i128;
  while i * i <= n {
    if n % i == 0 || n % (i + 2) == 0 {
      return false;
    }
    i += 6;
  }
  true
}

/// Element[x, domain] - Test or assert domain membership
pub fn element_ast(x: &Expr, domain: &Expr) -> Result<Expr, InterpreterError> {
  let domain_name = match domain {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Element".to_string(),
        args: vec![x.clone(), domain.clone()],
      });
    }
  };

  // Validate domain name
  if !VALID_DOMAINS.contains(&domain_name) {
    eprintln!();
    eprintln!(
      "Element::bset: The second argument {} of Element should be one of: Primes, Integers, Rationals, Algebraics, Reals, Complexes or Booleans.",
      domain_name
    );
    return Ok(Expr::FunctionCall {
      name: "Element".to_string(),
      args: vec![x.clone(), domain.clone()],
    });
  }

  // Handle Alternatives: Element[a | b | c, dom]
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Alternatives,
    ..
  } = x
  {
    let alts = collect_alternatives(x);
    let mut remaining = Vec::new();
    for alt in &alts {
      match is_member_of_domain(alt, domain_name) {
        Some(true) => {} // Known member, skip
        Some(false) => {
          return Ok(Expr::Identifier("False".to_string()));
        }
        None => remaining.push(alt.clone()),
      }
    }
    if remaining.is_empty() {
      return Ok(Expr::Identifier("True".to_string()));
    }
    // Rebuild Alternatives from remaining
    let alt_expr = remaining
      .into_iter()
      .reduce(|acc, e| Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Alternatives,
        left: Box::new(acc),
        right: Box::new(e),
      })
      .unwrap();
    return Ok(Expr::FunctionCall {
      name: "Element".to_string(),
      args: vec![alt_expr, domain.clone()],
    });
  }

  // Handle lists: Element[{a, b, c}, dom] → Element[a | b | c, dom]
  if let Expr::List(items) = x {
    if items.is_empty() {
      return Ok(Expr::Identifier("True".to_string()));
    }
    // Convert list to Alternatives and recurse
    let alt_expr = items
      .iter()
      .cloned()
      .reduce(|acc, e| Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Alternatives,
        left: Box::new(acc),
        right: Box::new(e),
      })
      .unwrap();
    return element_ast(&alt_expr, domain);
  }

  // Simple case: check single element
  match is_member_of_domain(x, domain_name) {
    Some(true) => Ok(Expr::Identifier("True".to_string())),
    Some(false) => Ok(Expr::Identifier("False".to_string())),
    None => Ok(Expr::FunctionCall {
      name: "Element".to_string(),
      args: vec![x.clone(), domain.clone()],
    }),
  }
}

/// NotElement[x, domain] - Test non-membership of an expression in a mathematical domain
pub fn not_element_ast(
  x: &Expr,
  domain: &Expr,
) -> Result<Expr, InterpreterError> {
  let result = element_ast(x, domain)?;
  match &result {
    Expr::Identifier(name) if name == "True" => {
      Ok(Expr::Identifier("False".to_string()))
    }
    Expr::Identifier(name) if name == "False" => {
      Ok(Expr::Identifier("True".to_string()))
    }
    _ => {
      // Element returned unevaluated, so NotElement stays unevaluated too
      Ok(Expr::FunctionCall {
        name: "NotElement".to_string(),
        args: vec![x.clone(), domain.clone()],
      })
    }
  }
}

/// Collect all alternatives from a nested Alternatives expression
pub fn collect_alternatives(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Alternatives,
      left,
      right,
    } => {
      let mut result = collect_alternatives(left);
      result.extend(collect_alternatives(right));
      result
    }
    _ => vec![expr.clone()],
  }
}

/// AST-based Assuming: Assuming[assum, body]
/// Evaluates body with $Assumptions set to assum.
pub fn assuming_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "Assuming expects 2 arguments; {} given",
      args.len()
    )));
  }

  let assumption = evaluate_expr_to_expr(&args[0])?;

  // Save current $Assumptions
  let prev = ENV.with(|e| e.borrow().get("$Assumptions").cloned());

  // Set $Assumptions to the assumption
  let val = expr_to_string(&assumption);
  ENV.with(|e| {
    e.borrow_mut()
      .insert("$Assumptions".to_string(), StoredValue::Raw(val))
  });

  // Evaluate the body expression
  let result = evaluate_expr_to_expr(&args[1]);

  // Restore previous $Assumptions (even if body returned an error)
  ENV.with(|e| {
    let mut env = e.borrow_mut();
    if let Some(v) = prev {
      env.insert("$Assumptions".to_string(), v);
    } else {
      env.remove("$Assumptions");
    }
  });

  result
}

/// FilterRules[{rules...}, keys] - filter rules by matching keys
pub fn filter_rules_ast(
  rules: &Expr,
  keys: &Expr,
) -> Result<Expr, InterpreterError> {
  let rule_list = match rules {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FilterRules".to_string(),
        args: vec![rules.clone(), keys.clone()],
      });
    }
  };

  // Build set of key names to keep
  let key_names: Vec<String> = match keys {
    Expr::List(items) => items.iter().map(expr_to_string).collect(),
    _ => vec![expr_to_string(keys)],
  };

  let mut result = Vec::new();
  for rule in rule_list {
    let rule_key = match rule {
      Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
        expr_to_string(pattern)
      }
      Expr::FunctionCall { name, args }
        if (name == "Rule" || name == "RuleDelayed") && !args.is_empty() =>
      {
        expr_to_string(&args[0])
      }
      _ => continue,
    };
    if key_names.contains(&rule_key) {
      result.push(rule.clone());
    }
  }

  Ok(Expr::List(result))
}

/// AST-based For loop: For[init, test, incr, body]
pub fn for_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 || args.len() > 4 {
    return Err(InterpreterError::EvaluationError(format!(
      "For expects 3 or 4 arguments; {} given",
      args.len()
    )));
  }

  let init = &args[0];
  let test = &args[1];
  let incr = &args[2];
  let body = args.get(3);

  const MAX_ITERATIONS: usize = 100000;

  // Evaluate the initialization
  evaluate_expr_to_expr(init)?;

  let mut iterations = 0;
  loop {
    // Evaluate the test condition
    let test_result = evaluate_expr_to_expr(test)?;
    match test_result {
      Expr::Identifier(ref s) if s == "True" => {}
      Expr::Identifier(ref s) if s == "False" => break,
      _ => break,
    }

    // Evaluate the body (if provided)
    if let Some(body) = body {
      match evaluate_expr_to_expr(body) {
        Ok(_) => {}
        Err(InterpreterError::BreakSignal) => break,
        Err(InterpreterError::ContinueSignal) => {}
        Err(e) => return Err(e),
      }
    }

    // Evaluate the increment
    evaluate_expr_to_expr(incr)?;

    iterations += 1;
    if iterations >= MAX_ITERATIONS {
      return Err(InterpreterError::EvaluationError(
        "For: maximum iterations exceeded".into(),
      ));
    }
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// AST-based With implementation - substitutes bindings into body before evaluation.
/// With[{x = val, y = val2}, body] replaces x and y in body with evaluated values.
pub fn with_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let vars_expr = &args[0];
  let body_expr = &args[1];

  // Parse variable declarations from the first argument (should be a List)
  let bindings: Vec<(String, Expr)> = match vars_expr {
    Expr::List(items) => {
      let mut vars = Vec::new();
      for item in items {
        match item {
          Expr::FunctionCall {
            name,
            args: set_args,
          } if name == "Set" && set_args.len() == 2 => {
            if let Expr::Identifier(var_name) = &set_args[0] {
              let val = evaluate_expr_to_expr(&set_args[1])?;
              vars.push((var_name.clone(), val));
            }
          }
          Expr::Rule {
            pattern,
            replacement,
          } => {
            if let Expr::Identifier(var_name) = pattern.as_ref() {
              let val = evaluate_expr_to_expr(replacement)?;
              vars.push((var_name.clone(), val));
            }
          }
          _ => {}
        }
      }
      vars
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "With expects a list of variable declarations as first argument".into(),
      ));
    }
  };

  // Substitute all bindings into the body
  let mut substituted = body_expr.clone();
  for (var_name, val) in &bindings {
    substituted =
      crate::syntax::substitute_variable(&substituted, var_name, val);
  }

  // Evaluate the substituted body
  evaluate_expr_to_expr(&substituted)
}

/// Recursively set a value at a path of indices within an Expr.
/// Supports lists and FunctionCall arguments (e.g., Grid[[1, row, col]]).
pub fn set_part_deep(
  expr: &mut Expr,
  indices: &[Expr],
  value: &Expr,
) -> Result<(), InterpreterError> {
  if indices.is_empty() {
    *expr = value.clone();
    return Ok(());
  }

  let idx = match &indices[0] {
    Expr::Integer(n) => *n as i64,
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i64().ok_or_else(|| {
        InterpreterError::EvaluationError(
          "Part assignment: index too large".into(),
        )
      })?
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Part assignment: index must be an integer".into(),
      ));
    }
  };

  match expr {
    Expr::List(items) => {
      let len = items.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx < 0 || actual_idx >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {} of list does not exist.",
          idx
        )));
      }
      set_part_deep(&mut items[actual_idx as usize], &indices[1..], value)
    }
    Expr::FunctionCall { args, .. } => {
      // Part 0 is the head, Part 1.. are arguments (1-indexed)
      if idx == 0 {
        return Err(InterpreterError::EvaluationError(
          "Cannot set Part 0 (head) of a function call".into(),
        ));
      }
      let actual_idx = (idx - 1) as usize;
      if actual_idx >= args.len() {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {} of expression does not exist.",
          idx
        )));
      }
      set_part_deep(&mut args[actual_idx], &indices[1..], value)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Part assignment: cannot index into this expression".into(),
    )),
  }
}
