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
    let stored = if let Some(expr) = init_expr {
      // Evaluate the initialization expression
      let evaluated = evaluate_expr_to_expr(expr)?;
      StoredValue::ExprVal(evaluated)
    } else {
      // Uninitialized variable - generate a unique symbol
      StoredValue::Raw(crate::functions::scoping::unique_symbol(var_name))
    };

    // Save current binding and set new one
    let pv = ENV.with(|e| e.borrow_mut().insert(var_name.clone(), stored));
    prev.push((var_name.clone(), pv));
  }

  // Evaluate body. Wolfram leaves Return[val] symbolic at Module's
  // boundary — wrap any ReturnValue exception into a literal `Return[val]`
  // so e.g. `Module[{}, Return[1]]` evaluates to `Return[1]`. The top-level
  // display path in interpret() unwraps it back to `1` for the REPL.
  let result = match evaluate_expr_to_expr(body_expr) {
    Err(InterpreterError::ReturnValue(val)) => Ok(Expr::FunctionCall {
      name: "Return".to_string(),
      args: vec![*val].into(),
    }),
    other => other,
  };

  // Handle Condition in body: evaluate test while locals are still in scope
  let result = match &result {
    Ok(Expr::FunctionCall { name, args })
      if name == "Condition" && args.len() == 2 =>
    {
      match evaluate_expr_to_expr(&args[1]) {
        Ok(Expr::Identifier(ref s)) if s == "True" => {
          evaluate_expr_to_expr(&args[0])
        }
        Ok(test_val) => Ok(Expr::FunctionCall {
          name: "Condition".to_string(),
          args: vec![args[0].clone(), test_val].into(),
        }),
        Err(e) => Err(e),
      }
    }
    _ => result,
  };

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
      // Evaluate initializer with the *previous* binding still active (so
      // `Block[{x = n+2, n}, ...]` sees the outer `n`).
      let evaluated = evaluate_expr_to_expr(expr)?;
      ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name.clone(), StoredValue::ExprVal(evaluated))
      })
    } else {
      // No initializer: Block preserves the current global value during the
      // body but restores it on exit. We snapshot the existing binding
      // (without removing it) so any assignments inside the block can be
      // reverted afterwards.
      ENV.with(|e| e.borrow().get(var_name).cloned())
    };
    prev.push((var_name.clone(), pv));
  }

  // Evaluate body. Block, like Module, wraps any escaping Return[val]
  // into a literal `Return[val]` Expr so the symbolic value matches
  // wolframscript: `Block[{}, Return[42]]` ⇒ `Return[42]`. The top-level
  // display path unwraps it for the REPL.
  let result = match evaluate_expr_to_expr(body_expr) {
    Err(InterpreterError::ReturnValue(val)) => Ok(Expr::FunctionCall {
      name: "Return".to_string(),
      args: vec![*val].into(),
    }),
    other => other,
  };

  // Handle Condition in body: evaluate test while locals are still in scope
  let result = match &result {
    Ok(Expr::FunctionCall { name, args })
      if name == "Condition" && args.len() == 2 =>
    {
      match evaluate_expr_to_expr(&args[1]) {
        Ok(Expr::Identifier(ref s)) if s == "True" => {
          evaluate_expr_to_expr(&args[0])
        }
        Ok(test_val) => Ok(Expr::FunctionCall {
          name: "Condition".to_string(),
          args: vec![args[0].clone(), test_val].into(),
        }),
        Err(e) => Err(e),
      }
    }
    _ => result,
  };

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
  "PositiveReals",
  "PositiveIntegers",
  "NonNegativeReals",
  "NonNegativeIntegers",
  "NegativeReals",
  "NegativeIntegers",
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
    "PositiveReals" => match expr {
      Expr::Integer(n) => Some(*n > 0),
      Expr::Real(f) => Some(*f > 0.0 && f.is_finite()),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some((*n > 0 && *d > 0) || (*n < 0 && *d < 0))
        } else {
          None
        }
      }
      Expr::Constant(c) if REAL_CONSTANTS.contains(&c.as_str()) => Some(true),
      Expr::Identifier(name) if REAL_CONSTANTS.contains(&name.as_str()) => {
        Some(true)
      }
      _ => None,
    },
    "PositiveIntegers" => match expr {
      Expr::Integer(n) => Some(*n > 0),
      Expr::Real(f) => Some(*f > 0.0 && *f == f.floor() && f.is_finite()),
      _ => None,
    },
    "NonNegativeReals" => match expr {
      Expr::Integer(n) => Some(*n >= 0),
      Expr::Real(f) => Some(*f >= 0.0 && f.is_finite()),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some((*n >= 0 && *d > 0) || (*n <= 0 && *d < 0))
        } else {
          None
        }
      }
      Expr::Constant(c) if REAL_CONSTANTS.contains(&c.as_str()) => Some(true),
      Expr::Identifier(name) if REAL_CONSTANTS.contains(&name.as_str()) => {
        Some(true)
      }
      _ => None,
    },
    "NonNegativeIntegers" => match expr {
      Expr::Integer(n) => Some(*n >= 0),
      Expr::Real(f) => Some(*f >= 0.0 && *f == f.floor() && f.is_finite()),
      _ => None,
    },
    "NegativeReals" => match expr {
      Expr::Integer(n) => Some(*n < 0),
      Expr::Real(f) => Some(*f < 0.0 && f.is_finite()),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some((*n < 0 && *d > 0) || (*n > 0 && *d < 0))
        } else {
          None
        }
      }
      _ => None,
    },
    "NegativeIntegers" => match expr {
      Expr::Integer(n) => Some(*n < 0),
      Expr::Real(f) => Some(*f < 0.0 && *f == f.floor() && f.is_finite()),
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
        args: vec![x.clone(), domain.clone()].into(),
      });
    }
  };

  // Validate domain name
  if !VALID_DOMAINS.contains(&domain_name) {
    crate::emit_message(&format!(
      "Element::bset: The second argument {} of Element should be one of: Primes, Integers, Rationals, Algebraics, Reals, Complexes or Booleans.",
      domain_name
    ));
    return Ok(Expr::FunctionCall {
      name: "Element".to_string(),
      args: vec![x.clone(), domain.clone()].into(),
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
      args: vec![alt_expr, domain.clone()].into(),
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

  // Element[Plus[a, b, c, ...], Reals] / Integers / Rationals: drop any
  // summands already known to be in the domain — the remainder is in the
  // domain iff the original sum is. If everything drops out, the result is
  // True; if a single term remains, re-emit `Element[term, dom]`.
  if matches!(
    domain_name,
    "Reals" | "Integers" | "Rationals" | "Algebraics" | "Complexes"
  ) && let Expr::FunctionCall { name, args } = x
    && name == "Plus"
    && args.len() >= 2
  {
    let mut remaining: Vec<Expr> = Vec::new();
    let mut any_dropped = false;
    let mut conflict = false;
    for a in args.iter() {
      match is_member_of_domain(a, domain_name) {
        Some(true) => any_dropped = true,
        Some(false) => {
          conflict = true;
          remaining.push(a.clone());
        }
        None => remaining.push(a.clone()),
      }
    }
    if !any_dropped && !conflict {
      // No simplification possible; fall through.
    } else if remaining.is_empty() {
      return Ok(Expr::Identifier("True".to_string()));
    } else if any_dropped && !conflict {
      let reduced = if remaining.len() == 1 {
        remaining.into_iter().next().unwrap()
      } else {
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: remaining.into(),
        }
      };
      return element_ast(&reduced, domain);
    }
  }

  // Simple case: check single element
  match is_member_of_domain(x, domain_name) {
    Some(true) => Ok(Expr::Identifier("True".to_string())),
    Some(false) => Ok(Expr::Identifier("False".to_string())),
    None => Ok(Expr::FunctionCall {
      name: "Element".to_string(),
      args: vec![x.clone(), domain.clone()].into(),
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
        args: vec![x.clone(), domain.clone()].into(),
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

  // If the assumption is a simple equality `var == value` (or a List of
  // such), AND the body contains an Integrate/Sum/Product/Limit (where
  // wolframscript's ConditionalExpression handling would specialise the
  // result), substitute `var → value` in the body before evaluating.
  // Matches wolframscript: `Assuming[n == 1, Integrate[x^n, {x, 0, 1}]]`
  // returns `1/2`. Cases like `Assuming[n == 1, x^n]` keep `x^n`
  // because wolframscript also doesn't substitute there.
  let body = if contains_assumption_consumer(&args[1]) {
    apply_assumption_substitutions(&args[1], &assumption)
  } else {
    args[1].clone()
  };

  // Evaluate the body expression
  let result = evaluate_expr_to_expr(&body);

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

/// Does `expr` contain a function whose result depends on
/// `$Assumptions` (Integrate / Sum / Product / Limit)? Used as a guard
/// before specialising the body of `Assuming`, so `Assuming[n == 1, x^n]`
/// keeps `x^n` (matching wolframscript) while
/// `Assuming[n == 1, Integrate[x^n, ...]]` substitutes.
fn contains_assumption_consumer(expr: &Expr) -> bool {
  fn is_consumer(name: &str) -> bool {
    matches!(name, "Integrate" | "Sum" | "Product" | "Limit")
  }
  match expr {
    Expr::FunctionCall { name, args } => {
      is_consumer(name) || args.iter().any(contains_assumption_consumer)
    }
    Expr::List(items) => items.iter().any(contains_assumption_consumer),
    Expr::BinaryOp { left, right, .. } => {
      contains_assumption_consumer(left) || contains_assumption_consumer(right)
    }
    Expr::UnaryOp { operand, .. } => contains_assumption_consumer(operand),
    Expr::Comparison { operands, .. } => {
      operands.iter().any(contains_assumption_consumer)
    }
    _ => false,
  }
}

/// Walk `assumption` looking for equality assumptions of the form
/// `Equal[symbol, value]` (parsed as Comparison or FunctionCall, with
/// optional List wrapper for conjunctions) and substitute each
/// `symbol → value` in `body`. Used by `assuming_ast` to specialise an
/// integration / sum / etc. before evaluating it.
fn apply_assumption_substitutions(body: &Expr, assumption: &Expr) -> Expr {
  fn extract_equalities(a: &Expr, out: &mut Vec<(String, Expr)>) {
    match a {
      // List of assumptions: walk each entry.
      Expr::List(items) => {
        for item in items.iter() {
          extract_equalities(item, out);
        }
      }
      // And[…] (sometimes from `&&`): walk each clause.
      Expr::FunctionCall { name, args } if name == "And" => {
        for arg in args.iter() {
          extract_equalities(arg, out);
        }
      }
      // FullForm Equal[var, value]
      Expr::FunctionCall { name, args }
        if name == "Equal" && args.len() == 2 =>
      {
        if let Expr::Identifier(var) = &args[0] {
          out.push((var.clone(), args[1].clone()));
        }
      }
      // Parsed `var == value` is `Comparison { operands: [var, value],
      // operators: [Equal] }` (not a FunctionCall).
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() == 2
        && operators.len() == 1
        && operators[0] == crate::syntax::ComparisonOp::Equal =>
      {
        if let Expr::Identifier(var) = &operands[0] {
          out.push((var.clone(), operands[1].clone()));
        }
      }
      _ => {}
    }
  }
  let mut pairs: Vec<(String, Expr)> = Vec::new();
  extract_equalities(assumption, &mut pairs);
  let mut result = body.clone();
  for (var, value) in &pairs {
    result = crate::syntax::substitute_variable(&result, var, value);
  }
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
        args: vec![rules.clone(), keys.clone()].into(),
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

  Ok(Expr::List(result.into()))
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

  // Mirror the While safety cap. Wolframscript has no For iteration limit;
  // we set a very high one so practical scripts run unhindered while a true
  // infinite loop still terminates eventually rather than hanging the host.
  const MAX_ITERATIONS: usize = 1_000_000_000;

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

    // Evaluate the body (if provided). Return[val] inside the body
    // exits the loop and yields the literal `Return[val]` symbolic
    // expression — wolframscript renders it as `Return[val]` in
    // InputForm; interpret()'s top-level display unwraps it.
    if let Some(body) = body {
      match evaluate_expr_to_expr(body) {
        Ok(_) => {}
        Err(InterpreterError::BreakSignal) => break,
        Err(InterpreterError::ContinueSignal) => {}
        Err(InterpreterError::ReturnValue(val)) => {
          return Ok(Expr::FunctionCall {
            name: "Return".to_string(),
            args: vec![*val].into(),
          });
        }
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

  // Substitute all bindings into the body simultaneously
  // to prevent variable name leakage across bindings
  let binding_refs: Vec<(&str, &Expr)> = bindings
    .iter()
    .map(|(name, val)| (name.as_str(), val))
    .collect();
  let substituted =
    crate::syntax::substitute_variables(body_expr, &binding_refs);

  // Evaluate the substituted body
  evaluate_expr_to_expr(&substituted)
}

/// Recursively set a value at a path of indices within an Expr.
/// Supports lists and FunctionCall arguments (e.g., Grid[[1, row, col]]).
/// Supports Span indices (e.g., `A[[;;, 2]] = {6, 7}`).
pub fn set_part_deep(
  expr: &mut Expr,
  indices: &[Expr],
  value: &Expr,
) -> Result<(), InterpreterError> {
  if indices.is_empty() {
    *expr = value.clone();
    return Ok(());
  }

  // Handle `All` index (a[[All]] = ...) — equivalent to Span[1, All].
  // Threads the assignment over every position in the current list.
  if matches!(&indices[0], Expr::Identifier(s) | Expr::Constant(s) if s == "All")
  {
    let len = match expr {
      Expr::List(items) => items.len(),
      Expr::FunctionCall { args, .. } => args.len(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Part assignment: cannot apply All to this expression".into(),
        ));
      }
    };
    let rhs_items = match value {
      Expr::List(items) if items.len() == len => Some(items.clone()),
      _ => None,
    };
    for i in 0..len {
      let elem_value = match &rhs_items {
        Some(items) => items[i].clone(),
        None => value.clone(),
      };
      let inner = match expr {
        Expr::List(items) => &mut items[i],
        Expr::FunctionCall { args, .. } => &mut args[i],
        _ => unreachable!(),
      };
      set_part_deep(inner, &indices[1..], &elem_value)?;
    }
    return Ok(());
  }

  // Handle Span index (e.g., 1;;n, ;;, 1;;-1) by threading the assignment
  // over each selected position in the current list.
  if let Expr::FunctionCall { name, args } = &indices[0]
    && name == "Span"
  {
    let len = match expr {
      Expr::List(items) => items.len() as i64,
      Expr::FunctionCall { args, .. } => args.len() as i64,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Part assignment: cannot apply Span to this expression".into(),
        ));
      }
    };
    let positions = resolve_span(args, len)?;
    // Match wolframscript: a List of the same length as the selection
    // distributes element-wise; any other RHS (scalar, or list with
    // different length) is broadcast as a whole to each position.
    let rhs_items = match value {
      Expr::List(items) if items.len() == positions.len() => {
        Some(items.clone())
      }
      _ => None,
    };
    for (i, &pos) in positions.iter().enumerate() {
      let actual_idx = (pos - 1) as usize;
      let elem_value = match &rhs_items {
        Some(items) => &items[i],
        None => value,
      };
      let inner = match expr {
        Expr::List(items) => &mut items[actual_idx],
        Expr::FunctionCall { args, .. } => &mut args[actual_idx],
        _ => unreachable!(),
      };
      set_part_deep(inner, &indices[1..], elem_value)?;
    }
    return Ok(());
  }

  // Handle List index (e.g., a[[{1, 3}]] = ...): assign to each selected
  // position. When the RHS is a List of the same length, distribute its
  // elements; otherwise broadcast the entire RHS to every position.
  if let Expr::List(index_items) = &indices[0] {
    let len = match expr {
      Expr::List(items) => items.len() as i64,
      Expr::FunctionCall { args, .. } => args.len() as i64,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Part assignment: cannot apply list index to this expression".into(),
        ));
      }
    };
    let mut positions = Vec::with_capacity(index_items.len());
    for item in index_items {
      let n = match item {
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
      let actual_idx = if n < 0 { len + n } else { n - 1 };
      if actual_idx < 0 || actual_idx >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {} of list does not exist.",
          n
        )));
      }
      positions.push(actual_idx as usize);
    }
    let distribute =
      matches!(value, Expr::List(items) if items.len() == positions.len());
    for (i, &actual_idx) in positions.iter().enumerate() {
      let elem_value: Expr = if distribute {
        if let Expr::List(items) = value {
          items[i].clone()
        } else {
          unreachable!()
        }
      } else {
        value.clone()
      };
      let inner = match expr {
        Expr::List(items) => &mut items[actual_idx],
        Expr::FunctionCall { args, .. } => &mut args[actual_idx],
        _ => unreachable!(),
      };
      set_part_deep(inner, &indices[1..], &elem_value)?;
    }
    return Ok(());
  }

  // Association assignment: an integer position selects the n-th value
  // (mirroring the Part read semantics), while a key selects the value for
  // that key (appending a new entry when the key is absent).
  if let Expr::Association(pairs) = expr {
    use num_traits::ToPrimitive;
    let pos = match &indices[0] {
      Expr::Integer(n) => Some(*n as i64),
      Expr::BigInteger(n) => n.to_i64(),
      _ => None,
    };
    if let Some(n) = pos {
      let len = pairs.len() as i64;
      let actual_idx = if n < 0 { len + n } else { n - 1 };
      if actual_idx < 0 || actual_idx >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {} of association does not exist.",
          n
        )));
      }
      return set_part_deep(
        &mut pairs[actual_idx as usize].1,
        &indices[1..],
        value,
      );
    }
    // Key-based index: match on the string form (quote-insensitive).
    let key_cmp = expr_to_string(&indices[0]).trim_matches('"').to_string();
    if let Some(pair) = pairs
      .iter_mut()
      .find(|(k, _)| expr_to_string(k).trim_matches('"') == key_cmp)
    {
      return set_part_deep(&mut pair.1, &indices[1..], value);
    }
    // Absent key: append it when this is the final index; descending deeper
    // into a key that does not exist is an error (matching wolframscript).
    if indices.len() == 1 {
      pairs.push((indices[0].clone(), value.clone()));
      return Ok(());
    }
    return Err(InterpreterError::EvaluationError(format!(
      "Part::partw: Part {} of association does not exist.",
      key_cmp
    )));
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

/// Resolve a Span[start, end] (or Span[start, end, step]) over a sequence of
/// `len` elements into the list of 1-based positions it selects.
fn resolve_span(args: &[Expr], len: i64) -> Result<Vec<i64>, InterpreterError> {
  let to_pos = |e: &Expr, default: i64| -> Result<i64, InterpreterError> {
    match e {
      Expr::Integer(n) => {
        let n = *n as i64;
        Ok(if n < 0 { len + n + 1 } else { n })
      }
      Expr::Identifier(s) if s == "All" => Ok(default),
      _ => Err(InterpreterError::EvaluationError(
        "Part assignment: unsupported Span endpoint".into(),
      )),
    }
  };
  let (start, end, step) = match args.len() {
    2 => (to_pos(&args[0], 1)?, to_pos(&args[1], len)?, 1i64),
    3 => {
      let step_val = match &args[2] {
        Expr::Integer(n) => *n as i64,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Part assignment: unsupported Span step".into(),
          ));
        }
      };
      (to_pos(&args[0], 1)?, to_pos(&args[1], len)?, step_val)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Part assignment: malformed Span".into(),
      ));
    }
  };
  if step == 0 {
    return Err(InterpreterError::EvaluationError(
      "Part assignment: Span step cannot be zero".into(),
    ));
  }
  let mut positions = Vec::new();
  let mut p = start;
  if step > 0 {
    while p <= end {
      if p < 1 || p > len {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {} of list does not exist.",
          p
        )));
      }
      positions.push(p);
      p += step;
    }
  } else {
    while p >= end {
      if p < 1 || p > len {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {} of list does not exist.",
          p
        )));
      }
      positions.push(p);
      p += step;
    }
  }
  Ok(positions)
}
