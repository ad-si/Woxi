#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, expr_to_string};

use crate::functions::calculus_ast::{is_constant_wrt, simplify};

/// Eliminate[{eq1, eq2, ...}, var]  or  Eliminate[{eq1, eq2, ...}, {v1, v2, ...}]
/// Eliminates variables from a system of equations.
pub fn eliminate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Eliminate expects exactly 2 arguments".into(),
    ));
  }

  // Extract equations
  let equations: Vec<Expr> = match &args[0] {
    Expr::List(items) => items.clone(),
    // Single equation
    eq @ Expr::Comparison { .. } => vec![eq.clone()],
    eq @ Expr::FunctionCall { name, .. } if name == "Equal" => vec![eq.clone()],
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Eliminate".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Extract variables to eliminate
  let vars_to_eliminate: Vec<String> = match &args[1] {
    Expr::Identifier(name) => vec![name.clone()],
    Expr::List(items) => {
      let mut vars = Vec::new();
      for item in items {
        if let Expr::Identifier(name) = item {
          vars.push(name.clone());
        } else {
          return Ok(Expr::FunctionCall {
            name: "Eliminate".to_string(),
            args: args.to_vec(),
          });
        }
      }
      vars
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Eliminate".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut eqs = equations;

  // Eliminate each variable one at a time
  for var in &vars_to_eliminate {
    eqs = eliminate_one_variable(&eqs, var)?;
    if eqs.is_empty() {
      return Ok(Expr::Identifier("True".to_string()));
    }
  }

  // Return the result
  if eqs.len() == 1 {
    Ok(eqs.into_iter().next().unwrap())
  } else {
    // Multiple equations: join with And
    Ok(Expr::FunctionCall {
      name: "And".to_string(),
      args: eqs,
    })
  }
}

/// Solve an equation for a variable, returning the value expression.
/// For `lhs == rhs`, rearranges to isolate `var`:
///   a*var + rest = 0 → var = -rest/a
pub fn solve_for_var(eq: &Expr, var: &str) -> Option<Expr> {
  let (lhs, rhs) = extract_eq_sides(eq)?;

  // Convert to standard form: lhs - rhs = 0
  let poly = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(lhs),
    right: Box::new(rhs),
  };
  let expanded = expand_and_combine(&poly);

  // Check degree
  let degree = max_power(&expanded, var)?;
  if degree != 1 {
    // For non-linear, fall back to Solve
    let solutions =
      solve_ast(&[eq.clone(), Expr::Identifier(var.to_string())]).ok()?;
    if let Expr::List(outer) = &solutions {
      for item in outer {
        if let Expr::List(inner) = item {
          for rule in inner {
            if let Expr::Rule { replacement, .. } = rule {
              return Some(simplify(expand_and_combine(replacement)));
            }
          }
        }
      }
    }
    return None;
  }

  // Linear: extract coefficient of var and constant part
  let terms = collect_additive_terms(&expanded);
  let mut coeff = Expr::Integer(0);
  let mut rest = Expr::Integer(0);

  for term in &terms {
    if let Some(c) = extract_coefficient_of_power(term, var, 1) {
      coeff = add_exprs(&coeff, &c);
    }
    if let Some(c) = extract_coefficient_of_power(term, var, 0) {
      rest = add_exprs(&rest, &c);
    }
  }

  let coeff = simplify(coeff);
  let rest = simplify(rest);

  // var = -rest / coeff
  let neg_rest = negate_expr(&rest);
  let neg_rest = simplify(expand_and_combine(&neg_rest));

  // Simplify division
  match &coeff {
    Expr::Integer(1) => Some(neg_rest),
    Expr::Integer(-1) => {
      Some(simplify(expand_and_combine(&negate_expr(&neg_rest))))
    }
    _ => Some(simplify(solve_divide(&neg_rest, &coeff))),
  }
}

/// Extract (lhs, rhs) from an equation expression
pub fn extract_eq_sides(eq: &Expr) -> Option<(Expr, Expr)> {
  match eq {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == crate::syntax::ComparisonOp::Equal =>
    {
      Some((operands[0].clone(), operands[1].clone()))
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      Some((args[0].clone(), args[1].clone()))
    }
    _ => None,
  }
}

/// Check if an expression contains a variable
pub fn contains_var(expr: &Expr, var: &str) -> bool {
  !is_constant_wrt(expr, var)
}

/// Eliminate one variable from a system of equations
pub fn eliminate_one_variable(
  equations: &[Expr],
  var: &str,
) -> Result<Vec<Expr>, InterpreterError> {
  // Find an equation containing the variable, preferring linear ones
  let mut solve_idx = None;
  let mut solve_degree = i128::MAX;

  for (i, eq) in equations.iter().enumerate() {
    if let Some((lhs, rhs)) = extract_eq_sides(eq) {
      let diff = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(lhs),
        right: Box::new(rhs),
      };
      let expanded = expand_and_combine(&diff);
      if contains_var(&expanded, var)
        && let Some(d) = max_power(&expanded, var)
        && d < solve_degree
      {
        solve_degree = d;
        solve_idx = Some(i);
        if d == 1 {
          break; // Prefer linear equations
        }
      }
    }
  }

  let solve_idx = match solve_idx {
    Some(i) => i,
    None => {
      // Variable not found in any equation — return equations unchanged
      return Ok(equations.to_vec());
    }
  };

  // Solve the chosen equation for the variable directly
  let chosen_eq = &equations[solve_idx];
  let val = match solve_for_var(chosen_eq, var) {
    Some(v) => v,
    None => return Ok(equations.to_vec()),
  };

  // Substitute into all other equations
  let mut result = Vec::new();
  for (i, eq) in equations.iter().enumerate() {
    if i == solve_idx {
      continue; // Skip the equation we solved
    }
    let substituted = crate::syntax::substitute_variable(eq, var, &val);
    // Simplify the substituted equation
    let simplified = simplify_equation(&substituted)?;
    // Skip trivially true equations (True, or a == a)
    if is_trivially_true(&simplified) {
      continue;
    }
    result.push(simplified);
  }

  Ok(result)
}

/// Simplify an equation by evaluating both sides
pub fn simplify_equation(eq: &Expr) -> Result<Expr, InterpreterError> {
  match eq {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == crate::syntax::ComparisonOp::Equal =>
    {
      let lhs = simplify(expand_and_combine(&operands[0]));
      let rhs = simplify(expand_and_combine(&operands[1]));
      Ok(Expr::Comparison {
        operands: vec![lhs, rhs],
        operators: vec![crate::syntax::ComparisonOp::Equal],
      })
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      let lhs = simplify(expand_and_combine(&args[0]));
      let rhs = simplify(expand_and_combine(&args[1]));
      Ok(Expr::Comparison {
        operands: vec![lhs, rhs],
        operators: vec![crate::syntax::ComparisonOp::Equal],
      })
    }
    other => Ok(simplify(expand_and_combine(other))),
  }
}

/// Check if an equation is trivially true (e.g., True, or 0 == 0)
pub fn is_trivially_true(eq: &Expr) -> bool {
  match eq {
    Expr::Identifier(s) if s == "True" => true,
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == crate::syntax::ComparisonOp::Equal =>
    {
      expr_to_string(&operands[0]) == expr_to_string(&operands[1])
    }
    _ => false,
  }
}

/// SolveAlways[eqn, vars] - find conditions on parameters for equation to hold for all values of vars
pub fn solve_always_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "SolveAlways expects exactly 2 arguments".into(),
    ));
  }

  let eqn = &args[0];
  let vars_arg = &args[1];

  // Extract variables: can be a single symbol or a list
  let vars: Vec<String> = match vars_arg {
    Expr::Identifier(s) => vec![s.clone()],
    Expr::List(items) => items
      .iter()
      .filter_map(|e| {
        if let Expr::Identifier(s) = e {
          Some(s.clone())
        } else {
          None
        }
      })
      .collect(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "SolveAlways".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if vars.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "SolveAlways".to_string(),
      args: args.to_vec(),
    });
  }

  // Handle True/False (e.g. 0 == 0 evaluates to True before reaching us)
  match eqn {
    Expr::Identifier(s) if s == "True" => {
      return Ok(Expr::List(vec![Expr::List(vec![])]));
    }
    Expr::Identifier(s) if s == "False" => {
      return Ok(Expr::List(vec![]));
    }
    _ => {}
  }

  // Extract the polynomial expression (LHS - RHS from equation)
  let poly_expr = match eqn {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == crate::syntax::ComparisonOp::Equal =>
    {
      // lhs == rhs => lhs - rhs
      let lhs_str = expr_to_string(&operands[0]);
      let rhs_str = expr_to_string(&operands[1]);
      format!("Expand[({}) - ({})]", lhs_str, rhs_str)
    }
    _ => {
      // Treat expression itself as the polynomial = 0
      let s = expr_to_string(eqn);
      format!("Expand[{}]", s)
    }
  };

  // Recursively extract all leaf coefficients by expanding through each variable.
  // Start with the polynomial expression string, then for each variable extract
  // CoefficientList and recurse on the remaining variables.
  let mut leaf_coeffs: Vec<String> = Vec::new();
  extract_leaf_coefficients(&poly_expr, &vars, 0, &mut leaf_coeffs)?;

  // Filter out zeros
  leaf_coeffs.retain(|s| s != "0");

  if leaf_coeffs.is_empty() {
    return Ok(Expr::List(vec![Expr::List(vec![])]));
  }

  // Now solve: each leaf coefficient must equal zero.
  // Process sequentially, substituting already-found solutions.
  let mut rules: Vec<Expr> = Vec::new();
  let mut solved_vars = std::collections::HashSet::new();

  for coeff_str in &leaf_coeffs {
    // Parse the coefficient expression
    let coeff_expr = match crate::syntax::string_to_expr(coeff_str) {
      Ok(e) => e,
      Err(_) => continue,
    };

    // Simple case: coefficient is a single variable
    if let Expr::Identifier(s) = &coeff_expr {
      if !solved_vars.contains(s) && !vars.contains(s) {
        solved_vars.insert(s.clone());
        rules.push(Expr::Rule {
          pattern: Box::new(Expr::Identifier(s.clone())),
          replacement: Box::new(Expr::Integer(0)),
        });
      }
      continue;
    }

    // Find free variables (parameters, not the quantified vars)
    let mut free_vars_here = std::collections::BTreeSet::new();
    collect_free_vars_sa(&coeff_expr, &vars, &mut free_vars_here);

    // Remove already-solved variables
    for sv in &solved_vars {
      free_vars_here.remove(sv);
    }

    if free_vars_here.is_empty() {
      // No free variables — nonzero constant, impossible
      // But first substitute solved values and check if it becomes zero
      let mut sub_code = format!("({})", coeff_str);
      for rule in &rules {
        if let Expr::Rule {
          pattern,
          replacement,
        } = rule
        {
          let p = expr_to_string(pattern);
          let r = expr_to_string(replacement);
          sub_code = format!("({} /. {} -> ({}))", sub_code, p, r);
        }
      }
      if let Ok(result) = crate::interpret(&sub_code)
        && result.trim() == "0"
      {
        continue;
      }
      return Ok(Expr::List(vec![]));
    } else if free_vars_here.len() == 1 {
      // Single free variable, try Solve
      let fv = free_vars_here.iter().next().unwrap().clone();
      let mut sub_code = format!("({})", coeff_str);
      for rule in &rules {
        if let Expr::Rule {
          pattern,
          replacement,
        } = rule
        {
          let p = expr_to_string(pattern);
          let r = expr_to_string(replacement);
          sub_code = format!("({} /. {} -> ({}))", sub_code, p, r);
        }
      }
      let solve_code = format!("Solve[({}) == 0, {}]", sub_code, fv);
      if let Ok(solve_result) = crate::interpret(&solve_code)
        && let Ok(result_expr) = crate::syntax::string_to_expr(&solve_result)
        && let Expr::List(outer) = &result_expr
        && let Some(Expr::List(inner)) = outer.first()
      {
        for rule_expr in inner {
          if let Expr::Rule { pattern, .. } = rule_expr
            && let Expr::Identifier(s) = pattern.as_ref()
          {
            solved_vars.insert(s.clone());
          }
          rules.push(rule_expr.clone());
        }
        continue;
      }
      // Fallback: set variable to 0
      solved_vars.insert(fv.clone());
      rules.push(Expr::Rule {
        pattern: Box::new(Expr::Identifier(fv)),
        replacement: Box::new(Expr::Integer(0)),
      });
    } else {
      // Multiple free variables: solve one at a time
      // Substitute known solutions first
      let mut sub_code = format!("({})", coeff_str);
      for rule in &rules {
        if let Expr::Rule {
          pattern,
          replacement,
        } = rule
        {
          let p = expr_to_string(pattern);
          let r = expr_to_string(replacement);
          sub_code = format!("({} /. {} -> ({}))", sub_code, p, r);
        }
      }
      // Try to solve for the first free variable
      let fv = free_vars_here.iter().next().unwrap().clone();
      let solve_code = format!("Solve[({}) == 0, {}]", sub_code, fv);
      if let Ok(solve_result) = crate::interpret(&solve_code)
        && let Ok(result_expr) = crate::syntax::string_to_expr(&solve_result)
        && let Expr::List(outer) = &result_expr
        && let Some(Expr::List(inner)) = outer.first()
      {
        for rule_expr in inner {
          if let Expr::Rule { pattern, .. } = rule_expr
            && let Expr::Identifier(s) = pattern.as_ref()
          {
            solved_vars.insert(s.clone());
          }
          rules.push(rule_expr.clone());
        }
        continue;
      }
      // Fallback: set all free vars to 0
      for fv in &free_vars_here {
        if !solved_vars.contains(fv) {
          solved_vars.insert(fv.clone());
          rules.push(Expr::Rule {
            pattern: Box::new(Expr::Identifier(fv.clone())),
            replacement: Box::new(Expr::Integer(0)),
          });
        }
      }
    }
  }

  Ok(Expr::List(vec![Expr::List(rules)]))
}

/// Recursively extract leaf coefficients from a polynomial expression.
/// For each variable in vars (starting at var_idx), extract CoefficientList,
/// then recurse on remaining variables for each coefficient.
pub fn extract_leaf_coefficients(
  poly_str: &str,
  vars: &[String],
  var_idx: usize,
  result: &mut Vec<String>,
) -> Result<(), InterpreterError> {
  if var_idx >= vars.len() {
    // No more variables to extract — this is a leaf coefficient
    // Simplify it
    let simplified = crate::interpret(&format!("Expand[{}]", poly_str))?;
    result.push(simplified.trim().to_string());
    return Ok(());
  }

  let var = &vars[var_idx];
  let coeffs_code = format!("CoefficientList[{}, {}]", poly_str, var);
  let coeffs_str = crate::interpret(&coeffs_code)?;
  let coeffs_expr = crate::syntax::string_to_expr(&coeffs_str)
    .unwrap_or(Expr::Identifier(coeffs_str));

  match &coeffs_expr {
    Expr::List(coeffs) => {
      for coeff in coeffs {
        let coeff_s = expr_to_string(coeff);
        extract_leaf_coefficients(&coeff_s, vars, var_idx + 1, result)?;
      }
    }
    _ => {
      let s = expr_to_string(&coeffs_expr);
      extract_leaf_coefficients(&s, vars, var_idx + 1, result)?;
    }
  }
  Ok(())
}

/// Collect all identifiers from an expression that are NOT in the excluded set
pub fn collect_free_vars_sa(
  expr: &Expr,
  excluded: &[String],
  result: &mut std::collections::BTreeSet<String>,
) {
  match expr {
    Expr::Identifier(s) => {
      if !excluded.contains(s)
        && !is_builtin_constant_sa(s)
        && s.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)
      {
        result.insert(s.clone());
      }
    }
    Expr::List(items) => {
      for item in items {
        collect_free_vars_sa(item, excluded, result);
      }
    }
    Expr::FunctionCall { args, .. } => {
      for arg in args {
        collect_free_vars_sa(arg, excluded, result);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_free_vars_sa(left, excluded, result);
      collect_free_vars_sa(right, excluded, result);
    }
    Expr::UnaryOp { operand, .. } => {
      collect_free_vars_sa(operand, excluded, result);
    }
    Expr::Comparison { operands, .. } => {
      for op in operands {
        collect_free_vars_sa(op, excluded, result);
      }
    }
    _ => {}
  }
}

pub fn is_builtin_constant_sa(s: &str) -> bool {
  matches!(
    s,
    "Pi"
      | "E"
      | "I"
      | "Infinity"
      | "True"
      | "False"
      | "Null"
      | "None"
      | "All"
      | "Automatic"
      | "ComplexInfinity"
      | "Indeterminate"
      | "EulerGamma"
      | "GoldenRatio"
      | "Degree"
      | "Catalan"
  )
}
