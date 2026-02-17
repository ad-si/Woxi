//! AST-native polynomial functions.
//!
//! Expand, Factor, Simplify, Coefficient, Exponent, PolynomialQ.

use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

use super::calculus_ast::{is_constant_wrt, simplify};

// ─── helpers ────────────────────────────────────────────────────────

use std::collections::HashMap;

/// Extract a map of variable_name → exponent from a list of variable factors.
/// E.g. [x^2, y] → {"x": 2, "y": 1}
fn extract_exponent_map(var_factors: &[Expr]) -> HashMap<String, i128> {
  let mut map = HashMap::new();
  for f in var_factors {
    match f {
      Expr::Identifier(name) => {
        *map.entry(name.clone()).or_insert(0) += 1;
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => {
        let name = expr_to_string(left);
        let exp = match right.as_ref() {
          Expr::Integer(n) => *n,
          _ => 1,
        };
        *map.entry(name).or_insert(0) += exp;
      }
      _ => {
        let name = expr_to_string(f);
        *map.entry(name).or_insert(0) += 1;
      }
    }
  }
  map
}

/// Helper to create boolean result
fn bool_expr(b: bool) -> Expr {
  Expr::Identifier(if b { "True" } else { "False" }.to_string())
}

// ─── PolynomialQ ────────────────────────────────────────────────────

/// PolynomialQ[expr, var] - Tests if expr is a polynomial in var
pub fn polynomial_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialQ expects 1 or 2 arguments".into(),
    ));
  }

  if args.len() == 1 {
    // 1-arg form: check if expr is a polynomial in all its variables
    let mut vars = std::collections::HashSet::new();
    collect_poly_vars(&args[0], &mut vars);
    if vars.is_empty() {
      // A constant is a polynomial
      return Ok(bool_expr(true));
    }
    return Ok(bool_expr(vars.iter().all(|v| is_polynomial(&args[0], v))));
  }

  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Second argument of PolynomialQ must be a symbol".into(),
      ));
    }
  };
  Ok(bool_expr(is_polynomial(&args[0], var)))
}

/// Collect variables that appear in polynomial context only (not inside functions like Sin)
fn collect_poly_vars(
  expr: &Expr,
  vars: &mut std::collections::HashSet<String>,
) {
  match expr {
    Expr::Identifier(name)
      if name != "True"
        && name != "False"
        && name != "Null"
        && name != "I"
        && name != "Pi"
        && name != "E"
        && name != "Infinity" =>
    {
      vars.insert(name.clone());
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_poly_vars(left, vars);
      collect_poly_vars(right, vars);
    }
    Expr::UnaryOp { operand, .. } => collect_poly_vars(operand, vars),
    Expr::FunctionCall { name, args } => {
      if name == "Plus" || name == "Times" || name == "Power" {
        for a in args {
          collect_poly_vars(a, vars);
        }
      }
      // For other functions like Sin[x], don't collect x as a polynomial variable
    }
    _ => {}
  }
}

/// Recursively check whether an expression is a polynomial in `var`.
fn is_polynomial(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => true,
    Expr::Identifier(_) => true, // either it IS the variable or a constant symbol – both ok
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus | BinaryOperator::Times => {
        is_polynomial(left, var) && is_polynomial(right, var)
      }
      BinaryOperator::Power => {
        // base must contain the variable and exponent must be a non-negative integer
        if is_constant_wrt(right, var) {
          if let Expr::Integer(n) = right.as_ref() {
            *n >= 0 && is_polynomial(left, var)
          } else {
            // non-integer exponent like x^y where y is a symbol ≠ var
            // Only polynomial if base is constant w.r.t. var
            is_constant_wrt(left, var)
          }
        } else {
          false
        }
      }
      BinaryOperator::Divide => {
        // polynomial / constant-in-var is still polynomial
        is_polynomial(left, var) && is_constant_wrt(right, var)
      }
      _ => is_constant_wrt(expr, var),
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_polynomial(operand, var),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" | "Times" => args.iter().all(|a| is_polynomial(a, var)),
      "Power" if args.len() == 2 => {
        if is_constant_wrt(&args[1], var) {
          if let Expr::Integer(n) = &args[1] {
            *n >= 0 && is_polynomial(&args[0], var)
          } else {
            is_constant_wrt(&args[0], var)
          }
        } else {
          false
        }
      }
      _ => is_constant_wrt(expr, var),
    },
    Expr::List(_) => false,
    _ => is_constant_wrt(expr, var),
  }
}

// ─── Exponent ───────────────────────────────────────────────────────

/// Exponent[expr, var] - Returns the maximum power of var in expr
pub fn exponent_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Exponent expects 2 or 3 arguments".into(),
    ));
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Second argument of Exponent must be a symbol".into(),
      ));
    }
  };

  // Exponent[0, x] -> -Infinity
  if matches!(&args[0], Expr::Integer(0)) {
    return Ok(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    });
  }

  // Expand and combine like terms first to handle things like (x^2+1)^3-1
  let expanded = expand_and_combine(&args[0]);

  // Determine if we need Max (default) or Min
  let use_min =
    args.len() == 3 && matches!(&args[2], Expr::Identifier(s) if s == "Min");

  if use_min {
    match min_power(&expanded, var) {
      Some(n) => Ok(Expr::Integer(n)),
      None => Ok(Expr::FunctionCall {
        name: "Exponent".to_string(),
        args: args.to_vec(),
      }),
    }
  } else {
    match max_power(&expanded, var) {
      Some(n) => Ok(Expr::Integer(n)),
      None => Ok(Expr::FunctionCall {
        name: "Exponent".to_string(),
        args: args.to_vec(),
      }),
    }
  }
}

/// Find the maximum power of `var` in `expr`.  Returns None for non-polynomial forms.
fn max_power(expr: &Expr, var: &str) -> Option<i128> {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) | Expr::String(_) => {
      Some(0)
    }
    Expr::Identifier(name) => {
      if name == var {
        Some(1)
      } else {
        Some(0)
      }
    }
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus => {
        let l = max_power(left, var)?;
        let r = max_power(right, var)?;
        Some(l.max(r))
      }
      BinaryOperator::Times => {
        let l = max_power(left, var)?;
        let r = max_power(right, var)?;
        Some(l + r)
      }
      BinaryOperator::Power => {
        if is_constant_wrt(left, var) {
          Some(0)
        } else if is_constant_wrt(right, var) {
          if let Expr::Integer(n) = right.as_ref() {
            let base_pow = max_power(left, var)?;
            Some(base_pow * n)
          } else {
            None
          }
        } else {
          None
        }
      }
      BinaryOperator::Divide => {
        if is_constant_wrt(right, var) {
          max_power(left, var)
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(0)
        } else {
          None
        }
      }
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => max_power(operand, var),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let mut m: i128 = 0;
        for a in args {
          m = m.max(max_power(a, var)?);
        }
        Some(m)
      }
      "Times" => {
        let mut s: i128 = 0;
        for a in args {
          s += max_power(a, var)?;
        }
        Some(s)
      }
      "Power" if args.len() == 2 => {
        if is_constant_wrt(&args[0], var) {
          Some(0)
        } else if let Expr::Integer(n) = &args[1] {
          let base_pow = max_power(&args[0], var)?;
          Some(base_pow * n)
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(0)
        } else {
          None
        }
      }
    },
    _ => {
      if is_constant_wrt(expr, var) {
        Some(0)
      } else {
        None
      }
    }
  }
}

/// Find the minimum power of `var` in `expr`.
fn min_power(expr: &Expr, var: &str) -> Option<i128> {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) | Expr::String(_) => {
      Some(0)
    }
    Expr::Identifier(name) => {
      if name == var {
        Some(1)
      } else {
        Some(0)
      }
    }
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus => {
        let l = min_power(left, var)?;
        let r = min_power(right, var)?;
        Some(l.min(r))
      }
      BinaryOperator::Times => {
        let l = min_power(left, var)?;
        let r = min_power(right, var)?;
        Some(l + r)
      }
      BinaryOperator::Power => {
        if is_constant_wrt(left, var) {
          Some(0)
        } else if is_constant_wrt(right, var) {
          if let Expr::Integer(n) = right.as_ref() {
            let base_pow = min_power(left, var)?;
            Some(base_pow * n)
          } else {
            None
          }
        } else {
          None
        }
      }
      BinaryOperator::Divide => {
        if is_constant_wrt(right, var) {
          min_power(left, var)
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(0)
        } else {
          None
        }
      }
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => min_power(operand, var),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let mut m: Option<i128> = None;
        for a in args {
          let p = min_power(a, var)?;
          m = Some(match m {
            None => p,
            Some(prev) => prev.min(p),
          });
        }
        m
      }
      "Times" => {
        let mut s: i128 = 0;
        for a in args {
          s += min_power(a, var)?;
        }
        Some(s)
      }
      "Power" if args.len() == 2 => {
        if is_constant_wrt(&args[0], var) {
          Some(0)
        } else if let Expr::Integer(n) = &args[1] {
          let base_pow = min_power(&args[0], var)?;
          Some(base_pow * n)
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(0)
        } else {
          None
        }
      }
    },
    _ => {
      if is_constant_wrt(expr, var) {
        Some(0)
      } else {
        None
      }
    }
  }
}

// ─── Coefficient ────────────────────────────────────────────────────

/// Coefficient[expr, var] or Coefficient[expr, var, n]
/// Returns the coefficient of var^n (default n=1).
pub fn coefficient_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Coefficient expects 2 or 3 arguments".into(),
    ));
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Second argument of Coefficient must be a symbol".into(),
      ));
    }
  };
  let power: i128 = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(n) => *n,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Coefficient".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    1
  };

  // First expand, then extract coefficient
  let expanded = expand_expr(&args[0]);
  let terms = collect_additive_terms(&expanded);
  let mut coeff_sum: Vec<Expr> = Vec::new();

  for term in &terms {
    if let Some(c) = extract_coefficient_of_power(term, var, power) {
      coeff_sum.push(c);
    }
  }

  if coeff_sum.is_empty() {
    Ok(Expr::Integer(0))
  } else if coeff_sum.len() == 1 {
    Ok(coeff_sum.remove(0))
  } else {
    // Sum all coefficient contributions
    let mut result = coeff_sum.remove(0);
    for c in coeff_sum {
      result = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(result),
        right: Box::new(c),
      };
    }
    Ok(simplify(result))
  }
}

/// CoefficientList[poly, var] - list of coefficients from power 0 to degree.
pub fn coefficient_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "CoefficientList expects 2 arguments".into(),
    ));
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CoefficientList".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Expand and combine the expression first
  let expanded = expand_and_combine(&args[0]);

  // Find the degree
  let degree = match max_power(&expanded, var) {
    Some(d) => d,
    None => {
      return Ok(Expr::FunctionCall {
        name: "CoefficientList".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Extract coefficient for each power from 0 to degree
  let mut coeffs = Vec::new();
  for power in 0..=degree {
    let coeff = coefficient_ast(&[
      args[0].clone(),
      args[1].clone(),
      Expr::Integer(power),
    ])?;
    let simplified = crate::evaluator::evaluate_expr_to_expr(&coeff)?;
    coeffs.push(simplified);
  }

  Ok(Expr::List(coeffs))
}

/// Collect all additive terms from an expression (flattening Plus).
fn collect_additive_terms(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let mut terms = collect_additive_terms(left);
      terms.extend(collect_additive_terms(right));
      terms
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      let mut terms = collect_additive_terms(left);
      let right_terms = collect_additive_terms(right);
      for t in right_terms {
        terms.push(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(t),
        });
      }
      terms
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut terms = Vec::new();
      for a in args {
        terms.extend(collect_additive_terms(a));
      }
      terms
    }
    _ => vec![expr.clone()],
  }
}

/// From a single term, extract the coefficient of `var^power`.
/// E.g. for term = 3*x^2, var = "x", power = 2 → Some(3)
/// For term = a*x^2, var = "x", power = 2 → Some(a)
fn extract_coefficient_of_power(
  term: &Expr,
  var: &str,
  power: i128,
) -> Option<Expr> {
  // Get the power of var and the remaining coefficient
  let (term_power, coeff) = term_var_power_and_coeff(term, var);
  if term_power == power {
    Some(coeff)
  } else {
    None
  }
}

/// Decompose a multiplicative term into (power_of_var, coefficient).
/// E.g. 3*x^2 → (2, 3); a*x → (1, a); 5 → (0, 5); x → (1, 1)
fn term_var_power_and_coeff(term: &Expr, var: &str) -> (i128, Expr) {
  match term {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) | Expr::String(_) => {
      (0, term.clone())
    }
    Expr::Identifier(name) => {
      if name == var {
        (1, Expr::Integer(1))
      } else {
        (0, term.clone())
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Identifier(name) = left.as_ref()
        && name == var
        && let Expr::Integer(n) = right.as_ref()
      {
        return (*n, Expr::Integer(1));
      }
      // Check if the whole thing is constant w.r.t. var
      if is_constant_wrt(term, var) {
        (0, term.clone())
      } else {
        // Complex expression — try to factor out var powers
        (-1, term.clone()) // sentinel: won't match any requested power
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let (lp, lc) = term_var_power_and_coeff(left, var);
      let (rp, rc) = term_var_power_and_coeff(right, var);
      let total_power = lp + rp;
      let coeff = multiply_exprs(&lc, &rc);
      (total_power, coeff)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (p, c) = term_var_power_and_coeff(operand, var);
      (
        p,
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(c),
        },
      )
    }
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Times" => {
        let mut total_power: i128 = 0;
        let mut coeffs: Vec<Expr> = Vec::new();
        for a in args {
          let (p, c) = term_var_power_and_coeff(a, var);
          total_power += p;
          coeffs.push(c);
        }
        let coeff = coeffs
          .into_iter()
          .reduce(|a, b| multiply_exprs(&a, &b))
          .unwrap_or(Expr::Integer(1));
        (total_power, coeff)
      }
      "Power" if args.len() == 2 => {
        if let Expr::Identifier(name) = &args[0]
          && name == var
          && let Expr::Integer(n) = &args[1]
        {
          return (*n, Expr::Integer(1));
        }
        if is_constant_wrt(term, var) {
          (0, term.clone())
        } else {
          (-1, term.clone())
        }
      }
      _ => {
        if is_constant_wrt(term, var) {
          (0, term.clone())
        } else {
          (-1, term.clone())
        }
      }
    },
    _ => {
      if is_constant_wrt(term, var) {
        (0, term.clone())
      } else {
        (-1, term.clone())
      }
    }
  }
}

/// Multiply two expressions, simplifying trivial cases.
fn multiply_exprs(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(1), _) => b.clone(),
    (_, Expr::Integer(1)) => a.clone(),
    (Expr::Integer(0), _) | (_, Expr::Integer(0)) => Expr::Integer(0),
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x * y),
    _ => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}

/// Add two expressions, simplifying trivial cases.
fn add_exprs(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(0), _) => b.clone(),
    (_, Expr::Integer(0)) => a.clone(),
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x + y),
    _ => Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}

// ─── Expand ─────────────────────────────────────────────────────────

/// Expand[expr] - Expands products and positive integer powers
pub fn expand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Expand expects exactly 1 argument".into(),
    ));
  }
  Ok(expand_and_combine(&args[0]))
}

/// Expand an expression and combine like terms.
fn expand_and_combine(expr: &Expr) -> Expr {
  let expanded = expand_expr(expr);
  let terms = collect_additive_terms(&expanded);
  combine_and_build(terms)
}

/// Recursively expand an expression.
fn expand_expr(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::Constant(_)
    | Expr::Identifier(_)
    | Expr::Slot(_) => expr.clone(),

    Expr::BinaryOp { op, left, right } => {
      let left_exp = expand_expr(left);
      let right_exp = expand_expr(right);
      match op {
        BinaryOperator::Plus => Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(left_exp),
          right: Box::new(right_exp),
        },
        BinaryOperator::Minus => Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(left_exp),
          right: Box::new(right_exp),
        },
        BinaryOperator::Times => distribute_product(&left_exp, &right_exp),
        BinaryOperator::Power => {
          // (sum)^n where n is positive integer
          if let Expr::Integer(n) = &right_exp
            && *n >= 2
            && is_sum(&left_exp)
          {
            return expand_power(&left_exp, *n);
          }
          // Try to simplify Power (e.g. I^2 → -1, Sqrt[x]^2 → x)
          if let Ok(simplified) = crate::functions::math_ast::power_ast(&[
            left_exp.clone(),
            right_exp.clone(),
          ]) {
            // Only use simplified result if it actually simplified
            if !matches!(
              &simplified,
              Expr::BinaryOp {
                op: BinaryOperator::Power,
                ..
              }
            ) {
              return simplified;
            }
          }
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(left_exp),
            right: Box::new(right_exp),
          }
        }
        _ => Expr::BinaryOp {
          op: *op,
          left: Box::new(left_exp),
          right: Box::new(right_exp),
        },
      }
    }

    Expr::UnaryOp { op, operand } => {
      let operand_exp = expand_expr(operand);
      match op {
        UnaryOperator::Minus => {
          // Distribute minus over sums
          let terms = collect_additive_terms(&operand_exp);
          let negated: Vec<Expr> =
            terms.into_iter().map(|t| negate_term(&t)).collect();
          build_sum(negated)
        }
        _ => Expr::UnaryOp {
          op: *op,
          operand: Box::new(operand_exp),
        },
      }
    }

    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let expanded_args: Vec<Expr> = args.iter().map(expand_expr).collect();
        let mut all_terms = Vec::new();
        for a in &expanded_args {
          all_terms.extend(collect_additive_terms(a));
        }
        build_sum(all_terms)
      }
      "Times" => {
        let expanded_args: Vec<Expr> = args.iter().map(expand_expr).collect();
        if expanded_args.is_empty() {
          return Expr::Integer(1);
        }
        let mut result = expanded_args[0].clone();
        for a in &expanded_args[1..] {
          result = distribute_product(&result, a);
        }
        result
      }
      "Power" if args.len() == 2 => {
        let base = expand_expr(&args[0]);
        let exp = expand_expr(&args[1]);
        if let Expr::Integer(n) = &exp
          && *n >= 2
          && is_sum(&base)
        {
          return expand_power(&base, *n);
        }
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![base, exp],
        }
      }
      _ => expr.clone(),
    },

    _ => expr.clone(),
  }
}

/// Check if an expression is a sum (Plus).
fn is_sum(expr: &Expr) -> bool {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus,
      ..
    } => true,
    Expr::FunctionCall { name, .. } if name == "Plus" => true,
    _ => false,
  }
}

/// Distribute the product of two expanded expressions.
/// If either is a sum, produce all cross-products.
fn distribute_product(left: &Expr, right: &Expr) -> Expr {
  let left_terms = collect_additive_terms(left);
  let right_terms = collect_additive_terms(right);

  if left_terms.len() == 1 && right_terms.len() == 1 {
    // Neither is a sum — just multiply
    return multiply_terms(&left_terms[0], &right_terms[0]);
  }

  let mut result_terms = Vec::new();
  for l in &left_terms {
    for r in &right_terms {
      result_terms.push(multiply_terms(l, r));
    }
  }
  build_sum(result_terms)
}

/// Multiply two non-sum terms (individual monomials).
fn multiply_terms(a: &Expr, b: &Expr) -> Expr {
  // Handle negation
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = a
  {
    return negate_term(&multiply_terms(operand, b));
  }
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = b
  {
    return negate_term(&multiply_terms(a, operand));
  }

  match (a, b) {
    (Expr::Integer(1), _) => b.clone(),
    (_, Expr::Integer(1)) => a.clone(),
    (Expr::Integer(0), _) | (_, Expr::Integer(0)) => Expr::Integer(0),
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x * y),
    (Expr::Real(x), Expr::Real(y)) => Expr::Real(x * y),
    (Expr::Integer(x), Expr::Real(y)) | (Expr::Real(y), Expr::Integer(x)) => {
      Expr::Real(*x as f64 * y)
    }
    _ => {
      // Combine like bases: x * x → x^2, x^a * x^b → x^(a+b)
      let mut a_factors = collect_multiplicative_factors(a);
      let b_factors = collect_multiplicative_factors(b);
      a_factors.extend(b_factors);
      combine_product_factors(a_factors)
    }
  }
}

/// Combine multiplicative factors, merging like bases into powers.
/// [x, x, y] → x^2 * y
fn combine_product_factors(factors: Vec<Expr>) -> Expr {
  // Group factors by base, sum exponents
  let mut base_exps: Vec<(String, Expr, Expr)> = Vec::new(); // (sort_key, base, exponent)
  let mut numeric_coeff = Expr::Integer(1);

  for f in &factors {
    match f {
      Expr::Integer(_) | Expr::Real(_) => {
        numeric_coeff = multiply_exprs(&numeric_coeff, f);
      }
      _ => {
        let (base, exp) = extract_base_and_exp(f);
        let key = expr_to_string(&base);
        if let Some(entry) = base_exps.iter_mut().find(|(k, _, _)| *k == key) {
          entry.2 = add_exprs(&entry.2, &exp);
        } else {
          base_exps.push((key, base, exp));
        }
      }
    }
  }

  // Build result
  let mut result_factors: Vec<Expr> = Vec::new();
  if !matches!(&numeric_coeff, Expr::Integer(1)) {
    result_factors.push(numeric_coeff);
  }

  for (_, base, exp) in base_exps {
    let exp = simplify(exp);
    if matches!(&exp, Expr::Integer(0)) {
      continue; // x^0 = 1, skip
    } else if matches!(&exp, Expr::Integer(1)) {
      result_factors.push(base);
    } else {
      // Try to evaluate the power (e.g. I^2 → -1)
      if let Ok(simplified) =
        crate::functions::math_ast::power_ast(&[base.clone(), exp.clone()])
      {
        result_factors.push(simplified);
      } else {
        result_factors.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(base),
          right: Box::new(exp),
        });
      }
    }
  }

  if result_factors.is_empty() {
    Expr::Integer(1)
  } else {
    build_product(result_factors)
  }
}

/// Extract base and exponent from a factor.
fn extract_base_and_exp(expr: &Expr) -> (Expr, Expr) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (args[0].clone(), args[1].clone())
    }
    _ => (expr.clone(), Expr::Integer(1)),
  }
}

/// Negate a term.
fn negate_term(t: &Expr) -> Expr {
  match t {
    Expr::Integer(n) => Expr::Integer(-n),
    Expr::Real(f) => Expr::Real(-f),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => *operand.clone(),
    _ => Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(t.clone()),
    },
  }
}

/// Expand (sum)^n by repeated distribution.
fn expand_power(base: &Expr, n: i128) -> Expr {
  if n == 0 {
    return Expr::Integer(1);
  }
  if n == 1 {
    return base.clone();
  }
  // Repeated multiplication
  let mut result = base.clone();
  for _ in 1..n {
    result = distribute_product(&result, base);
    // Combine like terms to keep expression manageable
    let terms = collect_additive_terms(&result);
    result = combine_and_build(terms);
  }
  result
}

/// Build a sum (BinaryOp::Plus chain) from terms.
fn build_sum(terms: Vec<Expr>) -> Expr {
  if terms.is_empty() {
    return Expr::Integer(0);
  }
  let mut iter = terms.into_iter();
  let mut result = iter.next().unwrap();
  for t in iter {
    // Handle negative terms: a + (-b) stays as BinaryOp::Plus with UnaryOp::Minus
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(t),
    };
  }
  result
}

/// Combine like terms and sort, then build the final expression.
fn combine_and_build(terms: Vec<Expr>) -> Expr {
  // Represent each term as (key, coefficient) where key identifies the "variable part"
  let mut term_map: Vec<(String, Vec<Expr>, Expr)> = Vec::new(); // (sort_key, var_factors, coeff)

  for term in &terms {
    let (coeff, var_key, var_factors) = decompose_term(term);
    // Find existing entry
    if let Some(entry) = term_map.iter_mut().find(|(k, _, _)| *k == var_key) {
      entry.2 = add_exprs(&entry.2, &coeff);
    } else {
      term_map.push((var_key, var_factors, coeff));
    }
  }

  // Sort terms using Wolfram's canonical ordering:
  // Reverse-variable lexicographic ascending — sort by last variable ascending,
  // then next-to-last ascending, etc. Constants come first naturally (all exponents 0).
  term_map.sort_by(|(ka, va, _), (kb, vb, _)| {
    // Constants first
    match (ka.is_empty(), kb.is_empty()) {
      (true, true) => return std::cmp::Ordering::Equal,
      (true, false) => return std::cmp::Ordering::Less,
      (false, true) => return std::cmp::Ordering::Greater,
      _ => {}
    }
    let ea = extract_exponent_map(va);
    let eb = extract_exponent_map(vb);
    // Collect all variable names, sort alphabetically
    let mut all_vars: Vec<&String> = ea.keys().chain(eb.keys()).collect();
    all_vars.sort();
    all_vars.dedup();
    // Compare from LAST variable ascending, then next-to-last ascending, etc.
    for var in all_vars.iter().rev() {
      let pa = ea.get(*var).copied().unwrap_or(0);
      let pb = eb.get(*var).copied().unwrap_or(0);
      if pa != pb {
        return pa.cmp(&pb); // ascending
      }
    }
    std::cmp::Ordering::Equal
  });

  // Build result terms
  let mut result_terms: Vec<Expr> = Vec::new();
  for (_, var_factors, coeff) in term_map {
    let coeff = simplify(coeff);
    if matches!(&coeff, Expr::Integer(0)) {
      continue; // skip zero terms
    }
    if var_factors.is_empty() {
      // Constant term
      result_terms.push(coeff);
    } else if matches!(&coeff, Expr::Integer(1)) {
      // Coefficient is 1, just use the variable part
      let var_expr = build_product(var_factors);
      result_terms.push(var_expr);
    } else if matches!(&coeff, Expr::Integer(-1)) {
      let var_expr = build_product(var_factors);
      result_terms.push(negate_term(&var_expr));
    } else {
      let var_expr = build_product(var_factors);
      result_terms.push(multiply_exprs(&coeff, &var_expr));
    }
  }

  if result_terms.is_empty() {
    Expr::Integer(0)
  } else {
    build_sum(result_terms)
  }
}

/// Decompose a term into (numeric_coefficient, sort_key, variable_factors).
/// E.g. 3*x^2*y → (3, "x^2*y", [x^2, y])
///      -x → (-1, "x^1", [x])
///      5 → (5, "", [])
fn decompose_term(term: &Expr) -> (Expr, String, Vec<Expr>) {
  match term {
    Expr::Integer(_) | Expr::Real(_) => (term.clone(), String::new(), vec![]),
    Expr::Identifier(_) => {
      (Expr::Integer(1), expr_to_string(term), vec![term.clone()])
    }
    Expr::Constant(_) => {
      (Expr::Integer(1), expr_to_string(term), vec![term.clone()])
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (c, k, v) = decompose_term(operand);
      (negate_term(&c), k, v)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    } => {
      let factors = collect_multiplicative_factors(term);
      let mut numeric_coeff = Expr::Integer(1);
      let mut var_factors: Vec<Expr> = Vec::new();

      for f in &factors {
        match f {
          Expr::Integer(_) | Expr::Real(_) => {
            numeric_coeff = multiply_exprs(&numeric_coeff, f);
          }
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } => {
            numeric_coeff = negate_term(&numeric_coeff);
            match operand.as_ref() {
              Expr::Integer(_) | Expr::Real(_) => {
                numeric_coeff = multiply_exprs(&numeric_coeff, operand);
              }
              _ => var_factors.push(*operand.clone()),
            }
          }
          _ => var_factors.push(f.clone()),
        }
      }

      // Sort variable factors for canonical key
      var_factors.sort_by_key(expr_to_string);
      let key = var_factors
        .iter()
        .map(expr_to_string)
        .collect::<Vec<_>>()
        .join("*");
      (numeric_coeff, key, var_factors)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      ..
    } => (Expr::Integer(1), expr_to_string(term), vec![term.clone()]),
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut numeric_coeff = Expr::Integer(1);
      let mut var_factors: Vec<Expr> = Vec::new();

      for f in args {
        match f {
          Expr::Integer(_) | Expr::Real(_) => {
            numeric_coeff = multiply_exprs(&numeric_coeff, f);
          }
          _ => var_factors.push(f.clone()),
        }
      }

      var_factors.sort_by_key(expr_to_string);
      let key = var_factors
        .iter()
        .map(expr_to_string)
        .collect::<Vec<_>>()
        .join("*");
      (numeric_coeff, key, var_factors)
    }
    _ => (Expr::Integer(1), expr_to_string(term), vec![term.clone()]),
  }
}

/// Collect multiplicative factors from nested Times.
fn collect_multiplicative_factors(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let mut factors = collect_multiplicative_factors(left);
      factors.extend(collect_multiplicative_factors(right));
      factors
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut factors = Vec::new();
      for a in args {
        factors.extend(collect_multiplicative_factors(a));
      }
      factors
    }
    _ => vec![expr.clone()],
  }
}

/// Build a product from factors.
fn build_product(factors: Vec<Expr>) -> Expr {
  if factors.is_empty() {
    return Expr::Integer(1);
  }
  let mut iter = factors.into_iter();
  let mut result = iter.next().unwrap();
  for f in iter {
    result = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(result),
      right: Box::new(f),
    };
  }
  result
}

// ─── Simplify ───────────────────────────────────────────────────────

/// Simplify[expr] - User-facing simplification
pub fn simplify_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Simplify expects exactly 1 argument".into(),
    ));
  }
  Ok(simplify_expr(&args[0]))
}

/// Full simplification: expand, combine like terms, simplify.
fn simplify_expr(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::Constant(_)
    | Expr::Identifier(_) => expr.clone(),

    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let num = simplify_expr(left);
      let den = simplify_expr(right);
      // Try to cancel: expand both and see if we can simplify
      simplify_division(&num, &den)
    }

    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let base = simplify_expr(left);
      let exp = simplify_expr(right);
      simplify(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(exp),
      })
    }

    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let l = simplify_expr(left);
      let r = simplify_expr(right);
      // Combine powers: x * x → x^2, x^a * x^b → x^(a+b)
      simplify_product(&l, &r)
    }

    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus,
      ..
    } => {
      let combined = expand_and_combine(expr);
      apply_trig_identities(&combined)
    }

    Expr::UnaryOp { op, operand } => {
      let inner = simplify_expr(operand);
      simplify(Expr::UnaryOp {
        op: *op,
        operand: Box::new(inner),
      })
    }

    // Handle FunctionCall forms of Plus, Times, Power
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let combined = expand_and_combine(expr);
        apply_trig_identities(&combined)
      }
      "Times" if args.len() == 2 => {
        let l = simplify_expr(&args[0]);
        let r = simplify_expr(&args[1]);
        simplify_product(&l, &r)
      }
      "Power" if args.len() == 2 => {
        let base = simplify_expr(&args[0]);
        let exp = simplify_expr(&args[1]);
        simplify(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(base),
          right: Box::new(exp),
        })
      }
      "Rational" if args.len() == 2 => expr.clone(),
      "ConditionalExpression" if args.len() == 2 => {
        simplify_conditional_expression(&args[0], &args[1])
      }
      _ => expr.clone(),
    },

    _ => simplify(expr.clone()),
  }
}

/// Simplify ConditionalExpression[value, cond] under current $Assumptions.
/// If cond matches $Assumptions → return Simplify[value]
/// If $Assumptions negates cond → return Undefined
/// Otherwise → ConditionalExpression[Simplify[value], cond]
fn simplify_conditional_expression(value: &Expr, cond: &Expr) -> Expr {
  let cond_str = expr_to_string(cond);

  // Get $Assumptions from environment (default: "True")
  let assumptions_str = crate::ENV
    .with(|e| {
      e.borrow().get("$Assumptions").map(|sv| match sv {
        crate::StoredValue::Raw(s) => s.clone(),
        crate::StoredValue::ExprVal(e) => expr_to_string(e),
        _ => "True".to_string(),
      })
    })
    .unwrap_or_else(|| "True".to_string());

  if cond_str == assumptions_str {
    // Condition matches assumptions → strip ConditionalExpression
    simplify_expr(value)
  } else if assumptions_str == format!("!{}", cond_str)
    || assumptions_str == format!("Not[{}]", cond_str)
  {
    // Assumptions negate the condition → Undefined
    Expr::Identifier("Undefined".to_string())
  } else {
    // Keep ConditionalExpression with simplified value
    Expr::FunctionCall {
      name: "ConditionalExpression".to_string(),
      args: vec![simplify_expr(value), cond.clone()],
    }
  }
}

/// Apply trigonometric identities to a sum expression.
/// Detects a*Sin[x]^2 + a*Cos[x]^2 → a and similar patterns.
fn apply_trig_identities(expr: &Expr) -> Expr {
  let terms = collect_additive_terms(expr);
  if terms.len() < 2 {
    return expr.clone();
  }

  // Look for pairs: coeff*Sin[arg]^2 + coeff*Cos[arg]^2 → coeff
  let mut used = vec![false; terms.len()];
  let mut result_terms: Vec<Expr> = Vec::new();

  for i in 0..terms.len() {
    if used[i] {
      continue;
    }
    if let Some((coeff_i, arg_i, is_sin_i)) = extract_trig_squared(&terms[i]) {
      // Look for matching pair
      for j in (i + 1)..terms.len() {
        if used[j] {
          continue;
        }
        if let Some((coeff_j, arg_j, is_sin_j)) =
          extract_trig_squared(&terms[j])
          && is_sin_i != is_sin_j
          && expr_to_string(&arg_i) == expr_to_string(&arg_j)
          && expr_to_string(&coeff_i) == expr_to_string(&coeff_j)
        {
          // Found matching pair: coeff*Sin[x]^2 + coeff*Cos[x]^2 = coeff
          result_terms.push(coeff_i.clone());
          used[i] = true;
          used[j] = true;
          break;
        }
      }
    }
    if !used[i] {
      result_terms.push(terms[i].clone());
    }
  }

  if result_terms.len() == terms.len() {
    // No simplification happened
    return expr.clone();
  }

  // Re-combine to simplify (e.g. 1 + 1 → 2)
  if let Ok(result) = crate::functions::math_ast::plus_ast(&result_terms) {
    result
  } else {
    build_sum(result_terms)
  }
}

/// Try to extract (coefficient, argument, is_sin) from a term like coeff*Sin[arg]^2 or coeff*Cos[arg]^2.
fn extract_trig_squared(term: &Expr) -> Option<(Expr, Expr, bool)> {
  // Pattern: Sin[arg]^2 or Cos[arg]^2 (coefficient = 1)
  if let Some((func, arg)) = match_trig_squared(term) {
    return Some((Expr::Integer(1), arg, func == "Sin"));
  }

  // Pattern: coeff * Sin[arg]^2 or coeff * Cos[arg]^2
  let factors = collect_multiplicative_factors(term);
  if factors.len() < 2 {
    return None;
  }

  // Find the trig^2 factor
  for (idx, f) in factors.iter().enumerate() {
    if let Some((func, arg)) = match_trig_squared(f) {
      let mut coeff_factors: Vec<Expr> = Vec::new();
      for (j, g) in factors.iter().enumerate() {
        if j != idx {
          coeff_factors.push(g.clone());
        }
      }
      let coeff = if coeff_factors.len() == 1 {
        coeff_factors.remove(0)
      } else {
        build_product(coeff_factors)
      };
      return Some((coeff, arg, func == "Sin"));
    }
  }
  None
}

/// Match Sin[arg]^2 or Cos[arg]^2, returning ("Sin"/"Cos", arg).
fn match_trig_squared(expr: &Expr) -> Option<(&str, Expr)> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } if matches!(right.as_ref(), Expr::Integer(2)) => {
      if let Expr::FunctionCall { name, args } = left.as_ref()
        && (name == "Sin" || name == "Cos")
        && args.len() == 1
      {
        return Some((name.as_str(), args[0].clone()));
      }
      None
    }
    _ => None,
  }
}

/// Simplify a product, combining powers.
fn simplify_product(a: &Expr, b: &Expr) -> Expr {
  // x * x → x^2
  if expr_to_string(a) == expr_to_string(b) {
    return Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(a.clone()),
      right: Box::new(Expr::Integer(2)),
    };
  }

  // x^a * x^b → x^(a+b)
  let (base_a, exp_a) = extract_base_exp(a);
  let (base_b, exp_b) = extract_base_exp(b);
  if expr_to_string(&base_a) == expr_to_string(&base_b) {
    let new_exp = simplify(Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(exp_a),
      right: Box::new(exp_b),
    });
    return simplify(Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(base_a),
      right: Box::new(new_exp),
    });
  }

  simplify(Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(a.clone()),
    right: Box::new(b.clone()),
  })
}

/// Extract base and exponent from a power expression.
fn extract_base_exp(expr: &Expr) -> (Expr, Expr) {
  extract_base_and_exp(expr)
}

/// Simplify a division by trying polynomial cancellation.
fn simplify_division(num: &Expr, den: &Expr) -> Expr {
  // If same expression, return 1
  if expr_to_string(num) == expr_to_string(den) {
    return Expr::Integer(1);
  }

  // Try: if denominator is a single factor, try polynomial division
  // E.g. (x^2 - 1) / (x - 1) → x + 1
  // We try to do this by expanding the numerator and attempting polynomial long division
  let num_expanded = expand_and_combine(num);
  let den_expanded = expand_and_combine(den);

  // Try to find the variable
  if let Some(var) = find_single_variable(&num_expanded)
    && let Some(quotient) =
      poly_divide_single_var(&num_expanded, &den_expanded, &var)
  {
    return quotient;
  }

  simplify(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(num_expanded),
    right: Box::new(den_expanded),
  })
}

/// Find a single variable in an expression (for univariate polynomial division).
fn find_single_variable(expr: &Expr) -> Option<String> {
  let mut vars = std::collections::HashSet::new();
  collect_variables(expr, &mut vars);
  if vars.len() == 1 {
    vars.into_iter().next()
  } else {
    None
  }
}

/// Collect all variable names from an expression.
fn collect_variables(
  expr: &Expr,
  vars: &mut std::collections::HashSet<String>,
) {
  match expr {
    Expr::Identifier(name)
      if name != "True" && name != "False" && name != "Null" =>
    {
      vars.insert(name.clone());
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_variables(left, vars);
      collect_variables(right, vars);
    }
    Expr::UnaryOp { operand, .. } => collect_variables(operand, vars),
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_variables(a, vars);
      }
    }
    Expr::List(items) => {
      for i in items {
        collect_variables(i, vars);
      }
    }
    _ => {}
  }
}

/// Try polynomial long division of num/den in a single variable.
/// Returns Some(quotient) if den divides num exactly.
fn poly_divide_single_var(num: &Expr, den: &Expr, var: &str) -> Option<Expr> {
  let num_coeffs = extract_poly_coeffs(num, var)?;
  let den_coeffs = extract_poly_coeffs(den, var)?;

  if den_coeffs.is_empty() {
    return None;
  }

  let num_deg = num_coeffs.len() as i128 - 1;
  let den_deg = den_coeffs.len() as i128 - 1;

  if num_deg < den_deg {
    return None;
  }

  // Polynomial long division with integer/rational coefficients
  let mut remainder = num_coeffs.clone();
  let mut quotient = vec![0i128; (num_deg - den_deg + 1) as usize];
  let lead_den = *den_coeffs.last()?;

  if lead_den == 0 {
    return None;
  }

  for i in (0..quotient.len()).rev() {
    let rem_idx = i + den_coeffs.len() - 1;
    if rem_idx >= remainder.len() {
      continue;
    }
    if remainder[rem_idx] % lead_den != 0 {
      return None; // Not exactly divisible with integers
    }
    let q = remainder[rem_idx] / lead_den;
    quotient[i] = q;
    for j in 0..den_coeffs.len() {
      remainder[i + j] -= q * den_coeffs[j];
    }
  }

  // Check remainder is zero
  if remainder.iter().any(|&c| c != 0) {
    return None;
  }

  // Build quotient polynomial
  Some(coeffs_to_expr(&quotient, var))
}

/// Extract integer polynomial coefficients from expr, indexed by power.
/// coeffs[i] = coefficient of var^i
fn extract_poly_coeffs(expr: &Expr, var: &str) -> Option<Vec<i128>> {
  let terms = collect_additive_terms(expr);
  let mut max_pow: i128 = 0;
  let mut term_data: Vec<(i128, i128)> = Vec::new(); // (power, integer_coeff)

  for term in &terms {
    let (power, coeff) = term_var_power_and_coeff(term, var);
    if power < 0 {
      return None; // non-polynomial term
    }
    let int_coeff = match &simplify(coeff) {
      Expr::Integer(n) => *n,
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => {
        if let Expr::Integer(n) = operand.as_ref() {
          -n
        } else {
          return None;
        }
      }
      _ => return None, // non-integer coefficient
    };
    max_pow = max_pow.max(power);
    term_data.push((power, int_coeff));
  }

  let mut coeffs = vec![0i128; (max_pow + 1) as usize];
  for (power, c) in term_data {
    coeffs[power as usize] += c;
  }

  Some(coeffs)
}

/// Build a polynomial expression from integer coefficients.
/// coeffs[i] = coefficient of var^i
fn coeffs_to_expr(coeffs: &[i128], var: &str) -> Expr {
  let mut terms: Vec<Expr> = Vec::new();

  for (i, &c) in coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let var_part = if i == 0 {
      None
    } else if i == 1 {
      Some(Expr::Identifier(var.to_string()))
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier(var.to_string())),
        right: Box::new(Expr::Integer(i as i128)),
      })
    };

    let term = match (c, var_part) {
      (c, None) => Expr::Integer(c),
      (1, Some(v)) => v,
      (-1, Some(v)) => negate_term(&v),
      (c, Some(v)) => Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(c)),
        right: Box::new(v),
      },
    };
    terms.push(term);
  }

  if terms.is_empty() {
    Expr::Integer(0)
  } else {
    build_sum(terms)
  }
}

// ─── Factor ─────────────────────────────────────────────────────────

/// Factor[expr] - Factor a polynomial expression
/// Threads over List.
pub fn factor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Factor expects exactly 1 argument".into(),
    ));
  }

  // Thread over List
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| factor_ast(&[item.clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }

  // First expand to canonical form
  let expanded = expand_and_combine(&args[0]);

  // Try to find the variable
  let var = match find_single_variable(&expanded) {
    Some(v) => v,
    None => return Ok(expanded), // multivariate or constant — return as is
  };

  // Extract integer coefficients
  let coeffs = match extract_poly_coeffs(&expanded, &var) {
    Some(c) => c,
    None => return Ok(expanded), // non-integer coefficients
  };

  // Factor out GCD of coefficients
  let gcd_coeff = coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if gcd_coeff == 0 {
    return Ok(Expr::Integer(0));
  }

  let reduced_coeffs: Vec<i128> =
    coeffs.iter().map(|c| c / gcd_coeff).collect();

  // Factor out leading negative
  let (sign, reduced_coeffs) =
    if reduced_coeffs.last().map(|&c| c < 0).unwrap_or(false) {
      (
        -1i128,
        reduced_coeffs.iter().map(|c| -c).collect::<Vec<_>>(),
      )
    } else {
      (1, reduced_coeffs)
    };
  let overall = gcd_coeff * sign;

  // Factor the monic-ish polynomial
  let factors = factor_integer_poly(&reduced_coeffs, &var);

  if factors.len() <= 1 && overall == 1 {
    // Couldn't factor further
    return Ok(expanded);
  }

  // Build result: overall * factor1 * factor2 * ...
  let mut result_factors: Vec<Expr> = Vec::new();

  if overall != 1 {
    result_factors.push(Expr::Integer(overall));
  }

  if factors.is_empty() {
    result_factors.push(coeffs_to_expr(&reduced_coeffs, &var));
  } else {
    // Group identical factors: (1+x)*(1+x) → (1+x)^2
    let mut grouped: Vec<(Expr, i128)> = Vec::new();
    for f in &factors {
      let key = expr_to_string(f);
      if let Some(entry) =
        grouped.iter_mut().find(|(e, _)| expr_to_string(e) == key)
      {
        entry.1 += 1;
      } else {
        grouped.push((f.clone(), 1));
      }
    }
    for (factor, count) in grouped {
      if count == 1 {
        result_factors.push(factor);
      } else {
        result_factors.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(factor),
          right: Box::new(Expr::Integer(count)),
        });
      }
    }
  }

  if result_factors.len() == 1 {
    return Ok(result_factors.remove(0));
  }

  Ok(build_product(result_factors))
}

/// GCD of two i128 values.
fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Factor an integer polynomial given as coefficients.
/// Returns a list of factor expressions, or empty if can't factor.
fn factor_integer_poly(coeffs: &[i128], var: &str) -> Vec<Expr> {
  // Remove trailing zeros (factor out x^k)
  let leading_zeros = coeffs.iter().take_while(|&&c| c == 0).count();
  let trimmed = &coeffs[leading_zeros..];

  let mut factors: Vec<Expr> = Vec::new();

  // Add x^k factor if leading zeros
  if leading_zeros > 0 {
    if leading_zeros == 1 {
      factors.push(Expr::Identifier(var.to_string()));
    } else {
      factors.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier(var.to_string())),
        right: Box::new(Expr::Integer(leading_zeros as i128)),
      });
    }
  }

  if trimmed.len() <= 1 {
    // Constant or empty
    if trimmed.len() == 1 && trimmed[0] != 1 {
      factors.push(Expr::Integer(trimmed[0]));
    }
    return factors;
  }

  // Try to find rational roots and factor
  let mut remaining = trimmed.to_vec();

  loop {
    if remaining.len() <= 1 {
      break;
    }
    if remaining.len() == 2 {
      // Linear: ax + b → (b + a*x) but represented as canonical form
      factors.push(linear_to_expr(remaining[0], remaining[1], var));
      break;
    }

    // Try rational root theorem
    match find_integer_root(&remaining) {
      Some(root) => {
        // Factor out (x - root) which in canonical form is (root_neg + x) → (-root + x)
        factors.push(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(-root)),
          right: Box::new(Expr::Identifier(var.to_string())),
        });
        remaining = divide_by_root(&remaining, root);
      }
      None => {
        // Can't find more integer roots — try polynomial trial division
        let sub_factors = try_factor_no_rational_roots(&remaining, var);
        if sub_factors.is_empty() {
          if remaining != [1] {
            factors.push(coeffs_to_expr(&remaining, var));
          }
        } else {
          factors.extend(sub_factors);
        }
        break;
      }
    }
  }

  // Sort factors by: constant term, then degree, then string representation
  factors.sort_by(|a, b| {
    let ca = factor_constant_term(a);
    let cb = factor_constant_term(b);
    ca.cmp(&cb)
      .then_with(|| factor_degree(a).cmp(&factor_degree(b)))
      .then_with(|| {
        // For same degree, sort by number of terms (fewer terms first)
        factor_term_count(a).cmp(&factor_term_count(b))
      })
      .then_with(|| {
        // For same degree and term count, sort by first non-constant coefficient
        factor_first_nonconst_coeff(a).cmp(&factor_first_nonconst_coeff(b))
      })
      .then_with(|| expr_to_string(a).cmp(&expr_to_string(b)))
  });

  factors
}

/// Get the constant term of a factor expression for sorting.
/// Recursively descends into the leftmost leaf of Plus chains.
fn factor_constant_term(expr: &Expr) -> i128 {
  match expr {
    Expr::Integer(n) => *n,
    Expr::Identifier(_) => 0, // x has constant term 0
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      ..
    } => {
      // Recurse into left child to find the constant term
      factor_constant_term(left)
    }
    _ => 0,
  }
}

/// Get the degree of a factor expression for sorting.
fn factor_degree(expr: &Expr) -> usize {
  let s = expr_to_string(expr);
  // Count the highest power of the variable
  // Look for x^N patterns
  let mut max_deg = 0usize;
  for cap in s.split('^') {
    // After '^', try to parse the number
    if let Ok(n) = cap
      .chars()
      .take_while(|c| c.is_ascii_digit())
      .collect::<String>()
      .parse::<usize>()
      && n > max_deg
    {
      max_deg = n;
    }
  }
  // If no power found but contains a variable, degree is 1
  if max_deg == 0 && s.chars().any(|c| c.is_ascii_lowercase()) {
    max_deg = 1;
  }
  max_deg
}

/// Count the number of additive terms in a factor expression for sorting.
fn factor_term_count(expr: &Expr) -> usize {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => factor_term_count(left) + factor_term_count(right),
    _ => 1,
  }
}

/// Get the first non-constant coefficient of a factor expression for sorting.
/// This finds the coefficient of the lowest-degree non-constant term.
fn factor_first_nonconst_coeff(expr: &Expr) -> i128 {
  // Convert the expression back to string and parse the first non-constant term's sign
  let s = expr_to_string(expr);
  // Find the first term with a variable (after the constant)
  // Parse the sign: look for first occurrence of the variable
  // The pattern will be like "1 + x", "1 - x", "1 + x^5", "1 - x^5"
  // After "1", look for " + " or " - " before the variable
  let parts: Vec<&str> =
    s.splitn(2, |c: char| c.is_ascii_lowercase()).collect();
  if parts.len() < 2 {
    return 0;
  }
  let prefix = parts[0].trim_end();
  if prefix.ends_with('+') || prefix.ends_with("+ ") {
    1
  } else if prefix.ends_with('-') || prefix.ends_with("- ") {
    -1
  } else if prefix.ends_with('*') {
    // coefficient * variable — extract the number before *
    let before_star = prefix.trim_end_matches('*').trim();
    // Find the last number in this prefix
    let num_str: String = before_star
      .chars()
      .rev()
      .take_while(|c| c.is_ascii_digit() || *c == '-')
      .collect::<String>()
      .chars()
      .rev()
      .collect();
    num_str.parse().unwrap_or(0)
  } else {
    0
  }
}

/// Build a linear expression from coefficients: c0 + c1*x
fn linear_to_expr(c0: i128, c1: i128, var: &str) -> Expr {
  if c1 == 1 {
    if c0 == 0 {
      Expr::Identifier(var.to_string())
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(c0)),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    }
  } else {
    let var_part = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(c1)),
      right: Box::new(Expr::Identifier(var.to_string())),
    };
    if c0 == 0 {
      var_part
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(c0)),
        right: Box::new(var_part),
      }
    }
  }
}

/// Find an integer root of polynomial with given coefficients.
/// Uses rational root theorem: possible roots are ±(factors of c0) / (factors of leading coeff).
fn find_integer_root(coeffs: &[i128]) -> Option<i128> {
  if coeffs.is_empty() {
    return None;
  }
  let c0 = coeffs[0]; // constant term
  let lead = coeffs[coeffs.len() - 1]; // leading coefficient

  if c0 == 0 {
    return Some(0);
  }

  // Get divisors of c0 and lead
  let c0_divs = integer_divisors(c0.abs());
  let lead_divs = integer_divisors(lead.abs());

  for &p in &c0_divs {
    for &q in &lead_divs {
      // Try root = p/q (only if it's an integer)
      if p % q == 0 {
        let root = p / q;
        if evaluate_poly(coeffs, root) == 0 {
          return Some(root);
        }
        if evaluate_poly(coeffs, -root) == 0 {
          return Some(-root);
        }
      }
    }
  }
  None
}

/// Evaluate polynomial at integer x.
fn evaluate_poly(coeffs: &[i128], x: i128) -> i128 {
  let mut result: i128 = 0;
  let mut power: i128 = 1;
  for &c in coeffs {
    result = result
      .checked_add(c.checked_mul(power).unwrap_or(i128::MAX))
      .unwrap_or(i128::MAX);
    if result == i128::MAX {
      return result; // overflow guard
    }
    power = power.checked_mul(x).unwrap_or(i128::MAX);
  }
  result
}

/// Get all positive divisors of n.
fn integer_divisors(n: i128) -> Vec<i128> {
  if n == 0 {
    return vec![1];
  }
  let n = n.abs();
  let mut divs = Vec::new();
  let mut i = 1i128;
  while i * i <= n {
    if n % i == 0 {
      divs.push(i);
      if i != n / i {
        divs.push(n / i);
      }
    }
    i += 1;
  }
  divs.sort();
  divs
}

/// Divide polynomial by (x - root) using synthetic division.
fn divide_by_root(coeffs: &[i128], root: i128) -> Vec<i128> {
  // coeffs[i] = coefficient of x^i, so coeffs = [c0, c1, c2, ...]
  // We need to divide by (x - root)
  let n = coeffs.len();
  if n <= 1 {
    return vec![];
  }
  let mut result = vec![0i128; n - 1];
  // Synthetic division: work from highest power down
  result[n - 2] = coeffs[n - 1];
  for i in (0..n - 2).rev() {
    result[i] = coeffs[i + 1] + root * result[i + 1];
  }
  result
}

/// Polynomial long division: divide `num` by `den`.
/// Returns (quotient, remainder) as coefficient vectors.
/// Coefficients are stored as [c0, c1, c2, ...] (c_i is coefficient of x^i).
/// Only works when division is exact over integers (remainder should be zero for our use).
fn poly_div(num: &[i128], den: &[i128]) -> Option<(Vec<i128>, Vec<i128>)> {
  let n = num.len();
  let m = den.len();
  if m == 0 || m > n {
    return None;
  }
  let lead_den = *den.last().unwrap();
  if lead_den == 0 {
    return None;
  }

  let mut rem = num.to_vec();
  let quot_len = n - m + 1;
  let mut quot = vec![0i128; quot_len];

  for i in (0..quot_len).rev() {
    if rem[i + m - 1] % lead_den != 0 {
      // Not exactly divisible over integers
      return None;
    }
    let coeff = rem[i + m - 1] / lead_den;
    quot[i] = coeff;
    for j in 0..m {
      rem[i + j] -= coeff * den[j];
    }
  }

  // Trim trailing zeros from remainder
  while rem.len() > 1 && *rem.last().unwrap() == 0 {
    rem.pop();
  }

  Some((quot, rem))
}

/// Compute the n-th cyclotomic polynomial as integer coefficients.
/// Φ_1(x) = x - 1
/// Φ_n(x) = (x^n - 1) / ∏_{d|n, d<n} Φ_d(x)
fn cyclotomic_poly(n: u64) -> Vec<i128> {
  if n == 1 {
    return vec![-1, 1]; // x - 1
  }

  // Start with x^n - 1
  let mut result = vec![0i128; (n + 1) as usize];
  result[0] = -1;
  result[n as usize] = 1;

  // Divide by Φ_d(x) for all proper divisors d of n
  let divs = divisors_of(n);
  for d in divs {
    if d < n {
      let phi_d = cyclotomic_poly(d);
      if let Some((q, _)) = poly_div(&result, &phi_d) {
        result = q;
      }
    }
  }

  result
}

/// Get all divisors of n in sorted order.
fn divisors_of(n: u64) -> Vec<u64> {
  let mut divs = Vec::new();
  let mut i = 1u64;
  while i * i <= n {
    if n.is_multiple_of(i) {
      divs.push(i);
      if i != n / i {
        divs.push(n / i);
      }
    }
    i += 1;
  }
  divs.sort();
  divs
}

/// Try to factor a polynomial (with no rational roots) using polynomial trial division.
/// First tries cyclotomic factors, then Kronecker-style trial division for small degrees.
fn try_factor_no_rational_roots(coeffs: &[i128], var: &str) -> Vec<Expr> {
  let deg = coeffs.len() - 1;
  if deg <= 1 {
    return vec![];
  }

  // Try cyclotomic polynomial division:
  // Check divisors up to reasonable size
  let mut remaining = coeffs.to_vec();
  let mut factors: Vec<Vec<i128>> = Vec::new();

  // Try cyclotomic polynomials Φ_n for n up to 2*deg
  // (any cyclotomic factor of a degree-d poly has n ≤ some bound)
  let max_n = (2 * deg) as u64;
  // Collect cyclotomic polys sorted by degree (ascending) for greedy factoring
  let mut cyclo_candidates: Vec<(u64, Vec<i128>)> = Vec::new();
  for n in 2..=max_n {
    let cp = cyclotomic_poly(n);
    if cp.len() - 1 <= deg {
      cyclo_candidates.push((n, cp));
    }
  }
  // Sort by degree so we try smaller factors first
  cyclo_candidates.sort_by_key(|(_, c)| c.len());

  let mut changed = true;
  while changed {
    changed = false;
    for (_n, cp) in &cyclo_candidates {
      if cp.len() > remaining.len() {
        continue;
      }
      if let Some((q, rem)) = poly_div(&remaining, cp)
        && rem.iter().all(|&c| c == 0)
      {
        factors.push(cp.clone());
        remaining = q;
        changed = true;
        // Trim trailing zeros
        while remaining.len() > 1 && *remaining.last().unwrap() == 0 {
          remaining.pop();
        }
        if remaining.len() <= 1 {
          break;
        }
      }
    }
    if remaining.len() <= 1 {
      break;
    }
  }

  if factors.is_empty() {
    // No cyclotomic factors found — try Kronecker's method for small-degree polynomials
    return try_kronecker_factor(coeffs, var);
  }

  // We found cyclotomic factors; handle the remainder
  if remaining.len() > 2 {
    // Try to factor the remainder further with Kronecker
    let sub = try_kronecker_factor(&remaining, var);
    if sub.is_empty() {
      factors.push(remaining);
    } else {
      return factors
        .iter()
        .map(|f| coeffs_to_expr(f, var))
        .chain(sub)
        .collect();
    }
  } else if remaining.len() > 1 || (remaining.len() == 1 && remaining[0] != 1) {
    factors.push(remaining);
  }

  factors.iter().map(|f| coeffs_to_expr(f, var)).collect()
}

/// Kronecker's method for factoring integer polynomials of small degree.
/// Try all monic integer polynomial divisors of degree 2..=deg/2
/// by evaluating at enough points to determine the candidate factor.
fn try_kronecker_factor(coeffs: &[i128], var: &str) -> Vec<Expr> {
  let deg = coeffs.len() - 1;
  if deg <= 3 {
    // For degree 2-3, if no rational roots exist, it's irreducible over Z
    return vec![];
  }

  // For degree 4+, try factoring as product of two polynomials
  // Evaluate at points 0, 1, -1, 2, -2, ...
  let max_trial_degree = deg / 2;
  if max_trial_degree > 20 {
    // Too expensive for Kronecker
    return vec![];
  }

  // Evaluate the polynomial at points 0, 1, -1, 2, -2, ... (need max_trial_degree + 1 points)
  let num_points = max_trial_degree + 1;
  let eval_points: Vec<i128> = (0..num_points as i128)
    .flat_map(|i| if i == 0 { vec![0] } else { vec![i, -i] })
    .take(num_points)
    .collect();

  let poly_vals: Vec<i128> = eval_points
    .iter()
    .map(|&x| evaluate_poly(coeffs, x))
    .collect();

  // For each trial degree d (2..=max_trial_degree), try to find a factor
  for trial_deg in 2..=max_trial_degree {
    let n_pts = trial_deg + 1;
    // Get divisors of poly value at each eval point
    let mut divisor_sets: Vec<Vec<i128>> = Vec::new();
    for i in 0..n_pts {
      let val = poly_vals[i];
      if val == 0 {
        // The trial factor also evaluates to 0 at this point
        divisor_sets.push(vec![0]);
      } else {
        let abs_divs = integer_divisors(val.abs());
        let mut divs: Vec<i128> = Vec::new();
        for &d in &abs_divs {
          divs.push(d);
          divs.push(-d);
        }
        divisor_sets.push(divs);
      }
    }

    // Try all combinations of divisor values and interpolate
    let combos = cartesian_product(&divisor_sets);
    for combo in combos {
      // Interpolate: find a polynomial of degree trial_deg with values combo at eval_points
      if let Some(factor_coeffs) =
        lagrange_interpolate_integer(&eval_points[..n_pts], &combo)
      {
        // Check if leading coeff is ±1 (monic or neg-monic) and degree matches
        if factor_coeffs.len() != trial_deg + 1 {
          continue;
        }
        let lead = *factor_coeffs.last().unwrap();
        if lead == 0 {
          continue;
        }

        // Try dividing
        if let Some((q, rem)) = poly_div(coeffs, &factor_coeffs)
          && rem.iter().all(|&c| c == 0)
        {
          // Found a factor! Recursively factor both parts
          let mut result = Vec::new();
          let sub1 = factor_sub_poly(&factor_coeffs, var);
          let sub2 = factor_sub_poly(&q, var);
          result.extend(sub1);
          result.extend(sub2);
          return result;
        }
      }
    }
  }

  vec![]
}

/// Factor a sub-polynomial by trying rational roots first, then trial division.
fn factor_sub_poly(coeffs: &[i128], var: &str) -> Vec<Expr> {
  if coeffs.len() <= 1 {
    if coeffs.len() == 1 && coeffs[0] != 1 {
      return vec![Expr::Integer(coeffs[0])];
    }
    return vec![];
  }
  if coeffs.len() == 2 {
    return vec![linear_to_expr(coeffs[0], coeffs[1], var)];
  }

  // Try rational roots first
  let mut remaining = coeffs.to_vec();
  let mut factors: Vec<Expr> = Vec::new();
  loop {
    match find_integer_root(&remaining) {
      Some(root) => {
        factors.push(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(-root)),
          right: Box::new(Expr::Identifier(var.to_string())),
        });
        remaining = divide_by_root(&remaining, root);
        if remaining.len() <= 1 {
          break;
        }
        if remaining.len() == 2 {
          factors.push(linear_to_expr(remaining[0], remaining[1], var));
          return factors;
        }
      }
      None => break,
    }
  }

  if remaining.len() > 1 {
    // Irreducible remainder
    factors.push(coeffs_to_expr(&remaining, var));
  }
  factors
}

/// Compute cartesian product of divisor sets (with size limit to avoid explosion).
fn cartesian_product(sets: &[Vec<i128>]) -> Vec<Vec<i128>> {
  if sets.is_empty() {
    return vec![vec![]];
  }

  // Limit total combinations to avoid exponential blowup
  let total: usize = sets.iter().map(|s| s.len()).product();
  if total > 10000 {
    return vec![];
  }

  let mut result = vec![vec![]];
  for set in sets {
    let mut new_result = Vec::new();
    for combo in &result {
      for &val in set {
        let mut new_combo = combo.clone();
        new_combo.push(val);
        new_result.push(new_combo);
      }
    }
    result = new_result;
  }
  result
}

/// Lagrange interpolation over integers.
/// Given points (x_i, y_i), find polynomial with integer coefficients.
/// Returns None if the interpolation doesn't yield integer coefficients.
fn lagrange_interpolate_integer(xs: &[i128], ys: &[i128]) -> Option<Vec<i128>> {
  let n = xs.len();
  if n == 0 {
    return None;
  }

  // Use the Newton form of interpolation (divided differences)
  let mut divided_diffs = ys.to_vec();

  for j in 1..n {
    for i in (j..n).rev() {
      let num = divided_diffs[i] - divided_diffs[i - 1];
      let den = xs[i] - xs[i - j];
      if den == 0 {
        return None;
      }
      if num % den != 0 {
        return None; // Not integer coefficients
      }
      divided_diffs[i] = num / den;
    }
  }

  // Convert Newton form to standard coefficients
  // P(x) = d[0] + d[1]*(x-x0) + d[2]*(x-x0)*(x-x1) + ...
  let mut coeffs = vec![0i128; n];
  // Build up: start with d[n-1], multiply by (x - x_{n-2}), add d[n-2], etc.
  coeffs[0] = divided_diffs[n - 1];
  for i in (0..n - 1).rev() {
    // Multiply current poly by (x - xs[i])
    // coeffs represents polynomial, multiply by (x - xs[i])
    // New coeffs[j] = coeffs[j-1] - xs[i]*coeffs[j]
    let mut new_coeffs = vec![0i128; n];
    for j in (0..n).rev() {
      new_coeffs[j] = if j > 0 { coeffs[j - 1] } else { 0 } - xs[i] * coeffs[j];
    }
    new_coeffs[0] += divided_diffs[i];
    coeffs = new_coeffs;
  }

  // Trim trailing zeros
  while coeffs.len() > 1 && *coeffs.last().unwrap() == 0 {
    coeffs.pop();
  }

  Some(coeffs)
}

// ─── ExpandAll ──────────────────────────────────────────────────────

/// ExpandAll[expr] - Recursively expands all subexpressions
pub fn expand_all_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ExpandAll expects exactly 1 argument".into(),
    ));
  }
  Ok(expand_all_recursive(&args[0]))
}

/// Recursively expand all subexpressions, then expand at the top level
fn expand_all_recursive(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::Constant(_)
    | Expr::Identifier(_)
    | Expr::Slot(_) => expr.clone(),

    Expr::BinaryOp { op, left, right } => {
      let left_exp = expand_all_recursive(left);
      let right_exp = expand_all_recursive(right);
      // After recursively expanding sub-expressions, expand at this level
      expand_and_combine(&Expr::BinaryOp {
        op: *op,
        left: Box::new(left_exp),
        right: Box::new(right_exp),
      })
    }

    Expr::UnaryOp { op, operand } => {
      let operand_exp = expand_all_recursive(operand);
      expand_and_combine(&Expr::UnaryOp {
        op: *op,
        operand: Box::new(operand_exp),
      })
    }

    Expr::FunctionCall { name, args } => {
      let expanded_args: Vec<Expr> =
        args.iter().map(expand_all_recursive).collect();
      // After expanding sub-expressions, expand at this level for Plus/Times/Power
      match name.as_str() {
        "Plus" | "Times" | "Power" => expand_and_combine(&Expr::FunctionCall {
          name: name.clone(),
          args: expanded_args,
        }),
        _ => Expr::FunctionCall {
          name: name.clone(),
          args: expanded_args,
        },
      }
    }

    _ => expr.clone(),
  }
}

// ─── Cancel ─────────────────────────────────────────────────────────

/// Cancel[expr] - Cancels common factors between numerator and denominator
pub fn cancel_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Cancel expects exactly 1 argument".into(),
    ));
  }
  Ok(cancel_expr(&args[0]))
}

fn cancel_expr(expr: &Expr) -> Expr {
  // Look for division
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let num = expand_and_combine(left);
      let den = expand_and_combine(right);

      // Try polynomial division
      if let Some(var) = find_single_variable_both(&num, &den)
        && let (Some(num_coeffs), Some(den_coeffs)) = (
          extract_poly_coeffs(&num, &var),
          extract_poly_coeffs(&den, &var),
        )
      {
        // Find the GCD of the two polynomials
        if let Some(gcd_coeffs) = poly_gcd(&num_coeffs, &den_coeffs)
          && (gcd_coeffs.len() > 1
            || (gcd_coeffs.len() == 1 && gcd_coeffs[0] != 1))
        {
          // Divide both by GCD
          if let (Some(mut new_num), Some(mut new_den)) = (
            poly_exact_divide(&num_coeffs, &gcd_coeffs),
            poly_exact_divide(&den_coeffs, &gcd_coeffs),
          ) {
            // Also cancel numeric content GCD (poly_gcd normalizes to
            // primitive, so numeric factors like gcd(2,4)=2 may remain)
            let num_content = new_num
              .iter()
              .copied()
              .filter(|&c| c != 0)
              .fold(0i128, gcd_i128);
            let den_content = new_den
              .iter()
              .copied()
              .filter(|&c| c != 0)
              .fold(0i128, gcd_i128);
            if num_content > 1 && den_content > 1 {
              let content_gcd = gcd_i128(num_content, den_content);
              if content_gcd > 1 {
                new_num = new_num.iter().map(|c| c / content_gcd).collect();
                new_den = new_den.iter().map(|c| c / content_gcd).collect();
              }
            }
            // Normalize sign: keep denominator positive
            if new_den.last().map(|&c| c < 0).unwrap_or(false) {
              new_num = new_num.iter().map(|c| -c).collect();
              new_den = new_den.iter().map(|c| -c).collect();
            }
            let num_expr = coeffs_to_expr(&new_num, &var);
            let den_expr = coeffs_to_expr(&new_den, &var);
            // If denominator is 1, just return numerator
            if new_den == [1] {
              return num_expr;
            }
            return Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(num_expr),
              right: Box::new(den_expr),
            };
          }
        }
      }

      // Try symbolic factor cancellation for products (e.g. (a*b)/(a*c) → b/c)
      let result = cancel_symbolic_factors(&num, &den);
      if let Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: ref rl,
        right: ref rr,
      } = result
      {
        // Only accept if something actually changed
        if expr_to_string(rl) != expr_to_string(&num)
          || expr_to_string(rr) != expr_to_string(&den)
        {
          return result;
        }
      } else {
        // Result is not a division (fully cancelled), return it
        return result;
      }

      // Fall back to simplify_division
      simplify_division(&num, &den)
    }
    _ => expand_and_combine(expr),
  }
}

/// Cancel common symbolic factors between numerator and denominator.
/// E.g. (a*b)/(a*c) → b/c, (a^2*b)/(a*b^2) → a/b
fn cancel_symbolic_factors(num: &Expr, den: &Expr) -> Expr {
  // Extract base and exponent from a factor
  fn base_and_exp(f: &Expr) -> (String, i128) {
    match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right: exp,
      } => {
        let base_str = expr_to_string(left);
        if let Expr::Integer(n) = exp.as_ref() {
          (base_str, *n)
        } else {
          (expr_to_string(f), 1)
        }
      }
      _ => (expr_to_string(f), 1),
    }
  }

  // Reconstruct a factor from base expr and exponent
  fn make_factor(base: &Expr, exp: i128) -> Option<Expr> {
    if exp == 0 {
      None
    } else if exp == 1 {
      Some(base.clone())
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base.clone()),
        right: Box::new(Expr::Integer(exp)),
      })
    }
  }

  let mut num_factors = collect_multiplicative_factors(num);
  let mut den_factors = collect_multiplicative_factors(den);

  // Separate numeric coefficients from symbolic factors
  let mut num_coeff: i128 = 1;
  let mut den_coeff: i128 = 1;
  num_factors.retain(|f| {
    if let Expr::Integer(n) = f {
      num_coeff *= n;
      false
    } else {
      true
    }
  });
  den_factors.retain(|f| {
    if let Expr::Integer(n) = f {
      den_coeff *= n;
      false
    } else {
      true
    }
  });

  // Cancel numeric GCD
  if num_coeff != 0 && den_coeff != 0 {
    let g = gcd_i128(num_coeff.abs(), den_coeff.abs());
    if g > 1 {
      num_coeff /= g;
      den_coeff /= g;
    }
    // Keep signs normalized: negative in numerator
    if den_coeff < 0 {
      num_coeff = -num_coeff;
      den_coeff = -den_coeff;
    }
  }

  // Build maps of base → (original_expr, exponent) for numerator and denominator
  let mut num_map: Vec<(Expr, String, i128)> = num_factors
    .iter()
    .map(|f| {
      let (base_str, exp) = base_and_exp(f);
      // Find the base expression (without exponent)
      let base_expr = match f {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } => {
          if let Expr::Integer(_) = right.as_ref() {
            left.as_ref().clone()
          } else {
            f.clone()
          }
        }
        _ => f.clone(),
      };
      (base_expr, base_str, exp)
    })
    .collect();

  let mut den_map: Vec<(Expr, String, i128)> = den_factors
    .iter()
    .map(|f| {
      let (base_str, exp) = base_and_exp(f);
      let base_expr = match f {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } => {
          if let Expr::Integer(_) = right.as_ref() {
            left.as_ref().clone()
          } else {
            f.clone()
          }
        }
        _ => f.clone(),
      };
      (base_expr, base_str, exp)
    })
    .collect();

  // Cancel common factors
  let mut changed = false;
  for (_, num_base_str, num_exp) in num_map.iter_mut() {
    for (_, den_base_str, den_exp) in den_map.iter_mut() {
      if *num_base_str == *den_base_str && *num_exp > 0 && *den_exp > 0 {
        let common = (*num_exp).min(*den_exp);
        *num_exp -= common;
        *den_exp -= common;
        changed = true;
      }
    }
  }

  if !changed && num_coeff == 1 && den_coeff == 1 {
    // Nothing was cancelled, return original
    return Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(num.clone()),
      right: Box::new(den.clone()),
    };
  }

  // Rebuild numerator and denominator
  let mut new_num_factors: Vec<Expr> = Vec::new();
  if num_coeff != 1 || (num_map.iter().all(|(_, _, e)| *e == 0)) {
    new_num_factors.push(Expr::Integer(num_coeff));
  }
  for (base_expr, _, exp) in &num_map {
    if let Some(f) = make_factor(base_expr, *exp) {
      new_num_factors.push(f);
    }
  }

  let mut new_den_factors: Vec<Expr> = Vec::new();
  if den_coeff != 1 || (den_map.iter().all(|(_, _, e)| *e == 0)) {
    new_den_factors.push(Expr::Integer(den_coeff));
  }
  for (base_expr, _, exp) in &den_map {
    if let Some(f) = make_factor(base_expr, *exp) {
      new_den_factors.push(f);
    }
  }

  let new_num = build_product(new_num_factors);
  let new_den = build_product(new_den_factors);

  if let Expr::Integer(1) = &new_den {
    new_num
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(new_num),
      right: Box::new(new_den),
    }
  }
}

/// Find the single variable that appears in either or both expressions
fn find_single_variable_both(a: &Expr, b: &Expr) -> Option<String> {
  let mut vars = std::collections::HashSet::new();
  collect_variables(a, &mut vars);
  collect_variables(b, &mut vars);
  if vars.len() == 1 {
    vars.into_iter().next()
  } else {
    None
  }
}

/// Compute GCD of two integer polynomials using Euclidean algorithm
fn poly_gcd(a: &[i128], b: &[i128]) -> Option<Vec<i128>> {
  if b.iter().all(|&c| c == 0) {
    return Some(a.to_vec());
  }
  if a.iter().all(|&c| c == 0) {
    return Some(b.to_vec());
  }

  let mut r0 = a.to_vec();
  let mut r1 = b.to_vec();

  // Trim trailing zeros
  while r0.last() == Some(&0) && r0.len() > 1 {
    r0.pop();
  }
  while r1.last() == Some(&0) && r1.len() > 1 {
    r1.pop();
  }

  // Euclidean algorithm for polynomials
  for _ in 0..100 {
    if r1.iter().all(|&c| c == 0) || r1.is_empty() {
      // Normalize: make leading coefficient positive and divide by GCD of coefficients
      let g = r0.iter().copied().filter(|&c| c != 0).fold(0i128, gcd_i128);
      if g > 0 {
        let result: Vec<i128> = r0.iter().map(|c| c / g).collect();
        if result.last().map(|&c| c < 0).unwrap_or(false) {
          return Some(result.iter().map(|c| -c).collect());
        }
        return Some(result);
      }
      return Some(r0);
    }
    if r0.len() < r1.len() {
      std::mem::swap(&mut r0, &mut r1);
    }
    // Pseudo-remainder
    let remainder = poly_pseudo_remainder(&r0, &r1)?;
    r0 = r1;
    r1 = remainder;
  }
  None
}

/// Compute pseudo-remainder of polynomial division
fn poly_pseudo_remainder(a: &[i128], b: &[i128]) -> Option<Vec<i128>> {
  if b.is_empty() || b.iter().all(|&c| c == 0) {
    return None;
  }
  let mut rem = a.to_vec();
  let b_lead = *b.last()?;
  let b_deg = b.len() - 1;

  while rem.len() > b.len()
    || (rem.len() == b.len() && !rem.iter().all(|&c| c == 0))
  {
    while rem.last() == Some(&0) && rem.len() > 1 {
      rem.pop();
    }
    if rem.len() < b.len() {
      break;
    }
    let rem_lead = *rem.last()?;
    let rem_deg = rem.len() - 1;
    if rem_deg < b_deg {
      break;
    }
    let shift = rem_deg - b_deg;
    // Multiply remainder by b_lead and subtract rem_lead * shifted b
    for c in &mut rem {
      *c *= b_lead;
    }
    for (i, &bc) in b.iter().enumerate() {
      rem[i + shift] -= rem_lead * bc;
    }
    // Trim trailing zeros
    while rem.last() == Some(&0) && rem.len() > 1 {
      rem.pop();
    }
  }

  // Simplify by GCD of coefficients
  let g = rem
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if g > 1 {
    rem = rem.iter().map(|c| c / g).collect();
  }
  Some(rem)
}

/// Exact polynomial division: returns quotient if a is divisible by b
fn poly_exact_divide(a: &[i128], b: &[i128]) -> Option<Vec<i128>> {
  if b.is_empty() || b.iter().all(|&c| c == 0) {
    return None;
  }
  let a_deg = a.len() as i128 - 1;
  let b_deg = b.len() as i128 - 1;
  if a_deg < b_deg {
    return None;
  }
  let mut remainder = a.to_vec();
  let mut quotient = vec![0i128; (a_deg - b_deg + 1) as usize];
  let lead_b = *b.last()?;
  if lead_b == 0 {
    return None;
  }

  for i in (0..quotient.len()).rev() {
    let rem_idx = i + b.len() - 1;
    if rem_idx >= remainder.len() {
      continue;
    }
    if remainder[rem_idx] % lead_b != 0 {
      return None;
    }
    let q = remainder[rem_idx] / lead_b;
    quotient[i] = q;
    for j in 0..b.len() {
      remainder[i + j] -= q * b[j];
    }
  }

  if remainder.iter().any(|&c| c != 0) {
    return None;
  }
  Some(quotient)
}

// ─── Collect ────────────────────────────────────────────────────────

/// Collect[expr, x] - Collects terms by powers of x
pub fn collect_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Collect expects exactly 2 arguments".into(),
    ));
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Collect".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Expand and collect terms by power of var
  let expanded = expand_and_combine(&args[0]);
  let terms = collect_additive_terms(&expanded);

  // Group terms by power of var
  let mut power_groups: Vec<(i128, Vec<Expr>)> = Vec::new();

  for term in &terms {
    let (power, coeff) = term_var_power_and_coeff(term, var);
    if let Some(entry) = power_groups.iter_mut().find(|(p, _)| *p == power) {
      entry.1.push(coeff);
    } else {
      power_groups.push((power, vec![coeff]));
    }
  }

  // Sort by power ascending
  power_groups.sort_by_key(|(p, _)| *p);

  // Build result: sum of (combined_coeff * var^power)
  let mut result_terms = Vec::new();
  for (power, coeffs) in power_groups {
    // Sum the coefficients
    let combined_coeff = if coeffs.len() == 1 {
      simplify(coeffs[0].clone())
    } else {
      let sum = coeffs
        .into_iter()
        .reduce(|a, b| add_exprs(&a, &b))
        .unwrap_or(Expr::Integer(0));
      simplify(sum)
    };

    if matches!(&combined_coeff, Expr::Integer(0)) {
      continue;
    }

    let var_part = if power == 0 {
      None
    } else if power == 1 {
      Some(Expr::Identifier(var.to_string()))
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier(var.to_string())),
        right: Box::new(Expr::Integer(power)),
      })
    };

    let term = match (combined_coeff.clone(), var_part) {
      (c, None) => c,
      (Expr::Integer(1), Some(v)) => v,
      (c, Some(v)) if matches!(&c, Expr::Integer(-1)) => negate_term(&v),
      (c, Some(v)) => {
        // Wolfram canonical Times ordering: coefficient goes first unless
        // the coefficient contains a variable that sorts after the collect
        // variable alphabetically, in which case the variable goes first.
        let mut coeff_vars = std::collections::HashSet::new();
        collect_variables(&c, &mut coeff_vars);
        let var_after = coeff_vars.iter().any(|cv| cv.as_str() > var);
        if var_after {
          multiply_exprs(&v, &c)
        } else {
          multiply_exprs(&c, &v)
        }
      }
    };
    result_terms.push(term);
  }

  if result_terms.is_empty() {
    Ok(Expr::Integer(0))
  } else {
    Ok(build_sum(result_terms))
  }
}

// ─── Together ───────────────────────────────────────────────────────

/// Together[expr] - Combines fractions over a common denominator
/// Threads over List.
pub fn together_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Together expects exactly 1 argument".into(),
    ));
  }
  // Thread over List
  if let Expr::List(items) = &args[0] {
    let results: Vec<Expr> = items.iter().map(together_expr).collect();
    return Ok(Expr::List(results));
  }
  Ok(together_expr(&args[0]))
}

/// Extract numerator and denominator from an expression.
/// Handles BinaryOp::Divide, Rational, Power[..., -1], and
/// Times[..., Power[..., -1]] forms.
fn extract_num_den(expr: &Expr) -> (Expr, Expr) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      (args[0].clone(), args[1].clone())
    }
    // Power[base, -1] => 1/base
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Some(neg_exp) = get_negative_integer(&args[1]) {
        if neg_exp == 1 {
          (Expr::Integer(1), args[0].clone())
        } else {
          (
            Expr::Integer(1),
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![args[0].clone(), Expr::Integer(neg_exp as i128)],
            },
          )
        }
      } else {
        (expr.clone(), Expr::Integer(1))
      }
    }
    // Times[factors...] — split into numerator factors and denominator factors
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      // Flatten nested Times first
      let flat_args = flatten_times_args(args);
      let mut num_factors = Vec::new();
      let mut den_factors = Vec::new();
      for arg in &flat_args {
        match arg {
          Expr::FunctionCall {
            name: pname,
            args: pargs,
          } if pname == "Power" && pargs.len() == 2 => {
            if let Some(neg_exp) = get_negative_integer(&pargs[1]) {
              if neg_exp == 1 {
                den_factors.push(pargs[0].clone());
              } else {
                den_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![pargs[0].clone(), Expr::Integer(neg_exp as i128)],
                });
              }
            } else {
              num_factors.push(arg.clone());
            }
          }
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } => {
            if let Some(neg_exp) = get_negative_integer(right) {
              if neg_exp == 1 {
                den_factors.push(*left.clone());
              } else {
                den_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![*left.clone(), Expr::Integer(neg_exp as i128)],
                });
              }
            } else {
              num_factors.push(arg.clone());
            }
          }
          // BinaryOp::Divide inside Times: split into num/den
          Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left,
            right,
          } => {
            num_factors.push(*left.clone());
            den_factors.push(*right.clone());
          }
          _ => num_factors.push(arg.clone()),
        }
      }
      if den_factors.is_empty() {
        (expr.clone(), Expr::Integer(1))
      } else {
        let num = build_product(num_factors);
        let den = build_product(den_factors);
        (num, den)
      }
    }
    // BinaryOp::Times — split into numerator and denominator factors
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      // Flatten into a Times FunctionCall and recurse
      let flat = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![*left.clone(), *right.clone()],
      };
      extract_num_den(&flat)
    }
    // UnaryOp::Minus — negate the numerator
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (num, den) = extract_num_den(operand);
      (negate_expr(&num), den)
    }
    _ => (expr.clone(), Expr::Integer(1)),
  }
}

/// Flatten nested Times args: Times[a, Times[b, c]] → [a, b, c]
fn flatten_times_args(args: &[Expr]) -> Vec<Expr> {
  let mut flat = Vec::new();
  for arg in args {
    match arg {
      Expr::FunctionCall { name, args: inner } if name == "Times" => {
        flat.extend(flatten_times_args(inner));
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        flat.extend(flatten_times_args(&[*left.clone(), *right.clone()]));
      }
      _ => flat.push(arg.clone()),
    }
  }
  flat
}

/// Check if an expression is a negative integer and return its absolute value
fn get_negative_integer(expr: &Expr) -> Option<i64> {
  match expr {
    Expr::Integer(n) if *n < 0 => Some((-*n) as i64),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => match operand.as_ref() {
      Expr::Integer(n) if *n > 0 => Some(*n as i64),
      _ => None,
    },
    _ => None,
  }
}

/// Negate an expression
fn negate_expr(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(n) => Expr::Integer(-n),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => *operand.clone(),
    _ => Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(expr.clone()),
    },
  }
}

fn together_expr(expr: &Expr) -> Expr {
  // Collect additive terms and put them over a common denominator
  let terms = collect_additive_terms(expr);
  if terms.len() <= 1 {
    return expr.clone();
  }

  // Extract numerator and denominator for each term
  let mut fractions: Vec<(Expr, Expr)> = Vec::new();
  for term in &terms {
    fractions.push(extract_num_den(term));
  }

  // Compute the common denominator (product of all denominators, simplified)
  let mut common_den = Expr::Integer(1);
  let mut unique_dens: Vec<Expr> = Vec::new();
  for (_, den) in &fractions {
    if !matches!(den, Expr::Integer(1)) {
      let den_str = expr_to_string(den);
      if !unique_dens.iter().any(|d| expr_to_string(d) == den_str) {
        unique_dens.push(den.clone());
        common_den = multiply_exprs(&common_den, den);
      }
    }
  }

  if matches!(&common_den, Expr::Integer(1)) {
    // No fractions to combine
    return expr.clone();
  }

  // Build numerator: sum of (num_i * common_den / den_i)
  let mut new_num_terms = Vec::new();
  for (num, den) in &fractions {
    if matches!(den, Expr::Integer(1)) {
      // Multiply by full common_den
      new_num_terms.push(multiply_exprs(num, &common_den));
    } else {
      // Multiply by common_den / den
      let mut factor = Expr::Integer(1);
      for ud in &unique_dens {
        let ud_str = expr_to_string(ud);
        let den_str = expr_to_string(den);
        if ud_str != den_str {
          factor = multiply_exprs(&factor, ud);
        }
      }
      new_num_terms.push(multiply_exprs(num, &factor));
    }
  }

  let combined_num = if new_num_terms.len() == 1 {
    expand_and_combine(&new_num_terms.remove(0))
  } else {
    expand_and_combine(&build_sum(new_num_terms))
  };
  // Keep denominator in factored form (Wolfram behavior),
  // but canonicalize each individual factor and sort them
  let combined_den = if unique_dens.len() == 1 {
    expand_and_combine(&unique_dens[0])
  } else {
    let mut canonical_dens: Vec<Expr> =
      unique_dens.iter().map(expand_and_combine).collect();
    canonical_dens.sort_by_key(expr_to_string);
    build_product(canonical_dens)
  };

  if matches!(&combined_den, Expr::Integer(1)) {
    combined_num
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(combined_num),
      right: Box::new(combined_den),
    }
  }
}

// ─── Apart ──────────────────────────────────────────────────────────

/// Apart[expr] - Partial fraction decomposition
pub fn apart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Apart expects 1 or 2 arguments".into(),
    ));
  }

  let var = if args.len() == 2 {
    match &args[1] {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Apart".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    // Find the variable automatically
    match find_single_variable(&args[0]) {
      Some(v) => v,
      None => {
        return Ok(Expr::FunctionCall {
          name: "Apart".to_string(),
          args: args.to_vec(),
        });
      }
    }
  };

  apart_expr(&args[0], &var)
}

fn apart_expr(expr: &Expr, var: &str) -> Result<Expr, InterpreterError> {
  // Extract numerator and denominator
  let (num, den) = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    _ => {
      return Ok(expr.clone());
    }
  };

  let num_expanded = expand_and_combine(&num);
  let den_expanded = expand_and_combine(&den);

  let num_coeffs = match extract_poly_coeffs(&num_expanded, var) {
    Some(c) => c,
    None => return Ok(expr.clone()),
  };
  let den_coeffs = match extract_poly_coeffs(&den_expanded, var) {
    Some(c) => c,
    None => return Ok(expr.clone()),
  };

  // If numerator degree >= denominator degree, do polynomial division first
  if num_coeffs.len() >= den_coeffs.len() {
    if let Some(quot_coeffs) = poly_exact_divide(&num_coeffs, &den_coeffs) {
      // Perfectly divisible
      return Ok(coeffs_to_expr(&quot_coeffs, var));
    }
    // Polynomial long division with remainder
    let (quotient, remainder) = poly_long_divide(&num_coeffs, &den_coeffs);
    if remainder.iter().all(|&c| c == 0) {
      return Ok(coeffs_to_expr(&quotient, var));
    }
    // result = quotient + Apart[remainder/den]
    let quot_expr = coeffs_to_expr(&quotient, var);
    let rem_expr = coeffs_to_expr(&remainder, var);
    let frac = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(rem_expr),
      right: Box::new(den_expanded.clone()),
    };
    let apart_remainder = apart_proper_fraction(&frac, var)?;
    return Ok(add_exprs(&quot_expr, &apart_remainder));
  }

  apart_proper_fraction(expr, var)
}

/// Perform partial fraction decomposition for a proper fraction (deg(num) < deg(den))
fn apart_proper_fraction(
  expr: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  let (num, den) = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    _ => return Ok(expr.clone()),
  };

  let den_expanded = expand_and_combine(&den);
  let den_coeffs = match extract_poly_coeffs(&den_expanded, var) {
    Some(c) => c,
    None => return Ok(expr.clone()),
  };

  // Factor the denominator
  let gcd_coeff = den_coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if gcd_coeff == 0 {
    return Ok(expr.clone());
  }
  let reduced: Vec<i128> = den_coeffs.iter().map(|c| c / gcd_coeff).collect();
  let (sign, reduced) = if reduced.last().map(|&c| c < 0).unwrap_or(false) {
    (-1i128, reduced.iter().map(|c| -c).collect::<Vec<_>>())
  } else {
    (1, reduced)
  };
  let _overall = gcd_coeff * sign;

  // Find roots of denominator
  let mut remaining = reduced.clone();
  let mut roots: Vec<i128> = Vec::new();

  loop {
    if remaining.len() <= 1 {
      break;
    }
    match find_integer_root(&remaining) {
      Some(root) => {
        roots.push(root);
        remaining = divide_by_root(&remaining, root);
      }
      None => break,
    }
  }

  if roots.len() < 2 {
    // Can't decompose further
    return Ok(expr.clone());
  }

  // Sort roots descending so linear factors (-root + x) appear in ascending order
  roots.sort_by(|a, b| b.cmp(a));

  // Simple partial fraction for distinct linear factors:
  // N(x) / ((x-r1)(x-r2)...) = A1/(x-r1) + A2/(x-r2) + ...
  // where Ai = N(ri) / product of (ri - rj) for j != i
  let num_expanded = expand_and_combine(&num);
  let num_coeffs = match extract_poly_coeffs(&num_expanded, var) {
    Some(c) => c,
    None => return Ok(expr.clone()),
  };

  // If there's a remaining irreducible factor, we can't do simple partial fractions
  if !remaining.iter().all(|&c| c == 0) && remaining.len() > 1 {
    return Ok(expr.clone());
  }

  let mut result_terms = Vec::new();
  let overall_factor = gcd_coeff * sign;

  for (i, &root) in roots.iter().enumerate() {
    let num_at_root = evaluate_poly(&num_coeffs, root);
    let mut den_product = 1i128;
    for (j, &other_root) in roots.iter().enumerate() {
      if i != j {
        den_product *= root - other_root;
      }
    }
    // Handle the remaining factor
    if remaining.len() == 1 && remaining[0] != 0 {
      den_product *= remaining[0];
    }
    den_product *= overall_factor;

    if den_product == 0 {
      return Ok(expr.clone());
    }

    // A_i = num_at_root / den_product
    // Term = A_i / (x - root) which in canonical form is A_i / (-root + x)
    let g = gcd_i128(num_at_root.abs(), den_product.abs());
    let (mut an, mut ad) = (num_at_root / g, den_product / g);
    if ad < 0 {
      an = -an;
      ad = -ad;
    }

    // Build linear factor: when root is 0, just use the variable directly
    let linear_factor = if root == 0 {
      Expr::Identifier(var.to_string())
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-root)),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    };

    let abs_an = an.abs();
    let frac = if ad == 1 && abs_an == 1 {
      // Wolfram canonical form: (expr)^(-1)
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(linear_factor),
        right: Box::new(Expr::Integer(-1)),
      }
    } else if ad == 1 {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(abs_an)),
        right: Box::new(linear_factor),
      }
    } else {
      // an / (ad * linear_factor) — Wolfram format: 1/(2*(-1 + x))
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(abs_an)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(ad)),
          right: Box::new(linear_factor),
        }),
      }
    };

    if an < 0 {
      result_terms.push(Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(frac),
      });
    } else {
      result_terms.push(frac);
    }
  }

  if result_terms.is_empty() {
    Ok(expr.clone())
  } else {
    Ok(build_sum(result_terms))
  }
}

/// Polynomial long division returning (quotient, remainder) as coefficient vectors
fn poly_long_divide(num: &[i128], den: &[i128]) -> (Vec<i128>, Vec<i128>) {
  let n_deg = num.len();
  let d_deg = den.len();
  if n_deg < d_deg {
    return (vec![0], num.to_vec());
  }
  let mut remainder = num.to_vec();
  let mut quotient = vec![0i128; n_deg - d_deg + 1];
  let lead_den = den[d_deg - 1];
  if lead_den == 0 {
    return (vec![0], num.to_vec());
  }

  for i in (0..quotient.len()).rev() {
    let rem_idx = i + d_deg - 1;
    if rem_idx >= remainder.len() {
      continue;
    }
    if remainder[rem_idx] % lead_den != 0 {
      // Non-integer quotient - stop here
      return (vec![0], num.to_vec());
    }
    let q = remainder[rem_idx] / lead_den;
    quotient[i] = q;
    for j in 0..d_deg {
      remainder[i + j] -= q * den[j];
    }
  }
  (quotient, remainder)
}

// ─── Solve ──────────────────────────────────────────────────────────

/// Solve[equation, var] — solve a polynomial equation for a variable.
///
/// Supports linear (degree 1) and quadratic (degree 2) equations.
/// The equation must be of the form `lhs == rhs`.
pub fn solve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Solve expects exactly 2 arguments".into(),
    ));
  }

  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    // Constants (E, Pi, Degree) are not valid variables
    Expr::Constant(name) => {
      eprintln!("Solve::ivar: {} is not a valid variable.", name);
      return Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec(),
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Check if variable has Constant attribute (user-defined constants)
  let is_constant = crate::FUNC_ATTRS.with(|m| {
    m.borrow()
      .get(var)
      .is_some_and(|attrs| attrs.contains(&"Constant".to_string()))
  });
  if is_constant {
    eprintln!("Solve::ivar: {} is not a valid variable.", var);
    return Ok(Expr::FunctionCall {
      name: "Solve".to_string(),
      args: args.to_vec(),
    });
  }

  // Extract equation: lhs == rhs → lhs - rhs
  let poly = match &args[0] {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == crate::syntax::ComparisonOp::Equal =>
    {
      // lhs - rhs
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(operands[0].clone()),
        right: Box::new(operands[1].clone()),
      }
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Equal" && fargs.len() == 2 =>
    {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(fargs[0].clone()),
        right: Box::new(fargs[1].clone()),
      }
    }
    Expr::Identifier(s) if s == "True" => {
      // x == x → True → all solutions
      return Ok(Expr::List(vec![Expr::List(vec![])]));
    }
    Expr::Identifier(s) if s == "False" => {
      // contradiction → no solutions
      return Ok(Expr::List(vec![]));
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Expand and collect polynomial coefficients
  let expanded = expand_and_combine(&poly);
  let terms = collect_additive_terms(&expanded);

  // Find maximum degree
  let degree = match max_power(&expanded, var) {
    Some(d) => d,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Extract coefficients for each power of var
  let mut coeffs: Vec<Expr> = Vec::new();
  for d in 0..=degree {
    let mut coeff_sum: Vec<Expr> = Vec::new();
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, d) {
        coeff_sum.push(c);
      }
    }
    if coeff_sum.is_empty() {
      coeffs.push(Expr::Integer(0));
    } else if coeff_sum.len() == 1 {
      coeffs.push(coeff_sum.remove(0));
    } else {
      let mut result = coeff_sum.remove(0);
      for c in coeff_sum {
        result = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(result),
          right: Box::new(c),
        };
      }
      coeffs.push(simplify(result));
    }
  }

  let make_rule = |solution: Expr| -> Expr {
    Expr::List(vec![Expr::Rule {
      pattern: Box::new(Expr::Identifier(var.to_string())),
      replacement: Box::new(solution),
    }])
  };

  match degree {
    0 => {
      // No variable present — check if constant is zero
      let c0 = &coeffs[0];
      if matches!(c0, Expr::Integer(0)) {
        Ok(Expr::List(vec![Expr::List(vec![])]))
      } else {
        Ok(Expr::List(vec![]))
      }
    }
    1 => {
      // Linear: a*x + b = 0  → x = -b/a
      let b = &coeffs[0]; // constant term
      let a = &coeffs[1]; // coefficient of x
      let neg_b = negate_expr(b);
      let solution = simplify(solve_divide(&neg_b, a));
      Ok(Expr::List(vec![make_rule(solution)]))
    }
    2 => {
      // Quadratic: a*x^2 + b*x + c = 0
      let c = &coeffs[0]; // constant term
      let b = &coeffs[1]; // coefficient of x
      let a = &coeffs[2]; // coefficient of x^2

      // Discriminant: b^2 - 4*a*c
      let b_sq = multiply_exprs(b, b);
      let four_ac = multiply_exprs(&Expr::Integer(4), &multiply_exprs(a, c));
      let discriminant = simplify(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(b_sq),
        right: Box::new(four_ac),
      });

      let neg_b = negate_expr(b);
      let two_a = multiply_exprs(&Expr::Integer(2), a);

      // For integer coefficients, use exact arithmetic with simplified Sqrt
      if let (Expr::Integer(ai), Expr::Integer(bi), Expr::Integer(ci)) =
        (a, b, c)
      {
        let ai = *ai;
        let bi = *bi;
        let ci = *ci;
        let disc_int = bi * bi - 4 * ai * ci;

        if disc_int >= 0 {
          let (sqrt_out, sqrt_in) = simplify_sqrt_parts(disc_int);
          // roots = (-bi ± sqrt_out * Sqrt[sqrt_in]) / (2*ai)
          if sqrt_in == 1 {
            // Perfect square discriminant: exact integer/rational roots
            let sol1 = solve_divide(
              &Expr::Integer(-bi - sqrt_out),
              &Expr::Integer(2 * ai),
            );
            let sol2 = solve_divide(
              &Expr::Integer(-bi + sqrt_out),
              &Expr::Integer(2 * ai),
            );
            return Ok(Expr::List(vec![make_rule(sol1), make_rule(sol2)]));
          } else {
            // Irrational roots: (-bi ± sqrt_out*Sqrt[sqrt_in]) / (2*ai)
            // Simplify by dividing common factors
            let g =
              gcd_i128(gcd_i128(-bi, sqrt_out).abs(), (2 * ai).abs()).abs();
            let nb = -bi / g;
            let so = sqrt_out / g;
            let den = 2 * ai / g;
            // Normalize sign
            let (nb, so, den) = if den < 0 {
              (-nb, -so, -den)
            } else {
              (nb, so, den)
            };
            let sqrt_part = if so == 1 {
              Expr::FunctionCall {
                name: "Sqrt".to_string(),
                args: vec![Expr::Integer(sqrt_in)],
              }
            } else {
              multiply_exprs(
                &Expr::Integer(so),
                &Expr::FunctionCall {
                  name: "Sqrt".to_string(),
                  args: vec![Expr::Integer(sqrt_in)],
                },
              )
            };
            let make_sol = |sign_minus: bool| -> Expr {
              let num = if nb == 0 {
                if sign_minus {
                  negate_expr(&sqrt_part)
                } else {
                  sqrt_part.clone()
                }
              } else {
                let nb_expr = Expr::Integer(nb);
                Expr::BinaryOp {
                  op: if sign_minus {
                    BinaryOperator::Minus
                  } else {
                    BinaryOperator::Plus
                  },
                  left: Box::new(nb_expr),
                  right: Box::new(sqrt_part.clone()),
                }
              };
              if den == 1 {
                num
              } else {
                Expr::BinaryOp {
                  op: BinaryOperator::Divide,
                  left: Box::new(num),
                  right: Box::new(Expr::Integer(den)),
                }
              }
            };
            let sol1 = make_sol(true);
            let sol2 = make_sol(false);
            return Ok(Expr::List(vec![make_rule(sol1), make_rule(sol2)]));
          }
        } else {
          // Complex roots: (-bi ± I*Sqrt[-disc]) / (2*ai)
          let neg_disc = -disc_int;
          let (sqrt_out, sqrt_in) = simplify_sqrt_parts(neg_disc);
          if sqrt_in == 1 {
            // Gaussian integer/rational roots
            let real_part =
              solve_divide(&Expr::Integer(-bi), &Expr::Integer(2 * ai));
            let imag_part =
              solve_divide(&Expr::Integer(sqrt_out), &Expr::Integer(2 * ai));
            let make_sol = |sign_minus: bool| -> Expr {
              let i_part =
                multiply_exprs(&Expr::Identifier("I".to_string()), &imag_part);
              simplify(Expr::BinaryOp {
                op: if sign_minus {
                  BinaryOperator::Minus
                } else {
                  BinaryOperator::Plus
                },
                left: Box::new(real_part.clone()),
                right: Box::new(i_part),
              })
            };
            let sol1 = make_sol(true);
            let sol2 = make_sol(false);
            return Ok(Expr::List(vec![make_rule(sol1), make_rule(sol2)]));
          } else {
            // Complex roots with irrational imaginary part
            let g =
              gcd_i128(gcd_i128(-bi, sqrt_out).abs(), (2 * ai).abs()).abs();
            let nb = -bi / g;
            let so = sqrt_out / g;
            let den = 2 * ai / g;
            let (nb, so, den) = if den < 0 {
              (-nb, -so, -den)
            } else {
              (nb, so, den)
            };
            let sqrt_part = multiply_exprs(
              &Expr::Identifier("I".to_string()),
              &if so == 1 {
                Expr::FunctionCall {
                  name: "Sqrt".to_string(),
                  args: vec![Expr::Integer(sqrt_in)],
                }
              } else {
                multiply_exprs(
                  &Expr::Integer(so),
                  &Expr::FunctionCall {
                    name: "Sqrt".to_string(),
                    args: vec![Expr::Integer(sqrt_in)],
                  },
                )
              },
            );
            let make_sol = |sign_minus: bool| -> Expr {
              let num = if nb == 0 {
                if sign_minus {
                  negate_expr(&sqrt_part)
                } else {
                  sqrt_part.clone()
                }
              } else {
                Expr::BinaryOp {
                  op: if sign_minus {
                    BinaryOperator::Minus
                  } else {
                    BinaryOperator::Plus
                  },
                  left: Box::new(Expr::Integer(nb)),
                  right: Box::new(sqrt_part.clone()),
                }
              };
              if den == 1 {
                num
              } else {
                Expr::BinaryOp {
                  op: BinaryOperator::Divide,
                  left: Box::new(num),
                  right: Box::new(Expr::Integer(den)),
                }
              }
            };
            let sol1 = make_sol(true);
            let sol2 = make_sol(false);
            return Ok(Expr::List(vec![make_rule(sol1), make_rule(sol2)]));
          }
        }
      }

      // Non-integer coefficients: use general symbolic formula
      let sqrt_disc = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![discriminant],
      };
      let sol1 = simplify(solve_divide(
        &simplify(Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(neg_b.clone()),
          right: Box::new(sqrt_disc.clone()),
        }),
        &two_a,
      ));
      let sol2 = simplify(solve_divide(
        &simplify(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(neg_b),
          right: Box::new(sqrt_disc),
        }),
        &two_a,
      ));
      Ok(Expr::List(vec![make_rule(sol1), make_rule(sol2)]))
    }
    _ => {
      // Higher degree: return unevaluated
      Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Divide two expressions symbolically, simplifying integer cases.
fn solve_divide(num: &Expr, den: &Expr) -> Expr {
  match (num, den) {
    (Expr::Integer(0), _) => Expr::Integer(0),
    (_, Expr::Integer(1)) => num.clone(),
    (Expr::Integer(n), Expr::Integer(d)) if *d != 0 => {
      let g = gcd_i128(*n, *d).abs();
      let mut rn = n / g;
      let mut rd = d / g;
      // Normalize sign: denominator always positive
      if rd < 0 {
        rn = -rn;
        rd = -rd;
      }
      if rd == 1 {
        Expr::Integer(rn)
      } else {
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(rn), Expr::Integer(rd)],
        }
      }
    }
    _ => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(num.clone()),
      right: Box::new(den.clone()),
    },
  }
}

/// Simplify Sqrt for integer arguments.
/// Returns (outside, inside) where Sqrt[n] = outside * Sqrt[inside].
/// E.g. Sqrt[20] = 2*Sqrt[5] → (2, 5), Sqrt[4] = 2 → (2, 1).
fn simplify_sqrt_parts(n: i128) -> (i128, i128) {
  if n == 0 {
    return (0, 1); // Sqrt[0] = 0 → (0, 1) so 0 * Sqrt[1] = 0
  }
  if n < 0 {
    return (1, n);
  }
  let mut outside = 1i128;
  let mut inside = n;
  // Extract perfect square factors
  let mut factor = 2i128;
  while factor * factor <= inside {
    while inside % (factor * factor) == 0 {
      inside /= factor * factor;
      outside *= factor;
    }
    factor += 1;
  }
  (outside, inside)
}
