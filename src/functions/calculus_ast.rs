//! AST-native calculus functions (D, Integrate).
//!
//! These functions work directly with `Expr` AST nodes for symbolic differentiation
//! and integration.

use crate::InterpreterError;
use crate::functions::math_ast::{is_sqrt, make_sqrt};
use crate::syntax::Expr;

/// D[expr, var] or D[expr, {var, n}] or D[expr, x, y, ...] - Symbolic differentiation
pub fn d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "D expects at least 2 arguments".into(),
    ));
  }

  // D[expr, x, y, ...] — mixed partial derivatives: differentiate sequentially
  if args.len() > 2 {
    // First differentiate with respect to the last variable
    let inner = d_ast(&[args[0].clone(), args[1].clone()])?;
    // Then differentiate the result with respect to remaining variables
    let mut remaining = vec![inner];
    remaining.extend_from_slice(&args[2..]);
    return d_ast(&remaining);
  }

  // Thread over lists in the first argument: D[{f1, f2, ...}, var] -> {D[f1, var], D[f2, var], ...}
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|item| d_ast(&[item.clone(), args[1].clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }

  // Handle D[expr, {{x, y, ...}}] — gradient/Jacobian
  if let Expr::List(outer_items) = &args[1]
    && outer_items.len() == 1
    && let Expr::List(vars) = &outer_items[0]
  {
    if let Expr::List(_) = &args[0] {
      // D[{f1, f2, ...}, {{x, y, ...}}] → Jacobian matrix
      let expr_list = match &args[0] {
        Expr::List(items) => items.clone(),
        _ => unreachable!(),
      };
      let mut rows = Vec::new();
      for f in &expr_list {
        let mut row = Vec::new();
        for v in vars {
          row.push(d_ast(&[f.clone(), v.clone()])?);
        }
        rows.push(Expr::List(row));
      }
      return Ok(Expr::List(rows));
    } else {
      // D[f, {{x, y, ...}}] → gradient vector
      let mut result = Vec::new();
      for v in vars {
        result.push(d_ast(&[args[0].clone(), v.clone()])?);
      }
      return Ok(Expr::List(result));
    }
  }

  // Handle D[expr, {var, n}] for higher-order derivatives
  if let Expr::List(items) = &args[1]
    && items.len() == 2
  {
    let var_name = match &items[0] {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Variable specification in D must be a symbol".into(),
        ));
      }
    };
    let n = match &items[1] {
      Expr::Integer(n) if *n >= 0 => *n as usize,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Derivative order in D must be a non-negative integer".into(),
        ));
      }
    };
    // Apply differentiation n times
    let mut result = args[0].clone();
    for _ in 0..n {
      result = differentiate(&result, &var_name)?;
      result = simplify(result);
    }
    return Ok(result);
  }

  // Get the variable name
  let var_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    // For indexed variables like x[k], differentiate with respect to the full expression
    // This treats x[k] as an atomic unit — D[x[i], x[k]] = 0 for symbolic i, k
    other => {
      return differentiate_wrt_expr(&args[0], other);
    }
  };

  // Differentiate the expression and cancel common factors in fractions
  let result = differentiate(&args[0], &var_name)?;
  // Only apply Cancel when the result is a fraction to avoid expanding products
  let (_, den) =
    crate::functions::polynomial_ast::together::extract_num_den(&result);
  if !matches!(&den, Expr::Integer(1)) {
    Ok(crate::functions::polynomial_ast::cancel_expr(&result))
  } else {
    Ok(result)
  }
}

/// Differentiate an expression with respect to a non-symbol expression (e.g., x[k]).
/// For symbolic indexed variables, D[f[x[i]], x[k]] = 0 when we can't determine equality.
fn differentiate_wrt_expr(
  expr: &Expr,
  var_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  // If the expression is structurally equal to the variable, derivative is 1
  if crate::syntax::expr_to_string(expr)
    == crate::syntax::expr_to_string(var_expr)
  {
    return Ok(Expr::Integer(1));
  }
  // For products: use product rule
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left,
    right,
  } = expr
  {
    let dl = differentiate_wrt_expr(left, var_expr)?;
    let dr = differentiate_wrt_expr(right, var_expr)?;
    let term1 = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(dl),
      right: right.clone(),
    });
    let term2 = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: left.clone(),
      right: Box::new(dr),
    });
    return Ok(simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(term1),
      right: Box::new(term2),
    }));
  }
  // For sums
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left,
    right,
  } = expr
  {
    let dl = differentiate_wrt_expr(left, var_expr)?;
    let dr = differentiate_wrt_expr(right, var_expr)?;
    return Ok(simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(dl),
      right: Box::new(dr),
    }));
  }
  // For FunctionCall with Plus/Times
  if let Expr::FunctionCall { name, args } = expr {
    if name == "Times" && args.len() >= 2 {
      // Product of multiple terms
      let mut result_terms = Vec::new();
      for (i, arg) in args.iter().enumerate() {
        let darg = differentiate_wrt_expr(arg, var_expr)?;
        if !matches!(darg, Expr::Integer(0)) {
          let mut factors = Vec::new();
          for (j, a) in args.iter().enumerate() {
            if i == j {
              factors.push(darg.clone());
            } else {
              factors.push(a.clone());
            }
          }
          result_terms.push(
            crate::functions::math_ast::times_ast(&factors).unwrap_or(
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: factors,
              },
            ),
          );
        }
      }
      if result_terms.is_empty() {
        return Ok(Expr::Integer(0));
      }
      return crate::functions::math_ast::plus_ast(&result_terms).or(Ok(
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: result_terms,
        },
      ));
    }
    if name == "Plus" {
      let derivs: Result<Vec<_>, _> = args
        .iter()
        .map(|a| differentiate_wrt_expr(a, var_expr))
        .collect();
      let d = derivs?;
      return crate::functions::math_ast::plus_ast(&d).or(Ok(
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: d,
        },
      ));
    }
  }
  // Otherwise, treat as constant w.r.t. var_expr → 0
  Ok(Expr::Integer(0))
}

/// Integrate[expr, var] or Integrate[expr, {var, lo, hi}] - Symbolic integration
pub fn integrate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Integrate expects exactly 2 arguments".into(),
    ));
  }

  // Check if the second argument is {var, lo, hi} (definite integral)
  if let Expr::List(items) = &args[1]
    && items.len() == 3
  {
    let var_name = match &items[0] {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Integrate".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let lo = &items[1];
    let hi = &items[2];

    // Try known definite integrals first
    if let Some(result) = try_definite_integral(&args[0], &var_name, lo, hi) {
      return Ok(result);
    }

    // Fall back: compute indefinite integral and evaluate at bounds
    if let Some(antideriv) = integrate(&args[0], &var_name) {
      let antideriv = simplify(antideriv);
      let antideriv = crate::evaluator::evaluate_expr_to_expr(&antideriv)
        .unwrap_or(antideriv);
      // F(hi) - F(lo)
      // When a boundary is ±Infinity, use Limit instead of direct substitution
      // to correctly handle indeterminate forms like 0 * Infinity.
      let at_hi = if is_infinity(hi) || is_negative_infinity(hi) {
        let limit_expr = Expr::FunctionCall {
          name: "Limit".to_string(),
          args: vec![
            antideriv.clone(),
            Expr::FunctionCall {
              name: "Rule".to_string(),
              args: vec![Expr::Identifier(var_name.clone()), hi.clone()],
            },
          ],
        };
        crate::evaluator::evaluate_expr_to_expr(&limit_expr)?
      } else {
        let sub = crate::syntax::substitute_variable(&antideriv, &var_name, hi);
        crate::evaluator::evaluate_expr_to_expr(&sub)?
      };
      let at_lo = if is_infinity(lo) || is_negative_infinity(lo) {
        let limit_expr = Expr::FunctionCall {
          name: "Limit".to_string(),
          args: vec![
            antideriv.clone(),
            Expr::FunctionCall {
              name: "Rule".to_string(),
              args: vec![Expr::Identifier(var_name.clone()), lo.clone()],
            },
          ],
        };
        crate::evaluator::evaluate_expr_to_expr(&limit_expr)?
      } else {
        let sub = crate::syntax::substitute_variable(&antideriv, &var_name, lo);
        crate::evaluator::evaluate_expr_to_expr(&sub)?
      };
      let result = simplify(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(at_hi),
        right: Box::new(at_lo),
      });
      return Ok(
        crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result),
      );
    }

    // Return unevaluated
    return Ok(Expr::FunctionCall {
      name: "Integrate".to_string(),
      args: args.to_vec(),
    });
  }

  // Indefinite integral: Integrate[expr, var]
  let var_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Integrate".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Integrate the expression
  match integrate(&args[0], &var_name) {
    Some(result) => {
      let simplified = simplify(result);
      let evaluated = crate::evaluator::evaluate_expr_to_expr(&simplified)
        .unwrap_or(simplified);
      Ok(evaluated)
    }
    None => Ok(Expr::FunctionCall {
      name: "Integrate".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Check if an expression represents Infinity
fn is_infinity(expr: &Expr) -> bool {
  matches!(expr, Expr::Identifier(name) if name == "Infinity")
}

/// Check if an expression represents -Infinity (via UnaryOp::Minus or Times[-1, Infinity])
fn is_negative_infinity(expr: &Expr) -> bool {
  match expr {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => is_infinity(operand),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      (matches!(left.as_ref(), Expr::Integer(-1)) && is_infinity(right))
        || (matches!(right.as_ref(), Expr::Integer(-1)) && is_infinity(left))
    }
    Expr::Integer(n) if *n < 0 => false, // negative number, not -Infinity
    _ => false,
  }
}

/// Try to evaluate a definite integral using known closed-form results
fn try_definite_integral(
  integrand: &Expr,
  var: &str,
  lo: &Expr,
  hi: &Expr,
) -> Option<Expr> {
  // Gaussian integral: ∫_{-∞}^{∞} E^(-a*x^2) dx = Sqrt[Pi/a]
  if is_negative_infinity(lo)
    && is_infinity(hi)
    && let Some(coeff) = match_gaussian(integrand, var)
  {
    // coeff is 'a' in E^(-a*x^2): result = Sqrt[Pi/a]
    return Some(match coeff {
      Expr::Integer(1) => make_sqrt(Expr::Constant("Pi".to_string())),
      _ => make_sqrt(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(coeff),
      }),
    });
  }

  // Half-Gaussian: ∫_0^{∞} E^(-a*x^2) dx = Sqrt[Pi/a]/2
  if matches!(lo, Expr::Integer(0))
    && is_infinity(hi)
    && let Some(coeff) = match_gaussian(integrand, var)
  {
    let sqrt_part = match coeff {
      Expr::Integer(1) => make_sqrt(Expr::Constant("Pi".to_string())),
      _ => make_sqrt(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(coeff),
      }),
    };
    return Some(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(sqrt_part),
      right: Box::new(Expr::Integer(2)),
    });
  }

  None
}

/// Try to match an expression as E^(-a*x^2) where a is a positive constant.
/// Returns Some(a) if it matches, None otherwise.
fn match_gaussian(expr: &Expr, var: &str) -> Option<Expr> {
  // Match E^(exponent) where E is the constant
  let exponent = match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Constant(c) if c == "E") {
        Some(right.as_ref())
      } else {
        None
      }
    }
    _ => None,
  }?;

  // Match -a*x^2 or -(x^2) forms in the exponent
  match_neg_a_x_squared(exponent, var)
}

/// Match an exponent expression as -a*x^2 and return 'a'.
/// Handles forms: -x^2, -(x^2), -a*x^2, Times[-1, x, x], etc.
fn match_neg_a_x_squared(expr: &Expr, var: &str) -> Option<Expr> {
  match expr {
    // UnaryOp::Minus wrapping something
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      // -x^2 => a=1
      if is_var_squared(operand, var) {
        return Some(Expr::Integer(1));
      }
      // -(a*x^2) => a
      if let Some(coeff) = match_a_x_squared(operand, var) {
        return Some(coeff);
      }
      None
    }
    // Times[-1, x^2] or Times[x^2, -1]
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      // -1 * x^2
      if matches!(left.as_ref(), Expr::Integer(-1))
        && is_var_squared(right, var)
      {
        return Some(Expr::Integer(1));
      }
      if matches!(right.as_ref(), Expr::Integer(-1))
        && is_var_squared(left, var)
      {
        return Some(Expr::Integer(1));
      }
      // -1 * (a * x^2)
      if matches!(left.as_ref(), Expr::Integer(-1))
        && let Some(coeff) = match_a_x_squared(right, var)
      {
        return Some(coeff);
      }
      if matches!(right.as_ref(), Expr::Integer(-1))
        && let Some(coeff) = match_a_x_squared(left, var)
      {
        return Some(coeff);
      }
      // (-a) * x^2 where a is a negative integer
      if let Expr::Integer(n) = left.as_ref()
        && *n < 0
        && is_var_squared(right, var)
      {
        return Some(Expr::Integer(-*n));
      }
      if let Expr::Integer(n) = right.as_ref()
        && *n < 0
        && is_var_squared(left, var)
      {
        return Some(Expr::Integer(-*n));
      }
      None
    }
    // FunctionCall("Times", [...]) form
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Times[-1, x^2] => a=1
      // Times[-a, x^2] => a (where a is positive)
      // Times[-1, a, x^2] => a
      // Find a negative integer factor and check remaining is a*x^2
      for (i, arg) in args.iter().enumerate() {
        if let Expr::Integer(n) = arg
          && *n < 0
        {
          let mut rest: Vec<Expr> = args
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, e)| e.clone())
            .collect();
          let rest_expr = if rest.len() == 1 {
            rest.remove(0)
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest,
            }
          };
          if *n == -1 {
            // Times[-1, x^2] => a=1
            if is_var_squared(&rest_expr, var) {
              return Some(Expr::Integer(1));
            }
            // Times[-1, a*x^2] => a
            if let Some(coeff) = match_a_x_squared(&rest_expr, var) {
              return Some(coeff);
            }
          } else {
            // Times[-a, x^2] => a
            if is_var_squared(&rest_expr, var) {
              return Some(Expr::Integer(-*n));
            }
          }
        }
      }
      None
    }
    // BinaryOp::Minus: 0 - x^2 or similar (unlikely but handle)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Integer(0)) {
        if is_var_squared(right, var) {
          return Some(Expr::Integer(1));
        }
        if let Some(coeff) = match_a_x_squared(right, var) {
          return Some(coeff);
        }
      }
      None
    }
    _ => None,
  }
}

/// Check if expr is x^2 (where x is the variable)
fn is_var_squared(expr: &Expr, var: &str) -> bool {
  matches!(
    expr,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Identifier(name) if name == var)
      && matches!(right.as_ref(), Expr::Integer(2))
  )
}

/// Match a*x^2 and return 'a'
fn match_a_x_squared(expr: &Expr, var: &str) -> Option<Expr> {
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left,
    right,
  } = expr
  {
    // a * x^2
    if is_constant_wrt(left, var) && is_var_squared(right, var) {
      return Some(*left.clone());
    }
    // x^2 * a
    if is_var_squared(left, var) && is_constant_wrt(right, var) {
      return Some(*right.clone());
    }
  }
  // FunctionCall("Times", [...]) form
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
  {
    // Find x^2 factor, rest is 'a'
    for (i, arg) in args.iter().enumerate() {
      if is_var_squared(arg, var) {
        let rest: Vec<Expr> = args
          .iter()
          .enumerate()
          .filter(|(j, _)| *j != i)
          .map(|(_, e)| e.clone())
          .collect();
        if rest.iter().all(|a| is_constant_wrt(a, var)) {
          return if rest.len() == 1 {
            Some(rest[0].clone())
          } else {
            Some(Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest,
            })
          };
        }
      }
    }
  }
  None
}

/// Check if expression is constant with respect to a variable
pub fn is_constant_wrt(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::String(_) | Expr::Constant(_) => {
      true
    }
    Expr::Identifier(name) => name != var,
    Expr::List(items) => items.iter().all(|e| is_constant_wrt(e, var)),
    Expr::BinaryOp { left, right, .. } => {
      is_constant_wrt(left, var) && is_constant_wrt(right, var)
    }
    Expr::UnaryOp { operand, .. } => is_constant_wrt(operand, var),
    Expr::FunctionCall { args, .. } => {
      args.iter().all(|e| is_constant_wrt(e, var))
    }
    Expr::CurriedCall { func, args } => {
      is_constant_wrt(func, var) && args.iter().all(|e| is_constant_wrt(e, var))
    }
    _ => false,
  }
}

/// Differentiate an expression with respect to a variable
/// Public wrapper for differentiate - used by Derivative[n][f][x] evaluation
pub fn differentiate_expr(
  expr: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  differentiate(expr, var)
}

fn differentiate(expr: &Expr, var: &str) -> Result<Expr, InterpreterError> {
  match expr {
    // Constants
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => {
      Ok(Expr::Integer(0))
    }

    // Variable
    Expr::Identifier(name) => {
      if name == var {
        Ok(Expr::Integer(1))
      } else {
        Ok(Expr::Integer(0))
      }
    }

    // Binary operations
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator::*;
      match op {
        Plus => {
          // d/dx[a + b] = d/dx[a] + d/dx[b]
          let da = differentiate(left, var)?;
          let db = differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Plus,
            left: Box::new(da),
            right: Box::new(db),
          }))
        }
        Minus => {
          // d/dx[a - b] = d/dx[a] - d/dx[b]
          let da = differentiate(left, var)?;
          let db = differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Minus,
            left: Box::new(da),
            right: Box::new(db),
          }))
        }
        Times => {
          // Product rule: d/dx[a * b] = a' * b + a * b'
          let da = differentiate(left, var)?;
          let db = differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Plus,
            left: Box::new(Expr::BinaryOp {
              op: Times,
              left: Box::new(da),
              right: right.clone(),
            }),
            right: Box::new(Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(db),
            }),
          }))
        }
        Divide => {
          // Rewrite a/b as a * b^(-1) to use power+product rule
          // instead of quotient rule (avoids exponential expression growth)
          if is_constant_wrt(right, var) {
            // d/dx[a / c] = (d/dx a) / c
            // Use times_ast with b^(-1) so integer coefficients cancel
            let da = differentiate(left, var)?;
            let result = crate::functions::math_ast::times_ast(&[
              da,
              Expr::BinaryOp {
                op: Power,
                left: right.clone(),
                right: Box::new(Expr::Integer(-1)),
              },
            ])
            .unwrap_or_else(|_| Expr::BinaryOp {
              op: Divide,
              left: Box::new(
                differentiate(left, var).unwrap_or(Expr::Integer(0)),
              ),
              right: right.clone(),
            });
            Ok(simplify(result))
          } else if is_constant_wrt(left, var) {
            // d/dx[c / b] = c * d/dx[b^(-1)] = -c * b' / b^2
            let rewritten = Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(Expr::BinaryOp {
                op: Power,
                left: right.clone(),
                right: Box::new(Expr::Integer(-1)),
              }),
            };
            differentiate(&rewritten, var)
          } else {
            // d/dx[a / b] = d/dx[a * b^(-1)] (product rule + power rule)
            let rewritten = Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(Expr::BinaryOp {
                op: Power,
                left: right.clone(),
                right: Box::new(Expr::Integer(-1)),
              }),
            };
            differentiate(&rewritten, var)
          }
        }
        Power => {
          // Power rule for x^n: n * x^(n-1) * x'
          // Use Plus[-1, n] to match Wolfram's canonical form (-1 + n)
          if is_constant_wrt(right, var) {
            let df = differentiate(left, var)?;
            Ok(simplify(Expr::BinaryOp {
              op: Times,
              left: Box::new(Expr::BinaryOp {
                op: Times,
                left: right.clone(),
                right: Box::new(Expr::BinaryOp {
                  op: Power,
                  left: left.clone(),
                  right: Box::new(Expr::BinaryOp {
                    op: Plus,
                    left: Box::new(Expr::Integer(-1)),
                    right: right.clone(),
                  }),
                }),
              }),
              right: Box::new(df),
            }))
          } else if matches!(left.as_ref(), Expr::Constant(c) if c == "E") {
            // d/dx[E^g(x)] = E^g(x) * g'(x)  (since Log[E] = 1)
            let dg = differentiate(right, var)?;
            Ok(simplify(Expr::BinaryOp {
              op: Times,
              left: Box::new(expr.clone()),
              right: Box::new(dg),
            }))
          } else if is_constant_wrt(left, var) {
            // d/dx[a^g(x)] = a^g(x) * ln(a) * g'(x)
            let dg = differentiate(right, var)?;
            Ok(simplify(Expr::BinaryOp {
              op: Times,
              left: Box::new(Expr::BinaryOp {
                op: Times,
                left: Box::new(expr.clone()),
                right: Box::new(Expr::FunctionCall {
                  name: "Log".to_string(),
                  args: vec![*left.clone()],
                }),
              }),
              right: Box::new(dg),
            }))
          } else {
            // General case: d/dx[f(x)^g(x)] = f(x)^g(x) * (g'(x)*Log[f(x)] + g(x)*f'(x)/f(x))
            // This is logarithmic differentiation
            let df = differentiate(left, var)?;
            let dg = differentiate(right, var)?;
            Ok(simplify(Expr::BinaryOp {
              op: Times,
              left: Box::new(expr.clone()), // f^g
              right: Box::new(Expr::BinaryOp {
                op: Plus,
                left: Box::new(Expr::BinaryOp {
                  op: Times,
                  left: Box::new(dg), // g'
                  right: Box::new(Expr::FunctionCall {
                    name: "Log".to_string(),
                    args: vec![*left.clone()], // Log[f]
                  }),
                }),
                right: Box::new(Expr::BinaryOp {
                  op: Times,
                  left: right.clone(), // g
                  right: Box::new(Expr::BinaryOp {
                    op: Times,
                    left: Box::new(df), // f'
                    right: Box::new(Expr::BinaryOp {
                      op: Power,
                      left: left.clone(),                 // f
                      right: Box::new(Expr::Integer(-1)), // f^(-1)
                    }),
                  }),
                }),
              }),
            }))
          }
        }
        _ => Ok(Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())],
        }),
      }
    }

    // Unary minus
    Expr::UnaryOp { op, operand } => {
      use crate::syntax::UnaryOperator;
      if matches!(op, UnaryOperator::Minus) {
        let d = differentiate(operand, var)?;
        Ok(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(d),
        })
      } else {
        Ok(Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())],
        })
      }
    }

    // Function calls
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Sin" if args.len() == 1 => {
          // d/dx[sin(f(x))] = cos(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Cos".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Cos" if args.len() == 1 => {
          // d/dx[cos(f(x))] = -sin(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::FunctionCall {
                name: "Sin".to_string(),
                args: args.clone(),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Tan" if args.len() == 1 => {
          // d/dx[tan(f(x))] = sec^2(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(Expr::FunctionCall {
                name: "Sec".to_string(),
                args: args.clone(),
              }),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(df),
          }))
        }
        "Sec" if args.len() == 1 => {
          // d/dx[sec(f(x))] = sec(f(x)) * tan(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(Expr::FunctionCall {
                name: "Sec".to_string(),
                args: args.clone(),
              }),
              right: Box::new(Expr::FunctionCall {
                name: "Tan".to_string(),
                args: args.clone(),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Csc" if args.len() == 1 => {
          // d/dx[csc(f(x))] = -csc(f(x)) * cot(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::FunctionCall {
                  name: "Csc".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::FunctionCall {
                  name: "Cot".to_string(),
                  args: args.clone(),
                }),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Cot" if args.len() == 1 => {
          // d/dx[cot(f(x))] = -csc^2(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(Expr::FunctionCall {
                  name: "Csc".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Sinh" if args.len() == 1 => {
          // d/dx[sinh(f(x))] = cosh(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Cosh".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Cosh" if args.len() == 1 => {
          // d/dx[cosh(f(x))] = sinh(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Sinh".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Tanh" if args.len() == 1 => {
          // d/dx[tanh(f(x))] = sech^2(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(Expr::FunctionCall {
                name: "Sech".to_string(),
                args: args.clone(),
              }),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(df),
          }))
        }
        "Sech" if args.len() == 1 => {
          // d/dx[sech(f(x))] = -sech(f(x)) * tanh(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::FunctionCall {
                  name: "Sech".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::FunctionCall {
                  name: "Tanh".to_string(),
                  args: args.clone(),
                }),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Csch" if args.len() == 1 => {
          // d/dx[csch(f(x))] = -coth(f(x)) * csch(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::FunctionCall {
                  name: "Coth".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::FunctionCall {
                  name: "Csch".to_string(),
                  args: args.clone(),
                }),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Coth" if args.len() == 1 => {
          // d/dx[coth(f(x))] = -csch^2(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(Expr::FunctionCall {
                  name: "Csch".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "ArcSin" if args.len() == 1 => {
          // d/dx[arcsin(f(x))] = f'(x) / sqrt(1 - f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_minus_f_sq = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(args[0].clone()),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
          };
          let sqrt_expr = make_sqrt(one_minus_f_sq);
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![sqrt_expr, Expr::Integer(-1)],
            }),
          }))
        }
        "ArcCos" if args.len() == 1 => {
          // d/dx[arccos(f(x))] = -f'(x) / sqrt(1 - f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_minus_f_sq = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(args[0].clone()),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
          };
          let sqrt_expr = make_sqrt(one_minus_f_sq);
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(df),
            }),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![sqrt_expr, Expr::Integer(-1)],
            }),
          }))
        }
        "ArcTan" if args.len() == 1 => {
          // d/dx[arctan(f(x))] = f'(x) / (1 + f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_plus_f_sq = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          };
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![one_plus_f_sq, Expr::Integer(-1)],
            }),
          }))
        }
        "ArcCot" if args.len() == 1 => {
          // d/dx[arccot(f(x))] = -f'(x) / (1 + f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_plus_f_sq = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          };
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(df),
            }),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![one_plus_f_sq, Expr::Integer(-1)],
            }),
          }))
        }
        "ArcSinh" if args.len() == 1 => {
          // d/dx[arcsinh(f(x))] = f'(x) / sqrt(1 + f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_plus_f_sq = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          };
          let sqrt_expr = make_sqrt(one_plus_f_sq);
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![sqrt_expr, Expr::Integer(-1)],
            }),
          }))
        }
        "ArcCosh" if args.len() == 1 => {
          // d/dx[arccosh(f(x))] = f'(x) / (sqrt(f(x) - 1) * sqrt(f(x) + 1))
          // Using factored form to match Wolfram's branch-cut-aware convention
          let df = differentiate(&args[0], var)?;
          let f_minus_one = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(args[0].clone()),
          };
          let f_plus_one = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(args[0].clone()),
          };
          let sqrt_minus = make_sqrt(f_minus_one);
          let sqrt_plus = make_sqrt(f_plus_one);
          // f'(x) / (Sqrt[f-1] * Sqrt[f+1])
          let denom = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(sqrt_minus),
            right: Box::new(sqrt_plus),
          };
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![denom, Expr::Integer(-1)],
            }),
          }))
        }
        "ArcTanh" if args.len() == 1 => {
          // d/dx[arctanh(f(x))] = f'(x) / (1 - f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_minus_f_sq = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(args[0].clone()),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
          };
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![one_minus_f_sq, Expr::Integer(-1)],
            }),
          }))
        }
        "Exp" if args.len() == 1 => {
          // d/dx[e^f(x)] = e^f(x) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Exp".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Log" if args.len() == 1 => {
          // d/dx[ln(f(x))] = f'(x) * f(x)^(-1)
          let df = differentiate(&args[0], var)?;
          let power_neg_one = Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![args[0].clone(), Expr::Integer(-1)],
          };
          if matches!(df, Expr::Integer(1)) {
            Ok(power_neg_one)
          } else {
            crate::functions::math_ast::times_ast(&[df, power_neg_one])
          }
        }
        // Handle evaluated Plus[a, b, ...] (FunctionCall form of +)
        "Plus" if args.len() >= 2 => {
          let mut result = differentiate(&args[0], var)?;
          for arg in &args[1..] {
            let d = differentiate(arg, var)?;
            result = simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(result),
              right: Box::new(d),
            });
          }
          Ok(result)
        }
        // Handle evaluated Times[a, b, ...] (FunctionCall form of *)
        "Times" if args.len() >= 2 => {
          // Generalized product rule:
          // D[f1*f2*...*fn, x] = sum_i(f1*...*D[fi]*...*fn)
          // Each term replaces one factor with its derivative.
          let mut sum_terms: Vec<Expr> = Vec::new();
          for i in 0..args.len() {
            let di = differentiate(&args[i], var)?;
            // Skip if derivative is zero (constant factor)
            if matches!(&di, Expr::Integer(0)) {
              continue;
            }
            if let Expr::Real(f) = &di
              && *f == 0.0
            {
              continue;
            }
            // Build the product: all factors with the i-th replaced by its derivative
            let mut product_args: Vec<Expr> = Vec::new();
            for (j, arg) in args.iter().enumerate() {
              if j == i {
                product_args.push(di.clone());
              } else {
                product_args.push(arg.clone());
              }
            }
            // Simplify this individual product term using times_ast
            let term = if product_args.len() == 1 {
              simplify(product_args.remove(0))
            } else {
              crate::functions::math_ast::times_ast(&product_args)?
            };
            if !matches!(&term, Expr::Integer(0)) {
              sum_terms.push(term);
            }
          }

          if sum_terms.is_empty() {
            Ok(Expr::Integer(0))
          } else if sum_terms.len() == 1 {
            Ok(sum_terms.remove(0))
          } else {
            // Combine terms using plus_ast for proper like-term collection
            // without expanding product sub-expressions
            crate::functions::math_ast::plus_ast(&sum_terms).or_else(|_| {
              Ok(Expr::FunctionCall {
                name: "Plus".to_string(),
                args: sum_terms,
              })
            })
          }
        }
        // Handle evaluated Power[base, exp] (FunctionCall form of ^)
        "Power" if args.len() == 2 => differentiate(
          &Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(args[0].clone()),
            right: Box::new(args[1].clone()),
          },
          var,
        ),
        "Sqrt" if args.len() == 1 => {
          // d/dx[sqrt(f(x))] = f'(x) / (2 * sqrt(f(x)))
          let df = differentiate(&args[0], var)?;
          if matches!(df, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(df),
            right: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(Expr::Integer(2)),
              right: Box::new(make_sqrt(args[0].clone())),
            }),
          }))
        }
        // Handle Abs[f(x)]: d/dx[|f|] = f'*Sign[f] (for real f ≠ 0)
        "Abs" if args.len() == 1 => {
          let df = differentiate(&args[0], var)?;
          if matches!(df, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Sign".to_string(),
              args: args.clone(),
            }),
          }))
        }
        // Sign derivative: D[Sign[f(x)], x] = Derivative[2][Abs][f(x)] * f'(x)
        // (Wolfram uses this form instead of 2*DiracDelta[x])
        "Sign" if args.len() == 1 => {
          let df = differentiate(&args[0], var)?;
          if matches!(df, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let deriv_expr = Expr::CurriedCall {
            func: Box::new(Expr::CurriedCall {
              func: Box::new(Expr::FunctionCall {
                name: "Derivative".to_string(),
                args: vec![Expr::Integer(2)],
              }),
              args: vec![Expr::Identifier("Abs".to_string())],
            }),
            args: args.clone(),
          };
          if matches!(df, Expr::Integer(1)) {
            Ok(deriv_expr)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(df),
              right: Box::new(deriv_expr),
            }))
          }
        }
        // BesselJ[n, z]: D[BesselJ[n,z], z] = (BesselJ[n-1,z] - BesselJ[n+1,z]) / 2
        "BesselJ" if args.len() == 2 => {
          let dn = differentiate(&args[0], var)?;
          let dz = differentiate(&args[1], var)?;
          if matches!(dn, Expr::Integer(0)) && matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          if !matches!(dn, Expr::Integer(0)) {
            // Derivative w.r.t. order: leave unevaluated
            return Ok(Expr::FunctionCall {
              name: "D".to_string(),
              args: vec![expr.clone(), Expr::Identifier(var.to_string())],
            });
          }
          // D[BesselJ[n,z], z] = (BesselJ[n-1,z] - BesselJ[n+1,z]) / 2
          // Use plus_ast for canonical ordering: n-1 → Plus[-1, n], n+1 → Plus[1, n]
          let n_minus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(-1),
            args[0].clone(),
          ])
          .unwrap_or_else(|_| {
            simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Minus,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(1)),
            })
          });
          let n_plus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(1),
            args[0].clone(),
          ])
          .unwrap_or_else(|_| {
            simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(1)),
            })
          });
          let bessel_nm1 = Expr::FunctionCall {
            name: "BesselJ".to_string(),
            args: vec![n_minus_1, args[1].clone()],
          };
          let bessel_np1 = Expr::FunctionCall {
            name: "BesselJ".to_string(),
            args: vec![n_plus_1, args[1].clone()],
          };
          let diff = simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Minus,
            left: Box::new(bessel_nm1),
            right: Box::new(bessel_np1),
          });
          let half_diff = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(diff),
            right: Box::new(Expr::Integer(2)),
          };
          // Chain rule: multiply by dz
          if matches!(dz, Expr::Integer(1)) {
            Ok(half_diff)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(half_diff),
            }))
          }
        }
        // ExpIntegralEi[z]: D[ExpIntegralEi[z], z] = E^z / z
        "ExpIntegralEi" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let exp_z = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(Expr::Constant("E".to_string())),
            right: Box::new(args[0].clone()),
          };
          let result = simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(exp_z),
            right: Box::new(args[0].clone()),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Gamma[z]: D[Gamma[z], z] = Gamma[z] * PolyGamma[0, z]
        "Gamma" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let result = simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Gamma".to_string(),
              args: args.clone(),
            }),
            right: Box::new(Expr::FunctionCall {
              name: "PolyGamma".to_string(),
              args: vec![Expr::Integer(0), args[0].clone()],
            }),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // UnitStep[x]: D[UnitStep[x], x] = Piecewise[{{Indeterminate, x == 0}}, 0]
        "UnitStep" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let result = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![
              Expr::List(vec![Expr::List(vec![
                Expr::Identifier("Indeterminate".to_string()),
                Expr::Comparison {
                  operands: vec![args[0].clone(), Expr::Integer(0)],
                  operators: vec![crate::syntax::ComparisonOp::Equal],
                },
              ])]),
              Expr::Integer(0),
            ],
          };
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Erf[z]: D[Erf[z], z] = 2*E^(-z^2)/Sqrt[Pi]
        "Erf" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let z_sq = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(args[0].clone()),
            right: Box::new(Expr::Integer(2)),
          };
          let exp_neg_z2 = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(Expr::Constant("E".to_string())),
            right: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(z_sq),
            }),
          };
          let two_over_sqrt_pi = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(Expr::Integer(2)),
            right: Box::new(make_sqrt(Expr::Constant("Pi".to_string()))),
          };
          let result = simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(two_over_sqrt_pi),
            right: Box::new(exp_neg_z2),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Erfc[z]: D[Erfc[z], z] = -2*E^(-z^2)/Sqrt[Pi]
        "Erfc" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let z_sq = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(args[0].clone()),
            right: Box::new(Expr::Integer(2)),
          };
          let neg_exp_neg_z2 = Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(Expr::Constant("E".to_string())),
              right: Box::new(Expr::UnaryOp {
                op: crate::syntax::UnaryOperator::Minus,
                operand: Box::new(z_sq),
              }),
            }),
          };
          let two_over_sqrt_pi = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(Expr::Integer(2)),
            right: Box::new(make_sqrt(Expr::Constant("Pi".to_string()))),
          };
          let result = simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(two_over_sqrt_pi),
            right: Box::new(neg_exp_neg_z2),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Erfi[z]: D[Erfi[z], z] = 2*E^(z^2)/Sqrt[Pi]
        "Erfi" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let z_sq = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(args[0].clone()),
            right: Box::new(Expr::Integer(2)),
          };
          let exp_z2 = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(Expr::Constant("E".to_string())),
            right: Box::new(z_sq),
          };
          let two_over_sqrt_pi = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(Expr::Integer(2)),
            right: Box::new(make_sqrt(Expr::Constant("Pi".to_string()))),
          };
          let result = simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(two_over_sqrt_pi),
            right: Box::new(exp_z2),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Handle Rational[n, d] as constant
        "Rational" if args.len() == 2 => Ok(Expr::Integer(0)),
        // Handle Integrate[f, {t, a, b}] via Leibniz integral rule
        "Integrate" if args.len() == 2 => {
          if let Expr::List(spec) = &args[1]
            && spec.len() == 3
          {
            // D[Integrate[f[t], {t, a(x), b(x)}], x]
            // = f[b(x)] * b'(x) - f[a(x)] * a'(x)
            // (Plus the partial derivative term, which vanishes if f doesn't contain x directly)
            let integrand = &args[0];
            let int_var = &spec[0];
            let lo = &spec[1];
            let hi = &spec[2];

            let int_var_name = match int_var {
              Expr::Identifier(n) => n.as_str(),
              _ => "",
            };

            let da = differentiate(lo, var)?;
            let db = differentiate(hi, var)?;

            // Substitute integration variable with upper/lower bound in integrand
            let f_at_hi = if !int_var_name.is_empty() {
              crate::syntax::substitute_variable(integrand, int_var_name, hi)
            } else {
              integrand.clone()
            };
            let f_at_lo = if !int_var_name.is_empty() {
              crate::syntax::substitute_variable(integrand, int_var_name, lo)
            } else {
              integrand.clone()
            };

            let mut terms = Vec::new();
            // f[b(x)] * b'(x) term
            if !matches!(db, Expr::Integer(0)) {
              let term = if matches!(db, Expr::Integer(1)) {
                simplify(f_at_hi)
              } else {
                simplify(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Times,
                  left: Box::new(simplify(f_at_hi)),
                  right: Box::new(db),
                })
              };
              terms.push(term);
            }
            // -f[a(x)] * a'(x) term
            if !matches!(da, Expr::Integer(0)) {
              let term = if matches!(da, Expr::Integer(1)) {
                simplify(f_at_lo)
              } else {
                simplify(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Times,
                  left: Box::new(simplify(f_at_lo)),
                  right: Box::new(da),
                })
              };
              terms.push(Expr::UnaryOp {
                op: crate::syntax::UnaryOperator::Minus,
                operand: Box::new(term),
              });
            }

            if terms.is_empty() {
              Ok(Expr::Integer(0))
            } else if terms.len() == 1 {
              Ok(simplify(terms.remove(0)))
            } else {
              Ok(simplify(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Plus,
                left: Box::new(terms[0].clone()),
                right: Box::new(terms[1].clone()),
              }))
            }
          } else {
            // Indefinite integral: return unevaluated
            Ok(Expr::FunctionCall {
              name: "D".to_string(),
              args: vec![expr.clone(), Expr::Identifier(var.to_string())],
            })
          }
        }
        // Flattened Derivative[n, f, x]: this is the evaluated form of
        // Derivative[n][f][x] (n-th derivative of f at x).
        // D[Derivative[n, f, x], x] = Derivative[n+1][f][x] * D[x, x]
        // via chain rule, where D[x,x] = 1 for simple variable.
        "Derivative" if args.len() == 3 => {
          if is_constant_wrt(expr, var) {
            return Ok(Expr::Integer(0));
          }
          let order = &args[0];
          let func = &args[1];
          let inner = &args[2];
          let d_inner = differentiate(inner, var)?;
          if matches!(d_inner, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          // Build Derivative[n+1][f][x]
          let new_order = if let Expr::Integer(k) = order {
            Expr::Integer(k + 1)
          } else {
            Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(order.clone()),
            }
          };
          let deriv_expr = Expr::CurriedCall {
            func: Box::new(Expr::CurriedCall {
              func: Box::new(Expr::FunctionCall {
                name: "Derivative".to_string(),
                args: vec![new_order],
              }),
              args: vec![func.clone()],
            }),
            args: vec![inner.clone()],
          };
          if matches!(&d_inner, Expr::Integer(1)) {
            Ok(simplify(deriv_expr))
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(d_inner),
              right: Box::new(deriv_expr),
            }))
          }
        }
        // General chain rule for unknown functions: D[f[g1(x),...,gn(x)], x]
        // = Sum_i Derivative[0,...,1,...,0][f][g1,...,gn] * D[gi, x]
        _ => {
          // If the function is entirely constant w.r.t. var, return 0
          if is_constant_wrt(expr, var) {
            return Ok(Expr::Integer(0));
          }

          // Compute derivatives of each argument
          let n = args.len();
          let mut dargs: Vec<Expr> = Vec::with_capacity(n);
          for arg in args {
            dargs.push(differentiate(arg, var)?);
          }

          // Build chain rule sum
          let mut terms: Vec<Expr> = Vec::new();
          for i in 0..n {
            if matches!(&dargs[i], Expr::Integer(0)) {
              continue;
            }

            // Build Derivative[0,...,1,...,0][f][g1,...,gn]
            let deriv_indices: Vec<Expr> = (0..n)
              .map(|j| {
                if j == i {
                  Expr::Integer(1)
                } else {
                  Expr::Integer(0)
                }
              })
              .collect();

            let deriv_expr = Expr::CurriedCall {
              func: Box::new(Expr::CurriedCall {
                func: Box::new(Expr::FunctionCall {
                  name: "Derivative".to_string(),
                  args: deriv_indices,
                }),
                args: vec![Expr::Identifier(name.clone())],
              }),
              args: args.clone(),
            };

            if matches!(&dargs[i], Expr::Integer(1)) {
              terms.push(deriv_expr);
            } else {
              terms.push(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(dargs[i].clone()),
                right: Box::new(deriv_expr),
              });
            }
          }

          if terms.is_empty() {
            Ok(Expr::Integer(0))
          } else if terms.len() == 1 {
            Ok(simplify(terms.remove(0)))
          } else {
            let mut result = terms[0].clone();
            for term in &terms[1..] {
              result = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Plus,
                left: Box::new(result),
                right: Box::new(term.clone()),
              };
            }
            Ok(simplify(result))
          }
        }
      }
    }

    // CurriedCall: handles Derivative[n1,...,nk][f][g1,...,gk] expressions
    // and InverseFunction[f][x]
    Expr::CurriedCall { func, args } => {
      // Handle InverseFunction[f][x]:
      // D[InverseFunction[f][x], x] = 1 / Derivative[1][f][InverseFunction[f][x]]
      if let Expr::FunctionCall {
        name: inv_name,
        args: inv_args,
      } = func.as_ref()
        && inv_name == "InverseFunction"
        && inv_args.len() == 1
        && args.len() == 1
      {
        let dx = differentiate(&args[0], var)?;
        if matches!(dx, Expr::Integer(0)) {
          return Ok(Expr::Integer(0));
        }
        // 1 / f'(InverseFunction[f][x])
        let deriv_f_at_inv = Expr::CurriedCall {
          func: Box::new(Expr::CurriedCall {
            func: Box::new(Expr::FunctionCall {
              name: "Derivative".to_string(),
              args: vec![Expr::Integer(1)],
            }),
            args: inv_args.clone(),
          }),
          args: vec![expr.clone()],
        };
        let result = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(deriv_f_at_inv),
          right: Box::new(Expr::Integer(-1)),
        };
        if matches!(dx, Expr::Integer(1)) {
          return Ok(result);
        } else {
          return Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(dx),
            right: Box::new(result),
          }));
        }
      }

      // Check if this is Derivative[...][f][args]
      if let Expr::CurriedCall {
        func: inner_func,
        args: func_names,
      } = func.as_ref()
        && let Expr::FunctionCall {
          name: deriv_name,
          args: indices,
        } = inner_func.as_ref()
        && deriv_name == "Derivative"
        && func_names.len() == 1
        && indices.len() == args.len()
      {
        // D[Derivative[n1,...,nk][f][g1,...,gk], x]
        // = Sum_i Derivative[n1,...,ni+1,...,nk][f][g1,...,gk] * D[gi, x]
        let n = args.len();
        let mut dargs: Vec<Expr> = Vec::with_capacity(n);
        for arg in args {
          dargs.push(differentiate(arg, var)?);
        }

        let mut terms: Vec<Expr> = Vec::new();
        for i in 0..n {
          if matches!(&dargs[i], Expr::Integer(0)) {
            continue;
          }

          // Increment the i-th derivative index
          let new_indices: Vec<Expr> = indices
            .iter()
            .enumerate()
            .map(|(j, idx)| {
              if j == i {
                if let Expr::Integer(k) = idx {
                  Expr::Integer(k + 1)
                } else {
                  Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Plus,
                    left: Box::new(Expr::Integer(1)),
                    right: Box::new(idx.clone()),
                  }
                }
              } else {
                idx.clone()
              }
            })
            .collect();

          let deriv_expr = Expr::CurriedCall {
            func: Box::new(Expr::CurriedCall {
              func: Box::new(Expr::FunctionCall {
                name: "Derivative".to_string(),
                args: new_indices,
              }),
              args: func_names.clone(),
            }),
            args: args.clone(),
          };

          if matches!(&dargs[i], Expr::Integer(1)) {
            terms.push(deriv_expr);
          } else {
            terms.push(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(dargs[i].clone()),
              right: Box::new(deriv_expr),
            });
          }
        }

        return if terms.is_empty() {
          Ok(Expr::Integer(0))
        } else if terms.len() == 1 {
          Ok(simplify(terms.remove(0)))
        } else {
          let mut result = terms[0].clone();
          for term in &terms[1..] {
            result = Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(result),
              right: Box::new(term.clone()),
            };
          }
          Ok(simplify(result))
        };
      }
      // Fallback for other CurriedCall forms
      if is_constant_wrt(expr, var) {
        Ok(Expr::Integer(0))
      } else {
        Ok(Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())],
        })
      }
    }

    _ => {
      if is_constant_wrt(expr, var) {
        Ok(Expr::Integer(0))
      } else {
        Ok(Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())],
        })
      }
    }
  }
}

/// Build `expr / divisor`, simplifying to just `expr` when `divisor == 1`.
/// For integer divisors, produces `Rational[1, n] * expr` to match Wolfram output.
fn make_divided(expr: Expr, divisor: Expr) -> Expr {
  match &divisor {
    Expr::Integer(1) => expr,
    // expr / (a/b) → expr * b/a = (b * expr) / a
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: num,
      right: den,
    } => {
      let result = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: den.clone(),
          right: Box::new(expr),
        }),
        right: num.clone(),
      };
      simplify(result)
    }
    // expr / Rational[a, b] → (b * expr) / a
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      let num = &args[0]; // a
      let den = &args[1]; // b
      let result = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(den.clone()),
          right: Box::new(expr),
        }),
        right: Box::new(num.clone()),
      };
      simplify(result)
    }
    _ => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(expr),
      right: Box::new(divisor),
    },
  }
}

/// Build `-expr / divisor`, simplifying to `-expr` when `divisor == 1`.
/// For integer divisors, produces `Rational[-1, n] * expr` to match Wolfram output.
fn make_neg_divided(expr: Expr, divisor: Expr) -> Expr {
  match &divisor {
    Expr::Integer(1) => Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(expr),
    },
    Expr::Integer(n) => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(crate::functions::math_ast::make_rational_pub(-1, *n)),
      right: Box::new(expr),
    },
    _ => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(expr),
      }),
      right: Box::new(divisor),
    },
  }
}

/// Extract a trig function with its power from an expression.
/// Returns (function_name, argument, power) for Sin[f]^n or Cos[f]^n patterns.
/// Power defaults to 1 if not explicitly raised.
fn extract_trig_factor(expr: &Expr) -> Option<(&str, &Expr, i64)> {
  // Sin[f] or Cos[f] (power = 1)
  if let Expr::FunctionCall { name, args } = expr
    && args.len() == 1
    && (name == "Sin" || name == "Cos")
  {
    return Some((name.as_str(), &args[0], 1));
  }
  // Sin[f]^n or Cos[f]^n as BinaryOp
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left,
    right,
  } = expr
    && let Expr::Integer(n) = right.as_ref()
    && *n >= 1
    && let Expr::FunctionCall { name, args } = left.as_ref()
    && args.len() == 1
    && (name == "Sin" || name == "Cos")
  {
    return Some((name.as_str(), &args[0], *n as i64));
  }
  // Power[Sin[f], n] or Power[Cos[f], n] as FunctionCall
  if let Expr::FunctionCall { name, args } = expr
    && name == "Power"
    && args.len() == 2
    && let Expr::Integer(n) = &args[1]
    && *n >= 1
    && let Expr::FunctionCall {
      name: trig_name,
      args: trig_args,
    } = &args[0]
    && trig_args.len() == 1
    && (trig_name == "Sin" || trig_name == "Cos")
  {
    return Some((trig_name.as_str(), &trig_args[0], *n as i64));
  }
  None
}

/// Try to integrate a product of Sin[f]^m * Cos[f]^n where f is linear in var.
/// Handles:
///   - Sin[f] * Cos[f]^n → -Cos[f]^(n+1) / ((n+1)*a)
///   - Sin[f]^m * Cos[f] → Sin[f]^(m+1) / ((m+1)*a)
///   - General odd power cases via reduction
fn try_integrate_sin_cos_product(factors: &[&Expr], var: &str) -> Option<Expr> {
  let mut sin_arg: Option<&Expr> = None;
  let mut sin_power: i64 = 0;
  let mut cos_arg: Option<&Expr> = None;
  let mut cos_power: i64 = 0;

  for factor in factors {
    if let Some((name, arg, power)) = extract_trig_factor(factor) {
      match name {
        "Sin" => {
          if sin_power > 0 && !expr_str_eq(sin_arg.unwrap(), arg) {
            return None;
          }
          sin_arg = Some(arg);
          sin_power += power;
        }
        "Cos" => {
          if cos_power > 0 && !expr_str_eq(cos_arg.unwrap(), arg) {
            return None;
          }
          cos_arg = Some(arg);
          cos_power += power;
        }
        _ => return None,
      }
    } else {
      // Non-trig factor that depends on var: can't handle
      if !is_constant_wrt(factor, var) {
        return None;
      }
    }
  }

  // Need both Sin and Cos present
  if sin_power == 0 || cos_power == 0 {
    return None;
  }

  let sin_a = sin_arg?;
  let cos_a = cos_arg?;
  // Arguments must be the same
  if !expr_str_eq(sin_a, cos_a) {
    return None;
  }

  let arg = sin_a;
  let coeff = try_match_linear_arg(arg, var)?;

  // When sin_power is odd (priority) or == 1: use u = Cos[f] substitution
  // ∫ Sin[f]^m * Cos[f]^n dx where m is odd:
  //   Factor out Sin[f], convert Sin[f]^(m-1) = (1-Cos[f]^2)^((m-1)/2)
  //   u = Cos[f], du = -a*Sin[f]dx
  //   = -1/a * ∫ (1-u^2)^((m-1)/2) * u^n du
  //
  // When sin_power == 1: ∫ Sin[f] * Cos[f]^n dx = -Cos[f]^(n+1) / ((n+1)*a)
  if sin_power == 1 {
    let new_power = cos_power + 1;
    let cos_expr = Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![arg.clone()],
    };
    let power_expr = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(cos_expr),
      right: Box::new(Expr::Integer(new_power as i128)),
    };
    // divisor = (n+1) * a
    let total_divisor = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Integer(new_power as i128)),
      right: Box::new(coeff),
    });
    return Some(make_neg_divided(power_expr, total_divisor));
  }

  // When cos_power == 1: ∫ Sin[f]^m * Cos[f] dx = Sin[f]^(m+1) / ((m+1)*a)
  if cos_power == 1 {
    let new_power = sin_power + 1;
    let sin_expr = Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![arg.clone()],
    };
    let power_expr = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(sin_expr),
      right: Box::new(Expr::Integer(new_power as i128)),
    };
    // divisor = (m+1) * a
    let total_divisor = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Integer(new_power as i128)),
      right: Box::new(coeff),
    });
    return Some(make_divided(power_expr, total_divisor));
  }

  // General case: both powers > 1
  // If sin_power is odd: reduce using Sin^2 = 1 - Cos^2
  if sin_power % 2 == 1 {
    // Sin[f]^m * Cos[f]^n = Sin[f] * (1-Cos[f]^2)^((m-1)/2) * Cos[f]^n
    // Expand (1-Cos[f]^2)^k and integrate each term with the sin_power=1 rule
    let k = (sin_power - 1) / 2;
    let cos_f = Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![arg.clone()],
    };
    // Expand (1-u^2)^k using binomial theorem
    // = sum_{j=0}^{k} C(k,j) * (-1)^j * u^(2j)
    // So integral = sum_{j=0}^{k} C(k,j) * (-1)^j * ∫ Sin[f] * Cos[f]^(n+2j) dx
    //            = sum_{j=0}^{k} C(k,j) * (-1)^j * (-Cos[f]^(n+2j+1) / ((n+2j+1)*a))
    let mut terms: Vec<Expr> = Vec::new();
    for j in 0..=k {
      let binom = binomial_coeff(k, j);
      let sign = if j % 2 == 0 { 1i128 } else { -1 };
      let new_cos_power = cos_power + 2 * j;
      let new_power = new_cos_power + 1;
      let cos_power_expr = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(cos_f.clone()),
        right: Box::new(Expr::Integer(new_power as i128)),
      };
      // coefficient = binom * sign * (-1) / ((new_power) * a)
      // = -binom * sign / (new_power * a)
      let numer = -sign * binom;
      let total_divisor = simplify(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::Integer(new_power as i128)),
        right: Box::new(coeff.clone()),
      });
      let term = if numer == 1 {
        make_divided(cos_power_expr, total_divisor)
      } else if numer == -1 {
        make_neg_divided(cos_power_expr, total_divisor)
      } else {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(make_divided(Expr::Integer(numer), total_divisor)),
          right: Box::new(cos_power_expr),
        }
      };
      terms.push(term);
    }
    return Some(if terms.len() == 1 {
      terms.remove(0)
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms,
      }
    });
  }

  // If cos_power is odd: reduce using Cos^2 = 1 - Sin^2
  if cos_power % 2 == 1 {
    let k = (cos_power - 1) / 2;
    let sin_f = Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![arg.clone()],
    };
    let mut terms: Vec<Expr> = Vec::new();
    for j in 0..=k {
      let binom = binomial_coeff(k, j);
      let sign = if j % 2 == 0 { 1i128 } else { -1 };
      let new_sin_power = sin_power + 2 * j;
      let new_power = new_sin_power + 1;
      let sin_power_expr = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(sin_f.clone()),
        right: Box::new(Expr::Integer(new_power as i128)),
      };
      let numer = sign * binom;
      let total_divisor = simplify(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::Integer(new_power as i128)),
        right: Box::new(coeff.clone()),
      });
      let term = if numer == 1 {
        make_divided(sin_power_expr, total_divisor)
      } else if numer == -1 {
        make_neg_divided(sin_power_expr, total_divisor)
      } else {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(make_divided(Expr::Integer(numer), total_divisor)),
          right: Box::new(sin_power_expr),
        }
      };
      terms.push(term);
    }
    return Some(if terms.len() == 1 {
      terms.remove(0)
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms,
      }
    });
  }

  // Both even: use double-angle reduction (not yet implemented)
  None
}

/// Compute binomial coefficient C(n, k)
fn binomial_coeff(n: i64, k: i64) -> i128 {
  if k < 0 || k > n {
    return 0;
  }
  if k == 0 || k == n {
    return 1;
  }
  let k = k.min(n - k) as i128;
  let n = n as i128;
  let mut result: i128 = 1;
  for i in 0..k {
    result = result * (n - i) / (i + 1);
  }
  result
}

/// Build the antiderivative of Exp[-a*x^2]:
///   Sqrt[Pi/a]/2 * Erf[Sqrt[a]*x]  (general a)
///   (Sqrt[Pi]*Erf[x])/2            (when a=1)
fn make_gaussian_antiderivative(var: &str, coeff: &Expr) -> Expr {
  let var_expr = Expr::Identifier(var.to_string());
  let (erf_arg, prefix) = match coeff {
    Expr::Integer(1) => {
      // a=1: Erf[x], prefix = Sqrt[Pi]
      (var_expr, make_sqrt(Expr::Constant("Pi".to_string())))
    }
    Expr::Integer(n) if *n != 1 => {
      // concrete integer a: (Sqrt[Pi/a]*Erf[Sqrt[a]*x])/2 — matches Wolfram output
      let sqrt_a = make_sqrt(coeff.clone());
      let erf_arg = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(sqrt_a),
        right: Box::new(var_expr),
      };
      let prefix = make_sqrt(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(coeff.clone()),
      });
      (erf_arg, prefix)
    }
    _ => {
      // symbolic a: (Sqrt[Pi]*Erf[Sqrt[a]*x])/(2*Sqrt[a]) — matches Wolfram output
      let sqrt_a = make_sqrt(coeff.clone());
      let erf_arg = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(sqrt_a.clone()),
        right: Box::new(var_expr),
      };
      let prefix = make_sqrt(Expr::Constant("Pi".to_string()));
      let erf_expr = Expr::FunctionCall {
        name: "Erf".to_string(),
        args: vec![erf_arg],
      };
      // (Sqrt[Pi] * Erf[Sqrt[a]*x]) / (2 * Sqrt[a])
      return Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(prefix),
          right: Box::new(erf_expr),
        }),
        right: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(2)),
          right: Box::new(sqrt_a),
        }),
      };
    }
  };
  let erf_expr = Expr::FunctionCall {
    name: "Erf".to_string(),
    args: vec![erf_arg],
  };
  // a=1 case: (Sqrt[Pi] * Erf[x]) / 2
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(prefix),
      right: Box::new(erf_expr),
    }),
    right: Box::new(Expr::Integer(2)),
  }
}

/// Try to match an expression as `a*var` where `a` is constant w.r.t. `var`,
/// or just `var` (returning `Integer(1)`).
/// Returns Some(a) if it matches, None otherwise.
fn try_match_linear_arg(expr: &Expr, var: &str) -> Option<Expr> {
  match expr {
    Expr::Identifier(name) if name == var => Some(Expr::Integer(1)),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if is_constant_wrt(left, var)
        && matches!(right.as_ref(), Expr::Identifier(name) if name == var)
      {
        Some(*left.clone())
      } else if is_constant_wrt(right, var)
        && matches!(left.as_ref(), Expr::Identifier(name) if name == var)
      {
        Some(*right.clone())
      } else {
        None
      }
    }
    // x/c form: var/const → coefficient is 1/const (i.e., Rational[1,c] for integer c)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Identifier(name) if name == var)
        && is_constant_wrt(right, var)
      {
        // coefficient = 1/right — use division_ast for proper Rational creation
        if let Ok(result) = crate::functions::math_ast::divide_ast(&[
          Expr::Integer(1),
          *right.clone(),
        ]) {
          Some(result)
        } else {
          Some(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(Expr::Integer(1)),
            right: right.clone(),
          })
        }
      } else if is_constant_wrt(right, var) {
        // (expr)/const where expr might be a*x → coefficient is a/const
        if let Some(inner_coeff) = try_match_linear_arg(left, var) {
          if let Ok(result) = crate::functions::math_ast::divide_ast(&[
            inner_coeff.clone(),
            *right.clone(),
          ]) {
            Some(result)
          } else {
            Some(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Divide,
              left: Box::new(inner_coeff),
              right: right.clone(),
            })
          }
        } else {
          None
        }
      } else {
        None
      }
    }
    // FunctionCall("Times", [coeff, var]) form
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Find the variable factor and collect the rest as coefficient
      let mut var_idx = None;
      for (i, arg) in args.iter().enumerate() {
        if matches!(arg, Expr::Identifier(n) if n == var) {
          var_idx = Some(i);
          break;
        }
      }
      if let Some(vi) = var_idx {
        let rest: Vec<Expr> = args
          .iter()
          .enumerate()
          .filter(|(i, _)| *i != vi)
          .map(|(_, e)| e.clone())
          .collect();
        if rest.iter().all(|a| is_constant_wrt(a, var)) {
          if rest.len() == 1 {
            Some(rest[0].clone())
          } else {
            Some(Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest,
            })
          }
        } else {
          None
        }
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Extract the coefficient of `var` from a linear expression `a*var + b`.
/// Returns `Some(a)` if the expression is linear in `var`, `None` otherwise.
/// Works by differentiating the expression: if d/dx(expr) is constant w.r.t. var,
/// then expr is linear and the derivative is the coefficient.
fn extract_linear_coefficient(expr: &Expr, var: &str) -> Option<Expr> {
  // Check that expr actually depends on var
  if is_constant_wrt(expr, var) {
    return None;
  }
  // Differentiate: if expr = a*x + b, then d/dx(expr) = a (constant)
  let deriv = differentiate(expr, var).ok()?;
  let deriv = simplify(deriv);
  if is_constant_wrt(&deriv, var) {
    Some(deriv)
  } else {
    None
  }
}

/// Try to integrate Sin[f(x)]^2 or Cos[f(x)]^2 using power-reduction identities.
/// sin²(a*x) = x/2 - sin(2*a*x)/(4*a)
/// cos²(a*x) = x/2 + sin(2*a*x)/(4*a)
fn try_integrate_trig_squared(base: &Expr, var: &str) -> Option<Expr> {
  if let Expr::FunctionCall { name, args } = base
    && args.len() == 1
  {
    let is_sin = name == "Sin";
    let is_cos = name == "Cos";
    let is_sec = name == "Sec";
    let is_csc = name == "Csc";
    // ∫ Sec[a*x]^2 dx = Tan[a*x]/a
    if is_sec {
      let coeff = try_match_linear_arg(&args[0], var)?;
      let tan_expr = Expr::FunctionCall {
        name: "Tan".to_string(),
        args: args.clone(),
      };
      return Some(make_divided(tan_expr, coeff));
    }
    // ∫ Csc[a*x]^2 dx = -Cot[a*x]/a
    if is_csc {
      let coeff = try_match_linear_arg(&args[0], var)?;
      let cot_expr = Expr::FunctionCall {
        name: "Cot".to_string(),
        args: args.clone(),
      };
      return Some(make_neg_divided(cot_expr, coeff));
    }
    if !is_sin && !is_cos {
      return None;
    }
    let coeff = try_match_linear_arg(&args[0], var)?;
    // Build: 2*a*x
    let double_arg = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(args[0].clone()),
    });
    // sin(2*a*x)
    let sin_double = Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![double_arg],
    };
    // 4*a
    let four_a = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Integer(4)),
      right: Box::new(coeff),
    });
    // x/2
    let x_half = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Identifier(var.to_string())),
      right: Box::new(Expr::Integer(2)),
    };
    // sin(2*a*x)/(4*a)
    let sin_term = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(sin_double),
      right: Box::new(four_a),
    };
    if is_sin {
      // x/2 - sin(2*a*x)/(4*a)
      Some(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(x_half),
        right: Box::new(sin_term),
      })
    } else {
      // x/2 + sin(2*a*x)/(4*a)
      Some(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(x_half),
        right: Box::new(sin_term),
      })
    }
  } else {
    None
  }
}

/// Compute binomial coefficient C(n, k)
fn binomial(n: i128, k: i128) -> i128 {
  if k < 0 || k > n {
    return 0;
  }
  let k = k.min(n - k); // optimization: C(n,k) = C(n,n-k)
  let mut result: i128 = 1;
  for i in 0..k {
    result = result * (n - i) / (i + 1);
  }
  result
}

/// Try to integrate Sin[a*x]^n or Cos[a*x]^n for positive integer n ≥ 3
/// using Chebyshev expansion (multiple angle formula).
///
/// For odd n = 2m+1:
///   sin^n(x) = (1/4^m) * Sum_{k=0}^{m} (-1)^(m-k) * C(n,k) * sin((n-2k)*x)
///   cos^n(x) = (1/4^m) * Sum_{k=0}^{m} C(n,k) * cos((n-2k)*x)
///
/// For even n = 2m:
///   sin^n(x) = (1/4^m) * [C(n,m) + 2 * Sum_{k=0}^{m-1} (-1)^(m-k) * C(n,k) * cos((n-2k)*x)]
///   cos^n(x) = (1/4^m) * [C(n,m) + 2 * Sum_{k=0}^{m-1} C(n,k) * cos((n-2k)*x)]
fn try_integrate_trig_power(base: &Expr, n: i128, var: &str) -> Option<Expr> {
  if n < 3 {
    return None;
  }
  let (name, arg) = match base {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      let is_sin = name == "Sin";
      let is_cos = name == "Cos";
      if !is_sin && !is_cos {
        return None;
      }
      (name.as_str(), &args[0])
    }
    _ => return None,
  };

  // Match linear argument a*x
  let coeff = try_match_linear_arg(arg, var)?;
  let is_sin = name == "Sin";

  // Build the Chebyshev expansion terms and integrate each
  let mut terms: Vec<Expr> = Vec::new();
  let m = n / 2;
  let is_odd = n % 2 != 0;

  if is_odd {
    // Odd power: n = 2m+1
    // For sin: integral of (-1)^(m-k)*C(n,k)*sin((n-2k)*x) → absorb -1/k into coefficient
    //   coeff = (-1)^(m-k+1) * C(n,k), trig = Cos[(n-2k)*x]
    // For cos: integral of C(n,k)*cos((n-2k)*x) → coeff = C(n,k)/k, trig = Sin[(n-2k)*x]
    for k in 0..=m {
      let freq = n - 2 * k; // always positive since k ≤ m and n=2m+1
      let binom = binomial(n, k);

      // Build the trig argument: freq * a * x
      let freq_arg = if matches!(&coeff, Expr::Integer(1)) {
        if freq == 1 {
          Expr::Identifier(var.to_string())
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(freq)),
            right: Box::new(Expr::Identifier(var.to_string())),
          }
        }
      } else {
        simplify(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(freq)),
          right: Box::new(arg.clone()),
        })
      };

      // For sin^n: integral coefficient includes the -1 from integrating sin
      // sign = (-1)^(m-k) for the Chebyshev expansion, then *(-1) for integral of sin
      // = (-1)^(m-k+1)
      // For cos^n: sign = +1 (Chebyshev) and integral of cos gives sin (no sign change)
      let coeff_num = if is_sin {
        let exp = (m - k + 1) % 2;
        if exp == 0 { binom } else { -binom }
      } else {
        binom
      };

      let integrated_trig = Expr::FunctionCall {
        name: if is_sin { "Cos" } else { "Sin" }.to_string(),
        args: vec![freq_arg],
      };

      // Total coefficient: coeff_num / (freq * 4^m)
      let denom = freq * (1i128 << (2 * m)); // freq * 4^m
      let term = if matches!(&coeff, Expr::Integer(1)) {
        let g = gcd_i64(
          coeff_num.unsigned_abs() as i128,
          denom.unsigned_abs() as i128,
        );
        let num = coeff_num / g;
        let den = denom / g;
        make_fraction_term(num, den, integrated_trig)
      } else {
        let g = gcd_i64(
          coeff_num.unsigned_abs() as i128,
          denom.unsigned_abs() as i128,
        );
        let num = coeff_num / g;
        let den = denom / g;
        let den_expr = simplify(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(den)),
          right: Box::new(coeff.clone()),
        });
        make_fraction_term_expr(num, den_expr, integrated_trig)
      };
      terms.push(term);
    }
  } else {
    // Even power: n = 2m
    // Constant term: C(n,m) / 4^m * x
    let binom_mid = binomial(n, m);
    let power_4m = 1i128 << (2 * m); // 4^m
    let g = gcd_i64(binom_mid, power_4m);
    let const_num = binom_mid / g;
    let const_den = power_4m / g;
    let const_term = if const_den == 1 {
      if const_num == 1 {
        Expr::Identifier(var.to_string())
      } else {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(const_num)),
          right: Box::new(Expr::Identifier(var.to_string())),
        }
      }
    } else if matches!(&coeff, Expr::Integer(1)) {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(if const_num == 1 {
          Expr::Identifier(var.to_string())
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(const_num)),
            right: Box::new(Expr::Identifier(var.to_string())),
          }
        }),
        right: Box::new(Expr::Integer(const_den)),
      }
    } else {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(if const_num == 1 {
          Expr::Identifier(var.to_string())
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(const_num)),
            right: Box::new(Expr::Identifier(var.to_string())),
          }
        }),
        right: Box::new(Expr::Integer(const_den)),
      }
    };
    terms.push(const_term);

    // Oscillating terms
    for k in 0..m {
      let freq = n - 2 * k;
      let binom = binomial(n, k);
      let sign = if is_sin {
        if (m - k) % 2 == 0 { 1i128 } else { -1i128 }
      } else {
        1
      };
      let coeff_num = 2 * sign * binom;

      let freq_arg = if matches!(&coeff, Expr::Integer(1)) {
        if freq == 1 {
          Expr::Identifier(var.to_string())
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(freq)),
            right: Box::new(Expr::Identifier(var.to_string())),
          }
        }
      } else {
        simplify(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(freq)),
          right: Box::new(arg.clone()),
        })
      };

      // Integrated: sin(freq*x)/(freq) for both sin^n and cos^n even powers
      let integrated_trig = Expr::FunctionCall {
        name: "Sin".to_string(),
        args: vec![freq_arg],
      };

      let denom = freq * power_4m;
      let term = if matches!(&coeff, Expr::Integer(1)) {
        let g = gcd_i64(
          coeff_num.unsigned_abs() as i128,
          denom.unsigned_abs() as i128,
        );
        let num = coeff_num / g;
        let den = denom / g;
        make_fraction_term(num, den, integrated_trig)
      } else {
        let g = gcd_i64(
          coeff_num.unsigned_abs() as i128,
          denom.unsigned_abs() as i128,
        );
        let num = coeff_num / g;
        let den = denom / g;
        let den_expr = simplify(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(den)),
          right: Box::new(coeff.clone()),
        });
        make_fraction_term_expr(num, den_expr, integrated_trig)
      };
      terms.push(term);
    }
  }

  if terms.is_empty() {
    return None;
  }

  // Combine terms using plus_ast for canonical ordering
  let result = crate::functions::math_ast::plus_ast(&terms).ok()?;
  Some(result)
}

/// Build a term: (num/den) * expr, simplified
fn make_fraction_term(num: i128, den: i128, expr: Expr) -> Expr {
  if den == 1 {
    if num == 1 {
      expr
    } else if num == -1 {
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(expr),
      }
    } else {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::Integer(num)),
        right: Box::new(expr),
      }
    }
  } else if num == 1 {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(expr),
      right: Box::new(Expr::Integer(den)),
    }
  } else if num == -1 {
    // -(expr/den) so plus_ast displays as "- expr/den"
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(expr),
        right: Box::new(Expr::Integer(den)),
      }),
    }
  } else if num < 0 {
    // -(|num|*expr/den) so plus_ast displays as "- num*expr/den"
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(-num)),
          right: Box::new(expr),
        }),
        right: Box::new(Expr::Integer(den)),
      }),
    }
  } else {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::Integer(num)),
        right: Box::new(expr),
      }),
      right: Box::new(Expr::Integer(den)),
    }
  }
}

/// Build a term: (num/den_expr) * expr, simplified
fn make_fraction_term_expr(num: i128, den_expr: Expr, expr: Expr) -> Expr {
  let num_expr = if num == 1 {
    expr
  } else if num == -1 {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(expr),
    }
  } else {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Integer(num)),
      right: Box::new(expr),
    }
  };
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(num_expr),
    right: Box::new(den_expr),
  }
}

fn gcd_i64(mut a: i128, mut b: i128) -> i128 {
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a.abs()
}

/// Try to match ∫ E^(a*x) / (c*x) dx = ExpIntegralEi[a*x] / c
/// Handles: E^(a*x) / x, E^(a*x) / (c*x), E^x / x, E^x / (c*x)
/// Also handles Exp[a*x] function form.
fn try_match_exp_over_linear(
  numerator: &Expr,
  denominator: &Expr,
  var: &str,
) -> Option<Expr> {
  // Check if numerator is E^(a*x) (Power form or Exp function form)
  let exp_linear_arg = match numerator {
    // E^(a*x) as BinaryOp::Power
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: base,
      right: exp,
    } if matches!(base.as_ref(), Expr::Constant(c) if c == "E") => {
      try_match_linear_arg(exp, var)
    }
    // Exp[a*x] as FunctionCall
    Expr::FunctionCall { name, args } if name == "Exp" && args.len() == 1 => {
      try_match_linear_arg(&args[0], var)
    }
    _ => None,
  };

  let linear_coeff = exp_linear_arg?; // a in E^(a*x)

  // Check if denominator is c*x or just x
  let denom_const = match denominator {
    // Just x
    Expr::Identifier(name) if name == var => Some(Expr::Integer(1)),
    // c*x (BinaryOp form)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if is_constant_wrt(left, var)
        && matches!(right.as_ref(), Expr::Identifier(name) if name == var)
      {
        Some(*left.clone())
      } else if is_constant_wrt(right, var)
        && matches!(left.as_ref(), Expr::Identifier(name) if name == var)
      {
        Some(*right.clone())
      } else {
        None
      }
    }
    // Times[c, x] (FunctionCall form)
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if is_constant_wrt(&args[0], var)
        && matches!(&args[1], Expr::Identifier(name) if name == var)
      {
        Some(args[0].clone())
      } else if is_constant_wrt(&args[1], var)
        && matches!(&args[0], Expr::Identifier(name) if name == var)
      {
        Some(args[1].clone())
      } else {
        None
      }
    }
    _ => None,
  };

  let denom_const = denom_const?; // c in c*x

  // Build ExpIntegralEi[a*x]
  let ei_arg = if matches!(&linear_coeff, Expr::Integer(1)) {
    Expr::Identifier(var.to_string())
  } else {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(linear_coeff),
      right: Box::new(Expr::Identifier(var.to_string())),
    }
  };
  let ei_expr = Expr::FunctionCall {
    name: "ExpIntegralEi".to_string(),
    args: vec![ei_arg],
  };

  // Return ExpIntegralEi[a*x] / c
  if matches!(&denom_const, Expr::Integer(1)) {
    Some(ei_expr)
  } else {
    Some(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(ei_expr),
      right: Box::new(denom_const),
    })
  }
}

/// Check if an expression is Rational[-1, 2] (i.e., exponent -1/2)
fn is_rational_neg_half(expr: &Expr) -> bool {
  matches!(
    expr,
    Expr::FunctionCall { name, args }
      if name == "Rational"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(-1))
        && matches!(&args[1], Expr::Integer(2))
  )
}

/// Try to integrate (base)^(-1/2) for special forms:
/// ∫ (1 - x^2)^(-1/2) dx = ArcSin[x]
/// ∫ (1 + x^2)^(-1/2) dx = ArcSinh[x]
fn try_integrate_inverse_sqrt(base: &Expr, var: &str) -> Option<Expr> {
  // Try to match the base as a + b*x^2 by extracting polynomial coefficients
  let base_eval =
    crate::evaluator::evaluate_expr_to_expr(base).unwrap_or(base.clone());
  let var_expr = Expr::Identifier(var.to_string());

  // Use CoefficientList to extract coefficients
  let coeff_result = crate::functions::polynomial_ast::coefficient_list_ast(&[
    base_eval, var_expr,
  ])
  .ok()?;
  let coeffs = match &coeff_result {
    Expr::List(items) => items,
    _ => return None,
  };

  // We need exactly a polynomial of degree 2 with no linear term: a + 0*x + b*x^2
  if coeffs.len() != 3 {
    return None;
  }
  // Linear coefficient must be zero
  let c1_val = crate::functions::math_ast::try_eval_to_f64(&coeffs[1])?;
  if c1_val.abs() > 1e-15 {
    return None;
  }

  let a = &coeffs[0]; // constant term
  let b = &coeffs[2]; // x^2 coefficient

  let a_val = crate::functions::math_ast::try_eval_to_f64(a)?;
  let b_val = crate::functions::math_ast::try_eval_to_f64(b)?;

  if a_val <= 0.0 {
    return None;
  }

  if b_val < 0.0 {
    // ∫ (a - |b|*x^2)^(-1/2) dx = (1/sqrt(|b|)) * ArcSin[x * sqrt(|b|/a)]
    let abs_b = simplify(Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(b.clone()),
    });
    let ratio = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(abs_b.clone()),
      right: Box::new(a.clone()),
    });
    let sqrt_ratio = simplify(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![ratio],
    });
    let sqrt_abs_b = simplify(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![abs_b],
    });
    let arg = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Identifier(var.to_string())),
      right: Box::new(sqrt_ratio),
    });
    let arcsin = Expr::FunctionCall {
      name: "ArcSin".to_string(),
      args: vec![arg],
    };
    Some(make_divided(arcsin, sqrt_abs_b))
  } else {
    // ∫ (a + b*x^2)^(-1/2) dx = (1/sqrt(b)) * ArcSinh[x * sqrt(b/a)]
    let ratio = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(b.clone()),
      right: Box::new(a.clone()),
    });
    let sqrt_ratio = simplify(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![ratio],
    });
    let sqrt_b = simplify(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![b.clone()],
    });
    let arg = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Identifier(var.to_string())),
      right: Box::new(sqrt_ratio),
    });
    let arcsinh = Expr::FunctionCall {
      name: "ArcSinh".to_string(),
      args: vec![arg],
    };
    Some(make_divided(arcsinh, sqrt_b))
  }
}

/// Try to integrate a rational function (numerator/denominator where both are polynomials).
/// Uses polynomial long division + partial fraction decomposition.
fn try_integrate_rational(
  num_expr: &Expr,
  den_expr: &Expr,
  var: &str,
) -> Option<Expr> {
  use crate::functions::polynomial_ast::{
    build_sum, coeffs_to_expr, divide_by_root, evaluate_poly,
    expand_and_combine, extract_poly_coeffs, find_integer_root, gcd_i128,
    poly_long_divide,
  };
  use crate::syntax::BinaryOperator;

  // Step 1: Extract polynomial coefficients
  let num_expanded = expand_and_combine(num_expr);
  let den_expanded = expand_and_combine(den_expr);
  let num_coeffs = extract_poly_coeffs(&num_expanded, var)?;
  let den_coeffs = extract_poly_coeffs(&den_expanded, var)?;

  // Need at least degree 1 denominator
  if den_coeffs.len() <= 1 {
    return None;
  }

  // Step 2: Polynomial long division (if needed)
  let (quotient_integral, proper_num) = if num_coeffs.len() >= den_coeffs.len()
  {
    let (quotient, remainder) = poly_long_divide(&num_coeffs, &den_coeffs);
    // Check that division actually worked (poly_long_divide returns (vec![0], original) on failure)
    if quotient == vec![0] && remainder == num_coeffs {
      return None;
    }
    let quot_expr = coeffs_to_expr(&quotient, var);
    let quot_integral = integrate(&quot_expr, var)?;
    // If remainder is all zeros, just return the quotient integral
    if remainder.iter().all(|&c| c == 0) {
      return Some(quot_integral);
    }
    (Some(quot_integral), remainder)
  } else {
    (None, num_coeffs)
  };

  // If proper numerator is all zeros, return just quotient integral
  if proper_num.iter().all(|&c| c == 0) {
    return quotient_integral;
  }

  // Step 3: Factor denominator
  let gcd_coeff = den_coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if gcd_coeff == 0 {
    return None;
  }
  let reduced: Vec<i128> = den_coeffs.iter().map(|c| c / gcd_coeff).collect();
  let (sign, reduced) = if reduced.last().map(|&c| c < 0).unwrap_or(false) {
    (-1i128, reduced.iter().map(|c| -c).collect::<Vec<_>>())
  } else {
    (1, reduced)
  };
  let overall_factor = gcd_coeff * sign;

  // Find integer roots
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

  // Remaining factor: if degree > 2, bail out
  // Trim trailing zeros from remaining
  while remaining.len() > 1 && remaining.last() == Some(&0) {
    remaining.pop();
  }
  let remaining_deg =
    if remaining.len() <= 1 && remaining.first().copied().unwrap_or(0) != 0 {
      0 // constant
    } else if remaining.len() <= 1 {
      0
    } else {
      remaining.len() - 1
    };
  if remaining_deg > 2 {
    return None;
  }

  // Must have at least one root or a quadratic remaining to do something useful
  if roots.is_empty() && remaining_deg < 2 {
    return None;
  }

  // Sort roots for consistent output (descending, so linear factors appear in ascending order)
  roots.sort_by(|a, b| b.cmp(a));

  // Step 4: Compute residues for linear roots
  let mut log_terms: Vec<Expr> = Vec::new();

  for (i, &root) in roots.iter().enumerate() {
    let num_at_root = evaluate_poly(&proper_num, root);
    let mut den_product = 1i128;
    for (j, &other_root) in roots.iter().enumerate() {
      if i != j {
        den_product = den_product.checked_mul(root - other_root)?;
      }
    }
    // Include remaining factor evaluation
    if remaining_deg > 0 {
      let rem_at_root = evaluate_poly(&remaining, root);
      den_product = den_product.checked_mul(rem_at_root)?;
    } else if remaining.len() == 1 && remaining[0] != 0 && remaining[0] != 1 {
      den_product = den_product.checked_mul(remaining[0])?;
    }
    den_product = den_product.checked_mul(overall_factor)?;

    if den_product == 0 {
      return None;
    }

    // A_i = num_at_root / den_product as reduced fraction
    let g = gcd_i128(num_at_root.abs(), den_product.abs());
    let (mut an, mut ad) = (num_at_root / g, den_product / g);
    if ad < 0 {
      an = -an;
      ad = -ad;
    }

    if an == 0 {
      continue;
    }

    // Step 5: Build Log terms
    // Convention: argument is positive at x=0
    // root > 0: Log[root - x], root < 0: Log[-root + x], root = 0: Log[x]
    let log_arg = if root == 0 {
      Expr::Identifier(var.to_string())
    } else if root > 0 {
      // Log[root - x]
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(Expr::Integer(root)),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    } else {
      // root < 0: Log[-root + x]
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-root)),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    };

    let log_expr = Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![log_arg],
    };

    // Build coefficient * Log[...]
    let term = if ad == 1 {
      if an == 1 {
        log_expr
      } else if an == -1 {
        Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(log_expr),
        }
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(an)),
          right: Box::new(log_expr),
        }
      }
    } else {
      // Use Rational form: Rational[an, ad] * Log[...]
      // But for positive fractions, Wolfram outputs Log[...]/ad
      // and for negative, -Log[...]/ad, etc.
      let abs_an = an.abs();
      let frac_expr = if abs_an == 1 {
        Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(log_expr.clone()),
          right: Box::new(Expr::Integer(ad)),
        }
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(crate::functions::math_ast::make_rational_pub(
            abs_an, ad,
          )),
          right: Box::new(log_expr.clone()),
        }
      };
      if an < 0 {
        Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(frac_expr),
        }
      } else {
        frac_expr
      }
    };

    log_terms.push(term);
  }

  // Step 6 & 7: Handle quadratic part
  let mut quad_terms: Vec<Expr> = Vec::new();

  if remaining_deg == 2 {
    // Remaining is a*x^2 + b*x + c (monic after normalization from factoring)
    // remaining = [c, b, a] where a should be 1 (monic from divide_by_root)
    let rem_a = remaining[2];
    let rem_b = remaining[1];
    let rem_c = remaining[0];

    // Normalize to monic: x^2 + (b/a)*x + (c/a)
    // Since we factored out integer roots, remaining should already be monic (a=1)
    if rem_a != 1 {
      return None; // Can't handle non-monic quadratics easily
    }

    let b = rem_b; // coefficient of x
    let c = rem_c; // constant term

    // discriminant: b^2 - 4c (for x^2 + bx + c)
    let disc = b * b - 4 * c;
    if disc >= 0 {
      // Reducible quadratic - shouldn't happen since we already extracted all integer roots
      // But if it does, bail out
      return None;
    }

    // Compute (Bx + C) numerator of partial fraction for quadratic factor.
    // Use polynomial identity:
    // proper_num(x) = sum(A_j * overall_factor * cofactor_j(x) * quad(x))
    //                 + (Bx + C) * overall_factor * linear_product(x)
    // where cofactor_j(x) = prod_{k!=j}(x - r_k), linear_product(x) = prod(x - r_j)
    //
    // Strategy: compute the sum as a rational polynomial, subtract from proper_num,
    // then divide by overall_factor * linear_product to get (Bx + C).

    // Build linear_product polynomial = prod(x - r_j)
    let mut lin_prod_poly = vec![1i128]; // start with constant 1
    for &root in &roots {
      // Multiply by (x - root): new[i] = old[i-1] - root * old[i]
      let mut new_poly = vec![0i128; lin_prod_poly.len() + 1];
      for (i, &coeff) in lin_prod_poly.iter().enumerate() {
        new_poly[i] -= root * coeff;
        new_poly[i + 1] += coeff;
      }
      lin_prod_poly = new_poly;
    }

    // Compute the sum of residue contributions as a rational polynomial.
    // Use a common denominator: multiply proper_num by LCM of all ad_j,
    // subtract the integer contributions, then divide.
    // First, collect residues as (an_j, ad_j) pairs.
    let mut residues: Vec<(i128, i128)> = Vec::new();
    for (i, &root) in roots.iter().enumerate() {
      let num_at_root = evaluate_poly(&proper_num, root);
      let mut den_prod = 1i128;
      for (j, &other_root) in roots.iter().enumerate() {
        if i != j {
          den_prod *= root - other_root;
        }
      }
      let rem_at_root = evaluate_poly(&remaining, root);
      den_prod *= rem_at_root;
      den_prod *= overall_factor;
      if den_prod == 0 {
        return None;
      }
      let g = gcd_i128(num_at_root.abs(), den_prod.abs());
      let (mut an, mut ad) = (num_at_root / g, den_prod / g);
      if ad < 0 {
        an = -an;
        ad = -ad;
      }
      residues.push((an, ad));
    }

    // Compute LCM of all denominators
    let mut lcm = 1i128;
    for &(_, ad) in &residues {
      let g = gcd_i128(lcm, ad);
      lcm = (lcm / g).checked_mul(ad)?;
    }

    // Build: lcm * proper_num - sum(lcm/ad_j * an_j * overall_factor * cofactor_j * quad)
    // = lcm * (Bx + C) * overall_factor * linear_product
    let mut residual: Vec<i128> = proper_num.iter().map(|&c| c * lcm).collect();
    // Pad to degree of den
    let target_deg = den_coeffs.len() - 1;
    while residual.len() <= target_deg {
      residual.push(0);
    }

    let quad_poly = remaining.clone(); // [c, b, 1]

    for (i, &_root) in roots.iter().enumerate() {
      let (an, ad) = residues[i];
      let scale = (lcm / ad) * an * overall_factor;

      // Build cofactor_j(x) = prod_{k!=j}(x - r_k)
      let mut cofactor = vec![1i128];
      for (j, &other_root) in roots.iter().enumerate() {
        if i != j {
          let mut new_poly = vec![0i128; cofactor.len() + 1];
          for (k, &coeff) in cofactor.iter().enumerate() {
            new_poly[k] -= other_root * coeff;
            new_poly[k + 1] += coeff;
          }
          cofactor = new_poly;
        }
      }

      // Multiply cofactor by quad_poly
      let mut product = vec![0i128; cofactor.len() + quad_poly.len() - 1];
      for (ci, &cv) in cofactor.iter().enumerate() {
        for (qi, &qv) in quad_poly.iter().enumerate() {
          product[ci + qi] += cv * qv;
        }
      }

      // Subtract scale * product from residual
      for (k, &pv) in product.iter().enumerate() {
        if k < residual.len() {
          residual[k] -= scale * pv;
        }
      }
    }

    // Now residual = lcm * (Bx + C) * overall_factor * linear_product
    // Divide residual by (overall_factor * linear_product)
    let mut divisor_poly = lin_prod_poly.clone();
    // Multiply divisor by overall_factor
    for c in &mut divisor_poly {
      *c *= overall_factor;
    }
    // Polynomial division: residual / divisor_poly should give (lcm * B)x + (lcm * C)
    // Use poly_long_divide
    let (bc_scaled, rem_check) = poly_long_divide(&residual, &divisor_poly);
    if !rem_check.iter().all(|&c| c == 0) {
      return None;
    }

    // Trim trailing zeros from bc_scaled
    let mut bc_scaled = bc_scaled;
    while bc_scaled.len() > 1 && bc_scaled.last() == Some(&0) {
      bc_scaled.pop();
    }

    // bc_scaled should be degree 1: [lcm*C, lcm*B]
    if bc_scaled.len() > 2 {
      return None;
    }
    let bc_c_scaled = if bc_scaled.is_empty() {
      0
    } else {
      bc_scaled[0]
    };
    let bc_b_scaled = if bc_scaled.len() < 2 { 0 } else { bc_scaled[1] };

    // Divide by lcm to get B and C as rationals
    let g_b = gcd_i128(bc_b_scaled.abs(), lcm);
    let (big_b_num, big_b_den) = (bc_b_scaled / g_b, lcm / g_b);
    let g_c = gcd_i128(bc_c_scaled.abs(), lcm);
    let (big_c_num, big_c_den) = (bc_c_scaled / g_c, lcm / g_c);

    // Convert to common B and C as integers (if possible) or rationals
    // For the integration formula, we need B and C as rationals
    // big_b = big_b_num / big_b_den, big_c = big_c_num / big_c_den
    // But our formula uses integer B and C. If they're not integers, we need to adjust.
    // Use common denominator for B and C as rationals
    let common_den = {
      let g = gcd_i128(big_b_den, big_c_den);
      (big_b_den / g).checked_mul(big_c_den)?
    };
    let big_b_int = big_b_num * (common_den / big_b_den);
    let big_c_int = big_c_num * (common_den / big_c_den);
    // The quadratic partial fraction is (big_b_int * x + big_c_int) / (common_den * (x^2+bx+c))

    // Step 7: Integrate (big_b_int * x + big_c_int) / (common_den * (x^2 + bx + c))
    // = (1/common_den) * Integrate[(big_b_int * x + big_c_int) / (x^2 + bx + c)]
    // Split: (Bx + C) = (B/2)(2x + b) + (C - Bb/2)
    // where B = big_b_int, C = big_c_int
    // Integral of (2x+b)/(x^2+bx+c) = Log[x^2+bx+c]
    // Integral of 1/(x^2+bx+c) = (2/sqrt(4c-b^2)) * ArcTan[(2x+b)/sqrt(4c-b^2)]

    let neg_disc = 4 * c - b * b; // = -(b^2-4c) > 0 since disc < 0
    if neg_disc <= 0 {
      return None;
    }

    // Build quadratic expr for Log: c + b*x + x^2 (Wolfram orders by ascending power)
    let quad_log_arg = if b == 0 && c == 1 {
      // 1 + x^2
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Identifier(var.to_string())),
          right: Box::new(Expr::Integer(2)),
        }),
      }
    } else {
      coeffs_to_expr(&[c, b, 1], var)
    };

    // Log part coefficient: B/(2*common_den)
    // = big_b_int / (2 * common_den)
    let log_total_num = big_b_int;
    let log_total_den = 2 * common_den;
    let g_log = gcd_i128(log_total_num.abs(), log_total_den);
    let (log_num, log_den) = (log_total_num / g_log, log_total_den / g_log);

    if log_num != 0 {
      let log_expr = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![quad_log_arg.clone()],
      };
      let log_term = build_coeff_times_expr(log_num, log_den, log_expr);
      quad_terms.push(log_term);
    }

    // ArcTan part coefficient: (2*big_c_int - big_b_int*b) / (common_den * sqrt(4c - b^2))
    let arctan_coeff_num = 2 * big_c_int - big_b_int * b;
    if arctan_coeff_num != 0 {
      // Extract perfect square factor from neg_disc: neg_disc = k^2 * m (m square-free)
      let (k, m) = {
        let mut outside = 1i128;
        let mut inside = neg_disc;
        let mut factor = 2i128;
        while factor * factor <= inside {
          while inside % (factor * factor) == 0 {
            outside *= factor;
            inside /= factor * factor;
          }
          factor += 1;
        }
        (outside, inside)
      };
      // sqrt(neg_disc) = k * sqrt(m), where sqrt(1) = 1

      // Simplify ArcTan argument: (b + 2*x) / (k * sqrt(m))
      // The numerator coefficients are b and 2; divide both and k by their common factor.
      let inner_gcd = if b == 0 { 2 } else { gcd_i128(b.abs(), 2) };
      let g = gcd_i128(inner_gcd, k);
      let k_reduced = k / g;
      let b_simplified = b / g;
      let two_simplified = 2 / g;
      let arctan_coeff_num = arctan_coeff_num / g;

      // Build ArcTan argument denominator expression
      let sqrt_denom = if m == 1 {
        if k_reduced <= 1 {
          None // denominator is 1
        } else {
          Some(Expr::Integer(k_reduced))
        }
      } else if k_reduced <= 1 {
        Some(make_sqrt(Expr::Integer(m)))
      } else {
        Some(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(k_reduced)),
          right: Box::new(make_sqrt(Expr::Integer(m))),
        })
      };

      // Build ArcTan argument numerator: b_simplified + two_simplified * x
      let arctan_numerator = if b_simplified == 0 {
        if two_simplified == 1 {
          Expr::Identifier(var.to_string())
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(two_simplified)),
            right: Box::new(Expr::Identifier(var.to_string())),
          }
        }
      } else {
        let x_term = if two_simplified == 1 {
          Expr::Identifier(var.to_string())
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(two_simplified)),
            right: Box::new(Expr::Identifier(var.to_string())),
          }
        };
        Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(b_simplified)),
          right: Box::new(x_term),
        }
      };

      // ArcTan argument: numerator / denom (or just numerator if denom is 1)
      let arctan_inner = if let Some(denom) = &sqrt_denom {
        Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(arctan_numerator),
          right: Box::new(denom.clone()),
        }
      } else {
        arctan_numerator
      };

      let arctan_expr = Expr::FunctionCall {
        name: "ArcTan".to_string(),
        args: vec![arctan_inner],
      };

      // Full coefficient: arctan_coeff_num / (common_den * k_reduced * sqrt(m))
      // Reduce: arctan_coeff_num / common_den first
      let g_at = gcd_i128(arctan_coeff_num.abs(), common_den);
      let at_num = arctan_coeff_num / g_at;
      let at_den_int = common_den / g_at;

      // Build effective sqrt expression combining at_den_int, k_reduced, and sqrt(m)
      let int_factor = at_den_int * k_reduced;
      let effective_sqrt = if m == 1 {
        Expr::Integer(int_factor)
      } else if int_factor == 1 {
        make_sqrt(Expr::Integer(m))
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(int_factor)),
          right: Box::new(make_sqrt(Expr::Integer(m))),
        }
      };

      let arctan_term = build_arctan_term(at_num, &effective_sqrt, arctan_expr);
      quad_terms.push(arctan_term);
    }
  } else if remaining_deg == 0 && roots.is_empty() {
    // No roots and no quadratic - can't decompose
    return None;
  }

  // Step 8: Combine all terms
  let mut all_terms: Vec<Expr> = Vec::new();
  if let Some(qi) = quotient_integral {
    all_terms.push(qi);
  }
  // ArcTan terms come before Log terms in Wolfram output ordering
  // Actually, looking at the expected outputs:
  // x/(1-x^3) → -(ArcTan[...]/Sqrt[3]) - Log[1 - x]/3 + Log[1 + x + x^2]/6
  // (2x+3)/(x^2+x+1) → (4*ArcTan[...])/Sqrt[3] + Log[...]
  // So ArcTan terms come first, then Log terms
  all_terms.extend(quad_terms);
  all_terms.extend(log_terms);

  if all_terms.is_empty() {
    return None;
  }

  Some(build_sum(all_terms))
}

/// Build an ArcTan term: coeff_num * arctan_expr / sqrt_expr
/// Handles simplification for common cases.
fn build_arctan_term(
  coeff_num: i128,
  sqrt_expr: &Expr,
  arctan_expr: Expr,
) -> Expr {
  use crate::syntax::BinaryOperator;

  let abs_coeff = coeff_num.abs();

  let term = if let Expr::Integer(s) = sqrt_expr {
    // sqrt is an integer - can simplify
    let g = gcd_i128_local(abs_coeff, *s);
    let reduced_num = abs_coeff / g;
    let reduced_den = *s / g;

    if reduced_den == 1 {
      if reduced_num == 1 {
        arctan_expr.clone()
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(reduced_num)),
          right: Box::new(arctan_expr.clone()),
        }
      }
    } else if reduced_num == 1 {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(arctan_expr.clone()),
        right: Box::new(Expr::Integer(reduced_den)),
      }
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(reduced_num)),
          right: Box::new(arctan_expr.clone()),
        }),
        right: Box::new(Expr::Integer(reduced_den)),
      }
    }
  } else {
    // sqrt is Sqrt[n] - put it in denominator
    if abs_coeff == 1 {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(arctan_expr.clone()),
        right: Box::new(sqrt_expr.clone()),
      }
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(abs_coeff)),
          right: Box::new(arctan_expr.clone()),
        }),
        right: Box::new(sqrt_expr.clone()),
      }
    }
  };

  if coeff_num < 0 {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(term),
    }
  } else {
    term
  }
}

/// Build (num/den) * expr, handling sign and simplifications.
fn build_coeff_times_expr(num: i128, den: i128, expr: Expr) -> Expr {
  use crate::syntax::BinaryOperator;

  let abs_num = num.abs();

  let unsigned_term = if den == 1 {
    if abs_num == 1 {
      expr
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(abs_num)),
        right: Box::new(expr),
      }
    }
  } else if abs_num == 1 {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(expr),
      right: Box::new(Expr::Integer(den)),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(crate::functions::math_ast::make_rational_pub(
        abs_num, den,
      )),
      right: Box::new(expr),
    }
  };

  if num < 0 {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(unsigned_term),
    }
  } else {
    unsigned_term
  }
}

/// Local gcd helper (avoids import issues)
fn gcd_i128_local(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// LIATE priority for integration by parts: higher = choose as u first
/// Log > Inverse trig > Algebraic > Trig > Exponential
fn liate_priority(expr: &Expr, var: &str) -> i32 {
  match expr {
    Expr::FunctionCall { name, .. } => match name.as_str() {
      "Log" => 5,
      "ArcSin" | "ArcCos" | "ArcTan" => 4,
      "Sin" | "Cos" | "Tan" | "Sec" | "Csc" | "Cot" | "Sinh" | "Cosh" => 2,
      "Exp" => 1,
      _ => 3,
    },
    // x, x^n => algebraic
    Expr::Identifier(_) => 3,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      // Any constant^(var-dependent) is exponential (E^x, e^x, 2^x, etc.)
      if is_constant_wrt(left, var) && !is_constant_wrt(right, var) {
        1
      } else {
        3
      }
    }
    _ => 3,
  }
}

/// Convert FunctionCall("Power", [base, exp]) → BinaryOp(Power, base, exp)
/// so that times_ast can combine powers with matching bases.
fn normalize_power(expr: Expr) -> Expr {
  if let Expr::FunctionCall { ref name, ref args } = expr
    && name == "Power"
    && args.len() == 2
  {
    return Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(args[0].clone()),
      right: Box::new(args[1].clone()),
    };
  }
  expr
}

/// Check if an expression is a pure exponential E^f(x).
fn is_exponential(expr: &Expr) -> bool {
  matches!(expr, Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left,
    ..
  } if matches!(left.as_ref(), Expr::Constant(c) if c == "E"))
}

/// Compare two expressions by their string representation.
fn expr_str_eq(a: &Expr, b: &Expr) -> bool {
  crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
}

/// Try to remove a specific factor from a product expression.
/// E.g., try_remove_factor(Times[2, E^x, (-1+x)], E^x) → Some(Times[2, (-1+x)])
/// E.g., try_remove_factor(E^x, E^x) → Some(Integer(1))
fn try_remove_factor(expr: &Expr, factor: &Expr) -> Option<Expr> {
  if expr_str_eq(expr, factor) {
    return Some(Expr::Integer(1));
  }
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      for (i, arg) in args.iter().enumerate() {
        if expr_str_eq(arg, factor) {
          let remaining: Vec<_> = args
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, a)| a.clone())
            .collect();
          return Some(if remaining.len() == 1 {
            remaining.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: remaining,
            }
          });
        }
      }
      None
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if expr_str_eq(left, factor) {
        Some(*right.clone())
      } else if expr_str_eq(right, factor) {
        Some(*left.clone())
      } else {
        None
      }
    }
    _ => None,
  }
}

std::thread_local! {
  // Recursion depth counter for integration by parts
  static IBP_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Check if expr is a polynomial in var (derivatives eventually reach 0).
fn is_polynomial_in(expr: &Expr, var: &str) -> bool {
  let mut current = expr.clone();
  for _ in 0..20 {
    if matches!(&current, Expr::Integer(0)) {
      return true;
    }
    match differentiate(&current, var) {
      Ok(d) => current = simplify(d),
      Err(_) => return false,
    }
  }
  false
}

/// Closed-form integration of polynomial × constant-base exponential:
/// ∫ P(x) * a^(c*x) dx = a^(c*x) * Σ_{k=0}^{deg} (-1)^k * P^(k)(x) / r^(k+1)
/// where r = c * Log[a] is the effective rate.
fn try_integrate_poly_times_const_exp(
  poly: &Expr,
  exponential: &Expr,
  base: &Expr,
  coeff: &Expr,
  var: &str,
) -> Option<Expr> {
  use crate::syntax::BinaryOperator::*;

  // For base E, Log[E] = 1, so rate = coeff directly.
  // For other bases, rate = coeff * Log[base].
  let rate = if matches!(base, Expr::Constant(c) if c == "E") {
    simplify(coeff.clone())
  } else {
    let log_base = Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![base.clone()],
    };
    simplify(Expr::BinaryOp {
      op: Times,
      left: Box::new(coeff.clone()),
      right: Box::new(log_base),
    })
  };

  // Collect derivatives of poly until we reach 0
  let mut derivs = vec![poly.clone()];
  let mut current = poly.clone();
  for _ in 0..20 {
    match differentiate(&current, var) {
      Ok(d) => {
        let d = simplify(d);
        if matches!(&d, Expr::Integer(0)) {
          break;
        }
        derivs.push(d.clone());
        current = d;
      }
      Err(_) => return None,
    }
  }

  // For numeric rates (e.g., base E with fractional coeff), compute each term
  // directly with 1/rate^(k+1) to get clean integer coefficients.
  // For symbolic rates (non-E bases involving Log), use the common-denominator
  // form: (exponential * Σ P^(k)(x)*rate^(n-1-k)) / rate^n.
  let is_numeric_rate = matches!(&rate, Expr::Integer(_))
    || matches!(&rate, Expr::FunctionCall { name, .. } if name == "Rational");

  let n = derivs.len();

  if is_numeric_rate {
    // Direct approach: Σ (-1)^k * P^(k)(x) / rate^(k+1)
    let inv_rate =
      crate::functions::math_ast::divide_ast(&[Expr::Integer(1), rate.clone()])
        .unwrap_or_else(|_| Expr::BinaryOp {
          op: Divide,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(rate.clone()),
        });

    let mut num_terms = Vec::new();
    for (k, deriv) in derivs.iter().enumerate() {
      let k1 = k as i128 + 1;
      let inv_rate_factor = if k1 == 1 {
        inv_rate.clone()
      } else {
        crate::functions::math_ast::power_two(&inv_rate, &Expr::Integer(k1))
          .unwrap_or_else(|_| Expr::BinaryOp {
            op: Power,
            left: Box::new(inv_rate.clone()),
            right: Box::new(Expr::Integer(k1)),
          })
      };

      let mut term = simplify(Expr::BinaryOp {
        op: Times,
        left: Box::new(deriv.clone()),
        right: Box::new(inv_rate_factor),
      });

      if k % 2 == 1 {
        term = simplify(Expr::BinaryOp {
          op: Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(term),
        });
      }

      num_terms.push(term);
    }

    let numerator = if num_terms.len() == 1 {
      num_terms.into_iter().next().unwrap()
    } else {
      let combined = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: num_terms,
      };
      crate::functions::polynomial_ast::expand_and_combine(&combined)
    };

    let result = simplify(Expr::BinaryOp {
      op: Times,
      left: Box::new(exponential.clone()),
      right: Box::new(numerator),
    });

    Some(result)
  } else {
    // Common-denominator approach: (exponential * Σ P^(k)(x)*rate^(n-1-k)) / rate^n
    let mut num_terms = Vec::new();
    for (k, deriv) in derivs.iter().enumerate() {
      let rate_power = n as i128 - 1 - k as i128;
      let rate_factor = if rate_power == 0 {
        Expr::Integer(1)
      } else if rate_power == 1 {
        rate.clone()
      } else {
        Expr::BinaryOp {
          op: Power,
          left: Box::new(rate.clone()),
          right: Box::new(Expr::Integer(rate_power)),
        }
      };

      let mut term = simplify(Expr::BinaryOp {
        op: Times,
        left: Box::new(deriv.clone()),
        right: Box::new(rate_factor),
      });

      if k % 2 == 1 {
        term = simplify(Expr::BinaryOp {
          op: Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(term),
        });
      }

      num_terms.push(term);
    }

    let numerator = if num_terms.len() == 1 {
      num_terms.into_iter().next().unwrap()
    } else {
      let combined = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: num_terms,
      };
      crate::functions::polynomial_ast::expand_and_combine(&combined)
    };

    let denom = if n == 1 {
      rate
    } else {
      Expr::BinaryOp {
        op: Power,
        left: Box::new(rate),
        right: Box::new(Expr::Integer(n as i128)),
      }
    };

    let result_num = simplify(Expr::BinaryOp {
      op: Times,
      left: Box::new(exponential.clone()),
      right: Box::new(numerator),
    });

    Some(Expr::BinaryOp {
      op: Divide,
      left: Box::new(result_num),
      right: Box::new(denom),
    })
  }
}

/// Try integration by parts: ∫ u dv = u*v - ∫ v du
/// `factors` are the factors that depend on `var`.
/// We pick `u` using the LIATE heuristic and `dv` is the product of the remaining factors.
fn try_integration_by_parts(factors: &[&Expr], var: &str) -> Option<Expr> {
  if factors.len() < 2 {
    return None;
  }

  // Limit recursion depth to prevent infinite loops
  let depth = IBP_DEPTH.with(|d| d.get());
  if depth >= 5 {
    return None;
  }

  // Find the factor with the highest LIATE priority → that becomes u
  let mut best_u_idx = 0;
  let mut best_priority = liate_priority(factors[0], var);
  for (i, f) in factors.iter().enumerate().skip(1) {
    let p = liate_priority(f, var);
    if p > best_priority {
      best_priority = p;
      best_u_idx = i;
    }
  }

  let u = factors[best_u_idx];

  // dv is the product of the remaining factors
  let dv_factors: Vec<&Expr> = factors
    .iter()
    .enumerate()
    .filter(|(i, _)| *i != best_u_idx)
    .map(|(_, f)| *f)
    .collect();

  // Special case: polynomial × constant-base exponential (including E base)
  // Use closed-form formula: ∫ P(x)*a^(cx) dx = a^(cx) * Σ (-1)^k P^(k)(x) / (c*Log[a])^(k+1)
  // For a=E, rate = c*Log[E] = c, giving direct polynomial-times-E^(cx) integration.
  if dv_factors.len() == 1
    && let Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: base,
      right: exp,
    } = dv_factors[0]
    && is_constant_wrt(base, var)
    && !is_constant_wrt(exp, var)
    && let Some(coeff) = try_match_linear_arg(exp, var)
    && is_polynomial_in(u, var)
  {
    return try_integrate_poly_times_const_exp(
      u,
      dv_factors[0],
      base,
      &coeff,
      var,
    );
  }

  let dv_expr = if dv_factors.len() == 1 {
    dv_factors[0].clone()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: dv_factors.iter().map(|f| (*f).clone()).collect(),
    }
  };

  // v = ∫ dv
  let v = integrate(&dv_expr, var)?;

  // du = D[u, var]
  let du = differentiate(u, var).ok()?;
  let du = simplify(du);

  // Decompose v into (v_core, v_denom) where v = v_core / v_denom
  // This allows proper fraction handling: u*v = (u*v_core)/v_denom
  let (v_core, v_denom) = if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: v_num,
    right: v_den,
  } = &v
  {
    if is_constant_wrt(v_den, var) {
      (*v_num.clone(), Some(*v_den.clone()))
    } else {
      (v.clone(), None)
    }
  } else {
    (v.clone(), None)
  };

  // Result: u*v - ∫ v*du dx
  // When v is a fraction a/b, compute uv as (u*a)/b for proper display
  // e.g. Log[x] * (x^2/2) → (x^2*Log[x])/2 instead of x^2/2*Log[x]
  let uv = {
    let u_times_core = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(u.clone()),
      right: Box::new(v_core.clone()),
    });
    if let Some(ref denom) = v_denom {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(u_times_core),
        right: Box::new(denom.clone()),
      }
    } else {
      u_times_core
    }
  };

  // If du is 0, the integral is just u*v
  if matches!(&du, Expr::Integer(0)) {
    return Some(uv);
  }

  // Normalize du: convert FunctionCall("Power", [base, exp]) → BinaryOp(Power)
  // so that times_ast can combine powers like x^2 * x^(-1) → x
  let du = normalize_power(du);

  // Compute v*du, decomposing v = numerator/constant for better simplification
  // (e.g., v = x^2/2, du = x^(-1) → numerator*du = x^2 * x^(-1) = x → v*du = x/2)
  let v_du = if let Some(ref denom) = v_denom {
    let num_du =
      crate::functions::math_ast::times_ast(&[v_core.clone(), du]).ok()?;
    simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(num_du),
      right: Box::new(denom.clone()),
    })
  } else {
    crate::functions::math_ast::times_ast(&[v, du]).ok()?
  };

  // Integrate v*du with increased depth
  IBP_DEPTH.with(|d| d.set(depth + 1));
  let int_v_du = integrate(&v_du, var);
  IBP_DEPTH.with(|d| d.set(depth));

  let int_v_du = int_v_du?;
  // Simplify to flatten nested products (e.g., BinaryOp(Times, 2, E^x*...) → Times[2, E^x, ...])
  let int_v_du = simplify(int_v_du);

  // Try to factor out common exponential factor (e.g., E^x from E^x*x - E^x → E^x*(-1+x))
  // This matches Wolfram's output form for exponential integrals.
  if is_exponential(&v_core)
    && let Some(quotient) = try_remove_factor(&int_v_du, &v_core)
  {
    // result = v_core * (u - quotient)
    // Expand the inner expression so e.g. x^2 - 2*(-1+x) becomes 2 - 2*x + x^2
    let inner =
      crate::functions::math_ast::subtract_ast(&[u.clone(), quotient]).ok()?;
    let inner = crate::functions::polynomial_ast::expand_and_combine(&inner);
    let result = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(v_core),
      right: Box::new(inner),
    });
    return Some(result);
  }

  Some(simplify(Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Minus,
    left: Box::new(uv),
    right: Box::new(int_v_du),
  }))
}

/// Integrate an expression with respect to a variable
fn integrate(expr: &Expr, var: &str) -> Option<Expr> {
  // General constant check: ∫ c dy = c*y for any expression c independent of y
  // (handles compound expressions like x^2, Sin[x], etc. when integrating w.r.t. a different variable)
  if is_constant_wrt(expr, var) {
    return Some(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(expr.clone()),
      right: Box::new(Expr::Identifier(var.to_string())),
    });
  }

  match expr {
    // Constant: ∫ c dx = c*x
    Expr::Integer(n) => Some(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Integer(*n)),
      right: Box::new(Expr::Identifier(var.to_string())),
    }),
    Expr::Real(f) => Some(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Real(*f)),
      right: Box::new(Expr::Identifier(var.to_string())),
    }),

    // Variable: ∫ x dx = x^2/2, ∫ c dx = c*x
    Expr::Identifier(name) => {
      if name == var {
        Some(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(Expr::Identifier(var.to_string())),
            right: Box::new(Expr::Integer(2)),
          }),
          right: Box::new(Expr::Integer(2)),
        })
      } else {
        // Constant * x
        Some(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Identifier(name.clone())),
          right: Box::new(Expr::Identifier(var.to_string())),
        })
      }
    }

    // Binary operations
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator::*;
      match op {
        Plus => {
          // ∫ (a + b) dx = ∫ a dx + ∫ b dx
          let int_a = integrate(left, var)?;
          let int_b = integrate(right, var)?;
          Some(Expr::BinaryOp {
            op: Plus,
            left: Box::new(int_a),
            right: Box::new(int_b),
          })
        }
        Minus => {
          // ∫ (a - b) dx = ∫ a dx - ∫ b dx
          let int_a = integrate(left, var)?;
          let int_b = integrate(right, var)?;
          Some(Expr::BinaryOp {
            op: Minus,
            left: Box::new(int_a),
            right: Box::new(int_b),
          })
        }
        Times => {
          // c * f(x) where c is constant
          if is_constant_wrt(left, var) {
            let int_b = integrate(right, var)?;
            Some(Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(int_b),
            })
          } else if is_constant_wrt(right, var) {
            let int_a = integrate(left, var)?;
            Some(Expr::BinaryOp {
              op: Times,
              left: right.clone(),
              right: Box::new(int_a),
            })
          } else {
            // Both factors depend on var: try trig product first
            if let Some(result) =
              try_integrate_sin_cos_product(&[left, right], var)
            {
              return Some(result);
            }
            // Fall back to integration by parts
            try_integration_by_parts(&[left, right], var)
          }
        }
        Divide => {
          // f(x) / c where c is constant
          if is_constant_wrt(right, var) {
            let int_a = integrate(left, var)?;
            Some(Expr::BinaryOp {
              op: Divide,
              left: Box::new(int_a),
              right: right.clone(),
            })
          } else {
            // If denominator is x^n, rewrite as numerator * x^(-n)
            if let Expr::BinaryOp {
              op: Power,
              left: base,
              right: exp,
            } = right.as_ref()
              && let Expr::Identifier(name) = base.as_ref()
              && name == var
              && is_constant_wrt(exp, var)
            {
              let neg_exp = Expr::UnaryOp {
                op: crate::syntax::UnaryOperator::Minus,
                operand: exp.clone(),
              };
              let x_neg_n = Expr::BinaryOp {
                op: Power,
                left: base.clone(),
                right: Box::new(neg_exp),
              };
              let rewritten = Expr::BinaryOp {
                op: Times,
                left: left.clone(),
                right: Box::new(x_neg_n),
              };
              if let Some(result) = integrate(&rewritten, var) {
                return Some(result);
              }
            }
            // ∫ E^(a*x) / (c*x) dx = ExpIntegralEi[a*x] / c
            // ∫ E^(a*x) / x dx = ExpIntegralEi[a*x]
            if let Some(result) = try_match_exp_over_linear(left, right, var) {
              return Some(result);
            }
            // Try rational function integration (partial fractions)
            try_integrate_rational(left, right, var)
          }
        }
        Power => {
          // ∫ x^n dx = x^(n+1)/(n+1) where n is constant
          if let Expr::Identifier(name) = left.as_ref()
            && name == var
            && is_constant_wrt(right, var)
          {
            let new_exp = simplify(Expr::BinaryOp {
              op: Plus,
              left: right.clone(),
              right: Box::new(Expr::Integer(1)),
            });
            // Special case: ∫ x^(-1) dx = Log[x]
            if matches!(&new_exp, Expr::Integer(0)) {
              return Some(Expr::FunctionCall {
                name: "Log".to_string(),
                args: vec![Expr::Identifier(var.to_string())],
              });
            }
            let power_expr = Expr::BinaryOp {
              op: Power,
              left: left.clone(),
              right: Box::new(new_exp.clone()),
            };
            // When new_exp is a negative integer, use Wolfram canonical form:
            // x^n / n where n < 0 → Times[Rational[1, n], Power[x, n]]
            // e.g. x^(-2)/(-2) → Times[Rational[-1, 2], Power[x, -2]] → -1/2*1/x^2
            if let Expr::Integer(n) = &new_exp {
              let n = *n;
              if n < 0 {
                let abs_n = -n;
                // Build Rational[-1, abs_n] (= 1/n since n < 0)
                let coeff = if abs_n == 1 {
                  Expr::Integer(-1)
                } else {
                  Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(-1), Expr::Integer(abs_n)],
                  }
                };
                return Some(Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![coeff, power_expr],
                });
              }
            }
            return Some(Expr::BinaryOp {
              op: Divide,
              left: Box::new(power_expr),
              right: Box::new(new_exp),
            });
          }
          // ∫ E^x dx = E^x, ∫ E^(a*x) dx = E^(a*x)/a,
          // ∫ E^(-a*x^2) dx = Gaussian integral
          if matches!(left.as_ref(), Expr::Constant(c) if c == "E") {
            let exp_arg = right.as_ref();
            // ∫ E^x dx = E^x
            if let Expr::Identifier(n) = exp_arg
              && n == var
            {
              return Some(expr.clone());
            }
            // ∫ E^(a*x) dx = E^(a*x)/a
            if let Some(coeff) = try_match_linear_arg(exp_arg, var) {
              return Some(make_divided(expr.clone(), coeff));
            }
            // ∫ E^(-a*x^2) dx = Sqrt[Pi/a]/2 * Erf[Sqrt[a]*x]
            if let Some(coeff) = match_neg_a_x_squared(exp_arg, var) {
              return Some(make_gaussian_antiderivative(var, &coeff));
            }
          }
          // ∫ a^x dx = a^x / Log[a], ∫ a^(c*x) dx = a^(c*x) / (c*Log[a])
          // where a is any constant base (not E, which is handled above)
          if is_constant_wrt(left, var) && !is_constant_wrt(right, var) {
            let exp_arg = right.as_ref();
            let log_a = Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![*left.clone()],
            };
            // ∫ a^x dx = a^x / Log[a]
            if let Expr::Identifier(n) = exp_arg
              && n == var
            {
              return Some(Expr::BinaryOp {
                op: Divide,
                left: Box::new(expr.clone()),
                right: Box::new(log_a),
              });
            }
            // ∫ a^(c*x) dx = a^(c*x) / (c * Log[a])
            if let Some(coeff) = try_match_linear_arg(exp_arg, var) {
              let divisor = simplify(Expr::BinaryOp {
                op: Times,
                left: Box::new(coeff),
                right: Box::new(log_a),
              });
              return Some(Expr::BinaryOp {
                op: Divide,
                left: Box::new(expr.clone()),
                right: Box::new(divisor),
              });
            }
          }
          // ∫ Sin[x]^n dx, ∫ Cos[x]^n dx using Chebyshev expansion
          if let Expr::Integer(n) = right.as_ref()
            && *n >= 2
            && let Some(result) = if *n == 2 {
              try_integrate_trig_squared(left, var)
            } else {
              try_integrate_trig_power(left, *n, var)
            }
          {
            return Some(result);
          }
          // ∫ (a*x + b)^n dx where n >= 3 and the base is linear in var:
          // Use substitution: result = (a*x + b)^(n+1) / ((n+1) * a)
          // For n == 2, Wolfram expands instead, so we skip to the expand path.
          if let Expr::Integer(n) = right.as_ref()
            && *n >= 3
            && !is_constant_wrt(left, var)
            && let Some(a) = extract_linear_coefficient(left, var)
          {
            let n1 = Expr::Integer(*n + 1);
            let base_pow = Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(left.as_ref().clone()),
              right: Box::new(n1.clone()),
            };
            let denom = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![n1, a],
            };
            return Some(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Divide,
              left: Box::new(base_pow),
              right: Box::new(denom),
            });
          }
          // ∫ f(x)^n dx where n is a positive integer: try expanding (used for n == 2)
          if let Expr::Integer(n) = right.as_ref()
            && *n >= 2
            && !is_constant_wrt(left, var)
          {
            let expanded =
              crate::functions::polynomial_ast::expand_and_combine(expr);
            if !expr_str_eq(&expanded, expr) {
              return integrate(&expanded, var);
            }
          }
          // ∫ (1 - x^2)^(-1/2) dx = ArcSin[x]
          // ∫ (1 + x^2)^(-1/2) dx = ArcSinh[x]
          if is_rational_neg_half(right)
            && !is_constant_wrt(left, var)
            && let Some(result) = try_integrate_inverse_sqrt(left, var)
          {
            return Some(result);
          }
          // base^(-n) where base depends on var: treat as 1/base^n (rational)
          if let Expr::Integer(n) = right.as_ref()
            && *n < 0
            && !is_constant_wrt(left, var)
          {
            let denom = if *n == -1 {
              left.as_ref().clone()
            } else {
              Expr::BinaryOp {
                op: Power,
                left: left.clone(),
                right: Box::new(Expr::Integer(-*n)),
              }
            };
            // Try exp over linear
            if let Some(result) =
              try_match_exp_over_linear(&Expr::Integer(1), &denom, var)
            {
              return Some(result);
            }
            // Try rational function integration
            if let Some(result) =
              try_integrate_rational(&Expr::Integer(1), &denom, var)
            {
              return Some(result);
            }
          }
          None
        }
        _ => None,
      }
    }

    // Function calls
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Plus" if args.len() >= 2 => {
          // ∫ (a + b + ...) dx = ∫ a dx + ∫ b dx + ...
          let integrals: Option<Vec<Expr>> =
            args.iter().map(|arg| integrate(arg, var)).collect();
          integrals.map(|ints| Expr::FunctionCall {
            name: "Plus".to_string(),
            args: ints,
          })
        }
        "Sin" if args.len() == 1 => {
          // ∫ sin(a*x) dx = -cos(a*x)/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let cos_expr = Expr::FunctionCall {
              name: "Cos".to_string(),
              args: args.clone(),
            };
            return Some(make_neg_divided(cos_expr, coeff));
          }
          None
        }
        "Cos" if args.len() == 1 => {
          // ∫ cos(a*x) dx = sin(a*x)/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let sin_expr = Expr::FunctionCall {
              name: "Sin".to_string(),
              args: args.clone(),
            };
            return Some(make_divided(sin_expr, coeff));
          }
          None
        }
        "Exp" if args.len() == 1 => {
          // ∫ e^x dx = e^x
          if let Expr::Identifier(n) = &args[0]
            && n == var
          {
            return Some(Expr::FunctionCall {
              name: "Exp".to_string(),
              args: args.clone(),
            });
          }
          // ∫ e^(a*x) dx = e^(a*x)/a  (linear argument)
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let exp_expr = Expr::FunctionCall {
              name: "Exp".to_string(),
              args: args.clone(),
            };
            return Some(make_divided(exp_expr, coeff));
          }
          // ∫ Exp[-a*x^2] dx = Sqrt[Pi/a]/2 * Erf[Sqrt[a]*x]
          // (when a=1: Sqrt[Pi]/2 * Erf[x])
          if let Some(coeff) = match_neg_a_x_squared(&args[0], var) {
            return Some(make_gaussian_antiderivative(var, &coeff));
          }
          None
        }
        "Sinh" if args.len() == 1 => {
          // ∫ sinh(a*x) dx = cosh(a*x)/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let cosh_expr = Expr::FunctionCall {
              name: "Cosh".to_string(),
              args: args.clone(),
            };
            return Some(make_divided(cosh_expr, coeff));
          }
          None
        }
        "Cosh" if args.len() == 1 => {
          // ∫ cosh(a*x) dx = sinh(a*x)/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let sinh_expr = Expr::FunctionCall {
              name: "Sinh".to_string(),
              args: args.clone(),
            };
            return Some(make_divided(sinh_expr, coeff));
          }
          None
        }
        "Tan" if args.len() == 1 => {
          // ∫ tan(a*x) dx = -Log[Cos[a*x]]/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let cos_expr = Expr::FunctionCall {
              name: "Cos".to_string(),
              args: args.clone(),
            };
            let log_cos = Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![cos_expr],
            };
            return Some(make_neg_divided(log_cos, coeff));
          }
          None
        }
        "Cot" if args.len() == 1 => {
          // ∫ cot(a*x) dx = Log[Sin[a*x]]/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let sin_expr = Expr::FunctionCall {
              name: "Sin".to_string(),
              args: args.clone(),
            };
            let log_sin = Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![sin_expr],
            };
            return Some(make_divided(log_sin, coeff));
          }
          None
        }
        "Log" if args.len() == 1 => {
          // ∫ Log[x] dx = -x + x*Log[x]
          if let Expr::Identifier(name) = &args[0]
            && name == var
          {
            let x = Expr::Identifier(var.to_string());
            return Some(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(Expr::UnaryOp {
                op: crate::syntax::UnaryOperator::Minus,
                operand: Box::new(x.clone()),
              }),
              right: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(x),
                right: Box::new(Expr::FunctionCall {
                  name: "Log".to_string(),
                  args: args.clone(),
                }),
              }),
            });
          }
          None
        }
        "Times" => {
          // ∫ (c1 * c2 * ... * f(x)) dx = c1 * c2 * ... * ∫ f(x) dx
          let (const_factors, var_factors): (Vec<_>, Vec<_>) =
            args.iter().partition(|a| is_constant_wrt(a, var));
          if var_factors.len() == 1 {
            let int_var = integrate(var_factors[0], var)?;
            let const_expr = if const_factors.is_empty() {
              return Some(int_var);
            } else if const_factors.len() == 1 {
              const_factors[0].clone()
            } else {
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: const_factors.into_iter().cloned().collect(),
              }
            };
            Some(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(const_expr),
              right: Box::new(int_var),
            })
          } else if var_factors.is_empty() {
            // All constant: ∫ c dx = c*x
            let const_expr = Expr::FunctionCall {
              name: "Times".to_string(),
              args: args.clone(),
            };
            Some(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(const_expr),
              right: Box::new(Expr::Identifier(var.to_string())),
            })
          } else {
            // Multiple variable-dependent factors: check for fraction form
            // Times[..., Power[den, -1]] → treat as numerator / denominator
            let mut num_var_factors: Vec<Expr> = Vec::new();
            let mut den_factors: Vec<Expr> = Vec::new();
            for vf in &var_factors {
              // Extract base and negative exponent from Power[base, -n]
              let neg_power = match vf {
                Expr::FunctionCall {
                  name: pname,
                  args: pargs,
                } if pname == "Power" && pargs.len() == 2 => {
                  if let Expr::Integer(n) = &pargs[1] {
                    if *n < 0 {
                      Some((pargs[0].clone(), *n))
                    } else {
                      None
                    }
                  } else {
                    None
                  }
                }
                Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Power,
                  left,
                  right,
                } => {
                  if let Expr::Integer(n) = right.as_ref() {
                    if *n < 0 {
                      Some((*left.clone(), *n))
                    } else {
                      None
                    }
                  } else {
                    None
                  }
                }
                _ => None,
              };
              if let Some((base, neg_exp)) = neg_power {
                if neg_exp == -1 {
                  den_factors.push(base);
                } else {
                  den_factors.push(Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Power,
                    left: Box::new(base),
                    right: Box::new(Expr::Integer(-neg_exp)),
                  });
                }
              } else {
                num_var_factors.push((*vf).clone());
              }
            }
            if !den_factors.is_empty() {
              let numerator = if num_var_factors.is_empty() {
                Expr::Integer(1)
              } else if num_var_factors.len() == 1 {
                num_var_factors.remove(0)
              } else {
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: num_var_factors,
                }
              };
              let denominator = if den_factors.len() == 1 {
                den_factors.remove(0)
              } else {
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: den_factors,
                }
              };
              // Helper to multiply constant factors back to a result
              let apply_const = |result: Expr| -> Expr {
                if const_factors.is_empty() {
                  result
                } else {
                  let const_expr = if const_factors.len() == 1 {
                    const_factors[0].clone()
                  } else {
                    Expr::FunctionCall {
                      name: "Times".to_string(),
                      args: const_factors
                        .iter()
                        .map(|e| (*e).clone())
                        .collect(),
                    }
                  };
                  Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Times,
                    left: Box::new(const_expr),
                    right: Box::new(result),
                  }
                }
              };
              // Try the same logic as the Divide arm
              if is_constant_wrt(&denominator, var)
                && let Some(int_num) = integrate(&numerator, var)
              {
                return Some(apply_const(
                  crate::functions::math_ast::make_divide(int_num, denominator),
                ));
              }
              // Try exp over linear: ∫ E^(a*x) / (c*x) dx
              if let Some(result) =
                try_match_exp_over_linear(&numerator, &denominator, var)
              {
                return Some(apply_const(result));
              }
              // Try rational function integration
              if let Some(result) =
                try_integrate_rational(&numerator, &denominator, var)
              {
                return Some(apply_const(result));
              }
            }
            // Try trig product: Sin[f]^m * Cos[f]^n
            let var_refs: Vec<&Expr> = var_factors.to_vec();
            if let Some(trig_result) =
              try_integrate_sin_cos_product(&var_refs, var)
            {
              if const_factors.is_empty() {
                return Some(trig_result);
              } else {
                let const_expr = if const_factors.len() == 1 {
                  const_factors[0].clone()
                } else {
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: const_factors.into_iter().cloned().collect(),
                  }
                };
                return Some(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Times,
                  left: Box::new(const_expr),
                  right: Box::new(trig_result),
                });
              }
            }
            // Fall through to integration by parts
            if let Some(ibp_result) = try_integration_by_parts(&var_refs, var) {
              // Multiply back the constant factors
              if const_factors.is_empty() {
                Some(ibp_result)
              } else {
                let const_expr = if const_factors.len() == 1 {
                  const_factors[0].clone()
                } else {
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: const_factors.into_iter().cloned().collect(),
                  }
                };
                Some(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Times,
                  left: Box::new(const_expr),
                  right: Box::new(ibp_result),
                })
              }
            } else {
              None
            }
          }
        }
        // Power[base, exp] as FunctionCall → normalize to BinaryOp and recurse
        "Power" if args.len() == 2 => {
          let as_binop = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(args[0].clone()),
            right: Box::new(args[1].clone()),
          };
          integrate(&as_binop, var)
        }
        _ => None,
      }
    }

    // Unary minus
    Expr::UnaryOp { op, operand } => {
      use crate::syntax::UnaryOperator;
      if matches!(op, UnaryOperator::Minus) {
        let int = integrate(operand, var)?;
        Some(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(int),
        })
      } else {
        None
      }
    }

    _ => None,
  }
}

/// Simplify an expression
pub fn simplify(mut expr: Expr) -> Expr {
  match &mut expr {
    Expr::BinaryOp { op, left, right } => {
      let op = *op;
      let left = simplify(*std::mem::replace(left, Box::new(Expr::Integer(0))));
      let right =
        simplify(*std::mem::replace(right, Box::new(Expr::Integer(0))));

      use crate::syntax::BinaryOperator::*;
      match (&op, &left, &right) {
        // 0 + x = x
        (Plus, Expr::Integer(0), _) => return right,
        // x + 0 = x
        (Plus, _, Expr::Integer(0)) => return left,
        // 0 * x = 0
        (Times, Expr::Integer(0), _) | (Times, _, Expr::Integer(0)) => {
          return Expr::Integer(0);
        }
        // 1 * x = x
        (Times, Expr::Integer(1), _) => return right,
        // x * 1 = x
        (Times, _, Expr::Integer(1)) => return left,
        // x - 0 = x
        (Minus, _, Expr::Integer(0)) => return left,
        // 0 - n = -n  (for integers)
        (Minus, Expr::Integer(0), Expr::Integer(n)) => {
          return Expr::Integer(-n);
        }
        // 0 - (-x) = x
        (
          Minus,
          Expr::Integer(0),
          Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand,
          },
        ) => {
          return *operand.clone();
        }
        // 0 - x = -x  (general)
        (Minus, Expr::Integer(0), _) => {
          return Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand: Box::new(right),
          };
        }
        // x / 1 = x
        (Divide, _, Expr::Integer(1)) => return left,
        // x^0 = 1
        (Power, _, Expr::Integer(0)) => return Expr::Integer(1),
        // x^1 = x
        (Power, _, Expr::Integer(1)) => return left,
        // 0^n = 0 (for n > 0)
        (Power, Expr::Integer(0), Expr::Integer(n)) if *n > 0 => {
          return Expr::Integer(0);
        }
        // 1^n = 1
        (Power, Expr::Integer(1), _) => return Expr::Integer(1),
        // Numeric simplification
        (Plus, Expr::Integer(a), Expr::Integer(b)) => {
          return Expr::Integer(a + b);
        }
        (Minus, Expr::Integer(a), Expr::Integer(b)) => {
          return Expr::Integer(a - b);
        }
        (Times, Expr::Integer(a), Expr::Integer(b)) => {
          return Expr::Integer(a * b);
        }
        _ => {}
      }

      // For Power, delegate to power_two for proper expansion (e.g. (3*x)^2 → 9*x^2)
      // Only for non-negative exponents to preserve canonical form
      if matches!(op, Power)
        && matches!(&right, Expr::Integer(n) if *n >= 0)
        && let Ok(result) = crate::functions::math_ast::power_two(&left, &right)
      {
        return result;
      }
      // For Times, delegate to times_ast for proper flattening and sorting
      if matches!(op, Times)
        && let Ok(result) =
          crate::functions::math_ast::times_ast(&[left.clone(), right.clone()])
      {
        return result;
      }
      // For Plus, delegate to plus_ast for proper sorting
      if matches!(op, Plus)
        && let Ok(result) =
          crate::functions::math_ast::plus_ast(&[left.clone(), right.clone()])
      {
        return result;
      }

      Expr::BinaryOp {
        op,
        left: Box::new(left),
        right: Box::new(right),
      }
    }
    Expr::UnaryOp { op, operand } => {
      let op = *op;
      let operand =
        simplify(*std::mem::replace(operand, Box::new(Expr::Integer(0))));
      use crate::syntax::UnaryOperator;
      if matches!(&op, UnaryOperator::Minus) {
        if let Expr::Integer(0) = operand {
          return Expr::Integer(0);
        }
        if let Expr::Integer(n) = operand {
          return Expr::Integer(-n);
        }
      }
      Expr::UnaryOp {
        op,
        operand: Box::new(operand),
      }
    }
    Expr::FunctionCall { name, args } => {
      let name = std::mem::take(name);
      let args: Vec<Expr> =
        std::mem::take(args).into_iter().map(simplify).collect();

      // Delegate Power[base, exp] to power_two for proper expansion
      // Only for non-negative exponents to avoid distributing (a*b)^(-n)
      // which changes canonical form (e.g. Sqrt[-1+x]*Sqrt[1+x] factor ordering)
      if name == "Power" && args.len() == 2 {
        let is_non_neg = matches!(&args[1], Expr::Integer(n) if *n >= 0);
        if is_non_neg
          && let Ok(result) =
            crate::functions::math_ast::power_two(&args[0], &args[1])
        {
          return result;
        }
      }
      // Delegate Sqrt to handle Sqrt[0] → 0, Sqrt[1] → 1, etc.
      if name == "Sqrt" && args.len() == 1 {
        let canonical = make_sqrt(args[0].clone());
        if let Some(inner) = is_sqrt(&canonical) {
          if matches!(inner, Expr::Integer(0)) {
            return Expr::Integer(0);
          }
          if matches!(inner, Expr::Integer(1)) {
            return Expr::Integer(1);
          }
        }
      }

      Expr::FunctionCall { name, args }
    }
    _ => expr,
  }
}

/// Extract base and exponent from Power expressions (both BinaryOp and FunctionCall forms)
fn extract_power(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => Some((*left.clone(), *right.clone())),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      Some((args[0].clone(), args[1].clone()))
    }
    _ => None,
  }
}

/// Check if an expression approaches 1 when var -> Infinity
fn eval_at_infinity_is_one(expr: &Expr, var: &str) -> bool {
  // Substitute a large value and check if close to 1
  let subst =
    crate::syntax::substitute_variable(expr, var, &Expr::Integer(1_000_000));
  if let Ok(val) = crate::evaluator::evaluate_expr_to_expr(&subst)
    && let Some(f) = crate::functions::math_ast::try_eval_to_f64(&val)
  {
    return (f - 1.0).abs() < 0.01;
  }
  false
}

/// Check if an expression diverges to infinity when var -> Infinity
fn eval_at_infinity_diverges(expr: &Expr, var: &str) -> Option<bool> {
  let subst =
    crate::syntax::substitute_variable(expr, var, &Expr::Integer(1_000_000));
  if let Ok(val) = crate::evaluator::evaluate_expr_to_expr(&subst)
    && let Some(f) = crate::functions::math_ast::try_eval_to_f64(&val)
  {
    if f > 1e5 {
      return Some(true); // positive infinity
    }
    if f < -1e5 {
      return Some(false); // negative infinity (returns Some(false) for sign)
    }
  }
  None
}

/// Handle limits at infinity.
/// Strategies:
/// 1. If expr is constant wrt var, return it directly
/// 2. Direct substitution heuristic (evaluate at large n) to classify the limit
/// 3. For f^g where f->1, g->inf: Limit = E^(Limit[g*(f-1)])
/// 4. For expressions going to 0 or a constant: detect via structure
fn limit_at_infinity(
  expr: &Expr,
  var_name: &str,
  point: &Expr,
) -> Result<Expr, InterpreterError> {
  // If the expression is constant wrt the variable, return it
  if is_constant_wrt(expr, var_name) {
    return crate::evaluator::evaluate_expr_to_expr(expr);
  }

  // Handle var itself: Limit[n, n -> Infinity] = Infinity
  if let Expr::Identifier(name) = expr
    && name == var_name
  {
    return Ok(point.clone());
  }

  // Handle f^g form (e.g., (1 + 1/n)^n -> E)
  if let Some((base, exp)) = extract_power(expr) {
    // Check if base -> 1 and exponent -> Infinity (1^Infinity indeterminate form)
    if eval_at_infinity_is_one(&base, var_name)
      && eval_at_infinity_diverges(&exp, var_name).is_some()
    {
      // Use identity: Limit[f^g] = E^(Limit[g * (f - 1)]) when f -> 1, g -> inf
      let f_minus_1 = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(base),
        right: Box::new(Expr::Integer(1)),
      };
      let product = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(exp),
        right: Box::new(f_minus_1),
      };
      // Take the limit of g * (f - 1) as var -> Infinity
      let rule = Expr::Rule {
        pattern: Box::new(Expr::Identifier(var_name.to_string())),
        replacement: Box::new(point.clone()),
      };
      let exponent_limit = limit_ast(&[product, rule])?;

      // If the exponent limit is a clean value, return E^limit
      if is_clean_value(&exponent_limit) {
        let result = simplify(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(Expr::Constant("E".to_string())),
          right: Box::new(exponent_limit),
        });
        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
    }
  }

  // For simple expressions, try evaluating at two large values to detect convergence
  let sign = if is_negative_infinity(point) { -1 } else { 1 };
  let val1 = eval_at_large_n(expr, var_name, sign * 1_000_000);
  let val2 = eval_at_large_n(expr, var_name, sign * 10_000_000);
  if let (Some(f1), Some(f2)) = (val1, val2) {
    // Both diverging to +infinity
    if f1 > 1e5 && f2 > f1 {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    // Both diverging to -infinity
    if f1 < -1e5 && f2 < f1 {
      return Ok(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
    // Approaching zero: both values small and getting smaller
    if f2.abs() < 1e-4 && f2.abs() < f1.abs() {
      return Ok(Expr::Integer(0));
    }
    // Check convergence: values should be close (relative difference)
    let diff = (f1 - f2).abs();
    let scale = f1.abs().max(f2.abs()).max(1e-15);
    if diff / scale < 0.01 {
      // Converging to a nonzero limit — determine the value
      // Check if the limit is a known integer
      let rounded = f2.round();
      if (f2 - rounded).abs() < 1e-4 {
        return Ok(Expr::Integer(rounded as i128));
      }
      // Check for known constants
      if (f2 - std::f64::consts::E).abs() < 1e-3 {
        return Ok(Expr::Constant("E".to_string()));
      }
      if (f2 - std::f64::consts::PI).abs() < 1e-3 {
        return Ok(Expr::Constant("Pi".to_string()));
      }
      // Check for common multiples/fractions of Pi
      let pi = std::f64::consts::PI;
      let pi_fractions: &[(f64, i128, i128)] = &[
        (pi / 2.0, 1, 2),
        (-pi / 2.0, -1, 2),
        (pi / 3.0, 1, 3),
        (-pi / 3.0, -1, 3),
        (pi / 4.0, 1, 4),
        (-pi / 4.0, -1, 4),
        (pi / 6.0, 1, 6),
        (-pi / 6.0, -1, 6),
        (2.0 * pi / 3.0, 2, 3),
        (-2.0 * pi / 3.0, -2, 3),
        (3.0 * pi / 4.0, 3, 4),
        (-3.0 * pi / 4.0, -3, 4),
        (5.0 * pi / 6.0, 5, 6),
        (-5.0 * pi / 6.0, -5, 6),
        (-pi, -1, 1),
        (2.0 * pi, 2, 1),
        (-2.0 * pi, -2, 1),
      ];
      for &(val, numer, denom) in pi_fractions {
        if (f2 - val).abs() < 1e-3 {
          if denom == 1 {
            if numer == -1 {
              return Ok(Expr::UnaryOp {
                op: crate::syntax::UnaryOperator::Minus,
                operand: Box::new(Expr::Constant("Pi".to_string())),
              });
            }
            return Ok(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(Expr::Integer(numer)),
              right: Box::new(Expr::Constant("Pi".to_string())),
            });
          }
          return Ok(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(numer), Expr::Integer(denom)],
              },
              Expr::Constant("Pi".to_string()),
            ],
          });
        }
      }
    }
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "Limit".to_string(),
    args: vec![
      expr.clone(),
      Expr::Rule {
        pattern: Box::new(Expr::Identifier(var_name.to_string())),
        replacement: Box::new(point.clone()),
      },
    ],
  })
}

/// Evaluate an expression numerically at var = n
fn eval_at_large_n(expr: &Expr, var: &str, n: i128) -> Option<f64> {
  let subst = crate::syntax::substitute_variable(expr, var, &Expr::Integer(n));
  let val = crate::evaluator::evaluate_expr_to_expr(&subst).ok()?;
  crate::functions::math_ast::try_eval_to_f64(&val)
}

/// Check if an expression is a "clean" value (integer, real, constant, or rational)
fn is_clean_value(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      true
    }
    _ => crate::functions::math_ast::try_eval_to_f64(expr).is_some(),
  }
}

/// MaxLimit[f, x -> a] - largest limiting value (from above/right)
/// MinLimit[f, x -> a] - smallest limiting value (from below/left)
pub fn max_limit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  one_sided_limit_ast(args, "MaxLimit", LimitDirection::FromAbove)
}

pub fn min_limit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  one_sided_limit_ast(args, "MinLimit", LimitDirection::FromBelow)
}

fn one_sided_limit_ast(
  args: &[Expr],
  fn_name: &str,
  direction: LimitDirection,
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: fn_name.to_string(),
      args: args.to_vec(),
    });
  }

  // Build Limit[expr, rule, Direction -> dir]
  let dir_str = match direction {
    LimitDirection::FromAbove => "FromAbove",
    LimitDirection::FromBelow => "FromBelow",
    LimitDirection::TwoSided => "FromAbove",
  };
  let direction_opt = Expr::Rule {
    pattern: Box::new(Expr::Identifier("Direction".to_string())),
    replacement: Box::new(Expr::String(dir_str.to_string())),
  };

  let result = limit_ast(&[args[0].clone(), args[1].clone(), direction_opt])?;

  // Check if result is unevaluated Limit
  if let Expr::FunctionCall { name, .. } = &result
    && name == "Limit"
  {
    return Ok(Expr::FunctionCall {
      name: fn_name.to_string(),
      args: args.to_vec(),
    });
  }

  Ok(result)
}

/// Direction for one-sided limits
#[derive(Debug, Clone, Copy, PartialEq)]
enum LimitDirection {
  /// Two-sided limit (default)
  TwoSided,
  /// From above (x -> x0+), i.e. from the right
  FromAbove,
  /// From below (x -> x0-), i.e. from the left
  FromBelow,
}

/// Parse the Direction option from a Rule like `Direction -> "FromAbove"`
fn parse_direction(option: &Expr) -> Option<LimitDirection> {
  if let Expr::Rule {
    pattern,
    replacement,
  } = option
    && let Expr::Identifier(name) = pattern.as_ref()
    && name == "Direction"
  {
    match replacement.as_ref() {
      Expr::String(s) if s == "FromAbove" => {
        return Some(LimitDirection::FromAbove);
      }
      Expr::String(s) if s == "FromBelow" => {
        return Some(LimitDirection::FromBelow);
      }
      // Direction -> -1 means from above (from the right, x -> x0+)
      Expr::Integer(n) if *n == -1 => {
        return Some(LimitDirection::FromAbove);
      }
      // Direction -> 1 means from below (from the left, x -> x0-)
      Expr::Integer(n) if *n == 1 => {
        return Some(LimitDirection::FromBelow);
      }
      _ => {}
    }
  }
  None
}

/// Check if an expression contains a Piecewise function call anywhere.
fn contains_piecewise(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      if name == "Piecewise" {
        return true;
      }
      args.iter().any(contains_piecewise)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_piecewise(left) || contains_piecewise(right)
    }
    Expr::UnaryOp { operand, .. } => contains_piecewise(operand),
    _ => false,
  }
}

/// Extract (numerator, denominator) from a canonicalized Times expression.
/// Recognizes patterns like Times[Power[den, -1], num] or
/// Times[num, Power[den, -1]] including multi-factor products.
fn extract_quotient_from_times(expr: &Expr) -> Option<(Expr, Expr)> {
  let factors: Vec<&Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().collect()
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => vec![left.as_ref(), right.as_ref()],
    _ => return None,
  };

  // Find Power[something, -1] factor(s) — those form the denominator
  let mut den_factors: Vec<Expr> = Vec::new();
  let mut num_factors: Vec<Expr> = Vec::new();

  for factor in &factors {
    // Check for Power[base, negative_integer] — these represent 1/base^|n|
    let inverse_base = match factor {
      Expr::FunctionCall { name, args }
        if name == "Power"
          && args.len() == 2
          && matches!(&args[1], Expr::Integer(n) if *n < 0) =>
      {
        if let Expr::Integer(n) = &args[1] {
          if *n == -1 {
            Some(args[0].clone())
          } else {
            // Power[base, -k] → denominator is Power[base, k]
            Some(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![args[0].clone(), Expr::Integer(-*n)],
            })
          }
        } else {
          None
        }
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left,
        right,
      } if matches!(right.as_ref(), Expr::Integer(n) if *n < 0) => {
        if let Expr::Integer(n) = right.as_ref() {
          if *n == -1 {
            Some(left.as_ref().clone())
          } else {
            Some(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: left.clone(),
              right: Box::new(Expr::Integer(-*n)),
            })
          }
        } else {
          None
        }
      }
      _ => None,
    };

    if let Some(den) = inverse_base {
      den_factors.push(den);
    } else {
      num_factors.push((*factor).clone());
    }
  }

  if den_factors.is_empty() {
    return None; // No denominator found
  }

  let numerator = if num_factors.len() == 1 {
    num_factors.pop().unwrap()
  } else if num_factors.is_empty() {
    Expr::Integer(1)
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: num_factors,
    }
  };

  let denominator = if den_factors.len() == 1 {
    den_factors.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: den_factors,
    }
  };

  Some((numerator, denominator))
}

/// Evaluate an expression numerically at var = point + delta
fn eval_near_point(
  expr: &Expr,
  var: &str,
  point: &Expr,
  delta: f64,
) -> Option<f64> {
  let point_val = crate::functions::math_ast::try_eval_to_f64(point)?;
  let val_at = point_val + delta;
  let subst =
    crate::syntax::substitute_variable(expr, var, &Expr::Real(val_at));
  let result = crate::evaluator::evaluate_expr_to_expr(&subst).ok()?;
  crate::functions::math_ast::try_eval_to_f64(&result)
}

/// Compute a one-sided limit numerically by evaluating at points approaching x0
fn numerical_one_sided_limit(
  expr: &Expr,
  var_name: &str,
  point: &Expr,
  direction: LimitDirection,
) -> Option<Expr> {
  let sign = match direction {
    LimitDirection::FromAbove => 1.0,
    LimitDirection::FromBelow => -1.0,
    LimitDirection::TwoSided => return None,
  };

  // Evaluate at decreasing distances from the point
  let deltas = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12];
  let mut vals = Vec::new();
  for &d in &deltas {
    if let Some(v) = eval_near_point(expr, var_name, point, sign * d) {
      if v.is_nan() {
        return None;
      }
      vals.push(v);
    } else {
      return None;
    }
  }

  // Check for immediate infinity (even at the first sample point)
  if vals.iter().any(|v| v.is_infinite()) {
    // Determine sign from the first non-infinite value, or from the infinite ones
    let sign_positive = vals
      .iter()
      .find(|v| !v.is_infinite())
      .map(|v| *v > 0.0)
      .unwrap_or_else(|| vals[0].is_sign_positive());
    if sign_positive {
      return Some(Expr::Identifier("Infinity".to_string()));
    } else {
      return Some(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
  }

  // Check if the values are monotonically diverging (sign consistent, magnitude increasing)
  let all_positive = vals.iter().all(|&v| v > 0.0);
  let all_negative = vals.iter().all(|&v| v < 0.0);
  let magnitudes_increasing = vals.windows(2).all(|w| w[1].abs() > w[0].abs());

  if magnitudes_increasing && (all_positive || all_negative) {
    // Check that the growth is unbounded (magnitude at least doubles over the range)
    if vals.last().unwrap().abs() > 2.0 * vals.first().unwrap().abs() {
      if all_positive {
        return Some(Expr::Identifier("Infinity".to_string()));
      } else {
        return Some(Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(Expr::Identifier("Infinity".to_string())),
        });
      }
    }
  }

  // Check convergence to a finite value
  let last = *vals.last().unwrap();
  let second_last = vals[vals.len() - 2];
  let diff = (last - second_last).abs();
  let scale = last.abs().max(second_last.abs()).max(1e-15);
  if diff / scale < 0.01 || diff < 1e-10 {
    // Converging — determine the value
    let rounded = last.round();
    if (last - rounded).abs() < 1e-6 {
      return Some(Expr::Integer(rounded as i128));
    }
    // Check for known constants
    if (last - std::f64::consts::E).abs() < 1e-4 {
      return Some(Expr::Constant("E".to_string()));
    }
    if (last - std::f64::consts::PI).abs() < 1e-4 {
      return Some(Expr::Constant("Pi".to_string()));
    }
    return Some(Expr::Real(last));
  }

  None
}

/// Compute a two-sided limit numerically by checking both sides agree
fn numerical_two_sided_limit(
  expr: &Expr,
  var_name: &str,
  point: &Expr,
) -> Option<Expr> {
  let from_above =
    numerical_one_sided_limit(expr, var_name, point, LimitDirection::FromAbove);
  let from_below =
    numerical_one_sided_limit(expr, var_name, point, LimitDirection::FromBelow);

  match (from_above, from_below) {
    (Some(a), Some(b)) => {
      // Check if both sides agree
      let a_val = crate::functions::math_ast::try_eval_to_f64(&a);
      let b_val = crate::functions::math_ast::try_eval_to_f64(&b);
      match (a_val, b_val) {
        (Some(av), Some(bv)) => {
          let diff = (av - bv).abs();
          let scale = av.abs().max(bv.abs()).max(1e-15);
          if diff / scale < 0.01 || diff < 1e-10 {
            // Both sides converge to the same value
            return Some(a);
          }
          // Sides disagree — indeterminate
          Some(Expr::Identifier("Indeterminate".to_string()))
        }
        _ => {
          // At least one side is infinite — check if they match symbolically
          let a_str = crate::syntax::expr_to_string(&a);
          let b_str = crate::syntax::expr_to_string(&b);
          if a_str == b_str {
            return Some(a);
          }
          // Different infinities — indeterminate
          Some(Expr::Identifier("Indeterminate".to_string()))
        }
      }
    }
    _ => None,
  }
}

/// Limit[expr, x -> x0] - Compute the limit of expr as x approaches x0
/// Limit[expr, x -> x0, Direction -> "FromAbove"] - One-sided limit
pub fn limit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Limit expects 2 or 3 arguments".into(),
    ));
  }

  // Parse optional Direction from 3rd argument
  let direction = if args.len() == 3 {
    parse_direction(&args[2]).unwrap_or(LimitDirection::TwoSided)
  } else {
    LimitDirection::TwoSided
  };

  // Second argument must be a Rule: x -> x0
  let (var_name, point) = match &args[1] {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let name = match pattern.as_ref() {
        Expr::Identifier(n) => n.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Limit".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (name, replacement.as_ref().clone())
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Limit".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Handle limits at Infinity
  if is_infinity(&point) || is_negative_infinity(&point) {
    return limit_at_infinity(&args[0], &var_name, &point);
  }

  // For one-sided limits, skip direct substitution when the expression
  // contains Piecewise, because substituting the exact boundary point may
  // select the wrong branch (e.g. the >= branch instead of the < branch).
  // In that case we fall through to numerical evaluation which correctly
  // approaches from the specified direction.
  let skip_direct_sub = contains_piecewise(&args[0]);

  // Strategy: try direct substitution first.
  // Suppress messages during trial substitution (e.g. Power::infy for Sin[x]/x at x=0)
  if !skip_direct_sub {
    let substituted =
      crate::syntax::substitute_variable(&args[0], &var_name, &point);
    let saved_warnings = crate::snapshot_warnings();
    crate::push_quiet();
    let result = crate::evaluator::evaluate_expr_to_expr(&substituted);
    crate::pop_quiet();
    crate::restore_warnings(saved_warnings);

    match result {
      Ok(ref val) => {
        // Check if the result is a valid numeric value (not Indeterminate, ComplexInfinity, etc.)
        match val {
          Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => {
            return result;
          }
          Expr::FunctionCall { name, args: fargs }
            if name == "Rational" && fargs.len() == 2 =>
          {
            // Check for 0/0 indeterminate form
            if matches!(&fargs[1], Expr::Integer(0)) {
              // Fall through to L'Hôpital
            } else {
              return result;
            }
          }
          Expr::FunctionCall { name, .. }
            if name == "DirectedInfinity" || name == "Indeterminate" =>
          {
            // Fall through to try other methods
          }
          _ => {
            // Check if it evaluates to a number via N[]
            if crate::functions::math_ast::try_eval_to_f64(val).is_some() {
              return result;
            }
            // Return unevaluated if substitution doesn't yield a clean result
          }
        }
      }
      Err(_) => {
        // Substitution failed (e.g., division by zero)
      }
    }
  }

  // Try L'Hôpital's rule for 0/0 forms
  // Extract numerator and denominator from either BinaryOp::Divide or
  // the canonical Times[Power[den, -1], num] form.
  let num_den: Option<(Expr, Expr)> = if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: num,
    right: den,
  } = &args[0]
  {
    Some((*num.clone(), *den.clone()))
  } else {
    extract_quotient_from_times(&args[0])
  };

  if let Some((numerator, denominator)) = num_den {
    let num_at_point =
      crate::syntax::substitute_variable(&numerator, &var_name, &point);
    let den_at_point =
      crate::syntax::substitute_variable(&denominator, &var_name, &point);
    let saved2 = crate::snapshot_warnings();
    crate::push_quiet();
    let num_val = crate::evaluator::evaluate_expr_to_expr(&num_at_point);
    let den_val = crate::evaluator::evaluate_expr_to_expr(&den_at_point);
    crate::pop_quiet();
    crate::restore_warnings(saved2);

    let num_is_zero = matches!(&num_val, Ok(Expr::Integer(0)))
      || matches!(&num_val, Ok(Expr::Real(f)) if *f == 0.0);
    let den_is_zero = matches!(&den_val, Ok(Expr::Integer(0)))
      || matches!(&den_val, Ok(Expr::Real(f)) if *f == 0.0);

    if num_is_zero && den_is_zero {
      // Apply L'Hôpital: Limit[f'/g', x -> x0]
      if let (Ok(df), Ok(dg)) = (
        differentiate(&numerator, &var_name),
        differentiate(&denominator, &var_name),
      ) {
        let new_expr = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(simplify(df)),
          right: Box::new(simplify(dg)),
        };
        return limit_ast(&[new_expr, args[1].clone()]);
      }
    }
  }

  // Numerical approach for one-sided or two-sided limits
  match direction {
    LimitDirection::FromAbove | LimitDirection::FromBelow => {
      if let Some(result) =
        numerical_one_sided_limit(&args[0], &var_name, &point, direction)
      {
        return Ok(result);
      }
    }
    LimitDirection::TwoSided => {
      if let Some(result) =
        numerical_two_sided_limit(&args[0], &var_name, &point)
      {
        return Ok(result);
      }
    }
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "Limit".to_string(),
    args: args.to_vec(),
  })
}

/// Rational arithmetic helpers for coefficient-based series computation
/// Represents a rational number as (numerator, denominator) with denominator > 0
fn rat_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  let num = a.0 * b.1 + b.0 * a.1;
  let den = a.1 * b.1;
  rat_reduce(num, den)
}

fn rat_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  rat_reduce(a.0 * b.0, a.1 * b.1)
}

fn rat_div(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  if b.0 < 0 {
    rat_reduce(-a.0 * b.1, a.1 * -b.0)
  } else {
    rat_reduce(a.0 * b.1, a.1 * b.0)
  }
}

fn rat_reduce(num: i128, den: i128) -> (i128, i128) {
  if num == 0 {
    return (0, 1);
  }
  let g = gcd(num.abs(), den.abs());
  let (n, d) = (num / g, den / g);
  if d < 0 { (-n, -d) } else { (n, d) }
}

fn rat_to_expr(r: (i128, i128)) -> Expr {
  if r.1 == 1 {
    Expr::Integer(r.0)
  } else {
    crate::functions::math_ast::make_rational_pub(r.0, r.1)
  }
}

/// Cauchy product of two coefficient vectors (convolution)
fn cauchy_product(
  a: &[(i128, i128)],
  b: &[(i128, i128)],
  n: usize,
) -> (i128, i128) {
  let mut sum = (0i128, 1i128);
  for k in 0..=n {
    if k < a.len() && (n - k) < b.len() {
      sum = rat_add(sum, rat_mul(a[k], b[n - k]));
    }
  }
  sum
}

/// Compute Tan[x] series coefficients around x=0 up to order n
/// Uses recurrence: tan'(x) = 1 + tan²(x), t(0) = 0
fn tan_series_coeffs(order: usize) -> Vec<(i128, i128)> {
  let mut t: Vec<(i128, i128)> = vec![(0, 1)]; // t_0 = 0
  for n in 0..order {
    // (n+1)*t_{n+1} = delta_{n,0} + sum_{k=0..n} t_k * t_{n-k}
    let mut rhs = cauchy_product(&t, &t, n);
    if n == 0 {
      rhs = rat_add(rhs, (1, 1));
    }
    // t_{n+1} = rhs / (n+1)
    t.push(rat_div(rhs, ((n + 1) as i128, 1)));
  }
  t
}

/// Compute Sec[x] series coefficients around x=0 up to order n
/// Uses: sec'(x) = sec(x)*tan(x), sec(0) = 1
/// If s = sec, t = tan: s'= s*t, t' = 1 + t^2 = s^2
fn sec_series_coeffs(order: usize) -> Vec<(i128, i128)> {
  let mut s: Vec<(i128, i128)> = vec![(1, 1)]; // s_0 = 1
  let mut t: Vec<(i128, i128)> = vec![(0, 1)]; // t_0 = 0
  for n in 0..order {
    // (n+1)*t_{n+1} = sum_{k=0..n} s_k * s_{n-k}
    let t_rhs = cauchy_product(&s, &s, n);
    t.push(rat_div(t_rhs, ((n + 1) as i128, 1)));
    // (n+1)*s_{n+1} = sum_{k=0..n} s_k * t_{n-k}
    let s_rhs = cauchy_product(&s, &t, n);
    s.push(rat_div(s_rhs, ((n + 1) as i128, 1)));
  }
  s
}

/// Compute Csc[x] series coefficients around x=0 up to order n
/// csc(x) = 1/sin(x), has a pole at x=0 so series starts at x^{-1}
/// Returns (coefficients, nmin) where nmin is the starting power
fn csc_series_coeffs(order: usize) -> (Vec<(i128, i128)>, i128) {
  // csc(x) = 1/x + x/6 + 7*x^3/360 + ...
  // Use: sin(x)*csc(x) = 1, solve for csc coefficients
  // sin coefficients: s_1 = 1, s_3 = -1/6, s_5 = 1/120, ...
  let total = order + 2; // need extra terms since csc starts at x^{-1}
  let mut sin_c: Vec<(i128, i128)> = Vec::new();
  let mut factorial = 1i128;
  for k in 0..=total {
    if k % 2 == 0 {
      sin_c.push((0, 1));
    } else {
      let sign = if (k / 2) % 2 == 0 { 1 } else { -1 };
      sin_c.push((sign, factorial));
    }
    if k < total {
      factorial *= (k + 1) as i128;
    }
  }
  // Recompute: sin_c[k] = coefficient of x^k in sin(x)
  // We need sin_c as rationals properly
  let mut sin_coeffs: Vec<(i128, i128)> = Vec::new();
  let mut fact = 1i128;
  for k in 0..=total {
    if k > 1 {
      fact *= k as i128;
    }
    if k % 2 == 0 {
      sin_coeffs.push((0, 1));
    } else {
      let sign = if (k / 2) % 2 == 0 { 1i128 } else { -1 };
      sin_coeffs.push(rat_reduce(sign, fact));
    }
  }

  // csc(x) = c_{-1}/x + c_1*x + c_3*x^3 + ...
  // sin(x)*csc(x) = 1
  // Shift: let csc(x) = (1/x) * C(x) where C(x) = c_0 + c_1*x + c_2*x^2 + ...
  // Then sin(x) * C(x) / x = 1, so sin(x)/x * C(x) = 1
  // sinc(x) = sin(x)/x = 1 - x^2/6 + x^4/120 - ...
  // sinc coefficients: sinc_k = sin_coeffs[k+1] (shift by one)
  let mut sinc: Vec<(i128, i128)> = Vec::new();
  for k in 0..=total {
    if k + 1 < sin_coeffs.len() {
      sinc.push(sin_coeffs[k + 1]);
    } else {
      sinc.push((0, 1));
    }
  }

  // C(x) = 1/sinc(x), so sinc*C = 1
  // c_0 = 1/sinc_0 = 1
  // c_n = -(1/sinc_0) * sum_{k=1..n} sinc_k * c_{n-k}
  let mut c: Vec<(i128, i128)> = vec![(1, 1)];
  for n in 1..=total {
    let mut s = (0i128, 1i128);
    for k in 1..=n {
      if k < sinc.len() && (n - k) < c.len() {
        s = rat_add(s, rat_mul(sinc[k], c[n - k]));
      }
    }
    c.push(rat_reduce(-s.0, s.1));
  }

  // csc(x) = c_0/x + c_1 + c_2*x + ... = c_0*x^{-1} + c_1*x^0 + ...
  // In SeriesData format, nmin = -1, coefficients are c_0, c_1, c_2, ...
  // But only keep up to order
  let keep = (order as i128 + 2) as usize; // from x^{-1} to x^{order}
  let coeffs: Vec<(i128, i128)> = c.into_iter().take(keep).collect();
  (coeffs, -1)
}

/// Compute Cot[x] series coefficients around x=0
/// Uses: cot'(x) = -1 - cot²(x) (but cot has pole at 0)
/// cot(x) = cos(x)/sin(x) = 1/x - x/3 - x^3/45 - ...
/// Returns (coefficients, nmin)
fn cot_series_coeffs(order: usize) -> (Vec<(i128, i128)>, i128) {
  // Use cos(x)/sin(x) = cot(x)
  // cos(x) = cot(x)*sin(x)
  // Similar to csc: let cot(x) = (1/x)*C(x)
  // cos(x) = C(x)*sin(x)/x = C(x)*sinc(x)
  let total = order + 2;

  let mut sin_coeffs: Vec<(i128, i128)> = Vec::new();
  let mut cos_coeffs: Vec<(i128, i128)> = Vec::new();
  let mut fact = 1i128;
  for k in 0..=total {
    if k > 1 {
      fact *= k as i128;
    }
    if k % 2 == 0 {
      let sign = if (k / 2) % 2 == 0 { 1i128 } else { -1 };
      cos_coeffs.push(rat_reduce(sign, fact));
      sin_coeffs.push((0, 1));
    } else {
      cos_coeffs.push((0, 1));
      let sign = if (k / 2) % 2 == 0 { 1i128 } else { -1 };
      sin_coeffs.push(rat_reduce(sign, fact));
    }
  }

  // sinc = sin(x)/x
  let mut sinc: Vec<(i128, i128)> = Vec::new();
  for k in 0..=total {
    if k + 1 < sin_coeffs.len() {
      sinc.push(sin_coeffs[k + 1]);
    } else {
      sinc.push((0, 1));
    }
  }

  // C(x)*sinc(x) = cos(x), where cot(x) = C(x)/x
  // c_0 = cos_0/sinc_0 = 1
  // c_n = (cos_n - sum_{k=1..n} sinc_k * c_{n-k}) / sinc_0
  let mut c: Vec<(i128, i128)> = vec![(1, 1)];
  for n in 1..=total {
    let mut s = if n < cos_coeffs.len() {
      cos_coeffs[n]
    } else {
      (0, 1)
    };
    for k in 1..=n {
      if k < sinc.len() && (n - k) < c.len() {
        s = rat_add(s, rat_mul((-sinc[k].0, sinc[k].1), c[n - k]));
      }
    }
    c.push(s);
  }

  let keep = (order as i128 + 2) as usize;
  let coeffs: Vec<(i128, i128)> = c.into_iter().take(keep).collect();
  (coeffs, -1)
}

/// Try coefficient-based series for known functions around x=0.
/// Returns Some((coefficients, nmin)) if handled, None otherwise.
fn try_fast_series(
  expr: &Expr,
  var_name: &str,
  x0: &Expr,
  order: i128,
) -> Option<(Vec<(i128, i128)>, i128)> {
  // Only for expansion around 0
  if !matches!(x0, Expr::Integer(0)) {
    return None;
  }

  // Only for f[var] form
  match expr {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      // Check inner arg is the variable
      if !matches!(&args[0], Expr::Identifier(v) if v == var_name) {
        return None;
      }
      let n = order as usize;
      match name.as_str() {
        "Tan" => Some((tan_series_coeffs(n), 0)),
        "Sec" => Some((sec_series_coeffs(n), 0)),
        "Csc" => Some(csc_series_coeffs(n)),
        "Cot" => Some(cot_series_coeffs(n)),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Series[expr, {x, x0, n}] - Taylor series expansion
pub fn series_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Series expects at least 2 arguments".into(),
    ));
  }

  // Extract options (e.g., Assumptions -> x > 0) from remaining args
  let mut option_args = Vec::new();
  let mut spec_args = Vec::new();
  for arg in &args[2..] {
    if let Expr::Rule { .. } = arg {
      option_args.push(arg.clone());
    } else {
      spec_args.push(arg.clone());
    }
  }

  // Handle multivariate: Series[expr, {x, x0, nx}, {y, y0, ny}, ...]
  if !spec_args.is_empty() {
    // First expand in the first variable (pass options too)
    let mut first_args = vec![args[0].clone(), args[1].clone()];
    first_args.extend(option_args.clone());
    let first_result = series_ast(&first_args)?;
    // Then expand coefficients in each subsequent variable
    let mut result = first_result;
    for spec in &spec_args {
      result = expand_series_data_coefficients(&result, spec)?;
    }
    return Ok(result);
  }

  // Second argument: {x, x0, n}
  let (var_name, x0, order) = match &args[1] {
    Expr::List(items) if items.len() == 3 => {
      let name = match &items[0] {
        Expr::Identifier(n) => n.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Series".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let order = match &items[2] {
        Expr::Integer(n) => *n,
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Series".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (name, items[1].clone(), order)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Series".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Try fast coefficient-based computation for known functions
  if let Some((rat_coeffs, nmin)) =
    try_fast_series(&args[0], &var_name, &x0, order)
  {
    let mut coefficients: Vec<Expr> =
      rat_coeffs.iter().map(|r| rat_to_expr(*r)).collect();
    let mut actual_nmin = nmin;

    // Strip leading zeros
    while !coefficients.is_empty()
      && matches!(coefficients[0], Expr::Integer(0))
    {
      coefficients.remove(0);
      actual_nmin += 1;
    }

    // Strip trailing zeros
    while coefficients.len() > 1
      && matches!(coefficients.last(), Some(Expr::Integer(0)))
    {
      coefficients.pop();
    }

    if coefficients.is_empty() {
      return Ok(Expr::Integer(0));
    }

    return Ok(Expr::FunctionCall {
      name: "SeriesData".to_string(),
      args: vec![
        Expr::Identifier(var_name),
        x0,
        Expr::List(coefficients),
        Expr::Integer(actual_nmin),
        Expr::Integer(order + 1),
        Expr::Integer(1),
      ],
    });
  }

  // Fast path for ExpIntegralEi series
  if let Expr::FunctionCall {
    name: fname,
    args: fargs,
  } = &args[0]
    && fname == "ExpIntegralEi"
    && fargs.len() == 1
    && matches!(&fargs[0], Expr::Identifier(v) if v == &var_name)
  {
    if matches!(&x0, Expr::Integer(0)) {
      // Series[ExpIntegralEi[x], {x, 0, n}]
      // Ei(x) = EulerGamma + Log[|x|] + Σ_{k=1}^n x^k / (k * k!)
      // For x > 0: Ei(x) = EulerGamma + Log[x] + ...
      // For x < 0: Ei(x) = EulerGamma + Log[-x] + ...
      // Check Assumptions option for sign of x
      let mut assume_negative = false;
      for opt in &option_args {
        if let Expr::Rule {
          pattern,
          replacement,
        } = opt
          && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Assumptions")
        {
          // Check if the assumption is x < 0
          if let Expr::Comparison {
            operands,
            operators,
          } = replacement.as_ref()
            && operands.len() == 2
            && matches!(&operands[0], Expr::Identifier(v) if v == &var_name)
            && matches!(&operands[1], Expr::Integer(0))
            && operators.len() == 1
            && matches!(&operators[0], crate::syntax::ComparisonOp::Less)
          {
            assume_negative = true;
          }
        }
      }

      let log_arg = if assume_negative {
        // Log[-x]
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), Expr::Identifier(var_name.clone())],
        }
      } else {
        // Log[x]
        Expr::Identifier(var_name.clone())
      };
      let c0 = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Identifier("EulerGamma".to_string()),
          Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![log_arg],
          },
        ],
      };
      let mut coefficients = vec![c0];
      let mut factorial: i128 = 1;
      for k in 1..=order {
        factorial *= k;
        // c_k = 1 / (k * k!)
        coefficients.push(rat_to_expr((1, k * factorial)));
      }

      return Ok(Expr::FunctionCall {
        name: "SeriesData".to_string(),
        args: vec![
          Expr::Identifier(var_name),
          x0,
          Expr::List(coefficients),
          Expr::Integer(0),
          Expr::Integer(order + 1),
          Expr::Integer(1),
        ],
      });
    }

    if matches!(&x0, Expr::Identifier(s) if s == "Infinity" || s == "DirectedInfinity")
      || matches!(&x0, Expr::FunctionCall { name, args: a } if name == "DirectedInfinity" && a.len() == 1 && matches!(&a[0], Expr::Integer(1)))
    {
      // Series[ExpIntegralEi[x], {x, Infinity, n}]
      // Asymptotic expansion: Ei(x) ~ E^x/x * Σ_{k=0}^{n-1} k!/x^k + regularization
      // The result in terms of SeriesData at Infinity:
      // SeriesData[x, Infinity, {coeffs...}, -1, -(n+1), -1]
      // where coefficients are k! (factorials)
      // But the output format from Wolfram is specific. Let me construct it differently.
      // Actually, Wolfram returns it as a proper SeriesData which // Normal gives the expression.
      //
      // Normal of the asymptotic expansion gives:
      // E^x * Sum[k!/x^(k+1), {k, 0, n-1}] + (Log[-1/x] - Log[-x] + 2*Log[x])/2
      //
      // Let me construct the SeriesData directly.
      // SeriesData[x, Infinity, {1, 1, 2, 6, 24, 120, ...}, 1, n+1, 1]
      // where the coefficients are k! and the powers are x^(-k-1)
      // The convention for series at infinity is different.

      // Build the Normal form directly since the SeriesData at infinity is complex
      // E^x*(n!/x^(n+1) + ... + 1/x^2 + 1/x) + (Log[-1/x] - Log[-x] + 2*Log[x])/2
      let mut exp_terms = Vec::new();
      let mut fact: i128 = 1;
      for k in 0..order {
        if k > 0 {
          fact *= k;
        }
        // k!/x^(k+1) = fact * Power[x, -(k+1)]
        let power = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::Identifier(var_name.clone()),
            Expr::Integer(-(k + 1)),
          ],
        };
        if fact == 1 {
          exp_terms.push(power);
        } else {
          exp_terms.push(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(fact), power],
          });
        }
      }
      // Reverse to show highest power first (matching Wolfram output order)
      exp_terms.reverse();

      // E^x * (sum of terms)
      let exp_x = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          Expr::Constant("E".to_string()),
          Expr::Identifier(var_name.clone()),
        ],
      };
      let exp_part = Expr::FunctionCall {
        name: "Times".to_string(),
        args: {
          let mut a = vec![exp_x];
          if exp_terms.len() == 1 {
            a.push(exp_terms.into_iter().next().unwrap());
          } else {
            a.push(Expr::FunctionCall {
              name: "Plus".to_string(),
              args: exp_terms,
            });
          }
          a
        },
      };

      // Regularization term: (Log[-1/x] - Log[-x] + 2*Log[x])/2
      let log_neg_inv_x = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(-1),
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![Expr::Identifier(var_name.clone()), Expr::Integer(-1)],
            },
          ],
        }],
      };
      let log_neg_x = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), Expr::Identifier(var_name.clone())],
        }],
      };
      let two_log_x = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(2),
          Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![Expr::Identifier(var_name.clone())],
          },
        ],
      };
      let log_sum = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          log_neg_inv_x,
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), log_neg_x],
          },
          two_log_x,
        ],
      };
      let reg_term = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(log_sum),
        right: Box::new(Expr::Integer(2)),
      };

      let result = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(exp_part),
        right: Box::new(reg_term),
      };
      return Ok(result);
    }
  }

  // Compute Taylor coefficients: f^(k)(x0) / k!
  let mut coefficients = Vec::new();
  let mut current_expr = args[0].clone();

  for k in 0..=order {
    // Evaluate current derivative at x0
    let substituted =
      crate::syntax::substitute_variable(&current_expr, &var_name, &x0);
    let value = crate::evaluator::evaluate_expr_to_expr(&substituted)?;

    // Compute k!
    let mut factorial = 1i128;
    for i in 2..=k {
      factorial *= i;
    }

    // Coefficient = value / k!
    let coeff = if matches!(&value, Expr::Integer(0)) {
      Expr::Integer(0)
    } else if factorial == 1 {
      value
    } else {
      // value / factorial
      match &value {
        Expr::Integer(n) => {
          let g = gcd(n.abs(), factorial);
          let (num, den) = (n / g, factorial / g);
          if den == 1 {
            Expr::Integer(num)
          } else {
            crate::functions::math_ast::make_rational_pub(num, den)
          }
        }
        // Handle Rational[n, d] / factorial → Rational[n, d*factorial] simplified
        Expr::FunctionCall { name, args: rargs }
          if name == "Rational"
            && rargs.len() == 2
            && matches!(&rargs[0], Expr::Integer(_))
            && matches!(&rargs[1], Expr::Integer(_)) =>
        {
          if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1]) {
            crate::functions::math_ast::make_rational_pub(*n, d * factorial)
          } else {
            Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Divide,
              left: Box::new(value),
              right: Box::new(Expr::Integer(factorial)),
            }
          }
        }
        _ => Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(value),
          right: Box::new(Expr::Integer(factorial)),
        },
      }
    };

    coefficients.push(coeff);

    // Differentiate for the next iteration (unless this is the last)
    if k < order {
      current_expr = match differentiate(&current_expr, &var_name) {
        Ok(d) => simplify(d),
        Err(_) => {
          return Ok(Expr::FunctionCall {
            name: "Series".to_string(),
            args: args.to_vec(),
          });
        }
      };
    }
  }

  // Strip leading zero coefficients and adjust nmin
  let mut nmin: i128 = 0;
  while !coefficients.is_empty() && matches!(coefficients[0], Expr::Integer(0))
  {
    coefficients.remove(0);
    nmin += 1;
  }

  // If all coefficients are zero, return 0
  if coefficients.is_empty() {
    return Ok(Expr::Integer(0));
  }

  // Build SeriesData[x, x0, {c0, c1, ...}, nmin, nmax, 1]
  Ok(Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name),
      x0,
      Expr::List(coefficients),
      Expr::Integer(nmin),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ],
  })
}

/// Expand each coefficient of a SeriesData in a new variable.
/// Used for multivariate Series: each coefficient becomes a SeriesData in the new variable.
fn expand_series_data_coefficients(
  series: &Expr,
  spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // series should be SeriesData[var, x0, {coeffs}, nmin, nmax, den]
  if let Expr::FunctionCall { name, args } = series
    && name == "SeriesData"
    && args.len() == 6
    && let Expr::List(coeffs) = &args[2]
  {
    // Expand each coefficient in the new variable
    let mut new_coeffs = Vec::new();
    for c in coeffs {
      let expanded = series_ast(&[c.clone(), spec.clone()])?;
      new_coeffs.push(expanded);
    }
    return Ok(Expr::FunctionCall {
      name: "SeriesData".to_string(),
      args: vec![
        args[0].clone(), // var
        args[1].clone(), // x0
        Expr::List(new_coeffs),
        args[3].clone(), // nmin
        args[4].clone(), // nmax
        args[5].clone(), // den
      ],
    });
  }
  // If not a SeriesData, just expand the expression
  series_ast(&[series.clone(), spec.clone()])
}

fn gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// NIntegrate[expr, {var, lo, hi}] - Numerical integration
pub fn nintegrate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "NIntegrate expects exactly 2 arguments".into(),
    ));
  }

  // Second argument must be {var, lo, hi}
  let (var_name, lo, hi) = match &args[1] {
    Expr::List(items) if items.len() == 3 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "NIntegrate: first element of integration range must be a symbol"
              .into(),
          ));
        }
      };
      // Evaluate bounds — support Infinity/-Infinity
      let lo_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
      let hi_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;
      let lo = expr_to_bound(&lo_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "NIntegrate: lower bound must be numeric or Infinity".into(),
        )
      })?;
      let hi = expr_to_bound(&hi_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "NIntegrate: upper bound must be numeric or Infinity".into(),
        )
      })?;
      (var_name, lo, hi)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "NIntegrate expects {var, lo, hi} as second argument".into(),
      ));
    }
  };

  let integrand = &args[0];

  // Evaluate the integrand at a point
  let eval_at = |x: f64| -> Option<f64> {
    let substituted =
      crate::syntax::substitute_variable(integrand, &var_name, &Expr::Real(x));
    let evaluated =
      crate::evaluator::evaluate_expr_to_expr(&substituted).ok()?;
    crate::functions::math_ast::try_eval_to_f64(&evaluated)
  };

  let lo_inf = lo.is_infinite() && lo < 0.0;
  let hi_inf = hi.is_infinite() && hi > 0.0;

  let result = if lo_inf && hi_inf {
    // (-∞, ∞): substitute x = tan(t), dx = sec²(t) dt, integrate over (-π/2, π/2)
    let eval_transformed = |t: f64| -> Option<f64> {
      let x = t.tan();
      let jacobian = 1.0 / (t.cos() * t.cos()); // sec²(t)
      eval_at(x).map(|v| v * jacobian)
    };
    let half_pi = std::f64::consts::FRAC_PI_2;
    // Use slightly inside to avoid tan(±π/2) = ±∞
    let eps = 1e-10;
    adaptive_simpson(
      &eval_transformed,
      -half_pi + eps,
      half_pi - eps,
      1e-10,
      50,
    )
  } else if lo_inf {
    // (-∞, b): substitute x = b - tan(t), dx = -sec²(t) dt, t in (0, π/2)
    let eval_transformed = |t: f64| -> Option<f64> {
      let x = hi - t.tan();
      let jacobian = 1.0 / (t.cos() * t.cos());
      eval_at(x).map(|v| v * jacobian)
    };
    let half_pi = std::f64::consts::FRAC_PI_2;
    let eps = 1e-10;
    adaptive_simpson(&eval_transformed, eps, half_pi - eps, 1e-10, 50)
  } else if hi_inf {
    // (a, ∞): substitute x = a + tan(t), dx = sec²(t) dt, t in (0, π/2)
    let eval_transformed = |t: f64| -> Option<f64> {
      let x = lo + t.tan();
      let jacobian = 1.0 / (t.cos() * t.cos());
      eval_at(x).map(|v| v * jacobian)
    };
    let half_pi = std::f64::consts::FRAC_PI_2;
    let eps = 1e-10;
    adaptive_simpson(&eval_transformed, eps, half_pi - eps, 1e-10, 50)
  } else {
    adaptive_simpson(&eval_at, lo, hi, 1e-12, 50)
  };

  match result {
    Some(val) => Ok(Expr::Real(val)),
    None => Err(InterpreterError::EvaluationError(
      "NIntegrate: failed to converge or integrand is not numeric".into(),
    )),
  }
}

/// Convert an expression to an f64 bound, supporting Infinity/-Infinity
fn expr_to_bound(expr: &Expr) -> Option<f64> {
  if matches!(expr, Expr::Identifier(s) if s == "Infinity") {
    return Some(f64::INFINITY);
  }
  if crate::functions::math_ast::is_neg_infinity(expr) {
    return Some(f64::NEG_INFINITY);
  }
  // DirectedInfinity[1] = Infinity, DirectedInfinity[-1] = -Infinity
  if let Expr::FunctionCall { name, args } = expr
    && name == "DirectedInfinity"
    && args.len() == 1
  {
    if matches!(&args[0], Expr::Integer(1)) {
      return Some(f64::INFINITY);
    }
    if matches!(&args[0], Expr::Integer(-1)) {
      return Some(f64::NEG_INFINITY);
    }
  }
  crate::functions::math_ast::try_eval_to_f64(expr)
}

/// Adaptive Simpson's quadrature
fn adaptive_simpson(
  f: &dyn Fn(f64) -> Option<f64>,
  a: f64,
  b: f64,
  tol: f64,
  max_depth: u32,
) -> Option<f64> {
  let fa = f(a)?;
  let fb = f(b)?;
  let m = (a + b) / 2.0;
  let fm = f(m)?;
  let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
  adaptive_simpson_rec(f, a, b, tol, whole, fa, fm, fb, max_depth)
}

fn adaptive_simpson_rec(
  f: &dyn Fn(f64) -> Option<f64>,
  a: f64,
  b: f64,
  tol: f64,
  whole: f64,
  fa: f64,
  fm: f64,
  fb: f64,
  depth: u32,
) -> Option<f64> {
  let m = (a + b) / 2.0;
  let m1 = (a + m) / 2.0;
  let m2 = (m + b) / 2.0;
  let fm1 = f(m1)?;
  let fm2 = f(m2)?;
  let h = b - a;
  let left = h / 12.0 * (fa + 4.0 * fm1 + fm);
  let right = h / 12.0 * (fm + 4.0 * fm2 + fb);
  let refined = left + right;
  let error = (refined - whole) / 15.0;

  if depth == 0 || error.abs() < tol {
    Some(refined + error)
  } else {
    let left_result =
      adaptive_simpson_rec(f, a, m, tol / 2.0, left, fa, fm1, fm, depth - 1)?;
    let right_result =
      adaptive_simpson_rec(f, m, b, tol / 2.0, right, fm, fm2, fb, depth - 1)?;
    Some(left_result + right_result)
  }
}

/// Grad[f, {x1, x2, ...}] - Gradient of a scalar function
pub fn grad_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Grad expects exactly 2 arguments".into(),
    ));
  }
  let vars = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Grad".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut components = Vec::with_capacity(vars.len());
  for var in vars {
    let var_name = match var {
      Expr::Identifier(s) => s,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Grad".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let deriv = differentiate_expr(&args[0], var_name)?;
    let evald = crate::evaluator::evaluate_expr_to_expr(&deriv)?;
    components.push(evald);
  }
  Ok(Expr::List(components))
}

/// Wronskian[{f1, ..., fn}, x] = determinant of matrix of derivatives
pub fn wronskian_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Wronskian expects exactly 2 arguments".into(),
    ));
  }
  let funcs = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Wronskian".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let var_name = match &args[1] {
    Expr::Identifier(s) => s,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Wronskian".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = funcs.len();
  if n == 0 {
    return Ok(Expr::Integer(1));
  }

  // Build matrix: M[i][j] = D^j[funcs[i], x]
  let mut matrix_rows = Vec::with_capacity(n);
  for f in funcs {
    let mut row = Vec::with_capacity(n);
    let mut current = crate::evaluator::evaluate_expr_to_expr(f)?;
    row.push(current.clone());
    for _j in 1..n {
      current = differentiate_expr(&current, var_name)?;
      current = crate::evaluator::evaluate_expr_to_expr(&current)?;
      row.push(current.clone());
    }
    matrix_rows.push(Expr::List(row));
  }

  let matrix = Expr::List(matrix_rows);
  let det = crate::functions::linear_algebra_ast::det_ast(&[matrix])?;
  let result = crate::evaluator::evaluate_expr_to_expr(&det)?;
  // Apply trig identities (e.g. Sin[x]^2 + Cos[x]^2 → 1) to simplify the determinant
  let simplified =
    crate::functions::polynomial_ast::apply_trig_identities(&result);
  let simplified_str = crate::syntax::expr_to_string(&simplified);
  let result_str = crate::syntax::expr_to_string(&result);
  if simplified_str != result_str {
    crate::evaluator::evaluate_expr_to_expr(&simplified)
  } else {
    Ok(result)
  }
}

/// Div[{f1, f2, ...}, {x1, x2, ...}] = divergence = Sum[D[fi, xi]]
pub fn div_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Div expects exactly 2 arguments".into(),
    ));
  }
  let funcs = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Div".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let vars = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Div".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if funcs.len() != vars.len() {
    return Ok(Expr::FunctionCall {
      name: "Div".to_string(),
      args: args.to_vec(),
    });
  }

  let mut terms = Vec::with_capacity(vars.len());
  for (f, var) in funcs.iter().zip(vars.iter()) {
    let var_name = match var {
      Expr::Identifier(s) => s,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Div".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let deriv = differentiate_expr(f, var_name)?;
    let evald = crate::evaluator::evaluate_expr_to_expr(&deriv)?;
    terms.push(evald);
  }

  if terms.len() == 1 {
    return Ok(terms.into_iter().next().unwrap());
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms,
  };
  crate::evaluator::evaluate_expr_to_expr(&sum)
}

/// Laplacian[f, {x1, x2, ...}] = Sum of second partial derivatives
pub fn laplacian_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Laplacian expects exactly 2 arguments".into(),
    ));
  }
  let vars = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Laplacian".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut terms = Vec::with_capacity(vars.len());
  for var in vars {
    let var_name = match var {
      Expr::Identifier(s) => s,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Laplacian".to_string(),
          args: args.to_vec(),
        });
      }
    };
    // Second derivative: D[D[f, x], x]
    let first = differentiate_expr(&args[0], var_name)?;
    let second = differentiate_expr(&first, var_name)?;
    let evald = crate::evaluator::evaluate_expr_to_expr(&second)?;
    terms.push(evald);
  }

  // Sum all terms
  if terms.len() == 1 {
    return Ok(terms.into_iter().next().unwrap());
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms,
  };
  crate::evaluator::evaluate_expr_to_expr(&sum)
}

/// Curl[{f1, f2}, {x1, x2}] - 2D curl (scalar)
/// Curl[{f1, f2, f3}, {x1, x2, x3}] - 3D curl (vector)
pub fn curl_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Curl expects exactly 2 arguments".into(),
    ));
  }
  let field = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Curl".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let vars = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Curl".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if field.len() == 2 && vars.len() == 2 {
    // 2D curl: dF2/dx1 - dF1/dx2
    let var1 = match &vars[0] {
      Expr::Identifier(s) => s,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Curl".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let var2 = match &vars[1] {
      Expr::Identifier(s) => s,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Curl".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let df2_dx1 = differentiate_expr(&field[1], var1)?;
    let df1_dx2 = differentiate_expr(&field[0], var2)?;
    let result = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(df2_dx1),
      right: Box::new(df1_dx2),
    };
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else if field.len() == 3 && vars.len() == 3 {
    // 3D curl: {dF3/dx2 - dF2/dx3, dF1/dx3 - dF3/dx1, dF2/dx1 - dF1/dx2}
    let var_names: Vec<&str> = vars
      .iter()
      .map(|v| match v {
        Expr::Identifier(s) => Ok(s.as_str()),
        _ => Err(InterpreterError::EvaluationError(
          "Curl: variables must be symbols".into(),
        )),
      })
      .collect::<Result<Vec<_>, _>>()?;

    let mut components = Vec::new();
    // Curl component i = dF_{(i+2)%3}/dx_{(i+1)%3} - dF_{(i+1)%3}/dx_{(i+2)%3}
    for i in 0..3 {
      let j = (i + 1) % 3;
      let k = (i + 2) % 3;
      let d1 = differentiate_expr(&field[k], var_names[j])?;
      let d2 = differentiate_expr(&field[j], var_names[k])?;
      let comp = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(d1),
        right: Box::new(d2),
      };
      components.push(crate::evaluator::evaluate_expr_to_expr(&comp)?);
    }
    Ok(Expr::List(components))
  } else {
    Ok(Expr::FunctionCall {
      name: "Curl".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Dt[expr, var] - Total derivative
pub fn dt_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Dt expects exactly 2 arguments".into(),
    ));
  }
  let var = match &args[1] {
    Expr::Identifier(s) => s.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Dt: second argument must be a variable".into(),
      ));
    }
  };
  let result = total_differentiate(&args[0], &var)?;
  Ok(simplify(result))
}

/// Check if an expression is a true constant (number or named constant).
/// Unlike is_constant_wrt, this does NOT consider other variables as constant.
fn is_true_constant(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_))
}

/// Total differentiation: like differentiate but treats all symbols as potentially
/// dependent on the differentiation variable (returns Dt[y, x] instead of 0).
fn total_differentiate(
  expr: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  use crate::syntax::BinaryOperator::*;

  match expr {
    // Constants
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => {
      Ok(Expr::Integer(0))
    }

    // Variable
    Expr::Identifier(name) => {
      if name == var {
        Ok(Expr::Integer(1))
      } else {
        // Other variables may depend on var: return Dt[y, x]
        Ok(Expr::FunctionCall {
          name: "Dt".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())],
        })
      }
    }

    // Binary operations
    Expr::BinaryOp { op, left, right } => match op {
      Plus => {
        let da = total_differentiate(left, var)?;
        let db = total_differentiate(right, var)?;
        Ok(simplify(Expr::BinaryOp {
          op: Plus,
          left: Box::new(da),
          right: Box::new(db),
        }))
      }
      Minus => {
        let da = total_differentiate(left, var)?;
        let db = total_differentiate(right, var)?;
        Ok(simplify(Expr::BinaryOp {
          op: Minus,
          left: Box::new(da),
          right: Box::new(db),
        }))
      }
      Times => {
        let da = total_differentiate(left, var)?;
        let db = total_differentiate(right, var)?;
        Ok(simplify(Expr::BinaryOp {
          op: Plus,
          left: Box::new(Expr::BinaryOp {
            op: Times,
            left: Box::new(da),
            right: right.clone(),
          }),
          right: Box::new(Expr::BinaryOp {
            op: Times,
            left: left.clone(),
            right: Box::new(db),
          }),
        }))
      }
      Divide => {
        let da = total_differentiate(left, var)?;
        let db = total_differentiate(right, var)?;
        Ok(simplify(Expr::BinaryOp {
          op: Divide,
          left: Box::new(Expr::BinaryOp {
            op: Minus,
            left: Box::new(Expr::BinaryOp {
              op: Times,
              left: Box::new(da),
              right: right.clone(),
            }),
            right: Box::new(Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(db),
            }),
          }),
          right: Box::new(Expr::BinaryOp {
            op: Power,
            left: right.clone(),
            right: Box::new(Expr::Integer(2)),
          }),
        }))
      }
      Power => {
        if is_true_constant(right) || is_constant_wrt(right, var) {
          // f(x)^n: n * f(x)^(n-1) * Dt[f, x]
          let df = total_differentiate(left, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(Expr::BinaryOp {
              op: Times,
              left: right.clone(),
              right: Box::new(Expr::BinaryOp {
                op: Power,
                left: left.clone(),
                right: Box::new(Expr::BinaryOp {
                  op: Plus,
                  left: Box::new(Expr::Integer(-1)),
                  right: right.clone(),
                }),
              }),
            }),
            right: Box::new(df),
          }))
        } else if matches!(left.as_ref(), Expr::Constant(c) if c == "E") {
          // E^g: E^g * Dt[g, x]
          let dg = total_differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(expr.clone()),
            right: Box::new(dg),
          }))
        } else {
          // General f^g: return unevaluated
          Ok(Expr::FunctionCall {
            name: "Dt".to_string(),
            args: vec![expr.clone(), Expr::Identifier(var.to_string())],
          })
        }
      }
      _ => Ok(Expr::FunctionCall {
        name: "Dt".to_string(),
        args: vec![expr.clone(), Expr::Identifier(var.to_string())],
      }),
    },

    // Unary minus
    Expr::UnaryOp { op, operand } => {
      use crate::syntax::UnaryOperator;
      if matches!(op, UnaryOperator::Minus) {
        let d = total_differentiate(operand, var)?;
        Ok(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(d),
        })
      } else {
        Ok(Expr::FunctionCall {
          name: "Dt".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())],
        })
      }
    }

    // Known function calls - chain rule
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Sin" if args.len() == 1 => {
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(Expr::FunctionCall {
              name: "Cos".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Cos" if args.len() == 1 => {
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::FunctionCall {
                name: "Sin".to_string(),
                args: args.clone(),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Tan" if args.len() == 1 => {
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(Expr::BinaryOp {
              op: Power,
              left: Box::new(Expr::FunctionCall {
                name: "Sec".to_string(),
                args: args.clone(),
              }),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(df),
          }))
        }
        "Log" if args.len() == 1 => {
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(Expr::BinaryOp {
              op: Power,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(-1)),
            }),
            right: Box::new(df),
          }))
        }
        "Exp" if args.len() == 1 => {
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(expr.clone()),
            right: Box::new(df),
          }))
        }
        "Sqrt" if args.len() == 1 => {
          // d/dx[sqrt(f)] = f'/(2*sqrt(f))
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Divide,
            left: Box::new(df),
            right: Box::new(Expr::BinaryOp {
              op: Times,
              left: Box::new(Expr::Integer(2)),
              right: Box::new(expr.clone()),
            }),
          }))
        }
        "Plus" => {
          // Sum rule
          let mut terms = Vec::new();
          for arg in args {
            terms.push(total_differentiate(arg, var)?);
          }
          if terms.is_empty() {
            return Ok(Expr::Integer(0));
          }
          let mut result = terms.remove(0);
          for t in terms {
            result = Expr::BinaryOp {
              op: Plus,
              left: Box::new(result),
              right: Box::new(t),
            };
          }
          Ok(simplify(result))
        }
        "Times" if args.len() >= 2 => {
          // Product rule for n factors
          let mut sum_terms = Vec::new();
          for i in 0..args.len() {
            let di = total_differentiate(&args[i], var)?;
            let mut product = di;
            for (j, arg) in args.iter().enumerate() {
              if j != i {
                product = Expr::BinaryOp {
                  op: Times,
                  left: Box::new(product),
                  right: Box::new(arg.clone()),
                };
              }
            }
            sum_terms.push(product);
          }
          let mut result = sum_terms.remove(0);
          for t in sum_terms {
            result = Expr::BinaryOp {
              op: Plus,
              left: Box::new(result),
              right: Box::new(t),
            };
          }
          Ok(simplify(result))
        }
        _ => {
          // Unknown function: return unevaluated
          Ok(Expr::FunctionCall {
            name: "Dt".to_string(),
            args: vec![expr.clone(), Expr::Identifier(var.to_string())],
          })
        }
      }
    }

    _ => Ok(Expr::FunctionCall {
      name: "Dt".to_string(),
      args: vec![expr.clone(), Expr::Identifier(var.to_string())],
    }),
  }
}

/// AsymptoticSolve[eqn, x -> x0, n] — find asymptotic solutions of eqn near x = x0 to order n.
///
/// Uses Series expansion and iterative coefficient solving.
pub fn asymptotic_solve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "AsymptoticSolve".to_string(),
      args: args.to_vec(),
    });
  }

  // Parse the equation: eqn can be f == 0 or just f (treated as f == 0)
  let f_expr = match &args[0] {
    Expr::Comparison {
      operands,
      operators,
    } if operators.len() == 1
      && operators[0] == crate::syntax::ComparisonOp::Equal
      && operands.len() == 2 =>
    {
      // f == g becomes f - g
      if matches!(&operands[1], Expr::Integer(0)) {
        operands[0].clone()
      } else {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(operands[0].clone()),
          right: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(operands[1].clone()),
          }),
        }
      }
    }
    // Also handle FunctionCall "Equal"
    Expr::FunctionCall {
      name,
      args: eq_args,
    } if name == "Equal" && eq_args.len() == 2 => {
      if matches!(&eq_args[1], Expr::Integer(0)) {
        eq_args[0].clone()
      } else {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(eq_args[0].clone()),
          right: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(eq_args[1].clone()),
          }),
        }
      }
    }
    other => other.clone(),
  };

  // Parse x -> x0
  let (var_name, x0) = match &args[1] {
    Expr::Rule {
      pattern,
      replacement,
    } => match pattern.as_ref() {
      Expr::Identifier(name) => (name.clone(), *replacement.clone()),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "AsymptoticSolve".to_string(),
          args: args.to_vec(),
        });
      }
    },
    _ => {
      return Ok(Expr::FunctionCall {
        name: "AsymptoticSolve".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Parse order: must be a list {param, val, n} or a rule param -> val.
  // A plain integer is not valid (Wolfram returns unevaluated).
  let order = match &args[2] {
    Expr::List(items) if items.len() == 3 => match &items[2] {
      Expr::Integer(n) => *n,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "AsymptoticSolve".to_string(),
          args: args.to_vec(),
        });
      }
    },
    Expr::Integer(n) => *n,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "AsymptoticSolve".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // When 3rd arg is a plain integer (not a list/rule with perturbation param),
  // Wolfram returns unevaluated.
  if matches!(&args[2], Expr::Integer(_)) {
    return Ok(Expr::FunctionCall {
      name: "AsymptoticSolve".to_string(),
      args: args.to_vec(),
    });
  }

  if order < 1 {
    return Ok(Expr::List(vec![]));
  }

  // Compute the series expansion of f around x0
  let series_result = series_ast(&[
    f_expr.clone(),
    Expr::List(vec![
      Expr::Identifier(var_name.clone()),
      x0.clone(),
      Expr::Integer(order),
    ]),
  ])?;

  // Extract SeriesData coefficients
  let (coeffs, _min_power) = match extract_series_coefficients(&series_result) {
    Some(c) => c,
    None => {
      return Ok(Expr::FunctionCall {
        name: "AsymptoticSolve".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if coeffs.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // Use the series to find solutions via InverseSeries approach:
  // f(x) = c0 + c1*(x-x0) + c2*(x-x0)^2 + ... = 0
  // If c0 == 0, x = x0 is already a solution. Look at the structure.
  // If c0 != 0, we need to solve for x-x0.

  // Use Newton-like iteration on the truncated polynomial
  // Build the polynomial: sum of c_k * t^(k + min_power) where t = x - x0
  // and solve this polynomial for t using Solve

  // Build the polynomial expression in a temporary variable
  let t_var = Expr::Identifier("AsymptoticSolve$t".to_string());

  let mut poly_terms: Vec<Expr> = Vec::new();
  for (i, coeff) in coeffs.iter().enumerate() {
    if matches!(coeff, Expr::Integer(0)) {
      continue;
    }
    let power = _min_power + i as i128;
    let term = if power == 0 {
      coeff.clone()
    } else if power == 1 {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(coeff.clone()),
        right: Box::new(t_var.clone()),
      }
    } else {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(coeff.clone()),
        right: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(t_var.clone()),
          right: Box::new(Expr::Integer(power)),
        }),
      }
    };
    poly_terms.push(term);
  }

  if poly_terms.is_empty() {
    // f is identically zero to this order — any x works
    return Ok(Expr::List(vec![Expr::List(vec![Expr::Rule {
      pattern: Box::new(Expr::Identifier(var_name)),
      replacement: Box::new(x0),
    }])]));
  }

  let poly_expr = if poly_terms.len() == 1 {
    poly_terms.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: poly_terms,
    }
  };

  // Solve poly_expr == 0 for t
  use crate::evaluator::evaluate_expr_to_expr;

  let solve_expr = Expr::FunctionCall {
    name: "Solve".to_string(),
    args: vec![
      Expr::Comparison {
        operands: vec![poly_expr, Expr::Integer(0)],
        operators: vec![crate::syntax::ComparisonOp::Equal],
      },
      Expr::Identifier("AsymptoticSolve$t".to_string()),
    ],
  };

  let solutions = evaluate_expr_to_expr(&solve_expr)?;

  // Convert solutions from t -> val to x -> x0 + val
  match &solutions {
    Expr::List(sol_list) => {
      let mut result = Vec::new();
      for sol in sol_list {
        if let Expr::List(rules) = sol {
          let mut new_rules = Vec::new();
          for rule in rules {
            if let Expr::Rule { replacement, .. } = rule {
              // x = x0 + t
              let x_val = if matches!(x0, Expr::Integer(0)) {
                *replacement.clone()
              } else {
                Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Plus,
                  left: Box::new(x0.clone()),
                  right: replacement.clone(),
                }
              };
              let simplified = evaluate_expr_to_expr(&x_val)?;
              new_rules.push(Expr::Rule {
                pattern: Box::new(Expr::Identifier(var_name.clone())),
                replacement: Box::new(simplified),
              });
            }
          }
          if !new_rules.is_empty() {
            result.push(Expr::List(new_rules));
          }
        }
      }
      Ok(Expr::List(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "AsymptoticSolve".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Extract coefficients from a SeriesData expression.
/// Returns (coefficients, min_power).
fn extract_series_coefficients(expr: &Expr) -> Option<(Vec<Expr>, i128)> {
  match expr {
    Expr::FunctionCall { name, args } if name == "SeriesData" => {
      // SeriesData[var, x0, coeffs_list, nmin, nmax, den]
      if args.len() >= 4 {
        let coeffs = match &args[2] {
          Expr::List(items) => items.clone(),
          _ => return None,
        };
        let nmin = match &args[3] {
          Expr::Integer(n) => *n,
          _ => return None,
        };
        let den = if args.len() >= 6 {
          match &args[5] {
            Expr::Integer(d) => *d,
            _ => 1,
          }
        } else {
          1
        };
        // The actual power of the i-th coefficient is (nmin + i) / den
        // For simplicity, handle den == 1
        if den == 1 { Some((coeffs, nmin)) } else { None }
      } else {
        None
      }
    }
    _ => None,
  }
}

/// DiscreteConvolve[f, g, n, m] — discrete convolution.
///
/// Computes Sum[f /. n -> k, g /. m -> (m - k), {k, -Infinity, Infinity}]
/// by building a Sum expression and evaluating it.
pub fn discrete_convolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 4 {
    return Ok(Expr::FunctionCall {
      name: "DiscreteConvolve".to_string(),
      args: args.to_vec(),
    });
  }

  let f_expr = &args[0];
  let g_expr = &args[1];
  let n_var = match &args[2] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DiscreteConvolve".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let m_var = match &args[3] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DiscreteConvolve".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Build: Sum[(f /. n -> k$dc) * (g /. m -> (m - k$dc)), {k$dc, -Infinity, Infinity}]
  let k_var = "DiscreteConvolve$k";
  let k_expr = Expr::Identifier(k_var.to_string());

  // f /. n -> k
  let f_sub = crate::syntax::substitute_variable(f_expr, &n_var, &k_expr);

  // g /. m -> (m - k)  — but the result variable is m (output var),
  // which is the same as the output. In WL convention, the result is in terms of m.
  let m_minus_k = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left: Box::new(Expr::Identifier(m_var.clone())),
    right: Box::new(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(k_expr.clone()),
    }),
  };
  let g_sub = crate::syntax::substitute_variable(g_expr, &m_var, &m_minus_k);

  // Build the product
  let product = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(f_sub),
    right: Box::new(g_sub),
  };

  // Build Sum[product, {k, -Infinity, Infinity}]
  let sum_expr = Expr::FunctionCall {
    name: "Sum".to_string(),
    args: vec![
      product,
      Expr::List(vec![
        k_expr,
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(-1),
            Expr::Identifier("Infinity".to_string()),
          ],
        },
        Expr::Identifier("Infinity".to_string()),
      ]),
    ],
  };

  crate::evaluator::evaluate_expr_to_expr(&sum_expr)
}

/// FrenetSerretSystem[{f1, f2, ...}, t] - Frenet-Serret system for a parametric curve
/// Returns {{curvatures...}, {tangent, normal, ...}} where:
/// - 2D: {{κ}, {T, N}}
/// - 3D: {{κ, τ}, {T, N, B}}
pub fn frenet_serret_system_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "FrenetSerretSystem expects exactly 2 arguments".into(),
    ));
  }

  // If the first argument is a scalar function f[t], treat it as the 2D curve {t, f[t]}
  let owned_components;
  let components = match &args[0] {
    Expr::List(items) => items,
    _ => {
      owned_components = vec![args[1].clone(), args[0].clone()];
      &owned_components
    }
  };

  let var_name = match &args[1] {
    Expr::Identifier(s) => s.as_str(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "FrenetSerretSystem: second argument must be a variable".into(),
      ));
    }
  };

  let n = components.len();
  if !(2..=3).contains(&n) {
    return Ok(Expr::FunctionCall {
      name: "FrenetSerretSystem".to_string(),
      args: args.to_vec(),
    });
  }

  let eval = |e: &Expr| -> Result<Expr, InterpreterError> {
    crate::evaluator::evaluate_expr_to_expr(e)
  };

  // Compute first and second derivatives of each component
  let mut r1 = Vec::with_capacity(n); // r'
  let mut r2 = Vec::with_capacity(n); // r''
  for c in components {
    let d1 = differentiate_expr(c, var_name)?;
    let d1 = eval(&d1)?;
    let d2 = differentiate_expr(&d1, var_name)?;
    let d2 = eval(&d2)?;
    r1.push(d1);
    r2.push(d2);
  }

  // speed_sq = sum of r'[i]^2
  let speed_sq = sum_of_squares(&r1);
  let speed_sq = eval(&speed_sq)?;

  // speed = Sqrt[speed_sq]
  let speed = make_sqrt(speed_sq.clone());

  // T = r' / speed (unit tangent)
  let tangent: Vec<Expr> = r1
    .iter()
    .map(|c| {
      eval(&Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(c.clone()),
        right: Box::new(speed.clone()),
      })
    })
    .collect::<Result<Vec<_>, _>>()?;

  if n == 2 {
    // 2D case
    // κ = (x'*y'' - y'*x'') / (speed_sq)^(3/2)
    let numerator = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(r1[0].clone()),
        right: Box::new(r2[1].clone()),
      }),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(r1[1].clone()),
        right: Box::new(r2[0].clone()),
      }),
    };
    let denom = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(speed_sq),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Integer(3)),
        right: Box::new(Expr::Integer(2)),
      }),
    };
    let kappa = eval(&Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(denom),
    })?;

    // N = {-T2, T1} (rotate tangent 90° counterclockwise)
    let normal = vec![
      eval(&Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(tangent[1].clone()),
      })?,
      tangent[0].clone(),
    ];

    Ok(Expr::List(vec![
      Expr::List(vec![kappa]),
      Expr::List(vec![Expr::List(tangent), Expr::List(normal)]),
    ]))
  } else {
    // 3D case
    // r''' for torsion
    let mut r3 = Vec::with_capacity(3);
    for c in &r2 {
      let d3 = differentiate_expr(c, var_name)?;
      r3.push(eval(&d3)?);
    }

    // cross = r' × r''
    let cross = cross_product_3d(&r1, &r2);
    let cross: Vec<Expr> = cross
      .into_iter()
      .map(|c| eval(&c))
      .collect::<Result<Vec<_>, _>>()?;

    // norm_cross_sq = ||cross||^2
    let norm_cross_sq = sum_of_squares(&cross);
    let norm_cross_sq = eval(&norm_cross_sq)?;

    // Check if curvature is zero (straight line case)
    let is_zero_curvature = matches!(&norm_cross_sq, Expr::Integer(0));
    if is_zero_curvature {
      let zero_vec =
        Expr::List(vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)]);
      return Ok(Expr::List(vec![
        Expr::List(vec![Expr::Integer(0), Expr::Integer(0)]),
        Expr::List(vec![Expr::List(tangent), zero_vec.clone(), zero_vec]),
      ]));
    }

    // norm_cross = ||cross||
    let norm_cross = make_sqrt(norm_cross_sq.clone());

    // κ = ||cross|| / ||r'||^3 = norm_cross / speed_sq^(3/2)
    let speed_cubed = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(speed_sq),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Integer(3)),
        right: Box::new(Expr::Integer(2)),
      }),
    };
    let kappa = eval(&Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(norm_cross.clone()),
      right: Box::new(speed_cubed),
    })?;

    // τ = (r' × r'') · r''' / ||r' × r''||^2
    let dot_cross_r3 = dot_product(&cross, &r3);
    let dot_cross_r3 = eval(&dot_cross_r3)?;
    let tau = eval(&Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(dot_cross_r3),
      right: Box::new(norm_cross_sq),
    })?;

    // B = (r' × r'') / ||r' × r''|| (unit binormal)
    let binormal: Vec<Expr> = cross
      .iter()
      .map(|c| {
        eval(&Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(c.clone()),
          right: Box::new(norm_cross.clone()),
        })
      })
      .collect::<Result<Vec<_>, _>>()?;

    // N = B × T (unit normal)
    let normal_cross = cross_product_3d(&binormal, &tangent);
    let normal: Vec<Expr> = normal_cross
      .into_iter()
      .map(|c| eval(&c))
      .collect::<Result<Vec<_>, _>>()?;

    Ok(Expr::List(vec![
      Expr::List(vec![kappa, tau]),
      Expr::List(vec![
        Expr::List(tangent),
        Expr::List(normal),
        Expr::List(binormal),
      ]),
    ]))
  }
}

/// Helper: compute sum of squares of expressions
fn sum_of_squares(items: &[Expr]) -> Expr {
  let squared: Vec<Expr> = items
    .iter()
    .map(|e| Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(e.clone()),
      right: Box::new(Expr::Integer(2)),
    })
    .collect();
  if squared.len() == 1 {
    squared.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: squared,
    }
  }
}

/// Helper: compute cross product of two 3D vectors
fn cross_product_3d(a: &[Expr], b: &[Expr]) -> Vec<Expr> {
  vec![
    // a[1]*b[2] - a[2]*b[1]
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(a[1].clone()),
        right: Box::new(b[2].clone()),
      }),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(a[2].clone()),
        right: Box::new(b[1].clone()),
      }),
    },
    // a[2]*b[0] - a[0]*b[2]
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(a[2].clone()),
        right: Box::new(b[0].clone()),
      }),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(a[0].clone()),
        right: Box::new(b[2].clone()),
      }),
    },
    // a[0]*b[1] - a[1]*b[0]
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(a[0].clone()),
        right: Box::new(b[1].clone()),
      }),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(a[1].clone()),
        right: Box::new(b[0].clone()),
      }),
    },
  ]
}

/// ArcCurvature[curve, t] - curvature of a parametric curve
/// For 2D: κ = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)
/// For 3D: κ = ||r' × r''|| / ||r'||^3
/// For scalar f[t]: treated as the 2D curve {t, f[t]}
/// AsymptoticIntegrate[f, x, {x, x0, n}] - series expansion of the antiderivative
/// Computes the antiderivative of f, then expands as a series to order n.
pub fn asymptotic_integrate_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "AsymptoticIntegrate".to_string(),
      args: args.to_vec(),
    });
  }

  let f = &args[0];
  let var = &args[1];
  let spec = &args[2];

  // spec must be {x, x0, n}
  if !matches!(spec, Expr::List(items) if items.len() == 3) {
    return Ok(Expr::FunctionCall {
      name: "AsymptoticIntegrate".to_string(),
      args: args.to_vec(),
    });
  }

  // Try to compute the exact antiderivative first
  let antideriv_result = integrate_ast(&[f.clone(), var.clone()]);

  if let Ok(ref antideriv) = antideriv_result {
    // Check if integrate succeeded (not unevaluated)
    let is_unevaluated = matches!(antideriv, Expr::FunctionCall { name, .. } if name == "Integrate");
    if !is_unevaluated {
      // Expand the antiderivative as a series
      let series_result = series_ast(&[antideriv.clone(), spec.clone()])?;
      // Convert SeriesData to Normal polynomial
      let normal = Expr::FunctionCall {
        name: "Normal".to_string(),
        args: vec![series_result],
      };
      return crate::evaluator::evaluate_expr_to_expr(&normal);
    }
  }

  // Fallback: integrate the series term by term
  let series_result = series_ast(&[f.clone(), spec.clone()])?;

  match &series_result {
    Expr::FunctionCall { name, args: sargs }
      if name == "SeriesData" && sargs.len() >= 6 =>
    {
      if let Expr::List(coeffs) = &sargs[2] {
        let x0 = &sargs[1];
        let nmin =
          crate::functions::math_ast::expr_to_i128(&sargs[3]).unwrap_or(0);
        let nmax =
          crate::functions::math_ast::expr_to_i128(&sargs[4]).unwrap_or(0);
        let den =
          crate::functions::math_ast::expr_to_i128(&sargs[5]).unwrap_or(1);

        let var_name = match var {
          Expr::Identifier(s) => s.clone(),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "AsymptoticIntegrate".to_string(),
              args: args.to_vec(),
            });
          }
        };

        // Integrate each series term: coeff * (x-x0)^(k/den) -> coeff/(k/den+1) * (x-x0)^(k/den+1)
        let mut terms = Vec::new();
        for (i, coeff) in coeffs.iter().enumerate() {
          if matches!(coeff, Expr::Integer(0)) {
            continue;
          }
          let power_num = nmin + i as i128;
          // After integration: new power = (power_num + den) / den
          let new_power_num = power_num + den;
          // Skip if the integrated power exceeds the requested order n
          // nmax is the truncation order of the original series
          if new_power_num > nmax {
            continue;
          }

          let base = if matches!(x0, Expr::Integer(0)) {
            Expr::Identifier(var_name.clone())
          } else {
            Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Minus,
              left: Box::new(Expr::Identifier(var_name.clone())),
              right: Box::new(x0.clone()),
            }
          };

          // coeff / (new_power_num / den) * (x - x0)^(new_power_num/den)
          // = coeff * den / new_power_num * (x - x0)^(new_power_num/den)
          let factor = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(coeff.clone()),
            right: Box::new(crate::functions::math_ast::make_rational(
              den,
              new_power_num,
            )),
          };
          let power_expr = if new_power_num == den {
            base.clone()
          } else {
            Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(base),
              right: Box::new(crate::functions::math_ast::make_rational(
                new_power_num,
                den,
              )),
            }
          };
          let term = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(factor),
            right: Box::new(power_expr),
          };
          terms.push(crate::evaluator::evaluate_expr_to_expr(&term)?);
        }

        if terms.is_empty() {
          return Ok(Expr::Integer(0));
        }

        let result = if terms.len() == 1 {
          terms.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: terms,
          }
        };

        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
    }
    _ => {}
  }

  Ok(Expr::FunctionCall {
    name: "AsymptoticIntegrate".to_string(),
    args: args.to_vec(),
  })
}

pub fn arc_curvature_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Reuse FrenetSerretSystem and extract the curvature
  let fss = frenet_serret_system_ast(args)?;
  // FrenetSerretSystem returns {{κ, ...}, {T, N, ...}}
  // Extract first element of first list
  if let Expr::List(outer) = &fss
    && let Some(Expr::List(curvatures)) = outer.first()
    && let Some(kappa) = curvatures.first()
  {
    return Ok(kappa.clone());
  }
  // Fallback: return unevaluated
  Ok(Expr::FunctionCall {
    name: "ArcCurvature".to_string(),
    args: args.to_vec(),
  })
}

/// Check if an expression contains a variable by name.
fn expr_has_var(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(name) => name == var,
    Expr::Integer(_) | Expr::Real(_) | Expr::String(_) | Expr::Constant(_) => {
      false
    }
    Expr::BinaryOp { left, right, .. } => {
      expr_has_var(left, var) || expr_has_var(right, var)
    }
    Expr::UnaryOp { operand, .. } => expr_has_var(operand, var),
    Expr::FunctionCall { name: n, args } => {
      if n == "Rational" {
        return false;
      }
      args.iter().any(|a| expr_has_var(a, var))
    }
    Expr::List(items) => items.iter().any(|a| expr_has_var(a, var)),
    _ => false,
  }
}

/// Try to compute DifferenceDelta for exponential expressions a^f(x).
/// Returns Some(simplified_expr) if the expression is Power[base, exponent]
/// where base doesn't depend on the variable.
/// Δ[a^f(x), {x, h}] = a^f(x) * (a^(f(x+h)-f(x)) - 1)
fn try_exponential_delta(expr: &Expr, var: &str, step: &Expr) -> Option<Expr> {
  let (base, exponent) = match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => (left.as_ref(), right.as_ref()),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    _ => return None,
  };

  // Base must not contain the variable
  if expr_has_var(base, var) {
    return None;
  }

  // Exponent must contain the variable
  if !expr_has_var(exponent, var) {
    return None;
  }

  // Compute f(x+h) - f(x) for the exponent
  let x_plus_h = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left: Box::new(Expr::Identifier(var.to_string())),
    right: Box::new(step.clone()),
  };
  let shifted_exp =
    crate::syntax::substitute_variable(exponent, var, &x_plus_h);

  // delta_exp = Expand[shifted_exp - exponent]
  let delta_exp = Expr::FunctionCall {
    name: "Expand".to_string(),
    args: vec![Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(shifted_exp),
      right: Box::new(exponent.clone()),
    }],
  };

  // Result: base^exponent * (base^delta_exp - 1)
  let result = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(expr.clone()),
    right: Box::new(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(base.clone()),
        right: Box::new(delta_exp),
      }),
      right: Box::new(Expr::Integer(1)),
    }),
  };

  Some(result)
}

/// Try to compute DifferenceDelta for Sin/Cos using sum-to-product identities.
/// Δ[Sin[f(x)]] = 2*Sin[Δf/2]*Sin[Pi/2 + (f(x+h)+f(x))/2]
/// Δ[Cos[f(x)]] = -2*Sin[(f(x+h)+f(x))/2]*Sin[Δf/2]
/// where Δf = f(x+h) - f(x).
fn try_trig_delta(expr: &Expr, var: &str, step: &Expr) -> Option<Expr> {
  let (fn_name, arg) = match expr {
    Expr::FunctionCall { name, args }
      if (name == "Sin" || name == "Cos") && args.len() == 1 =>
    {
      (name.as_str(), &args[0])
    }
    _ => return None,
  };

  // Argument must contain the variable
  if !expr_has_var(arg, var) {
    return None;
  }

  let x_plus_h = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left: Box::new(Expr::Identifier(var.to_string())),
    right: Box::new(step.clone()),
  };
  let shifted_arg = crate::syntax::substitute_variable(arg, var, &x_plus_h);

  // Compute half_delta = (shifted_arg - arg) / 2 via evaluation
  let half_delta_raw = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(shifted_arg.clone()),
        right: Box::new(arg.clone()),
      }],
    }),
    right: Box::new(Expr::Integer(2)),
  };
  let half_delta =
    crate::evaluator::evaluate_expr_to_expr(&half_delta_raw).ok()?;

  // Build second Sin argument:
  // For Sin: arg + (step + Pi)/2  (= arg + half_delta + Pi/2, combined fraction)
  // For Cos: arg + (step - Pi)/2
  // We construct (step ± Pi)/2 as a single fraction, then add arg.
  let pi_term = if fn_name == "Sin" {
    Expr::Constant("Pi".to_string())
  } else {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(Expr::Constant("Pi".to_string())),
    }
  };
  // Build (2*half_delta + Pi) / 2  for the constant part, but use direct
  // (step + Pi) / 2 to get a cleaner fraction when possible.
  // General form: half_delta + Pi/2 + arg, combined as arg + (2*half_delta ± Pi)/2
  let const_part = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(half_delta.clone()),
      }),
      right: Box::new(pi_term),
    }),
    right: Box::new(Expr::Integer(2)),
  };
  let second_arg_expr = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left: Box::new(const_part),
    right: Box::new(arg.clone()),
  };

  let coeff = if fn_name == "Sin" {
    Expr::Integer(2)
  } else {
    Expr::Integer(-2)
  };

  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      coeff,
      Expr::FunctionCall {
        name: "Sin".to_string(),
        args: vec![half_delta],
      },
      Expr::FunctionCall {
        name: "Sin".to_string(),
        args: vec![second_arg_expr],
      },
    ],
  };

  Some(result)
}

/// DifferenceDelta[f, x] = f(x+1) - f(x)
/// DifferenceDelta[f, {x, n}] = n-th order forward difference with step 1
/// DifferenceDelta[f, {x, n, h}] = n-th order forward difference with step h
pub fn difference_delta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Ok(Expr::FunctionCall {
      name: "DifferenceDelta".to_string(),
      args: args.to_vec(),
    });
  }

  let expr = &args[0];

  // Parse second argument: x, {x, n}, or {x, n, h}
  let (var_name, order, step) = if args.len() == 1 {
    return Ok(Expr::FunctionCall {
      name: "DifferenceDelta".to_string(),
      args: args.to_vec(),
    });
  } else {
    match &args[1] {
      Expr::Identifier(name) => (name.clone(), 1usize, Expr::Integer(1)),
      Expr::List(items) if !items.is_empty() => {
        let var = match &items[0] {
          Expr::Identifier(name) => name.clone(),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "DifferenceDelta".to_string(),
              args: args.to_vec(),
            });
          }
        };
        let n = if items.len() >= 2 {
          match crate::functions::math_ast::expr_to_i128(&items[1]) {
            Some(n) if n >= 0 => n as usize,
            _ => {
              return Ok(Expr::FunctionCall {
                name: "DifferenceDelta".to_string(),
                args: args.to_vec(),
              });
            }
          }
        } else {
          1
        };
        let h = if items.len() >= 3 {
          items[2].clone()
        } else {
          Expr::Integer(1)
        };
        (var, n, h)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "DifferenceDelta".to_string(),
          args: args.to_vec(),
        });
      }
    }
  };

  if order == 0 {
    return crate::evaluator::evaluate_expr_to_expr(expr);
  }

  // Apply forward difference operator n times
  let mut current = expr.clone();
  for _ in 0..order {
    // Special case: Power[base, exponent] where base is independent of the variable.
    // Δ[base^f(x), {x, h}] = base^f(x) * (base^Δ[f(x)] - 1), which for f(x)=x
    // gives base^x * (base^h - 1). This avoids unsimplified a^(x+h) - a^x forms.
    if let Some(result) = try_exponential_delta(&current, &var_name, &step) {
      current = crate::evaluator::evaluate_expr_to_expr(&result)?;
      continue;
    }

    // Special case: Sin[f(x)] or Cos[f(x)] — use sum-to-product identities.
    // Δ[Sin[f(x)]] = 2*Cos[(f(x+h)+f(x))/2]*Sin[(f(x+h)-f(x))/2]
    // Δ[Cos[f(x)]] = -2*Sin[(f(x+h)+f(x))/2]*Sin[(f(x+h)-f(x))/2]
    // Then Cos[θ] → Sin[Pi/2 + θ] and -Sin[θ] → Sin[-Pi/2 + θ] to match Wolfram.
    if let Some(result) = try_trig_delta(&current, &var_name, &step) {
      current = crate::evaluator::evaluate_expr_to_expr(&result)?;
      continue;
    }

    // f(x + h)
    let x_plus_h = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(Expr::Identifier(var_name.clone())),
      right: Box::new(step.clone()),
    };
    let shifted =
      crate::syntax::substitute_variable(&current, &var_name, &x_plus_h);
    // f(x + h) - f(x)
    let diff = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(shifted),
      right: Box::new(current.clone()),
    };
    let expanded = Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![diff],
    };
    current = crate::evaluator::evaluate_expr_to_expr(&expanded)?;
  }

  Ok(current)
}

/// DifferenceQuotient[f, {x, h}] = (f(x+h) - f(x)) / h
/// DifferenceQuotient[f, x] = f(x+1) - f(x) (i.e. DifferenceDelta with step 1)
/// DifferenceQuotient[f, {x, h}, n] = n-th order difference quotient
pub fn difference_quotient_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "DifferenceQuotient".to_string(),
      args: args.to_vec(),
    });
  }

  let expr = &args[0];

  // Parse second argument: only {x, h} form is supported (bare x returns unevaluated)
  let (var_name, step) = if args.len() >= 2 {
    match &args[1] {
      Expr::List(items) if items.len() == 2 => {
        let var = match &items[0] {
          Expr::Identifier(name) => name.clone(),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "DifferenceQuotient".to_string(),
              args: args.to_vec(),
            });
          }
        };
        (var, items[1].clone())
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "DifferenceQuotient".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    return Ok(Expr::FunctionCall {
      name: "DifferenceQuotient".to_string(),
      args: args.to_vec(),
    });
  };

  let order = if args.len() == 3 {
    match crate::functions::math_ast::expr_to_i128(&args[2]) {
      Some(n) if n >= 0 => n as usize,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "DifferenceQuotient".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    1
  };

  if order == 0 {
    return crate::evaluator::evaluate_expr_to_expr(expr);
  }

  // Apply difference quotient n times: (DifferenceDelta[f, {x, 1, h}]) / h^n
  // Build DifferenceDelta args
  let delta_args = vec![
    expr.clone(),
    Expr::List(vec![
      Expr::Identifier(var_name.clone()),
      Expr::Integer(order as i128),
      step.clone(),
    ]),
  ];
  let delta_result = difference_delta_ast(&delta_args)?;

  // Divide by h^n
  let divisor = if order == 1 {
    step.clone()
  } else {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(step.clone()),
      right: Box::new(Expr::Integer(order as i128)),
    }
  };

  let quotient = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(delta_result),
    right: Box::new(divisor),
  };
  let result = Expr::FunctionCall {
    name: "Cancel".to_string(),
    args: vec![quotient],
  };

  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Helper: compute dot product of two vectors
fn dot_product(a: &[Expr], b: &[Expr]) -> Expr {
  let terms: Vec<Expr> = a
    .iter()
    .zip(b.iter())
    .map(|(ai, bi)| Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(ai.clone()),
      right: Box::new(bi.clone()),
    })
    .collect();
  if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms,
    }
  }
}
