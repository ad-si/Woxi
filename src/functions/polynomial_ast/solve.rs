#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

use crate::functions::calculus_ast::simplify;

// ─── Solve ──────────────────────────────────────────────────────────

/// Roots[equation, var] — find roots of a polynomial equation.
///
/// Returns solutions as `x == val1 || x == val2 || ...`
pub fn roots_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Roots expects exactly 2 arguments".into(),
    ));
  }

  let var = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Roots".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Use Solve to find solutions
  let solutions = solve_ast(args)?;

  // Convert {{var -> val1}, {var -> val2}, ...} to x == val1 || x == val2 || ...
  match &solutions {
    Expr::List(outer) => {
      let mut conditions: Vec<Expr> = Vec::new();
      for item in outer {
        if let Expr::List(inner) = item {
          if inner.is_empty() {
            // {{}} means all values (identity)
            return Ok(Expr::Identifier("True".to_string()));
          }
          for rule in inner {
            if let Expr::Rule { replacement, .. } = rule {
              conditions.push(Expr::Comparison {
                operands: vec![
                  Expr::Identifier(var.clone()),
                  *replacement.clone(),
                ],
                operators: vec![crate::syntax::ComparisonOp::Equal],
              });
            }
          }
        }
      }
      // Deduplicate conditions
      conditions.dedup_by(|a, b| expr_to_string(a) == expr_to_string(b));
      if conditions.is_empty() {
        Ok(Expr::Identifier("False".to_string()))
      } else if conditions.len() == 1 {
        Ok(conditions.into_iter().next().unwrap())
      } else {
        Ok(Expr::FunctionCall {
          name: "Or".to_string(),
          args: conditions,
        })
      }
    }
    // Solve returned unevaluated
    _ => Ok(Expr::FunctionCall {
      name: "Roots".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// ToRules[eqns] — converts logical combinations of equations to lists of rules.
/// Takes output from Roots/Reduce (Or/And of equations) and converts to Solve-style rules.
/// Discards inequalities (!=).
pub fn to_rules_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ToRules expects exactly 1 argument".into(),
    ));
  }

  fn eq_to_rule(expr: &Expr) -> Option<Expr> {
    // Convert x == val to {x -> val}
    if let Expr::Comparison {
      operands,
      operators,
    } = expr
      && operators.len() == 1
      && operators[0] == crate::syntax::ComparisonOp::Equal
      && operands.len() == 2
    {
      return Some(Expr::Rule {
        pattern: Box::new(operands[0].clone()),
        replacement: Box::new(operands[1].clone()),
      });
    }
    None
  }

  fn collect_and_rules(expr: &Expr) -> Vec<Expr> {
    // Collect all rules from And (conjunction) of equations
    match expr {
      Expr::FunctionCall { name, args } if name == "And" => {
        let mut rules = Vec::new();
        for arg in args {
          if let Some(rule) = eq_to_rule(arg) {
            rules.push(rule);
          }
          // Discard non-equations (inequalities, etc.)
        }
        rules
      }
      _ => {
        if let Some(rule) = eq_to_rule(expr) {
          vec![rule]
        } else {
          vec![]
        }
      }
    }
  }

  let input = &args[0];
  match input {
    // Or[x == a, x == b, ...] → {{x -> a}, {x -> b}, ...}
    Expr::FunctionCall { name, args } if name == "Or" => {
      let result: Vec<Expr> = args
        .iter()
        .map(|arg| Expr::List(collect_and_rules(arg)))
        .filter(|list| {
          if let Expr::List(items) = list {
            !items.is_empty()
          } else {
            false
          }
        })
        .collect();
      Ok(Expr::List(result))
    }
    // And[x == a, y == b] → {{x -> a, y -> b}}
    Expr::FunctionCall { name, .. } if name == "And" => {
      let rules = collect_and_rules(input);
      if rules.is_empty() {
        Ok(Expr::List(vec![]))
      } else {
        Ok(Expr::List(vec![Expr::List(rules)]))
      }
    }
    // Single equation: x == a → {{x -> a}}
    Expr::Comparison { .. } => {
      let rules = collect_and_rules(input);
      if rules.is_empty() {
        Ok(Expr::List(vec![]))
      } else {
        Ok(Expr::List(vec![Expr::List(rules)]))
      }
    }
    // True → {{}} (trivially satisfied)
    Expr::Identifier(s) if s == "True" => {
      Ok(Expr::List(vec![Expr::List(vec![])]))
    }
    // False → {} (no solutions)
    Expr::Identifier(s) if s == "False" => Ok(Expr::List(vec![])),
    // Anything else: return unevaluated
    _ => Ok(Expr::FunctionCall {
      name: "ToRules".to_string(),
      args: vec![input.clone()],
    }),
  }
}

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
pub fn solve_divide(num: &Expr, den: &Expr) -> Expr {
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
pub fn simplify_sqrt_parts(n: i128) -> (i128, i128) {
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

// ─── FindRoot ────────────────────────────────────────────────────────

/// FindRoot[expr, {var, x0}] — numerically find a root using Newton's method.
///
/// `expr` can be an expression (finds where it equals 0) or an equation `lhs == rhs`.
/// Returns `{var -> root_value}`.
pub fn find_root_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "FindRoot expects 2 arguments".into(),
    ));
  }

  // Parse second argument: {var, x0}
  let (var, x0) = match &args[1] {
    Expr::List(items) if items.len() == 2 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "FindRoot: variable must be a symbol".into(),
          ));
        }
      };
      let x0 = find_root_eval_number(&items[1])?;
      (var_name, x0)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "FindRoot: second argument must be {var, x0}".into(),
      ));
    }
  };

  // Extract the function to find root of: expr or lhs - rhs for equations
  let func = match &args[0] {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == crate::syntax::ComparisonOp::Equal =>
    {
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
    other => other.clone(),
  };

  // Compute derivative symbolically
  let deriv = crate::functions::calculus_ast::differentiate_expr(&func, &var)?;
  let deriv = simplify(deriv);

  // Newton's method
  let max_iter = 100;
  let tol = 1e-15;
  let mut x = x0;

  for _ in 0..max_iter {
    let fx = find_root_eval_at(&func, &var, x)?;
    if fx.abs() < tol {
      break;
    }
    let fpx = find_root_eval_at(&deriv, &var, x)?;
    if fpx.abs() < 1e-30 {
      // Derivative too small — try secant method step
      let h = 1e-8;
      let fx_plus = find_root_eval_at(&func, &var, x + h)?;
      let fpx_approx = (fx_plus - fx) / h;
      if fpx_approx.abs() < 1e-30 {
        return Err(InterpreterError::EvaluationError(
          "FindRoot: derivative is zero, cannot converge".into(),
        ));
      }
      x -= fx / fpx_approx;
    } else {
      x -= fx / fpx;
    }
  }

  // Format the result
  let result_val = if x == 0.0 || (x.abs() > 1e-15 && x.abs() < 1e15) {
    Expr::Real(x)
  } else {
    Expr::Real(x)
  };

  // Clean up -0.0
  let result_val = if x == 0.0 {
    Expr::Real(0.0)
  } else {
    result_val
  };

  Ok(Expr::List(vec![Expr::Rule {
    pattern: Box::new(Expr::Identifier(var)),
    replacement: Box::new(result_val),
  }]))
}

/// Evaluate an expression numerically at a specific value of var.
pub fn find_root_eval_at(
  expr: &Expr,
  var: &str,
  x: f64,
) -> Result<f64, InterpreterError> {
  let substituted =
    crate::syntax::substitute_variable(expr, var, &Expr::Real(x));
  let evaled = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
  match &evaled {
    Expr::Integer(n) => Ok(*n as f64),
    Expr::Real(r) => Ok(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Ok(*n as f64 / *d as f64)
      } else {
        Err(InterpreterError::EvaluationError(
          "FindRoot: cannot evaluate expression numerically".into(),
        ))
      }
    }
    _ => {
      // Try N[] evaluation
      let n_result = crate::functions::math_ast::n_ast(&[evaled])?;
      match &n_result {
        Expr::Real(r) => Ok(*r),
        Expr::Integer(n) => Ok(*n as f64),
        _ => Err(InterpreterError::EvaluationError(
          "FindRoot: cannot evaluate expression numerically".into(),
        )),
      }
    }
  }
}

/// Parse a number from an expression for FindRoot starting point.
pub fn find_root_eval_number(expr: &Expr) -> Result<f64, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(*n as f64),
    Expr::Real(r) => Ok(*r),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Ok(-find_root_eval_number(operand)?),
    _ => {
      // Try evaluating
      let evaled = crate::evaluator::evaluate_expr_to_expr(expr)?;
      match &evaled {
        Expr::Integer(n) => Ok(*n as f64),
        Expr::Real(r) => Ok(*r),
        _ => Err(InterpreterError::EvaluationError(
          "FindRoot: starting point must be numeric".into(),
        )),
      }
    }
  }
}

// ─── FindMinimum / FindMaximum ───────────────────────────────────────

/// FindMinimum[f, {x, x0}] — find a local minimum of f starting at x0
/// FindMinimum[f, {{x, x0}, {y, y0}}] — multivariable
/// Returns {min_value, {x -> x_min, ...}}
///
/// FindMaximum is implemented by negating f and negating the result.
pub fn find_minimum_ast(
  args: &[Expr],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  let func_name = if maximize {
    "FindMaximum"
  } else {
    "FindMinimum"
  };
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{func_name} expects 2 arguments"
    )));
  }

  let f = &args[0];

  // Parse variables and starting points: {x, x0} or {{x, x0}, {y, y0}}
  let var_specs = match &args[1] {
    Expr::List(items)
      if !items.is_empty() && matches!(&items[0], Expr::List(_)) =>
    {
      // Multivariable: {{x, x0}, {y, y0}, ...}
      let mut specs = Vec::new();
      for item in items {
        if let Expr::List(pair) = item
          && pair.len() == 2
          && let Expr::Identifier(name) = &pair[0]
        {
          let x0 = find_root_eval_number(&pair[1])?;
          specs.push((name.clone(), x0));
        } else {
          return Err(InterpreterError::EvaluationError(format!(
            "{func_name}: variable spec must be {{var, start}}"
          )));
        }
      }
      specs
    }
    Expr::List(items) if items.len() == 2 => {
      // Single variable: {x, x0}
      if let Expr::Identifier(name) = &items[0] {
        let x0 = find_root_eval_number(&items[1])?;
        vec![(name.clone(), x0)]
      } else {
        return Err(InterpreterError::EvaluationError(format!(
          "{func_name}: variable spec must be {{var, start}}"
        )));
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "{func_name}: second argument must be {{var, start}} or {{{{x, x0}}, {{y, y0}}}}"
      )));
    }
  };

  let vars: Vec<String> = var_specs.iter().map(|(v, _)| v.clone()).collect();
  let mut x: Vec<f64> = var_specs.iter().map(|(_, x0)| *x0).collect();
  let n = vars.len();

  // Compute symbolic gradients (partial derivatives)
  let mut grad_exprs: Vec<Expr> = Vec::with_capacity(n);
  for var in &vars {
    let deriv = crate::functions::calculus_ast::differentiate_expr(f, var)?;
    grad_exprs.push(simplify(deriv));
  }

  // Compute symbolic Hessian (for Newton's method in 1D, second derivative)
  let mut hess_exprs: Vec<Vec<Expr>> = Vec::new();
  for i in 0..n {
    let mut row = Vec::new();
    for j in 0..n {
      let h = crate::functions::calculus_ast::differentiate_expr(
        &grad_exprs[i],
        &vars[j],
      )?;
      row.push(simplify(h));
    }
    hess_exprs.push(row);
  }

  // Evaluate expression at point
  let eval_at = |expr: &Expr, point: &[f64]| -> Result<f64, InterpreterError> {
    let mut e = expr.clone();
    for (i, var) in vars.iter().enumerate() {
      e = crate::syntax::substitute_variable(&e, var, &Expr::Real(point[i]));
    }
    let evaled = crate::evaluator::evaluate_expr_to_expr(&e)?;
    expr_to_f64(&evaled)
  };

  let sign = if maximize { -1.0 } else { 1.0 };
  let max_iter = 200;
  let tol = 1e-15;

  if n == 1 {
    // Single variable: damped Newton's method on the derivative
    // Uses line search to ensure we actually decrease/increase the function
    for _ in 0..max_iter {
      let gval = eval_at(&grad_exprs[0], &x)?;
      if gval.abs() < tol {
        break;
      }
      let hval = eval_at(&hess_exprs[0][0], &x)?;

      // Compute Newton direction
      let step = if hval.abs() < 1e-30 {
        // Hessian too small — use gradient descent step
        sign * gval * 0.1
      } else if (maximize && hval > 0.0) || (!maximize && hval < 0.0) {
        // Hessian has wrong sign for our goal (saddle point or max when seeking min)
        // Use gradient descent instead
        sign * gval * 0.1
      } else {
        gval / hval
      };

      // Line search along Newton direction to ensure improvement
      let current_f = eval_at(f, &x)? * sign;
      let mut alpha = 1.0;
      let mut best_x = x[0] - step;
      let mut best_f = eval_at(f, &[best_x])? * sign;

      // Backtracking: reduce step if it doesn't improve
      for _ in 0..30 {
        if best_f < current_f {
          break;
        }
        alpha *= 0.5;
        best_x = x[0] - alpha * step;
        best_f = eval_at(f, &[best_x])? * sign;
      }
      x[0] = best_x;
    }
  } else {
    // Multivariable: use BFGS-like gradient descent with line search
    for _ in 0..max_iter {
      // Evaluate gradient
      let mut grad = vec![0.0; n];
      for i in 0..n {
        grad[i] = eval_at(&grad_exprs[i], &x)?;
      }

      let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
      if grad_norm < tol {
        break;
      }

      // Descent direction
      let dir: Vec<f64> = grad.iter().map(|g| -sign * g).collect();

      // Backtracking line search
      let mut alpha = 1.0;
      let c = 1e-4;
      let current_f = eval_at(f, &x)? * sign;

      for _ in 0..50 {
        let x_new: Vec<f64> = x
          .iter()
          .zip(dir.iter())
          .map(|(xi, di)| xi + alpha * di)
          .collect();
        let new_f = eval_at(f, &x_new)? * sign;
        let decrease: f64 =
          grad.iter().zip(dir.iter()).map(|(g, d)| g * d).sum::<f64>();
        if new_f <= current_f + c * alpha * decrease * sign {
          x = x_new;
          break;
        }
        alpha *= 0.5;
        if alpha < 1e-15 {
          x = x_new;
          break;
        }
      }
    }
  }

  // Compute final function value
  let min_val = eval_at(f, &x)?;
  let min_val_expr = Expr::Real(min_val);

  // Build result: {min_val, {x -> x_min, y -> y_min, ...}}
  let rules: Vec<Expr> = vars
    .iter()
    .zip(x.iter())
    .map(|(var, val)| Expr::Rule {
      pattern: Box::new(Expr::Identifier(var.clone())),
      replacement: Box::new(Expr::Real(*val)),
    })
    .collect();

  Ok(Expr::List(vec![min_val_expr, Expr::List(rules)]))
}

/// Convert an evaluated expression to f64
pub fn expr_to_f64(expr: &Expr) -> Result<f64, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(*n as f64),
    Expr::Real(r) => Ok(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Ok(*n as f64 / *d as f64)
      } else {
        Err(InterpreterError::EvaluationError(
          "Cannot evaluate expression numerically".into(),
        ))
      }
    }
    _ => {
      let n_result = crate::functions::math_ast::n_ast(&[expr.clone()])?;
      match &n_result {
        Expr::Real(r) => Ok(*r),
        Expr::Integer(n) => Ok(*n as f64),
        _ => Err(InterpreterError::EvaluationError(
          "Cannot evaluate expression numerically".into(),
        )),
      }
    }
  }
}
