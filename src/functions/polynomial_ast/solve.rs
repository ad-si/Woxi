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

// ─── Minimize / Maximize ─────────────────────────────────────────────

/// Minimize[f, x] or Minimize[f, {x, y, ...}] — find the global minimum.
/// Minimize[{f, cons1, cons2, ...}, vars] — constrained minimization.
/// Returns {min_val, {x -> x_min, ...}} with exact results when possible.
///
/// Maximize[f, vars] is the dual (negates objective and result).
pub fn minimize_ast(
  args: &[Expr],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  let func_name = if maximize { "Maximize" } else { "Minimize" };
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{func_name} expects 2 arguments"
    )));
  }

  // Parse variable list: x, {x}, {x, y}, ...
  let vars = minimize_parse_vars(&args[1], func_name)?;

  // Parse objective and constraints: f or {f, cons1, cons2, ...}
  let (objective, constraints) = minimize_parse_objective(&args[0]);

  if constraints.is_empty() {
    // Unconstrained
    if vars.len() == 1 {
      minimize_single_var(&objective, &vars[0], maximize, func_name)
    } else {
      minimize_multi_var(&objective, &vars, maximize, func_name)
    }
  } else {
    minimize_constrained(&objective, &constraints, &vars, maximize, func_name)
  }
}

fn minimize_parse_vars(
  expr: &Expr,
  func_name: &str,
) -> Result<Vec<String>, InterpreterError> {
  match expr {
    Expr::Identifier(name) => Ok(vec![name.clone()]),
    Expr::List(items) => {
      let mut vars = Vec::new();
      for item in items {
        match item {
          Expr::Identifier(name) => vars.push(name.clone()),
          _ => {
            return Err(InterpreterError::EvaluationError(format!(
              "{func_name}: variables must be symbols"
            )));
          }
        }
      }
      if vars.is_empty() {
        return Err(InterpreterError::EvaluationError(format!(
          "{func_name}: variable list cannot be empty"
        )));
      }
      Ok(vars)
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "{func_name}: second argument must be a variable or list of variables"
    ))),
  }
}

fn minimize_parse_objective(expr: &Expr) -> (Expr, Vec<Expr>) {
  if let Expr::List(items) = expr
    && !items.is_empty()
  {
    return (items[0].clone(), items[1..].to_vec());
  }
  (expr.clone(), vec![])
}

/// Evaluate f at a specific value of var, returning an exact expression when possible.
/// Falls back to numerical evaluation and recognizes simple integers/rationals.
fn minimize_eval_exact(
  f: &Expr,
  var: &str,
  val: &Expr,
) -> Result<Expr, InterpreterError> {
  let substituted = crate::syntax::substitute_variable(f, var, val);
  let evaled = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
  let simplified = simplify(evaled);

  // If already a simple exact number, return it
  if matches!(&simplified, Expr::Integer(_) | Expr::Real(_)) {
    return Ok(simplified);
  }
  if let Expr::FunctionCall { name, .. } = &simplified
    && name == "Rational"
  {
    return Ok(simplified);
  }
  if let Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand,
  } = &simplified
  {
    if matches!(operand.as_ref(), Expr::Integer(_)) {
      return Ok(simplified);
    }
    if let Expr::FunctionCall { name, .. } = operand.as_ref()
      && name == "Rational"
    {
      return Ok(simplified);
    }
  }

  // Try numerical evaluation to recognize exact integer/rational value
  if let Some(num_val) = minimize_try_f64(&simplified) {
    return Ok(minimize_recognize_exact(num_val));
  }

  Ok(simplified)
}

/// Try to recognize a float as an exact integer or rational.
fn minimize_recognize_exact(v: f64) -> Expr {
  if !v.is_finite() {
    return Expr::Real(v);
  }
  let rounded = v.round();
  if (rounded - v).abs() < 1e-8 {
    return Expr::Integer(rounded as i128);
  }
  for q in 2i128..=20 {
    let p = (v * q as f64).round() as i128;
    if ((p as f64 / q as f64) - v).abs() < 1e-8 {
      let (rn, rd) = reduce_fraction(p, q);
      return if rd == 1 {
        Expr::Integer(rn)
      } else {
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(rn), Expr::Integer(rd)],
        }
      };
    }
  }
  Expr::Real(v)
}

/// Evaluate f at multiple variables, returning exact result when possible.
fn minimize_eval_exact_multi(
  f: &Expr,
  vars: &[String],
  vals: &[Expr],
) -> Result<Expr, InterpreterError> {
  let mut expr = f.clone();
  for (var, val) in vars.iter().zip(vals.iter()) {
    expr = crate::syntax::substitute_variable(&expr, var, val);
  }
  let evaled = crate::evaluator::evaluate_expr_to_expr(&expr)?;
  let simplified = simplify(evaled);

  // If already a simple exact number, return it
  if matches!(&simplified, Expr::Integer(_) | Expr::Real(_)) {
    return Ok(simplified);
  }
  if let Expr::FunctionCall { name, .. } = &simplified
    && name == "Rational"
  {
    return Ok(simplified);
  }

  // Try numerical evaluation to recognize exact integer/rational value
  if let Some(num_val) = minimize_try_f64(&simplified) {
    return Ok(minimize_recognize_exact(num_val));
  }

  Ok(simplified)
}

/// Try to get f64 from an Expr (for comparison).
fn minimize_try_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(r) => Some(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        if *d != 0 {
          Some(*n as f64 / *d as f64)
        } else {
          None
        }
      } else {
        None
      }
    }
    _ => {
      if let Ok(n_result) = crate::functions::math_ast::n_ast(&[expr.clone()]) {
        match n_result {
          Expr::Real(r) => Some(r),
          Expr::Integer(n) => Some(n as f64),
          _ => None,
        }
      } else {
        None
      }
    }
  }
}

/// Find roots of a univariate polynomial given its integer coefficients.
/// coeffs[i] = coefficient of x^i.
/// Returns real roots as exact Expr values.
fn minimize_poly_roots_int(coeffs: &[i128], var: &str) -> Vec<Expr> {
  let _ = var; // variable name not needed for roots computation
  let degree = coeffs.len().saturating_sub(1);
  let mut roots = Vec::new();

  match degree {
    0 => {
      // Constant: no roots (if constant != 0) or all x (if 0)
    }
    1 => {
      // a*x + b = 0 → x = -b/a
      let a = coeffs[1];
      let b = coeffs[0];
      if a != 0 {
        let (num, den) = reduce_fraction(-b, a);
        let root = if den == 1 {
          Expr::Integer(num)
        } else {
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(num), Expr::Integer(den)],
          }
        };
        roots.push(root);
      }
    }
    2 => {
      // a*x^2 + b*x + c = 0
      let a = coeffs[2];
      let b = coeffs[1];
      let c = coeffs[0];
      if a != 0 {
        let disc = b * b - 4 * a * c;
        if disc >= 0 {
          let (sqrt_out, sqrt_in) = simplify_sqrt_parts(disc);
          if sqrt_in == 1 {
            // Perfect square
            let (n1, d1) = reduce_fraction(-b - sqrt_out, 2 * a);
            let (n2, d2) = reduce_fraction(-b + sqrt_out, 2 * a);
            roots.push(if d1 == 1 {
              Expr::Integer(n1)
            } else {
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(n1), Expr::Integer(d1)],
              }
            });
            if n1 != n2 || d1 != d2 {
              roots.push(if d2 == 1 {
                Expr::Integer(n2)
              } else {
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(n2), Expr::Integer(d2)],
                }
              });
            }
          } else {
            // Irrational roots: (-b ± sqrt_out * √sqrt_in) / (2a)
            let g =
              gcd_i128(gcd_i128((-b).abs(), sqrt_out.abs()), (2 * a).abs())
                .abs();
            let nb = -b / g;
            let so = sqrt_out / g;
            let den = 2 * a / g;
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
            for sign_minus in [true, false] {
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
              let root = if den == 1 {
                num
              } else {
                Expr::BinaryOp {
                  op: BinaryOperator::Divide,
                  left: Box::new(num),
                  right: Box::new(Expr::Integer(den)),
                }
              };
              roots.push(simplify(root));
            }
          }
        }
        // disc < 0: complex roots, no real roots
      }
    }
    3 => {
      // Try constant term = 0 (x is a factor)
      if coeffs[0] == 0 {
        roots.push(Expr::Integer(0));
        // Remaining: coeffs[3]*x^2 + coeffs[2]*x + coeffs[1]
        let sub_roots =
          minimize_poly_roots_int(&[coeffs[1], coeffs[2], coeffs[3]], var);
        for r in sub_roots {
          if !roots.iter().any(|existing| {
            minimize_try_f64(&r)
              .zip(minimize_try_f64(existing))
              .is_some_and(|(a, b)| (a - b).abs() < 1e-12)
          }) {
            roots.push(r);
          }
        }
        return roots;
      }

      // Rational root theorem: try ±(factors of coeffs[0]) / (factors of coeffs[3])
      let a = coeffs[3];
      let d = coeffs[0];
      // Collect actual divisors
      let mut divs_d: Vec<i128> = Vec::new();
      for i in 1i128..=(d.abs()) {
        if d % i == 0 {
          divs_d.push(i);
        }
      }
      let mut divs_a: Vec<i128> = Vec::new();
      for i in 1i128..=(a.abs()) {
        if a % i == 0 {
          divs_a.push(i);
        }
      }
      'outer: for &p in &divs_d {
        for &q in &divs_a {
          for &sign in &[1i128, -1i128] {
            let r = sign * p;
            let q_val = q;
            // Test if r/q is a root: a*(r/q)^3 + b*(r/q)^2 + c*(r/q) + d == 0
            // Multiply through by q^3: a*r^3 + b*r^2*q + c*r*q^2 + d*q^3 == 0
            let val = a * r * r * r
              + coeffs[2] * r * r * q_val
              + coeffs[1] * r * q_val * q_val
              + d * q_val * q_val * q_val;
            if val == 0 {
              let (rn, rd) = reduce_fraction(r, q_val);
              let root = if rd == 1 {
                Expr::Integer(rn)
              } else {
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(rn), Expr::Integer(rd)],
                }
              };
              roots.push(root);

              // Polynomial division to get quadratic
              // Divide a*x^3 + b*x^2 + c*x + d by (q*x - r)
              // Synthetic division with root = r/q
              // q1 = a
              // q2 = a*(r/q) + b = (a*r + b*q)/q
              // q3 = q2*(r/q) + c = ...
              // Multiply through: coefficients of (a*x^2 + (a*r/q + b)*x + ...) * q
              // Use polynomial long division:
              // (a*x^3 + b*x^2 + c*x + d) / (x - r/q)
              // = a*x^2 + (a*r/q + b)*x + (a*(r/q)^2 + b*(r/q) + c)
              // Multiply by q^2 to get integer coefficients:
              // a*q^2 * x^2 + (a*r*q + b*q^2)*x + (a*r^2 + b*r*q + c*q^2)
              // But we want integer coefficients, divide by gcd
              let qa = a;
              let qb = a * r / q_val + coeffs[2];
              let qc =
                a * r * r / (q_val * q_val) + coeffs[2] * r / q_val + coeffs[1];
              // Only proceed if exact (no fractions)
              if a * r % q_val == 0 && a * r * r % (q_val * q_val) == 0 {
                let sub_roots = minimize_poly_roots_int(&[qc, qb, qa], var);
                for sr in sub_roots {
                  if !roots.iter().any(|existing| {
                    minimize_try_f64(&sr)
                      .zip(minimize_try_f64(existing))
                      .is_some_and(|(a, b)| (a - b).abs() < 1e-12)
                  }) {
                    roots.push(sr);
                  }
                }
              }
              break 'outer;
            }
          }
        }
      }
    }
    _ => {
      // Higher degree: try numerical root finding with multiple starting points
      // We'll handle this in the caller via numerical fallback
    }
  }
  roots
}

/// Reduce fraction n/d to lowest terms with positive denominator.
fn reduce_fraction(n: i128, d: i128) -> (i128, i128) {
  if d == 0 {
    return (n, d);
  }
  let g = gcd_i128(n.abs(), d.abs()).abs();
  let mut rn = n / g;
  let mut rd = d / g;
  if rd < 0 {
    rn = -rn;
    rd = -rd;
  }
  (rn, rd)
}

/// Extract integer polynomial coefficients of `poly` in `var`.
/// Returns Some(coeffs) where coeffs[i] = coefficient of var^i.
/// Returns None if not a polynomial with integer coefficients.
fn minimize_extract_int_coeffs(poly: &Expr, var: &str) -> Option<Vec<i128>> {
  let expanded = expand_and_combine(poly);
  let degree = max_power(&expanded, var)? as usize;
  let terms = collect_additive_terms(&expanded);
  // Pre-check: ensure all terms are polynomial in var (sentinel -1 = non-polynomial)
  for term in &terms {
    let (power, _) = term_var_power_and_coeff(term, var);
    if power == -1 {
      return None; // term contains var non-polynomially (e.g. E^x, Sin[x])
    }
  }
  let mut coeffs = vec![0i128; degree + 1];
  for d in 0..=degree {
    let mut sum = 0i128;
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, d as i128) {
        match c {
          Expr::Integer(n) => sum += n,
          _ => return None, // non-integer coefficient
        }
      }
    }
    coeffs[d] = sum;
  }
  Some(coeffs)
}

/// Check if a polynomial f in var is bounded below.
/// Returns Some(true) if bounded below, Some(false) if not, None if unknown.
/// Only handles true polynomials with integer coefficients.
fn minimize_poly_bounded_below(f: &Expr, var: &str) -> Option<bool> {
  // Only use polynomial analysis for verified polynomials with integer coefficients
  let expanded = expand_and_combine(f);
  let degree = max_power(&expanded, var)?;
  if degree == 0 {
    // Might be a constant OR a non-polynomial term like E^x with "degree 0"
    // We can't distinguish here, return None to use numerical check
    return None;
  }
  // Verify the function is truly a polynomial by checking that all
  // integer polynomial coefficients can be extracted
  let coeffs = minimize_extract_int_coeffs(&expanded, var)?;
  if coeffs.len() < 2 {
    return None;
  }
  let d = coeffs.len() - 1;
  let lead_coeff = coeffs[d];

  if d % 2 == 1 {
    // Odd degree: always unbounded in both directions
    Some(false)
  } else if lead_coeff > 0 {
    Some(true)
  } else if lead_coeff < 0 {
    Some(false)
  } else {
    None
  }
}

/// Check if f is bounded below by evaluating numerically at large values.
fn minimize_bounded_below_numerical(f: &Expr, var: &str) -> bool {
  let test_points: &[f64] = &[-1e6, 1e6];
  let threshold = -1e8;
  for &x in test_points {
    let substituted =
      crate::syntax::substitute_variable(f, var, &Expr::Real(x));
    if let Ok(evaled) = crate::evaluator::evaluate_expr_to_expr(&substituted)
      && let Some(val) = minimize_try_f64(&evaled)
      && val < threshold
    {
      return false;
    }
  }
  true
}

/// Build the -Infinity result for minimize (no minimum exists).
fn minimize_neg_infinity_result(vars: &[String], maximize: bool) -> Expr {
  let inf_val = if maximize {
    // Maximize returns {Infinity, {x -> Infinity}}
    Expr::Identifier("Infinity".to_string())
  } else {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    }
  };
  let x_val = if maximize {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    }
  } else {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    }
  };
  let rules: Vec<Expr> = vars
    .iter()
    .map(|v| Expr::Rule {
      pattern: Box::new(Expr::Identifier(v.clone())),
      replacement: Box::new(x_val.clone()),
    })
    .collect();
  Expr::List(vec![inf_val, Expr::List(rules)])
}

/// Single-variable unconstrained minimize.
fn minimize_single_var(
  f: &Expr,
  var: &str,
  maximize: bool,
  func_name: &str,
) -> Result<Expr, InterpreterError> {
  // For maximize, negate f and negate the result at the end
  let f_inner = if maximize {
    simplify(negate_expr(f))
  } else {
    f.clone()
  };

  // Compute symbolic derivative
  let df = simplify(crate::functions::calculus_ast::differentiate_expr(
    &f_inner, var,
  )?);

  // Check if f is bounded below (polynomial check first)
  let bounded = if let Some(b) = minimize_poly_bounded_below(&f_inner, var) {
    b
  } else {
    // Non-polynomial: try numerical check
    minimize_bounded_below_numerical(&f_inner, var)
  };

  if !bounded {
    return Ok(minimize_neg_infinity_result(&[var.to_string()], maximize));
  }

  // Find critical points: solve df == 0
  let critical_points = minimize_find_critical_points_1d(&df, var, &f_inner)?;

  if critical_points.is_empty() {
    // Bounded function with no critical points: return unevaluated
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![f.clone(), Expr::Identifier(var.to_string())],
    });
  }

  // Evaluate f at each critical point, find the minimum
  let mut best_val: Option<f64> = None;
  let mut best_exact: Option<Expr> = None;
  let mut best_x: Option<Expr> = None;

  for cp in &critical_points {
    let fval_exact = minimize_eval_exact(&f_inner, var, cp)?;
    let fval_num = minimize_try_f64(&fval_exact);

    if let Some(fv) = fval_num {
      let is_better = match best_val {
        None => true,
        Some(bv) => fv < bv,
      };
      if is_better {
        best_val = Some(fv);
        best_exact = Some(fval_exact);
        best_x = Some(cp.clone());
      }
    }
  }

  let (min_val, min_x) = match (best_exact, best_x) {
    (Some(v), Some(x)) => (v, x),
    _ => {
      return Ok(Expr::FunctionCall {
        name: func_name.to_string(),
        args: vec![f.clone(), Expr::Identifier(var.to_string())],
      });
    }
  };

  // For maximize, negate the value back
  let result_val = if maximize {
    simplify(negate_expr(&min_val))
  } else {
    min_val
  };

  let rule = Expr::Rule {
    pattern: Box::new(Expr::Identifier(var.to_string())),
    replacement: Box::new(min_x),
  };
  Ok(Expr::List(vec![result_val, Expr::List(vec![rule])]))
}

/// Find critical points of f' = 0 in one variable.
fn minimize_find_critical_points_1d(
  df: &Expr,
  var: &str,
  f: &Expr,
) -> Result<Vec<Expr>, InterpreterError> {
  // Try polynomial root finding
  let expanded_df = expand_and_combine(df);

  if let Some(coeffs) = minimize_extract_int_coeffs(&expanded_df, var) {
    let roots = minimize_poly_roots_int(&coeffs, var);
    if !roots.is_empty() || matches!(coeffs.len(), 0 | 1) {
      return Ok(roots);
    }
  }

  // Fallback: try Solve[df == 0, var]
  let df_eq = Expr::Comparison {
    operands: vec![df.clone(), Expr::Integer(0)],
    operators: vec![crate::syntax::ComparisonOp::Equal],
  };
  match solve_ast(&[df_eq, Expr::Identifier(var.to_string())]) {
    Ok(solutions) => {
      if let Expr::List(sol_sets) = &solutions {
        // If Solve returned unevaluated pieces or empty list, try numerical
        if sol_sets.iter().any(|s| {
          !matches!(s, Expr::List(_))
            || matches!(s, Expr::FunctionCall { name, .. } if name == "Solve")
        }) {
          return minimize_find_critical_points_numerical(f, var);
        }
        // If Solve returned empty (no solutions found), also try numerical
        // as it might have incorrectly classified the equation
        if sol_sets.is_empty() {
          return minimize_find_critical_points_numerical(f, var);
        }
        let mut roots = Vec::new();
        for sol_set in sol_sets {
          if let Expr::List(rules) = sol_set {
            for rule in rules {
              if let Expr::Rule { replacement, .. } = rule {
                roots.push(*replacement.clone());
              }
            }
          }
        }
        // If Solve found no actual roots (all empty rule sets), try numerical
        if roots.is_empty() {
          return minimize_find_critical_points_numerical(f, var);
        }
        return Ok(roots);
      }
      // Unevaluated Solve result - try numerical
      minimize_find_critical_points_numerical(f, var)
    }
    Err(_) => minimize_find_critical_points_numerical(f, var),
  }
}

/// Numerically find critical points of f using Newton's method with multiple starts.
fn minimize_find_critical_points_numerical(
  f: &Expr,
  var: &str,
) -> Result<Vec<Expr>, InterpreterError> {
  let df =
    simplify(crate::functions::calculus_ast::differentiate_expr(f, var)?);

  let starts: &[f64] = &[-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0];
  let mut roots: Vec<f64> = Vec::new();
  let tol = 1e-10;

  for &x0 in starts {
    let mut x = x0;
    for _ in 0..100 {
      let gval = find_root_eval_at(&df, var, x).unwrap_or(f64::NAN);
      if gval.is_nan() || gval.is_infinite() {
        break;
      }
      if gval.abs() < tol {
        break;
      }
      let hval = {
        let h = 1e-7;
        let g1 = find_root_eval_at(&df, var, x + h).unwrap_or(f64::NAN);
        if g1.is_nan() {
          break;
        }
        (g1 - gval) / h
      };
      if hval.abs() < 1e-30 {
        break;
      }
      x -= gval / hval;
      if !x.is_finite() {
        break;
      }
    }
    if x.is_finite() {
      let gval = find_root_eval_at(&df, var, x).unwrap_or(f64::INFINITY);
      if gval.abs() < 1e-6 {
        // Check if this root is already found
        if !roots.iter().any(|&r| (r - x).abs() < 1e-6) {
          roots.push(x);
        }
      }
    }
  }

  Ok(roots.into_iter().map(minimize_recognize_exact).collect())
}

/// Multi-variable unconstrained minimize.
fn minimize_multi_var(
  f: &Expr,
  vars: &[String],
  maximize: bool,
  func_name: &str,
) -> Result<Expr, InterpreterError> {
  let f_inner = if maximize {
    simplify(negate_expr(f))
  } else {
    f.clone()
  };

  let n = vars.len();

  // Compute symbolic gradient
  let mut grad: Vec<Expr> = Vec::new();
  for var in vars {
    let dfi =
      crate::functions::calculus_ast::differentiate_expr(&f_inner, var)?;
    grad.push(simplify(dfi));
  }

  // Try to solve the gradient system symbolically
  // For independent linear equations in each variable, solve separately
  let mut solutions: Vec<Option<Expr>> = vec![None; n];
  let mut all_solved = true;

  for (i, var) in vars.iter().enumerate() {
    let grad_eq = Expr::Comparison {
      operands: vec![grad[i].clone(), Expr::Integer(0)],
      operators: vec![crate::syntax::ComparisonOp::Equal],
    };
    match solve_ast(&[grad_eq, Expr::Identifier(var.clone())]) {
      Ok(sol) => {
        if let Expr::List(sol_sets) = &sol
          && sol_sets.len() == 1
          && let Some(Expr::List(rules)) = sol_sets.first()
          && rules.len() == 1
          && let Some(Expr::Rule { replacement, .. }) = rules.first()
        {
          solutions[i] = Some(*replacement.clone());
          continue;
        }
        all_solved = false;
        break;
      }
      Err(_) => {
        all_solved = false;
        break;
      }
    }
  }

  if all_solved {
    let vals: Vec<Expr> = solutions.into_iter().flatten().collect();
    if vals.len() == n {
      // Evaluate f at the critical point
      let fval = minimize_eval_exact_multi(&f_inner, vars, &vals)?;
      let result_val = if maximize {
        simplify(negate_expr(&fval))
      } else {
        fval
      };
      let rules: Vec<Expr> = vars
        .iter()
        .zip(vals.iter())
        .map(|(v, val)| Expr::Rule {
          pattern: Box::new(Expr::Identifier(v.clone())),
          replacement: Box::new(val.clone()),
        })
        .collect();
      return Ok(Expr::List(vec![result_val, Expr::List(rules)]));
    }
  }

  // Fallback: numerical multi-variable minimize (gradient descent from origin)
  let mut x: Vec<f64> = vec![0.0; n];
  let tol = 1e-12;
  let max_iter = 500;

  for _ in 0..max_iter {
    let mut grad_vals = vec![0.0f64; n];
    let mut grad_norm = 0.0f64;
    for i in 0..n {
      // For multi-var we need proper substitution, use eval_at_multi
      let mut gexpr = grad[i].clone();
      for (j, vj) in vars.iter().enumerate() {
        gexpr =
          crate::syntax::substitute_variable(&gexpr, vj, &Expr::Real(x[j]));
      }
      let gval = crate::evaluator::evaluate_expr_to_expr(&gexpr)
        .ok()
        .and_then(|e| minimize_try_f64(&e))
        .unwrap_or(0.0);
      grad_vals[i] = gval;
      grad_norm += gval * gval;
    }
    grad_norm = grad_norm.sqrt();
    if grad_norm < tol {
      break;
    }

    // Gradient descent step
    let alpha = 0.01 / (1.0 + grad_norm);
    for i in 0..n {
      x[i] -= alpha * grad_vals[i];
    }
  }

  // Evaluate f at the numerical minimum
  let mut fexpr = f_inner.clone();
  for (i, var) in vars.iter().enumerate() {
    fexpr = crate::syntax::substitute_variable(&fexpr, var, &Expr::Real(x[i]));
  }
  let fval = crate::evaluator::evaluate_expr_to_expr(&fexpr)
    .ok()
    .and_then(|e| minimize_try_f64(&e))
    .unwrap_or(f64::NAN);

  if !fval.is_finite() {
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![
        f.clone(),
        Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect()),
      ],
    });
  }

  let result_val = if maximize {
    Expr::Real(-fval)
  } else {
    Expr::Real(fval)
  };
  let rules: Vec<Expr> = vars
    .iter()
    .zip(x.iter())
    .map(|(v, &val)| Expr::Rule {
      pattern: Box::new(Expr::Identifier(v.clone())),
      replacement: Box::new(Expr::Real(val)),
    })
    .collect();
  Ok(Expr::List(vec![result_val, Expr::List(rules)]))
}

/// Flatten And[a, b, c, ...] recursively into a flat list of constraints.
fn flatten_and_constraints(constraints: &[Expr]) -> Vec<Expr> {
  let mut result = Vec::new();
  for c in constraints {
    flatten_and_expr(c, &mut result);
  }
  result
}

fn flatten_and_expr(expr: &Expr, result: &mut Vec<Expr>) {
  match expr {
    Expr::FunctionCall { name, args } if name == "And" => {
      for arg in args {
        flatten_and_expr(arg, result);
      }
    }
    _ => result.push(expr.clone()),
  }
}

/// Constrained minimization.
fn minimize_constrained(
  f: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
  func_name: &str,
) -> Result<Expr, InterpreterError> {
  // Flatten And[...] chains into individual constraints
  let constraints = flatten_and_constraints(constraints);

  let f_inner = if maximize {
    simplify(negate_expr(f))
  } else {
    f.clone()
  };

  // Try ILP if any Element[x, Integers] constraint is present
  if constraints
    .iter()
    .any(|c| matches!(c, Expr::FunctionCall { name, .. } if name == "Element"))
    && let Some(result) =
      minimize_try_ilp(&f_inner, &constraints, vars, maximize, func_name)?
  {
    return Ok(result);
  }

  // Single variable with simple bound constraints
  if vars.len() == 1 {
    let var = &vars[0];
    return minimize_constrained_1d(
      &f_inner,
      &constraints,
      var,
      maximize,
      func_name,
    );
  }

  // Multi-variable: try linear programming for linear constraints + linear/quadratic objective
  minimize_constrained_nd(&f_inner, &constraints, vars, maximize, func_name)
}

/// Try Integer Linear Programming. Returns Some(result) if ILP was solved, None if unsupported.
fn minimize_try_ilp(
  f: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
  func_name: &str,
) -> Result<Option<Expr>, InterpreterError> {
  use std::collections::HashSet;

  // Separate Element[x, Integers] from actual constraints
  let mut integer_vars: HashSet<String> = HashSet::new();
  let mut actual_constraints: Vec<&Expr> = Vec::new();
  for c in constraints {
    match c {
      Expr::FunctionCall { name, args }
        if name == "Element" && args.len() == 2 =>
      {
        if let (Expr::Identifier(var), Expr::Identifier(dom)) =
          (&args[0], &args[1])
          && dom == "Integers"
        {
          integer_vars.insert(var.clone());
        }
        // Don't add to actual_constraints
      }
      _ => actual_constraints.push(c),
    }
  }

  // All problem variables must be integer-constrained
  if !vars.iter().all(|v| integer_vars.contains(v)) {
    return Ok(None);
  }

  // Extract linear objective coefficients
  let obj_coeffs = match minimize_extract_linear_expr(f, vars) {
    Some((c, _)) => c,
    None => return Ok(None), // non-linear objective
  };

  // Extract linear constraints: one equality + non-negativity inequalities
  let mut equalities: Vec<(Vec<f64>, f64)> = Vec::new(); // (coeffs, rhs)
  let mut lb: Vec<f64> = vec![0.0; vars.len()]; // lower bounds (default 0)

  for con in &actual_constraints {
    if let Some((coeffs, rhs, sense)) =
      minimize_extract_linear_constraint(con, vars)
    {
      match sense {
        0 => equalities.push((coeffs, rhs)), // ==
        1 => {
          // coeffs · x >= rhs  →  update lower bounds if simple bound
          // Check if it's x_i >= c (single variable)
          let nonzero: Vec<usize> = coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| c.abs() > 1e-12)
            .map(|(i, _)| i)
            .collect();
          if nonzero.len() == 1 {
            let i = nonzero[0];
            let bound = rhs / coeffs[i];
            if bound > lb[i] {
              lb[i] = bound;
            }
          }
          // Multi-variable ineq: only handle non-negativity for ILP DP
        }
        _ => {} // LessEqual and other constraints not handled in DP
      }
    } else {
      return Ok(None); // non-linear constraint
    }
  }

  // Only support single equality constraint for DP
  if equalities.len() != 1 {
    return Ok(None);
  }
  let (eq_coeffs, eq_rhs) = &equalities[0];

  // All lb must be 0 (non-negativity only) and all eq_coeffs must be positive integers
  if lb.iter().any(|&b| b.abs() > 1e-12) {
    // Non-zero lower bounds: shift variables and recurse? For now, skip.
    return Ok(None);
  }

  // Verify all equality coefficients are positive integers and rhs is positive integer
  let mut weights: Vec<i64> = Vec::with_capacity(vars.len());
  for &c in eq_coeffs {
    let ci = c.round() as i64;
    if (c - ci as f64).abs() > 1e-8 || ci <= 0 {
      return Ok(None); // non-integer or non-positive weight
    }
    weights.push(ci);
  }
  let target_f = eq_rhs.round();
  if (eq_rhs - target_f).abs() > 1e-8 || target_f < 0.0 {
    return Ok(None);
  }
  let target = target_f as usize;

  // Verify objective coefficients are non-negative integers (for DP correctness)
  let mut obj_int: Vec<i64> = Vec::with_capacity(vars.len());
  for &c in &obj_coeffs {
    let ci = c.round() as i64;
    if (c - ci as f64).abs() > 1e-8 || ci < 0 {
      return Ok(None);
    }
    obj_int.push(ci);
  }

  // DP: dp[t] = minimum objective value to achieve weight t
  // dp[0] = 0, dp[t] = min_i(dp[t - weights[i]] + obj_int[i]) for t >= weights[i]
  let n = vars.len();
  const INF: i64 = i64::MAX / 2;
  let mut dp = vec![INF; target + 1];
  let mut coin_used = vec![0usize; target + 1];
  dp[0] = 0;

  for t in 1..=target {
    for i in 0..n {
      let wi = weights[i] as usize;
      if wi <= t && dp[t - wi] != INF {
        let new_val = dp[t - wi] + obj_int[i];
        if new_val < dp[t] {
          dp[t] = new_val;
          coin_used[t] = i;
        }
      }
    }
  }

  if dp[target] == INF {
    // Infeasible
    return Ok(Some(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![
        f.clone(),
        Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect()),
      ],
    }));
  }

  // Backtrack to find variable assignments
  let mut x = vec![0i64; n];
  let mut t = target;
  while t > 0 {
    let i = coin_used[t];
    x[i] += 1;
    t -= weights[i] as usize;
  }

  let obj_val = dp[target];
  let result_val = if maximize {
    Expr::Integer(-(obj_val as i128))
  } else {
    Expr::Integer(obj_val as i128)
  };
  let rules: Vec<Expr> = vars
    .iter()
    .zip(x.iter())
    .map(|(v, &val)| Expr::Rule {
      pattern: Box::new(Expr::Identifier(v.clone())),
      replacement: Box::new(Expr::Integer(val as i128)),
    })
    .collect();
  Ok(Some(Expr::List(vec![result_val, Expr::List(rules)])))
}

/// Extract linear expression coefficients: f = sum(coeffs[i] * vars[i]) + constant.
/// Returns None if f is not linear in vars.
fn minimize_extract_linear_expr(
  f: &Expr,
  vars: &[String],
) -> Option<(Vec<f64>, f64)> {
  let expanded = expand_and_combine(f);
  let mut coeffs = vec![0.0f64; vars.len()];

  // Check degree <= 1 in each variable
  for var in vars {
    let deg = max_power(&expanded, var);
    if matches!(deg, Some(d) if d > 1) {
      return None;
    }
  }

  let terms = collect_additive_terms(&expanded);
  for (i, var) in vars.iter().enumerate() {
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, 1) {
        if let Some(cv) = minimize_try_f64(&c) {
          coeffs[i] += cv;
        } else {
          return None;
        }
      }
    }
  }

  // Constant term: set all vars to 0
  let mut const_expr = expanded.clone();
  for var in vars {
    const_expr =
      crate::syntax::substitute_variable(&const_expr, var, &Expr::Integer(0));
  }
  let constant = crate::evaluator::evaluate_expr_to_expr(&const_expr)
    .ok()
    .and_then(|e| minimize_try_f64(&e))
    .unwrap_or(0.0);

  Some((coeffs, constant))
}

/// Single-variable constrained minimize.
fn minimize_constrained_1d(
  f: &Expr,
  constraints: &[Expr],
  var: &str,
  maximize: bool,
  func_name: &str,
) -> Result<Expr, InterpreterError> {
  // Collect boundary points from constraints: x >= a, x <= b, x == c
  let mut lb: Option<f64> = None; // lower bound
  let mut ub: Option<f64> = None; // upper bound
  let mut eq_constraints: Vec<Expr> = Vec::new();
  let mut other_constraints = false;

  for con in constraints {
    match con {
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() == 2 && operators.len() == 1 => {
        use crate::syntax::ComparisonOp;
        let lhs = &operands[0];
        let rhs = &operands[1];
        match &operators[0] {
          ComparisonOp::GreaterEqual => {
            // lhs >= rhs
            // Check if it's var >= const or const <= var
            if matches!(lhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(rhs) {
                lb = Some(lb.map_or(v, |cur: f64| cur.max(v)));
              } else {
                other_constraints = true;
              }
            } else if matches!(rhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(lhs) {
                ub = Some(ub.map_or(v, |cur: f64| cur.min(v)));
              } else {
                other_constraints = true;
              }
            } else {
              other_constraints = true;
            }
          }
          ComparisonOp::LessEqual => {
            // lhs <= rhs
            if matches!(lhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(rhs) {
                ub = Some(ub.map_or(v, |cur: f64| cur.min(v)));
              } else {
                other_constraints = true;
              }
            } else if matches!(rhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(lhs) {
                lb = Some(lb.map_or(v, |cur: f64| cur.max(v)));
              } else {
                other_constraints = true;
              }
            } else {
              other_constraints = true;
            }
          }
          ComparisonOp::Equal => {
            eq_constraints.push(con.clone());
          }
          ComparisonOp::Greater => {
            // Strict inequality: treat as >= for boundary
            if matches!(lhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(rhs) {
                lb = Some(lb.map_or(v, |cur: f64| cur.max(v)));
              } else {
                other_constraints = true;
              }
            } else {
              other_constraints = true;
            }
          }
          ComparisonOp::Less => {
            if matches!(lhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(rhs) {
                ub = Some(ub.map_or(v, |cur: f64| cur.min(v)));
              } else {
                other_constraints = true;
              }
            } else {
              other_constraints = true;
            }
          }
          _ => other_constraints = true,
        }
      }
      _ => other_constraints = true,
    }
  }

  if other_constraints {
    // Cannot handle, return unevaluated
    let obj_with_cons = Expr::List(
      std::iter::once(f.clone())
        .chain(constraints.iter().cloned())
        .collect(),
    );
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![obj_with_cons, Expr::Identifier(var.to_string())],
    });
  }

  // Collect candidate x values: bounds + unconstrained critical points
  let mut candidates: Vec<f64> = Vec::new();

  // Add boundary points
  if let Some(l) = lb {
    candidates.push(l);
  }
  if let Some(u) = ub {
    candidates.push(u);
  }

  // Find unconstrained critical points and filter to feasible region
  let df =
    simplify(crate::functions::calculus_ast::differentiate_expr(f, var)?);
  let cps = minimize_find_critical_points_1d(&df, var, f)?;
  for cp in &cps {
    if let Some(v) = minimize_try_f64(cp) {
      let feasible =
        lb.is_none_or(|l| v >= l - 1e-10) && ub.is_none_or(|u| v <= u + 1e-10);
      if feasible {
        candidates.push(v);
      }
    }
  }

  if candidates.is_empty() {
    let obj_with_cons = Expr::List(
      std::iter::once(f.clone())
        .chain(constraints.iter().cloned())
        .collect(),
    );
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![obj_with_cons, Expr::Identifier(var.to_string())],
    });
  }

  // Find the minimum among candidates
  let mut best_f = f64::INFINITY;
  let mut best_x_f64 = candidates[0];
  for &cx in &candidates {
    let fx = find_root_eval_at(f, var, cx).unwrap_or(f64::INFINITY);
    if fx < best_f {
      best_f = fx;
      best_x_f64 = cx;
    }
  }

  // Try to find exact expression for best_x from critical points
  let best_x_exact = cps.iter().find(|cp| {
    minimize_try_f64(cp).is_some_and(|v| (v - best_x_f64).abs() < 1e-8)
  });

  let (result_val, result_x) = if let Some(exact_cp) = best_x_exact {
    let fval = minimize_eval_exact(f, var, exact_cp)?;
    let rv = if maximize {
      simplify(negate_expr(&fval))
    } else {
      fval
    };
    (rv, exact_cp.clone())
  } else {
    // Check if best_x_f64 is a boundary (integer or simple rational)
    let bx_rounded = best_x_f64.round();
    let result_x_expr = if (bx_rounded - best_x_f64).abs() < 1e-10 {
      Expr::Integer(bx_rounded as i128)
    } else {
      Expr::Real(best_x_f64)
    };
    let fval = minimize_eval_exact(f, var, &result_x_expr)?;
    let rv = if maximize {
      simplify(negate_expr(&fval))
    } else {
      fval
    };
    (rv, result_x_expr)
  };

  let rule = Expr::Rule {
    pattern: Box::new(Expr::Identifier(var.to_string())),
    replacement: Box::new(result_x),
  };
  Ok(Expr::List(vec![result_val, Expr::List(vec![rule])]))
}

/// Multi-variable constrained minimize.
/// Handles: LP (linear objective + linear constraints), and
/// non-linear objectives with linear equality/inequality constraints.
fn minimize_constrained_nd(
  f: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
  func_name: &str,
) -> Result<Expr, InterpreterError> {
  // First try pure LP (linear objective + linear constraints)
  if vars.len() >= 2
    && let Some(result) = minimize_lp_2d(f, constraints, vars, maximize)
  {
    return Ok(result);
  }

  // For any dimension, try boundary reduction for linear constraints
  if let Some(result) =
    minimize_constrained_boundary(f, constraints, vars, maximize)?
  {
    return Ok(result);
  }

  // Return unevaluated
  let obj_with_cons = Expr::List(
    std::iter::once(f.clone())
      .chain(constraints.iter().cloned())
      .collect(),
  );
  Ok(Expr::FunctionCall {
    name: func_name.to_string(),
    args: vec![
      obj_with_cons,
      Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect()),
    ],
  })
}

/// Check if a point (given as var→val map) satisfies all constraints numerically.
fn minimize_satisfies_constraints(
  constraints: &[Expr],
  vars: &[String],
  vals: &[f64],
) -> bool {
  use crate::syntax::ComparisonOp;
  for con in constraints {
    if let Expr::Comparison {
      operands,
      operators,
    } = con
      && operands.len() == 2
      && operators.len() == 1
    {
      let mut lhs_expr = operands[0].clone();
      let mut rhs_expr = operands[1].clone();
      for (var, &val) in vars.iter().zip(vals.iter()) {
        lhs_expr =
          crate::syntax::substitute_variable(&lhs_expr, var, &Expr::Real(val));
        rhs_expr =
          crate::syntax::substitute_variable(&rhs_expr, var, &Expr::Real(val));
      }
      let lhs_val = crate::evaluator::evaluate_expr_to_expr(&lhs_expr)
        .ok()
        .and_then(|e| minimize_try_f64(&e));
      let rhs_val = crate::evaluator::evaluate_expr_to_expr(&rhs_expr)
        .ok()
        .and_then(|e| minimize_try_f64(&e));
      if let (Some(l), Some(r)) = (lhs_val, rhs_val) {
        let ok = match &operators[0] {
          ComparisonOp::GreaterEqual => l >= r - 1e-8,
          ComparisonOp::LessEqual => l <= r + 1e-8,
          ComparisonOp::Greater => l > r - 1e-8,
          ComparisonOp::Less => l < r + 1e-8,
          ComparisonOp::Equal => (l - r).abs() <= 1e-8,
          _ => true,
        };
        if !ok {
          return false;
        }
      }
    }
  }
  true
}

/// Extract linear constraint coefficients for the form: a*x + b*y + ... >= c.
/// Returns None if constraint is not linear.
fn minimize_extract_linear_constraint(
  con: &Expr,
  vars: &[String],
) -> Option<(Vec<f64>, f64, i32)> {
  use crate::syntax::ComparisonOp;
  let (operands, operators) = match con {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => (operands, operators),
    _ => return None,
  };
  let sense = match &operators[0] {
    ComparisonOp::GreaterEqual | ComparisonOp::Greater => 1,
    ComparisonOp::LessEqual | ComparisonOp::Less => -1,
    ComparisonOp::Equal => 0,
    _ => return None,
  };

  // diff = lhs - rhs, should be linear
  let diff = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(operands[0].clone()),
    right: Box::new(operands[1].clone()),
  };
  let expanded = expand_and_combine(&diff);

  let mut coeffs = vec![0.0f64; vars.len()];
  let mut constant = 0.0f64;

  // Check this is a polynomial of degree <= 1 in all vars
  for (i, var) in vars.iter().enumerate() {
    let deg = max_power(&expanded, var);
    match deg {
      Some(d) if d > 1 => return None, // non-linear
      _ => {}
    }
    let terms = collect_additive_terms(&expanded);
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, 1) {
        if let Some(cv) = minimize_try_f64(&c) {
          coeffs[i] += cv;
        } else {
          return None; // non-constant coefficient
        }
      }
    }
  }
  // Constant term: evaluate with all vars set to 0
  let mut const_expr = expanded.clone();
  for var in vars {
    const_expr =
      crate::syntax::substitute_variable(&const_expr, var, &Expr::Integer(0));
  }
  if let Ok(evaled) = crate::evaluator::evaluate_expr_to_expr(&const_expr) {
    constant = minimize_try_f64(&evaled).unwrap_or(0.0);
  }
  // diff = sum(coeffs[i] * vars[i]) + constant >= 0 (for sense 1)
  // So sum(coeffs[i] * vars[i]) >= -constant
  Some((coeffs, -constant, sense))
}

/// Minimize with linear constraints by trying constraint boundaries.
/// For each linear constraint, substitute it as equality into f and minimize the
/// resulting lower-dimensional problem.
fn minimize_constrained_boundary(
  f: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
) -> Result<Option<Expr>, InterpreterError> {
  let n = vars.len();

  // Collect all linear constraints
  let mut lin_cons = Vec::new();
  for con in constraints {
    if let Some(lc) = minimize_extract_linear_constraint(con, vars) {
      lin_cons.push((con.clone(), lc));
    }
  }

  let mut candidates: Vec<(f64, Vec<f64>)> = Vec::new();

  // First try the unconstrained minimum
  if n == 1 {
    if let Ok(result) = minimize_single_var(f, &vars[0], false, "Minimize")
      && let Expr::List(items) = &result
      && items.len() == 2
      && let Some(fval) = minimize_try_f64(&items[0])
      && let Expr::List(rules) = &items[1]
      && let Some(Expr::Rule { replacement, .. }) = rules.first()
      && let Some(xval) = minimize_try_f64(replacement)
    {
      let feasible = minimize_satisfies_constraints(constraints, vars, &[xval]);
      if feasible {
        candidates.push((fval, vec![xval]));
      }
    }
  } else if n == 2
    && let Ok(result) = minimize_multi_var(f, vars, false, "Minimize")
    && let Expr::List(items) = &result
    && items.len() == 2
    && let Some(fval) = minimize_try_f64(&items[0])
    && let Expr::List(rules) = &items[1]
  {
    let mut vals = vec![0.0f64; n];
    let mut all_ok = true;
    for rule in rules {
      if let Expr::Rule {
        pattern,
        replacement,
      } = rule
        && let Expr::Identifier(vname) = pattern.as_ref()
        && let Some(pos) = vars.iter().position(|v| v == vname)
      {
        if let Some(val) = minimize_try_f64(replacement) {
          vals[pos] = val;
        } else {
          all_ok = false;
        }
      }
    }
    if all_ok {
      let feasible = minimize_satisfies_constraints(constraints, vars, &vals);
      if feasible {
        candidates.push((fval, vals));
      }
    }
  }

  // Try each linear constraint as equality boundary
  for (_, (coeffs, rhs, _)) in &lin_cons {
    // Find a variable with non-zero coefficient to eliminate
    let Some(elim_idx) = coeffs.iter().position(|&c| c.abs() > 1e-12) else {
      continue;
    };
    let elim_var = &vars[elim_idx];
    let elim_coeff = coeffs[elim_idx];

    // Solve: coeffs[elim_idx] * elim_var + sum(others) = rhs
    // elim_var = (rhs - sum(others)) / elim_coeff
    // Build expression: (rhs - sum(coeff_j * var_j for j != elim_idx)) / elim_coeff
    let mut elim_expr: Expr = Expr::Real(*rhs);
    for (j, var_j) in vars.iter().enumerate() {
      if j != elim_idx && coeffs[j].abs() > 1e-12 {
        let term = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Real(coeffs[j])),
          right: Box::new(Expr::Identifier(var_j.clone())),
        };
        elim_expr = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(elim_expr),
          right: Box::new(term),
        };
      }
    }
    if (elim_coeff - 1.0).abs() > 1e-12 {
      elim_expr = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(elim_expr),
        right: Box::new(Expr::Real(elim_coeff)),
      };
    }

    // Substitute elim_var = elim_expr in f
    let f_reduced = crate::syntax::substitute_variable(f, elim_var, &elim_expr);
    let f_reduced = simplify(f_reduced);

    // Get remaining variables (all except elim_var)
    let remaining_vars: Vec<String> =
      vars.iter().filter(|v| *v != elim_var).cloned().collect();

    if remaining_vars.is_empty() {
      // All variables eliminated - evaluate f
      if let Ok(fval_expr) = crate::evaluator::evaluate_expr_to_expr(&f_reduced)
        && let Some(fval) = minimize_try_f64(&fval_expr)
      {
        // Get the eliminated variable's value
        let elim_val_expr = crate::evaluator::evaluate_expr_to_expr(&elim_expr)
          .unwrap_or(elim_expr.clone());
        if let Some(elim_val) = minimize_try_f64(&elim_val_expr) {
          let mut vals = vec![0.0f64; n];
          vals[elim_idx] = elim_val;
          if minimize_satisfies_constraints(constraints, vars, &vals) {
            candidates.push((fval, vals));
          }
        }
      }
    } else if remaining_vars.len() == 1 {
      // 1D reduced problem
      let rem_var = &remaining_vars[0];
      let rem_idx = vars.iter().position(|v| v == rem_var).unwrap();

      if let Ok(result) =
        minimize_single_var(&f_reduced, rem_var, false, "Minimize")
        && let Expr::List(items) = &result
        && items.len() == 2
        && let Some(fval) = minimize_try_f64(&items[0])
        && let Expr::List(rules) = &items[1]
        && let Some(Expr::Rule { replacement, .. }) = rules.first()
        && let Some(rem_val) = minimize_try_f64(replacement)
      {
        // Compute elim_var value
        let elim_val_expr = crate::syntax::substitute_variable(
          &elim_expr,
          rem_var,
          &Expr::Real(rem_val),
        );
        if let Ok(evaled) =
          crate::evaluator::evaluate_expr_to_expr(&elim_val_expr)
          && let Some(elim_val) = minimize_try_f64(&evaled)
        {
          let mut vals = vec![0.0f64; n];
          vals[elim_idx] = elim_val;
          vals[rem_idx] = rem_val;
          if minimize_satisfies_constraints(constraints, vars, &vals) {
            candidates.push((fval, vals));
          }
        }
      }
    }
  }

  if candidates.is_empty() {
    return Ok(None);
  }

  // Find minimum candidate
  let best = candidates
    .iter()
    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
    .unwrap();

  let result_fval = if maximize {
    minimize_recognize_exact(-best.0)
  } else {
    minimize_recognize_exact(best.0)
  };

  let rules: Vec<Expr> = vars
    .iter()
    .zip(best.1.iter())
    .map(|(v, &val)| {
      let exact_val = minimize_recognize_exact(val);
      Expr::Rule {
        pattern: Box::new(Expr::Identifier(v.clone())),
        replacement: Box::new(exact_val),
      }
    })
    .collect();

  Ok(Some(Expr::List(vec![result_fval, Expr::List(rules)])))
}

/// Try to solve a 2D linear program by enumerating vertices.
/// Returns None if the problem is not a linear program.
fn minimize_lp_2d(
  f: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
) -> Option<Expr> {
  let (x_name, y_name) = (&vars[0], &vars[1]);

  // Each constraint ax + by >= c or ax + by <= c or ax + by == c
  // We store as (a, b, c, sense) where sense: 1 = >=, -1 = <=, 0 = ==
  let mut linear_cons: Vec<(f64, f64, f64, i32)> = Vec::new();

  for con in constraints {
    let Expr::Comparison {
      operands,
      operators,
    } = con
    else {
      return None;
    };
    if operands.len() != 2 || operators.len() != 1 {
      return None;
    }
    let lhs = &operands[0];
    let rhs = &operands[1];

    use crate::syntax::ComparisonOp;
    let sense = match &operators[0] {
      ComparisonOp::GreaterEqual => 1,
      ComparisonOp::LessEqual => -1,
      ComparisonOp::Greater => 1,
      ComparisonOp::Less => -1,
      ComparisonOp::Equal => 0,
      _ => return None,
    };

    // Extract coefficients from lhs - rhs as linear function of vars
    let diff = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(lhs.clone()),
      right: Box::new(rhs.clone()),
    };
    let expanded = expand_and_combine(&diff);
    let terms = collect_additive_terms(&expanded);

    let mut ax = 0.0f64;
    let mut ay = 0.0f64;
    let mut ac = 0.0f64;

    for term in &terms {
      let cx = extract_coefficient_of_power(term, x_name, 1);
      let cy = extract_coefficient_of_power(term, y_name, 1);

      if let Some(ref c) = cx {
        ax += minimize_try_f64(c)?;
      }
      if let Some(ref c) = cy {
        ay += minimize_try_f64(c)?;
      }
      // Constant term: coefficient of x^0 that doesn't contain y
      if cx.is_none() && cy.is_none() {
        ac += minimize_try_f64(term)?;
      }
    }

    linear_cons.push((ax, ay, -ac, sense));
  }

  // Extract linear objective coefficients
  let expanded_f = expand_and_combine(f);
  let terms_f = collect_additive_terms(&expanded_f);
  let mut fx = 0.0f64;
  let mut fy = 0.0f64;
  let mut fc = 0.0f64;
  for term in &terms_f {
    let cx = extract_coefficient_of_power(term, x_name, 1);
    let cy = extract_coefficient_of_power(term, y_name, 1);
    if let Some(ref c) = cx {
      fx += minimize_try_f64(c)?;
    }
    if let Some(ref c) = cy {
      fy += minimize_try_f64(c)?;
    }
    if cx.is_none() && cy.is_none() {
      fc += minimize_try_f64(term)?;
    }
  }

  // Enumerate vertices: intersections of all pairs of constraint lines
  let mut vertices: Vec<(f64, f64)> = Vec::new();

  // Add intersections of pairs of constraints (treating each as equality)
  for i in 0..linear_cons.len() {
    for j in (i + 1)..linear_cons.len() {
      let (a1, b1, c1, _) = linear_cons[i];
      let (a2, b2, c2, _) = linear_cons[j];
      let det = a1 * b2 - a2 * b1;
      if det.abs() < 1e-12 {
        continue;
      }
      let xv = (c1 * b2 - c2 * b1) / det;
      let yv = (a1 * c2 - a2 * c1) / det;
      // Check feasibility
      let feasible = linear_cons.iter().all(|&(a, b, c, sense)| {
        let val = a * xv + b * yv - c;
        match sense {
          1 => val >= -1e-8,
          -1 => val <= 1e-8,
          0 => val.abs() <= 1e-8,
          _ => true,
        }
      });
      if feasible {
        vertices.push((xv, yv));
      }
    }
  }

  if vertices.is_empty() {
    return None;
  }

  // Find the vertex that minimizes the objective
  let mut best_val = f64::INFINITY;
  let mut best_vertex = vertices[0];

  for &(xv, yv) in &vertices {
    let val = fx * xv + fy * yv + fc;
    if val < best_val {
      best_val = val;
      best_vertex = (xv, yv);
    }
  }

  // Try to make exact values from approximate
  let make_exact = |v: f64| -> Expr {
    let rounded = v.round();
    if (rounded - v).abs() < 1e-8 {
      Expr::Integer(rounded as i128)
    } else {
      // Check if v = p/q for small q
      for q in 1i128..=10 {
        let p = (v * q as f64).round() as i128;
        if ((p as f64 / q as f64) - v).abs() < 1e-8 {
          let (rn, rd) = reduce_fraction(p, q);
          return if rd == 1 {
            Expr::Integer(rn)
          } else {
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(rn), Expr::Integer(rd)],
            }
          };
        }
      }
      Expr::Real(v)
    }
  };

  // Also try to make the objective value exact
  let result_val = {
    let v = if maximize { -best_val } else { best_val };
    let rounded = v.round();
    if (rounded - v).abs() < 1e-8 {
      Expr::Integer(rounded as i128)
    } else {
      for q in 1i128..=10 {
        let p = (v * q as f64).round() as i128;
        if ((p as f64 / q as f64) - v).abs() < 1e-8 {
          let (rn, rd) = reduce_fraction(p, q);
          let e = if rd == 1 {
            Expr::Integer(rn)
          } else {
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(rn), Expr::Integer(rd)],
            }
          };
          return Some(Expr::List(vec![
            e,
            Expr::List(vec![
              Expr::Rule {
                pattern: Box::new(Expr::Identifier(x_name.clone())),
                replacement: Box::new(make_exact(best_vertex.0)),
              },
              Expr::Rule {
                pattern: Box::new(Expr::Identifier(y_name.clone())),
                replacement: Box::new(make_exact(best_vertex.1)),
              },
            ]),
          ]));
        }
      }
      Expr::Real(v)
    }
  };

  Some(Expr::List(vec![
    result_val,
    Expr::List(vec![
      Expr::Rule {
        pattern: Box::new(Expr::Identifier(x_name.clone())),
        replacement: Box::new(make_exact(best_vertex.0)),
      },
      Expr::Rule {
        pattern: Box::new(Expr::Identifier(y_name.clone())),
        replacement: Box::new(make_exact(best_vertex.1)),
      },
    ]),
  ]))
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
