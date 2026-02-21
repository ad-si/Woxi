#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

use crate::functions::calculus_ast::{is_constant_wrt, simplify};

// ─── Reduce ──────────────────────────────────────────────────────────

/// Reduce[expr, var] or Reduce[expr, {vars}] or Reduce[expr, vars, domain]
///
/// Reduces equations and inequalities to a canonical disjunctive form.
pub fn reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Reduce expects 2 or 3 arguments".into(),
    ));
  }

  let expr = &args[0];
  let domain = if args.len() == 3 {
    match &args[2] {
      Expr::Identifier(d) => Some(d.as_str()),
      _ => None,
    }
  } else {
    None
  };

  // Extract variable names
  let vars = extract_reduce_vars(&args[1]);
  if vars.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Reduce".to_string(),
      args: args.to_vec(),
    });
  }

  let result = reduce_expr(expr, &vars, domain)?;
  Ok(result)
}

/// Extract variable names from the second argument of Reduce.
pub fn extract_reduce_vars(expr: &Expr) -> Vec<String> {
  match expr {
    Expr::Identifier(name) => {
      // Check it's not a constant
      let is_constant = crate::FUNC_ATTRS.with(|m| {
        m.borrow()
          .get(name.as_str())
          .is_some_and(|attrs| attrs.contains(&"Constant".to_string()))
      });
      if is_constant
        || matches!(
          name.as_str(),
          "E" | "Pi" | "I" | "True" | "False" | "Infinity"
        )
      {
        vec![]
      } else {
        vec![name.clone()]
      }
    }
    Expr::Constant(_) => vec![],
    Expr::List(items) => items.iter().flat_map(extract_reduce_vars).collect(),
    _ => vec![],
  }
}

/// Core reduction logic.
pub fn reduce_expr(
  expr: &Expr,
  vars: &[String],
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Handle True/False
  if let Expr::Identifier(s) = expr {
    if s == "True" {
      return Ok(if domain == Some("Reals") || domain == Some("Integers") {
        Expr::Identifier("True".to_string())
      } else {
        Expr::Identifier("True".to_string())
      });
    }
    if s == "False" {
      return Ok(Expr::Identifier("False".to_string()));
    }
  }

  // Handle List of constraints → convert to And
  if let Expr::List(items) = expr {
    if items.is_empty() {
      return Ok(Expr::Identifier("True".to_string()));
    }
    let and_expr =
      items
        .iter()
        .skip(1)
        .fold(items[0].clone(), |acc, item| Expr::BinaryOp {
          op: BinaryOperator::And,
          left: Box::new(acc),
          right: Box::new(item.clone()),
        });
    return reduce_expr(&and_expr, vars, domain);
  }

  // Handle Or (disjunction)
  if let Expr::BinaryOp {
    op: BinaryOperator::Or,
    left,
    right,
  } = expr
  {
    let left_result = reduce_expr(left, vars, domain)?;
    let right_result = reduce_expr(right, vars, domain)?;
    return Ok(or_results(&left_result, &right_result));
  }

  // Handle FunctionCall Or
  if let Expr::FunctionCall { name, args: fargs } = expr {
    if name == "Or" {
      let mut result = Expr::Identifier("False".to_string());
      for a in fargs {
        let r = reduce_expr(a, vars, domain)?;
        result = or_results(&result, &r);
      }
      return Ok(result);
    }
    if name == "And" && !fargs.is_empty() {
      if fargs.len() == 1 {
        return reduce_expr(&fargs[0], vars, domain);
      }
      let combined =
        fargs
          .iter()
          .skip(1)
          .fold(fargs[0].clone(), |acc, a| Expr::BinaryOp {
            op: BinaryOperator::And,
            left: Box::new(acc),
            right: Box::new(a.clone()),
          });
      return reduce_expr(&combined, vars, domain);
    }
  }

  // Handle And (conjunction)
  if let Expr::BinaryOp {
    op: BinaryOperator::And,
    left,
    right,
  } = expr
  {
    return reduce_and(left, right, vars, domain);
  }

  // Single constraint
  if vars.len() == 1 {
    return reduce_single_var(expr, &vars[0], domain);
  }

  // Multi-variable: try sequential elimination
  reduce_multi_var(expr, vars, domain)
}

/// Reduce a single constraint in one variable.
pub fn reduce_single_var(
  expr: &Expr,
  var: &str,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Check if this is an equation (==)
  if let Some((lhs, rhs, op)) = extract_comparison(expr) {
    match op {
      CompOp::Equal => {
        return reduce_equation(&lhs, &rhs, var, domain);
      }
      CompOp::NotEqual => {
        return reduce_not_equal(&lhs, &rhs, var, domain);
      }
      CompOp::Less
      | CompOp::LessEqual
      | CompOp::Greater
      | CompOp::GreaterEqual => {
        return reduce_inequality(&lhs, &rhs, op, var, domain);
      }
    }
  }

  // If expression is already a simple passthrough (like x > 0)
  Ok(Expr::FunctionCall {
    name: "Reduce".to_string(),
    args: vec![expr.clone(), Expr::Identifier(var.to_string())],
  })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompOp {
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
}

/// Extract a comparison: returns (lhs, rhs, op)
pub fn extract_comparison(expr: &Expr) -> Option<(Expr, Expr, CompOp)> {
  match expr {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => {
      let op = match operators[0] {
        crate::syntax::ComparisonOp::Equal => CompOp::Equal,
        crate::syntax::ComparisonOp::NotEqual => CompOp::NotEqual,
        crate::syntax::ComparisonOp::Less => CompOp::Less,
        crate::syntax::ComparisonOp::LessEqual => CompOp::LessEqual,
        crate::syntax::ComparisonOp::Greater => CompOp::Greater,
        crate::syntax::ComparisonOp::GreaterEqual => CompOp::GreaterEqual,
        _ => return None,
      };
      Some((operands[0].clone(), operands[1].clone(), op))
    }
    Expr::FunctionCall { name, args } if args.len() == 2 => {
      let op = match name.as_str() {
        "Equal" => CompOp::Equal,
        "Unequal" => CompOp::NotEqual,
        "Less" => CompOp::Less,
        "LessEqual" => CompOp::LessEqual,
        "Greater" => CompOp::Greater,
        "GreaterEqual" => CompOp::GreaterEqual,
        _ => return None,
      };
      Some((args[0].clone(), args[1].clone(), op))
    }
    _ => None,
  }
}

/// Reduce an equation lhs == rhs for variable var.
pub fn reduce_equation(
  lhs: &Expr,
  rhs: &Expr,
  var: &str,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Build lhs - rhs
  let poly = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(lhs.clone()),
    right: Box::new(rhs.clone()),
  };

  let expanded = expand_and_combine(&poly);

  // Check if variable is present
  if is_constant_wrt(&expanded, var) {
    // No variable — check if constant is zero
    let simplified = simplify(expanded);
    return Ok(if matches!(simplified, Expr::Integer(0)) {
      Expr::Identifier("True".to_string())
    } else {
      Expr::Identifier("False".to_string())
    });
  }

  // Use the Solve infrastructure to find roots
  let eq_expr = Expr::Comparison {
    operands: vec![lhs.clone(), rhs.clone()],
    operators: vec![crate::syntax::ComparisonOp::Equal],
  };
  let solve_result = solve_ast(&[eq_expr, Expr::Identifier(var.to_string())])?;

  // Convert Solve result (list of rules) to Or of equalities
  if let Expr::List(solutions) = &solve_result {
    if solutions.is_empty() {
      return Ok(Expr::Identifier("False".to_string()));
    }

    let mut equalities: Vec<Expr> = Vec::new();
    for sol in solutions {
      if let Expr::List(rules) = sol {
        if rules.is_empty() {
          // Solve returned {{}} meaning all values are solutions
          return Ok(Expr::Identifier("True".to_string()));
        }
        for rule in rules {
          if let Expr::Rule {
            pattern: _,
            replacement,
          } = rule
          {
            let value = replacement.as_ref().clone();

            // Domain filtering
            if let Some(dom) = domain
              && !is_in_domain(&value, dom)
            {
              continue;
            }

            equalities
              .push(make_equality(&Expr::Identifier(var.to_string()), &value));
          }
        }
      }
    }

    if equalities.is_empty() {
      return Ok(Expr::Identifier("False".to_string()));
    }

    // Sort equalities for canonical output
    equalities.sort_by(compare_exprs);

    // Deduplicate
    equalities.dedup_by(|a, b| expr_to_string(a) == expr_to_string(b));

    return Ok(build_or(equalities));
  }

  // Solve returned unevaluated — try factoring
  let factored = factor_ast(&[expanded.clone()])?;
  let factors = collect_multiplicative_factors(&factored);
  if factors.len() > 1 {
    // Each factor == 0 gives solutions
    let mut all_equalities: Vec<Expr> = Vec::new();
    for factor in &factors {
      if is_constant_wrt(factor, var) {
        continue; // Skip constant factors
      }
      let sub_result = reduce_equation(factor, &Expr::Integer(0), var, domain)?;
      let sub_terms = collect_or_terms(&sub_result);
      for t in sub_terms {
        if !matches!(&t, Expr::Identifier(s) if s == "False") {
          all_equalities.push(t);
        }
      }
    }
    if !all_equalities.is_empty() {
      all_equalities.sort_by(compare_exprs);
      all_equalities.dedup_by(|a, b| expr_to_string(a) == expr_to_string(b));
      return Ok(build_or(all_equalities));
    }
  }

  // Return Reduce unevaluated
  Ok(Expr::FunctionCall {
    name: "Reduce".to_string(),
    args: if let Some(dom) = domain {
      vec![
        make_equality(lhs, rhs),
        Expr::Identifier(var.to_string()),
        Expr::Identifier(dom.to_string()),
      ]
    } else {
      vec![make_equality(lhs, rhs), Expr::Identifier(var.to_string())]
    },
  })
}

/// Reduce a != b for variable var.
pub fn reduce_not_equal(
  lhs: &Expr,
  rhs: &Expr,
  var: &str,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // For simple cases, just return the inequality
  let poly = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(lhs.clone()),
    right: Box::new(rhs.clone()),
  };
  let expanded = expand_and_combine(&poly);
  if is_constant_wrt(&expanded, var) {
    let simplified = simplify(expanded);
    return Ok(if matches!(simplified, Expr::Integer(0)) {
      Expr::Identifier("False".to_string())
    } else {
      Expr::Identifier("True".to_string())
    });
  }
  // Return the not-equal condition as-is for now
  let _ = domain;
  Ok(Expr::Comparison {
    operands: vec![lhs.clone(), rhs.clone()],
    operators: vec![crate::syntax::ComparisonOp::NotEqual],
  })
}

/// Reduce an inequality (lhs op rhs) for one variable.
pub fn reduce_inequality(
  lhs: &Expr,
  rhs: &Expr,
  op: CompOp,
  var: &str,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Move everything to one side: lhs - rhs op 0
  let poly = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(lhs.clone()),
    right: Box::new(rhs.clone()),
  };
  let expanded = expand_and_combine(&poly);

  // Check if variable is present
  if is_constant_wrt(&expanded, var) {
    // No variable — evaluate the comparison numerically
    let val = simplify(expanded);
    let result = evaluate_constant_ineq(&val, op);
    return Ok(result);
  }

  // Try to find the degree
  let degree = max_power(&expanded, var);

  match degree {
    Some(1) => reduce_linear_inequality(&expanded, var, op, domain),
    Some(2) => reduce_quadratic_inequality(&expanded, var, op, domain),
    _ => {
      // Try to factor and handle
      let factored =
        try_factor_and_reduce_inequality(lhs, rhs, op, var, domain);
      if let Some(result) = factored {
        return Ok(result);
      }
      // Return unevaluated
      Ok(Expr::FunctionCall {
        name: "Reduce".to_string(),
        args: vec![
          make_comparison(lhs, rhs, op),
          Expr::Identifier(var.to_string()),
        ],
      })
    }
  }
}

/// Evaluate a constant inequality (no variable present).
pub fn evaluate_constant_ineq(val: &Expr, op: CompOp) -> Expr {
  // Try to get a numeric value
  if let Some(n) = expr_to_number(val) {
    let result = match op {
      CompOp::Less => n < 0.0,
      CompOp::LessEqual => n <= 0.0,
      CompOp::Greater => n > 0.0,
      CompOp::GreaterEqual => n >= 0.0,
      CompOp::Equal => n == 0.0,
      CompOp::NotEqual => n != 0.0,
    };
    return Expr::Identifier(if result { "True" } else { "False" }.to_string());
  }
  // Can't evaluate
  Expr::Identifier("True".to_string())
}

/// Try to extract a numeric value from an expression.
pub fn expr_to_number(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(r) => Some(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some(*n as f64 / *d as f64)
      } else {
        None
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => expr_to_number(operand).map(|n| -n),
    _ => None,
  }
}

/// Reduce a linear inequality (a*x + b op 0)
pub fn reduce_linear_inequality(
  poly: &Expr,
  var: &str,
  op: CompOp,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  let terms = collect_additive_terms(poly);
  let mut a_parts: Vec<Expr> = Vec::new();
  let mut b_parts: Vec<Expr> = Vec::new();

  for term in &terms {
    if let Some(c) = extract_coefficient_of_power(term, var, 1) {
      a_parts.push(c);
    }
    if let Some(c) = extract_coefficient_of_power(term, var, 0) {
      b_parts.push(c);
    }
  }

  let a = simplify(build_sum_from_parts(&a_parts));
  let b = simplify(build_sum_from_parts(&b_parts));

  // a*x + b op 0 → x op' -b/a
  // Sign of a determines if inequality flips
  if let Expr::Integer(ai) = &a {
    let neg_b = negate_expr(&b);
    let bound = simplify(solve_divide(&neg_b, &a));

    let final_op = if *ai < 0 { flip_op(op) } else { op };

    let _ = domain;
    return Ok(make_comparison(
      &Expr::Identifier(var.to_string()),
      &bound,
      final_op,
    ));
  }

  // Symbolic coefficient — can't determine sign, return unevaluated
  Ok(Expr::FunctionCall {
    name: "Reduce".to_string(),
    args: vec![
      make_comparison(
        &Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(a),
            right: Box::new(Expr::Identifier(var.to_string())),
          }),
          right: Box::new(b),
        },
        &Expr::Integer(0),
        op,
      ),
      Expr::Identifier(var.to_string()),
    ],
  })
}

/// Reduce a quadratic inequality (a*x^2 + b*x + c op 0)
pub fn reduce_quadratic_inequality(
  poly: &Expr,
  var: &str,
  op: CompOp,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  let terms = collect_additive_terms(poly);
  let mut coeffs: Vec<Expr> = Vec::new();
  for d in 0..=2 {
    let mut coeff_sum: Vec<Expr> = Vec::new();
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, d) {
        coeff_sum.push(c);
      }
    }
    coeffs.push(simplify(build_sum_from_parts(&coeff_sum)));
  }

  let c = &coeffs[0];
  let b = &coeffs[1];
  let a = &coeffs[2];

  // Need integer coefficients for now
  if let (Expr::Integer(ai), Expr::Integer(bi), Expr::Integer(ci)) = (a, b, c) {
    let ai = *ai;
    let bi = *bi;
    let ci = *ci;
    let disc = bi * bi - 4 * ai * ci;

    if disc < 0 {
      // No real roots
      // If a > 0: polynomial is always positive
      // If a < 0: polynomial is always negative
      let always_positive = ai > 0;
      let result = match op {
        CompOp::Greater => always_positive,
        CompOp::GreaterEqual => always_positive,
        CompOp::Less => !always_positive,
        CompOp::LessEqual => !always_positive,
        _ => false,
      };
      return Ok(if result {
        // Always true over reals
        if domain == Some("Reals") {
          Expr::Identifier("True".to_string())
        } else {
          Expr::FunctionCall {
            name: "Element".to_string(),
            args: vec![
              Expr::Identifier(var.to_string()),
              Expr::Identifier("Reals".to_string()),
            ],
          }
        }
      } else {
        Expr::Identifier("False".to_string())
      });
    }

    if disc == 0 {
      // One repeated root: r = -b/(2a)
      let root = solve_divide(&Expr::Integer(-bi), &Expr::Integer(2 * ai));
      // If a > 0: poly = a*(x - r)^2 >= 0, equals 0 at r
      let _ = domain;
      return match op {
        CompOp::Equal => {
          Ok(make_equality(&Expr::Identifier(var.to_string()), &root))
        }
        CompOp::GreaterEqual if ai > 0 => {
          if domain == Some("Reals") {
            Ok(Expr::Identifier("True".to_string()))
          } else {
            Ok(Expr::FunctionCall {
              name: "Element".to_string(),
              args: vec![
                Expr::Identifier(var.to_string()),
                Expr::Identifier("Reals".to_string()),
              ],
            })
          }
        }
        CompOp::Greater if ai > 0 => Ok(Expr::Comparison {
          operands: vec![Expr::Identifier(var.to_string()), root],
          operators: vec![crate::syntax::ComparisonOp::NotEqual],
        }),
        CompOp::LessEqual if ai > 0 => {
          Ok(make_equality(&Expr::Identifier(var.to_string()), &root))
        }
        CompOp::Less if ai > 0 => Ok(Expr::Identifier("False".to_string())),
        CompOp::LessEqual if ai < 0 => {
          if domain == Some("Reals") {
            Ok(Expr::Identifier("True".to_string()))
          } else {
            Ok(Expr::FunctionCall {
              name: "Element".to_string(),
              args: vec![
                Expr::Identifier(var.to_string()),
                Expr::Identifier("Reals".to_string()),
              ],
            })
          }
        }
        CompOp::Less if ai < 0 => Ok(Expr::Comparison {
          operands: vec![Expr::Identifier(var.to_string()), root],
          operators: vec![crate::syntax::ComparisonOp::NotEqual],
        }),
        CompOp::GreaterEqual if ai < 0 => {
          Ok(make_equality(&Expr::Identifier(var.to_string()), &root))
        }
        CompOp::Greater if ai < 0 => Ok(Expr::Identifier("False".to_string())),
        _ => Ok(Expr::Identifier("False".to_string())),
      };
    }

    // Two distinct roots
    let (sqrt_out, sqrt_in) = simplify_sqrt_parts(disc);
    let (r1, r2) = if sqrt_in == 1 {
      // Rational roots
      let root1 =
        solve_divide(&Expr::Integer(-bi - sqrt_out), &Expr::Integer(2 * ai));
      let root2 =
        solve_divide(&Expr::Integer(-bi + sqrt_out), &Expr::Integer(2 * ai));
      order_roots(root1, root2)
    } else {
      // Irrational roots
      let root1 = make_quadratic_root(-bi, -sqrt_out, sqrt_in, 2 * ai);
      let root2 = make_quadratic_root(-bi, sqrt_out, sqrt_in, 2 * ai);
      order_roots(root1, root2)
    };

    // For a > 0: poly > 0 when x < r1 or x > r2
    //            poly < 0 when r1 < x < r2
    // For a < 0: reversed
    let positive_outside = ai > 0;
    let _ = domain;

    match (op, positive_outside) {
      (CompOp::Greater, true) | (CompOp::Less, false) => {
        // x < r1 || x > r2
        Ok(build_or(vec![
          make_comparison(
            &Expr::Identifier(var.to_string()),
            &r1,
            CompOp::Less,
          ),
          make_comparison(
            &Expr::Identifier(var.to_string()),
            &r2,
            CompOp::Greater,
          ),
        ]))
      }
      (CompOp::GreaterEqual, true) | (CompOp::LessEqual, false) => {
        // x <= r1 || x >= r2
        Ok(build_or(vec![
          make_comparison(
            &Expr::Identifier(var.to_string()),
            &r1,
            CompOp::LessEqual,
          ),
          make_comparison(
            &Expr::Identifier(var.to_string()),
            &r2,
            CompOp::GreaterEqual,
          ),
        ]))
      }
      (CompOp::Less, true) | (CompOp::Greater, false) => {
        // r1 < x < r2 → Inequality[r1, Less, x, Less, r2]
        Ok(make_compound_inequality(
          &r1,
          CompOp::Less,
          var,
          CompOp::Less,
          &r2,
        ))
      }
      (CompOp::LessEqual, true) | (CompOp::GreaterEqual, false) => {
        // r1 <= x <= r2 → Inequality[r1, LessEqual, x, LessEqual, r2]
        Ok(make_compound_inequality(
          &r1,
          CompOp::LessEqual,
          var,
          CompOp::LessEqual,
          &r2,
        ))
      }
      _ => Ok(Expr::Identifier("False".to_string())),
    }
  } else {
    // Non-integer coefficients — return unevaluated
    Ok(Expr::FunctionCall {
      name: "Reduce".to_string(),
      args: vec![
        Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(poly.clone()),
          right: Box::new(Expr::Integer(0)),
        },
        Expr::Identifier(var.to_string()),
      ],
    })
  }
}

/// Make a quadratic root expression: (nb + so*Sqrt[si]) / den
pub fn make_quadratic_root(nb: i128, so: i128, si: i128, den: i128) -> Expr {
  let g = gcd_i128(gcd_i128(nb, so).abs(), den.abs()).abs();
  let nb = nb / g;
  let so = so / g;
  let den = den / g;
  let (nb, so, den) = if den < 0 {
    (-nb, -so, -den)
  } else {
    (nb, so, den)
  };

  let sqrt_part = if si == 1 {
    Expr::Integer(1)
  } else if so == 1 {
    Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![Expr::Integer(si)],
    }
  } else if so == -1 {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(si)],
      }),
    }
  } else {
    multiply_exprs(
      &Expr::Integer(so),
      &Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(si)],
      },
    )
  };

  let num = if nb == 0 {
    sqrt_part
  } else {
    add_exprs(&Expr::Integer(nb), &sqrt_part)
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
}

/// Order two roots so that the smaller one comes first.
pub fn order_roots(r1: Expr, r2: Expr) -> (Expr, Expr) {
  if let (Some(v1), Some(v2)) = (expr_to_number(&r1), expr_to_number(&r2)) {
    if v1 <= v2 { (r1, r2) } else { (r2, r1) }
  } else {
    // Try evaluating symbolically
    let s1 = expr_to_string(&r1);
    let s2 = expr_to_string(&r2);
    // For expressions like -Sqrt[3] and Sqrt[3], the negative one should come first
    if s1.starts_with('-') && !s2.starts_with('-') {
      (r1, r2)
    } else if !s1.starts_with('-') && s2.starts_with('-') {
      (r2, r1)
    } else {
      (r1, r2)
    }
  }
}

/// Try to factor and solve higher-degree polynomial inequalities.
pub fn try_factor_and_reduce_inequality(
  lhs: &Expr,
  rhs: &Expr,
  op: CompOp,
  var: &str,
  domain: Option<&str>,
) -> Option<Expr> {
  // Try using Solve to find roots, then determine sign intervals
  let eq = Expr::Comparison {
    operands: vec![lhs.clone(), rhs.clone()],
    operators: vec![crate::syntax::ComparisonOp::Equal],
  };
  let solve_result =
    solve_ast(&[eq, Expr::Identifier(var.to_string())]).ok()?;

  if let Expr::List(solutions) = &solve_result {
    let mut roots: Vec<(Expr, f64)> = Vec::new();
    for sol in solutions {
      if let Expr::List(rules) = sol {
        for rule in rules {
          if let Expr::Rule { replacement, .. } = rule
            && let Some(val) = expr_to_number(replacement)
          {
            roots.push((replacement.as_ref().clone(), val));
          }
        }
      }
    }

    if roots.is_empty() {
      return None;
    }

    // Sort roots
    roots.sort_by(|a, b| {
      a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate
    roots.dedup_by(|a, b| (a.1 - b.1).abs() < 1e-12);

    // Filter by domain
    if let Some(dom) = domain {
      roots.retain(|(expr, _)| is_in_domain(expr, dom));
    }

    // Determine the sign in each interval by testing a point
    let poly = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(lhs.clone()),
      right: Box::new(rhs.clone()),
    };

    let mut intervals: Vec<Expr> = Vec::new();

    // Test point before first root
    if !roots.is_empty() {
      let test_x = roots[0].1 - 1.0;
      let sign = eval_poly_at(&poly, var, test_x);
      if sign_matches(sign, op) {
        intervals.push(make_comparison(
          &Expr::Identifier(var.to_string()),
          &roots[0].0,
          CompOp::Less,
        ));
      }
    }

    // Test at roots (for <= and >=)
    if matches!(op, CompOp::LessEqual | CompOp::GreaterEqual | CompOp::Equal) {
      for (root_expr, _) in &roots {
        intervals
          .push(make_equality(&Expr::Identifier(var.to_string()), root_expr));
      }
    }

    // Test between roots
    for i in 0..roots.len().saturating_sub(1) {
      let test_x = (roots[i].1 + roots[i + 1].1) / 2.0;
      let sign = eval_poly_at(&poly, var, test_x);
      if sign_matches(sign, op) {
        let ineq_op = if matches!(op, CompOp::LessEqual | CompOp::GreaterEqual)
        {
          // Use matching inclusive operators
          match op {
            CompOp::LessEqual => CompOp::LessEqual,
            CompOp::GreaterEqual => CompOp::GreaterEqual,
            _ => CompOp::Less,
          }
        } else {
          CompOp::Less
        };
        intervals.push(make_compound_inequality(
          &roots[i].0,
          ineq_op,
          var,
          ineq_op,
          &roots[i + 1].0,
        ));
      }
    }

    // Test point after last root
    if !roots.is_empty() {
      let test_x = roots.last().unwrap().1 + 1.0;
      let sign = eval_poly_at(&poly, var, test_x);
      if sign_matches(sign, op) {
        intervals.push(make_comparison(
          &Expr::Identifier(var.to_string()),
          &roots.last().unwrap().0,
          CompOp::Greater,
        ));
      }
    }

    // For strict inequalities with "between roots" intervals, clean up
    // by merging equality points into the intervals
    if matches!(op, CompOp::LessEqual | CompOp::GreaterEqual) {
      // Already handled by the inclusive interval construction
      // Remove standalone equalities that are covered by intervals
      let mut cleaned: Vec<Expr> = Vec::new();
      for interval in &intervals {
        if let Some((_, _, CompOp::Equal)) = extract_comparison(interval) {
          // Skip standalone equalities that are part of an interval
          continue;
        }
        cleaned.push(interval.clone());
      }
      if !cleaned.is_empty() {
        return Some(build_or(cleaned));
      }
    }

    if intervals.is_empty() {
      return Some(Expr::Identifier("False".to_string()));
    }

    return Some(build_or(intervals));
  }

  None
}

/// Evaluate a polynomial at a specific point.
pub fn eval_poly_at(poly: &Expr, var: &str, x: f64) -> f64 {
  let substituted =
    crate::syntax::substitute_variable(poly, var, &Expr::Real(x));
  let simplified = simplify(substituted);
  expr_to_number(&simplified).unwrap_or(0.0)
}

/// Check if a sign value matches the comparison operator (compared to 0).
pub fn sign_matches(val: f64, op: CompOp) -> bool {
  match op {
    CompOp::Less => val < 0.0,
    CompOp::LessEqual => val <= 0.0,
    CompOp::Greater => val > 0.0,
    CompOp::GreaterEqual => val >= 0.0,
    CompOp::Equal => val.abs() < 1e-12,
    CompOp::NotEqual => val.abs() >= 1e-12,
  }
}

/// Handle And conjunction of two constraints.
pub fn reduce_and(
  left: &Expr,
  right: &Expr,
  vars: &[String],
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Flatten And chains
  let mut constraints = Vec::new();
  collect_and_constraints(left, &mut constraints);
  collect_and_constraints(right, &mut constraints);

  // Remove True constraints (identity for And)
  constraints.retain(|c| !matches!(c, Expr::Identifier(s) if s == "True"));

  // Check for False
  if constraints
    .iter()
    .any(|c| matches!(c, Expr::Identifier(s) if s == "False"))
  {
    return Ok(Expr::Identifier("False".to_string()));
  }

  if constraints.is_empty() {
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Separate equations from inequalities
  let mut equations: Vec<(Expr, Expr)> = Vec::new(); // (lhs, rhs) for ==
  let mut inequalities: Vec<Expr> = Vec::new();
  let mut other: Vec<Expr> = Vec::new();

  for c in &constraints {
    if let Some((lhs, rhs, CompOp::Equal)) = extract_comparison(c) {
      equations.push((lhs, rhs));
    } else if let Some((_, _, _)) = extract_comparison(c) {
      inequalities.push(c.clone());
    } else {
      other.push(c.clone());
    }
  }

  if vars.len() == 1 {
    let var = &vars[0];

    // If we have equations, solve them first and check inequalities
    if !equations.is_empty() {
      let eq = &equations[0];
      let eq_result = reduce_equation(&eq.0, &eq.1, var, domain)?;

      if matches!(&eq_result, Expr::Identifier(s) if s == "False") {
        return Ok(Expr::Identifier("False".to_string()));
      }
      if matches!(&eq_result, Expr::Identifier(s) if s == "True") {
        // Equation is trivially true, reduce remaining
        let remaining: Vec<Expr> = equations[1..]
          .iter()
          .map(|(l, r)| make_equality(l, r))
          .chain(inequalities.iter().cloned())
          .chain(other.iter().cloned())
          .collect();
        if remaining.is_empty() {
          return Ok(Expr::Identifier("True".to_string()));
        }
        let combined =
          remaining
            .iter()
            .skip(1)
            .fold(remaining[0].clone(), |acc, c| Expr::BinaryOp {
              op: BinaryOperator::And,
              left: Box::new(acc),
              right: Box::new(c.clone()),
            });
        return reduce_expr(&combined, vars, domain);
      }

      // Equation gives concrete solutions — filter by remaining constraints
      let solutions = collect_or_terms(&eq_result);
      let mut valid_solutions: Vec<Expr> = Vec::new();

      for sol in &solutions {
        if let Some((_, rhs_val, CompOp::Equal)) = extract_comparison(sol) {
          // Check if this value satisfies all other constraints
          let mut satisfies = true;

          // Check remaining equations
          for other_eq in &equations[1..] {
            let subst_lhs =
              crate::syntax::substitute_variable(&other_eq.0, var, &rhs_val);
            let subst_rhs =
              crate::syntax::substitute_variable(&other_eq.1, var, &rhs_val);
            let diff = simplify(Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(simplify(subst_lhs)),
              right: Box::new(simplify(subst_rhs)),
            });
            if !matches!(diff, Expr::Integer(0))
              && let Some(n) = expr_to_number(&diff)
              && n.abs() > 1e-12
            {
              satisfies = false;
              break;
            }
          }

          // Check inequalities
          if satisfies {
            for ineq in &inequalities {
              let subst =
                crate::syntax::substitute_variable(ineq, var, &rhs_val);
              let evaled = crate::evaluator::evaluate_expr_to_expr(&subst);
              if let Ok(result) = evaled
                && matches!(&result, Expr::Identifier(s) if s == "False")
              {
                satisfies = false;
                break;
              }
            }
          }

          if satisfies {
            valid_solutions.push(sol.clone());
          }
        }
      }

      if valid_solutions.is_empty() {
        return Ok(Expr::Identifier("False".to_string()));
      }
      return Ok(build_or(valid_solutions));
    }

    // Only inequalities — try to combine them
    if !inequalities.is_empty() && other.is_empty() {
      return reduce_combined_inequalities(&inequalities, var, domain);
    }
  } else {
    // Multi-variable And: try to solve equations for each variable
    return reduce_multi_var_and(&constraints, vars, domain);
  }

  // Fallback: reduce each side and combine
  let left_r = reduce_expr(left, vars, domain)?;
  let right_r = reduce_expr(right, vars, domain)?;
  Ok(and_results(&left_r, &right_r))
}

/// Collect all And-connected terms.
pub fn collect_and_constraints(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => {
      collect_and_constraints(left, out);
      collect_and_constraints(right, out);
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      for a in args {
        collect_and_constraints(a, out);
      }
    }
    _ => out.push(expr.clone()),
  }
}

/// Collect equalities (var == val) from an And expression.
pub fn collect_and_equalities(expr: &Expr) -> Vec<(String, Expr)> {
  let mut result = Vec::new();
  let mut constraints = Vec::new();
  collect_and_constraints(expr, &mut constraints);
  for c in &constraints {
    if let Some((lhs, rhs, CompOp::Equal)) = extract_comparison(c)
      && let Expr::Identifier(name) = &lhs
    {
      result.push((name.clone(), rhs));
    }
  }
  result
}

/// Collect all Or-connected terms.
pub fn collect_or_terms(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Or,
      left,
      right,
    } => {
      let mut terms = collect_or_terms(left);
      terms.extend(collect_or_terms(right));
      terms
    }
    Expr::FunctionCall { name, args } if name == "Or" => {
      args.iter().flat_map(collect_or_terms).collect()
    }
    _ => vec![expr.clone()],
  }
}

/// Try to combine multiple inequalities on the same variable.
pub fn reduce_combined_inequalities(
  ineqs: &[Expr],
  var: &str,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Reduce each inequality individually first
  let mut reduced: Vec<Expr> = Vec::new();
  for ineq in ineqs {
    let r = reduce_single_var(ineq, var, domain)?;
    reduced.push(r);
  }

  // Try to find bounds: lower < x < upper
  let mut lower_bound: Option<(Expr, bool)> = None; // (value, inclusive)
  let mut upper_bound: Option<(Expr, bool)> = None;

  for r in &reduced {
    // Handle Inequality[low, Op1, var, Op2, high]
    if let Expr::FunctionCall { name, args: iargs } = r
      && name == "Inequality"
      && iargs.len() == 5
    {
      let low_val = &iargs[0];
      let op1_name = if let Expr::Identifier(s) = &iargs[1] {
        s.as_str()
      } else {
        ""
      };
      let mid = &iargs[2];
      let op2_name = if let Expr::Identifier(s) = &iargs[3] {
        s.as_str()
      } else {
        ""
      };
      let high_val = &iargs[4];

      if expr_to_string(mid) == var {
        // Extract lower bound
        let low_inc = matches!(op1_name, "LessEqual" | "GreaterEqual");
        lower_bound =
          update_bound(lower_bound, low_val.clone(), low_inc, false);
        // Extract upper bound
        let high_inc = matches!(op2_name, "LessEqual" | "GreaterEqual");
        upper_bound =
          update_bound(upper_bound, high_val.clone(), high_inc, true);
        continue;
      }
    }

    if let Some((lhs, rhs, op)) = extract_comparison(r) {
      let lhs_str = expr_to_string(&lhs);
      let rhs_str = expr_to_string(&rhs);
      let var_str = var;

      if lhs_str == var_str {
        // x op value
        match op {
          CompOp::Less => {
            upper_bound = update_bound(upper_bound, rhs, false, true);
          }
          CompOp::LessEqual => {
            upper_bound = update_bound(upper_bound, rhs, true, true);
          }
          CompOp::Greater => {
            lower_bound = update_bound(lower_bound, rhs, false, false);
          }
          CompOp::GreaterEqual => {
            lower_bound = update_bound(lower_bound, rhs, true, false);
          }
          _ => {}
        }
      } else if rhs_str == var_str {
        // value op x
        match op {
          CompOp::Less => {
            lower_bound = update_bound(lower_bound, lhs, false, false);
          }
          CompOp::LessEqual => {
            lower_bound = update_bound(lower_bound, lhs, true, false);
          }
          CompOp::Greater => {
            upper_bound = update_bound(upper_bound, lhs, false, true);
          }
          CompOp::GreaterEqual => {
            upper_bound = update_bound(upper_bound, lhs, true, true);
          }
          _ => {}
        }
      }
    }
  }

  match (&lower_bound, &upper_bound) {
    (Some((low, low_inc)), Some((high, high_inc))) => {
      let low_op = if *low_inc {
        CompOp::LessEqual
      } else {
        CompOp::Less
      };
      let high_op = if *high_inc {
        CompOp::LessEqual
      } else {
        CompOp::Less
      };
      return Ok(make_compound_inequality(low, low_op, var, high_op, high));
    }
    (Some((low, low_inc)), None) => {
      let op = if *low_inc {
        CompOp::GreaterEqual
      } else {
        CompOp::Greater
      };
      return Ok(make_comparison(&Expr::Identifier(var.to_string()), low, op));
    }
    (None, Some((high, high_inc))) => {
      let op = if *high_inc {
        CompOp::LessEqual
      } else {
        CompOp::Less
      };
      return Ok(make_comparison(
        &Expr::Identifier(var.to_string()),
        high,
        op,
      ));
    }
    (None, None) => {}
  }

  // Couldn't combine — return And of all reduced forms
  Ok(
    reduced
      .iter()
      .skip(1)
      .fold(reduced[0].clone(), |acc, r| Expr::BinaryOp {
        op: BinaryOperator::And,
        left: Box::new(acc),
        right: Box::new(r.clone()),
      }),
  )
}

/// Update a bound, keeping the tighter one.
pub fn update_bound(
  current: Option<(Expr, bool)>,
  new_val: Expr,
  inclusive: bool,
  is_upper: bool,
) -> Option<(Expr, bool)> {
  if let Some((old_val, _old_inc)) = &current
    && let (Some(old_n), Some(new_n)) =
      (expr_to_number(old_val), expr_to_number(&new_val))
  {
    if is_upper {
      // For upper bound, keep the smaller one
      if new_n < old_n || (new_n == old_n && !inclusive) {
        return Some((new_val, inclusive));
      }
      return current;
    } else {
      // For lower bound, keep the larger one
      if new_n > old_n || (new_n == old_n && !inclusive) {
        return Some((new_val, inclusive));
      }
      return current;
    }
  }
  Some((new_val, inclusive))
}

/// Handle multi-variable systems.
pub fn reduce_multi_var(
  expr: &Expr,
  vars: &[String],
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Collect all constraints
  let mut constraints = Vec::new();
  collect_and_constraints(expr, &mut constraints);

  reduce_multi_var_and(&constraints, vars, domain)
}

/// Solve a multi-variable system by sequential elimination.
pub fn reduce_multi_var_and(
  constraints: &[Expr],
  vars: &[String],
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  if vars.is_empty() || constraints.is_empty() {
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Try to find an equation to solve for the first variable
  for (i, constraint) in constraints.iter().enumerate() {
    if let Some((lhs, rhs, CompOp::Equal)) = extract_comparison(constraint) {
      // Try to solve for first variable that appears
      for var in vars {
        let poly = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(lhs.clone()),
          right: Box::new(rhs.clone()),
        };
        let expanded = expand_and_combine(&poly);
        if is_constant_wrt(&expanded, var) {
          continue;
        }

        let eq = Expr::Comparison {
          operands: vec![lhs.clone(), rhs.clone()],
          operators: vec![crate::syntax::ComparisonOp::Equal],
        };
        let solve_result = solve_ast(&[eq, Expr::Identifier(var.to_string())])?;

        if let Expr::List(solutions) = &solve_result {
          let mut all_results: Vec<Expr> = Vec::new();

          for sol in solutions {
            if let Expr::List(rules) = sol {
              for rule in rules {
                if let Expr::Rule { replacement, .. } = rule {
                  let value = replacement.as_ref();

                  // Substitute into remaining constraints
                  let remaining: Vec<Expr> = constraints
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, c)| {
                      crate::syntax::substitute_variable(c, var, value)
                    })
                    .collect();

                  let remaining_vars: Vec<String> = vars
                    .iter()
                    .filter(|v| v.as_str() != var.as_str())
                    .cloned()
                    .collect();

                  if remaining.is_empty() && remaining_vars.is_empty() {
                    // No remaining constraints
                    let var_eq =
                      make_equality(&Expr::Identifier(var.to_string()), value);
                    all_results.push(var_eq);
                  } else if remaining_vars.is_empty() {
                    // Check if remaining constraints are satisfied
                    let combined = remaining.iter().skip(1).fold(
                      remaining[0].clone(),
                      |acc, c| Expr::BinaryOp {
                        op: BinaryOperator::And,
                        left: Box::new(acc),
                        right: Box::new(c.clone()),
                      },
                    );
                    let evaled =
                      crate::evaluator::evaluate_expr_to_expr(&combined);
                    if let Ok(result) = evaled
                      && !matches!(&result, Expr::Identifier(s) if s == "False")
                    {
                      let var_eq = make_equality(
                        &Expr::Identifier(var.to_string()),
                        value,
                      );
                      all_results.push(var_eq);
                    }
                  } else {
                    // Reduce remaining with fewer variables
                    let combined = remaining.iter().skip(1).fold(
                      remaining[0].clone(),
                      |acc, c| Expr::BinaryOp {
                        op: BinaryOperator::And,
                        left: Box::new(acc),
                        right: Box::new(c.clone()),
                      },
                    );
                    let sub_result =
                      reduce_expr(&combined, &remaining_vars, domain)?;

                    if !matches!(&sub_result, Expr::Identifier(s) if s == "False")
                    {
                      if matches!(&sub_result, Expr::Identifier(s) if s == "True")
                      {
                        let var_eq = make_equality(
                          &Expr::Identifier(var.to_string()),
                          value,
                        );
                        all_results.push(var_eq);
                      } else {
                        // Back-substitute solved values into the var expression
                        let mut final_value = value.clone();
                        let sub_equalities =
                          collect_and_equalities(&sub_result);
                        for (sv, sval) in &sub_equalities {
                          final_value = crate::syntax::substitute_variable(
                            &final_value,
                            sv,
                            sval,
                          );
                        }
                        let final_value =
                          crate::evaluator::evaluate_expr_to_expr(&final_value)
                            .unwrap_or(simplify(final_value));
                        let var_eq = make_equality(
                          &Expr::Identifier(var.to_string()),
                          &final_value,
                        );
                        all_results.push(Expr::BinaryOp {
                          op: BinaryOperator::And,
                          left: Box::new(var_eq),
                          right: Box::new(sub_result),
                        });
                      }
                    }
                  }
                }
              }
            }
          }

          if all_results.is_empty() {
            return Ok(Expr::Identifier("False".to_string()));
          }
          return Ok(build_or(all_results));
        }
      }
    }
  }

  // No equation found to solve — return unevaluated
  Ok(Expr::FunctionCall {
    name: "Reduce".to_string(),
    args: {
      let combined =
        constraints
          .iter()
          .skip(1)
          .fold(constraints[0].clone(), |acc, c| Expr::BinaryOp {
            op: BinaryOperator::And,
            left: Box::new(acc),
            right: Box::new(c.clone()),
          });
      let vars_expr = if vars.len() == 1 {
        Expr::Identifier(vars[0].clone())
      } else {
        Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect())
      };
      if let Some(dom) = domain {
        vec![combined, vars_expr, Expr::Identifier(dom.to_string())]
      } else {
        vec![combined, vars_expr]
      }
    },
  })
}

/// Check if a value is in a given domain.
pub fn is_in_domain(expr: &Expr, domain: &str) -> bool {
  match domain {
    "Reals" => {
      // Check that the expression doesn't contain I (imaginary unit)
      !contains_imaginary(expr)
    }
    "Integers" => {
      matches!(expr, Expr::Integer(_))
    }
    "Complexes" => true,
    "Rationals" => matches!(
      expr,
      Expr::Integer(_) | Expr::FunctionCall { .. }
        if is_rational(expr)
    ),
    _ => true,
  }
}

pub fn is_rational(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      true
    }
    _ => false,
  }
}

/// Check if an expression contains the imaginary unit I.
pub fn contains_imaginary(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(s) if s == "I" => true,
    Expr::BinaryOp { left, right, .. } => {
      contains_imaginary(left) || contains_imaginary(right)
    }
    Expr::UnaryOp { operand, .. } => contains_imaginary(operand),
    Expr::FunctionCall { name, args } => {
      if name == "Complex" {
        return true;
      }
      args.iter().any(contains_imaginary)
    }
    Expr::List(items) => items.iter().any(contains_imaginary),
    _ => false,
  }
}
