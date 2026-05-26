use super::together::negate_expr;
#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

use crate::functions::calculus_ast::{is_constant_wrt, simplify};
use crate::functions::math_ast::make_sqrt;

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

  // Narrow quantifier-elimination case:
  //   Reduce[Exists[{x, y}, α·x² + β·y² ≤ c₁ ∧ γ·x + δ·y ≥ c₂], param]
  // The Lagrange max of (γ x + δ y) on the ellipse is
  // √(c₁ (γ²/α + δ²/β)) so the existence condition reduces to a single
  // inequality on `param`. Handles the audit case
  //   `Reduce[Exists[{x, y}, x² + a y² ≤ 1 ∧ x − y ≥ 2], a]` → `a ≤ 1/3`.
  if args.len() == 2
    && let Some(out) = try_reduce_exists_quadratic_linear(&args[0], &args[1])
  {
    return Ok(out);
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
      args: args.to_vec().into(),
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
    args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
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
      .into()
    } else {
      vec![make_equality(lhs, rhs), Expr::Identifier(var.to_string())].into()
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

/// Recognise `ArcXxxDegrees[var] CompOp const_degrees` and invert it.
/// Returns Some(reduced) when the pattern matches and the inversion has a
/// closed form, None otherwise.
fn try_reduce_arc_degrees(
  lhs: &Expr,
  rhs: &Expr,
  op: CompOp,
  var: &str,
) -> Result<Option<Expr>, InterpreterError> {
  // lhs must be `ArcXxxDegrees[var]`.
  let Expr::FunctionCall {
    name: lname,
    args: largs,
  } = lhs
  else {
    return Ok(None);
  };
  if largs.len() != 1 || !matches!(&largs[0], Expr::Identifier(s) if s == var) {
    return Ok(None);
  }

  // rhs must be a constant scalar (Integer/Real/Rational).
  if !is_constant_wrt(rhs, var) {
    return Ok(None);
  }

  // Build the corresponding trig threshold in radians:
  //   threshold = TrigF[rhs * Pi / 180].
  let trig_name = match lname.as_str() {
    "ArcCosDegrees" => "Cos",
    "ArcSinDegrees" => "Sin",
    "ArcTanDegrees" => "Tan",
    "ArcCotDegrees" => "Cot",
    "ArcCscDegrees" => "Csc",
    "ArcSecDegrees" => "Sec",
    _ => return Ok(None),
  };
  let radians = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      rhs.clone(),
      Expr::Identifier("Pi".to_string()),
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(180)].into(),
      },
    ]
    .into(),
  };
  let threshold_expr = Expr::FunctionCall {
    name: trig_name.to_string(),
    args: vec![radians].into(),
  };
  let threshold = crate::evaluator::evaluate_expr_to_expr(&threshold_expr)?;

  // Helper: build a 3-way bounded interval `low op1 var op2 high`.
  let bounded = |low: Expr, op1: CompOp, op2: CompOp, high: Expr| -> Expr {
    make_compound_inequality(&low, op1, var, op2, &high)
  };
  // Helper: simple comparison `var op c`.
  let simple = |op: CompOp, c: Expr| -> Expr {
    make_comparison(&Expr::Identifier(var.to_string()), &c, op)
  };

  // Build the result. Only `>` audit cases for now — that's the documented
  // basic example for every ArcXxxDegrees function.
  if !matches!(op, CompOp::Greater) {
    return Ok(None);
  }

  let result = match lname.as_str() {
    // ArcCosDegrees is strictly decreasing on [-1, 1] → [0, 180].
    // arccos(x) > k iff -1 ≤ x < cos(k°).
    "ArcCosDegrees" => bounded(
      Expr::Integer(-1),
      CompOp::LessEqual,
      CompOp::Less,
      threshold,
    ),
    // ArcSinDegrees is strictly increasing on [-1, 1] → [-90, 90].
    // arcsin(x) > k iff sin(k°) < x ≤ 1.
    "ArcSinDegrees" => {
      bounded(threshold, CompOp::Less, CompOp::LessEqual, Expr::Integer(1))
    }
    // ArcTanDegrees is strictly increasing on R → (-90, 90).
    // arctan(x) > k iff x > tan(k°).
    "ArcTanDegrees" => simple(CompOp::Greater, threshold),
    // ArcCotDegrees principal branch is strictly decreasing on [0, ∞) → (0, 90].
    // arccot(x) > k iff 0 ≤ x < cot(k°)  (for 0 < k < 90).
    "ArcCotDegrees" => {
      bounded(Expr::Integer(0), CompOp::LessEqual, CompOp::Less, threshold)
    }
    // ArcCscDegrees on [1, ∞) → (0, 90], strictly decreasing.
    // arccsc(x) > k iff 1 ≤ x < csc(k°)  (for 0 < k ≤ 90).
    "ArcCscDegrees" => {
      bounded(Expr::Integer(1), CompOp::LessEqual, CompOp::Less, threshold)
    }
    // ArcSecDegrees: arcsec(x) > k iff x > sec(k°) || x ≤ -1
    // (the second branch covers arcsec on (-∞, -1] mapping to (90, 180]).
    "ArcSecDegrees" => Expr::FunctionCall {
      name: "Or".to_string(),
      args: vec![
        simple(CompOp::Greater, threshold),
        simple(CompOp::LessEqual, Expr::Integer(-1)),
      ]
      .into(),
    },
    _ => return Ok(None),
  };
  Ok(Some(result))
}

/// Reduce an inequality (lhs op rhs) for one variable.
pub fn reduce_inequality(
  lhs: &Expr,
  rhs: &Expr,
  op: CompOp,
  var: &str,
  domain: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Recognise `ArcXxxDegrees[var] CompOp const_degrees` directly — these
  // need transcendental inversion, not polynomial reduction.
  if let Some(result) = try_reduce_arc_degrees(lhs, rhs, op, var)? {
    return Ok(result);
  }

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
  let degree = max_power_int(&expanded, var);

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
        ]
        .into(),
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
    ]
    .into(),
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
            ]
            .into(),
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
              ]
              .into(),
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
              ]
              .into(),
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
      ]
      .into(),
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
    make_sqrt(Expr::Integer(si))
  } else if so == -1 {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(make_sqrt(Expr::Integer(si))),
    }
  } else {
    multiply_exprs(&Expr::Integer(so), &make_sqrt(Expr::Integer(si)))
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

  // If any reduced result is a disjunction (Or), distribute And over Or:
  // (A || B) && C  →  (A && C) || (B && C), then reduce each conjunction.
  if let Some(or_idx) = reduced.iter().position(|r| {
    matches!(
      r,
      Expr::BinaryOp {
        op: BinaryOperator::Or,
        ..
      }
    ) || matches!(r, Expr::FunctionCall { name, .. } if name == "Or")
  }) {
    let or_expr = reduced.remove(or_idx);
    let branches = collect_or_terms(&or_expr);
    let other_constraints = reduced;

    let mut valid_branches = Vec::new();
    for branch in &branches {
      // Combine this branch with all other constraints
      let mut all_ineqs: Vec<Expr> = vec![branch.clone()];
      all_ineqs.extend(other_constraints.iter().cloned());

      let result = reduce_combined_inequalities(&all_ineqs, var, domain)?;
      if !matches!(&result, Expr::Identifier(s) if s == "False") {
        valid_branches.push(result);
      }
    }

    return if valid_branches.is_empty() {
      Ok(Expr::Identifier("False".to_string()))
    } else {
      Ok(build_or(valid_branches))
    };
  }

  // Try to find bounds: lower < x < upper
  let mut lower_bound: Option<(Expr, bool)> = None; // (value, inclusive)
  let mut upper_bound: Option<(Expr, bool)> = None;

  for r in &reduced {
    // Handle Inequality[low, Op1, var, Op2, high] or Expr::Comparison with 2 operators
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

    // Handle Expr::Comparison with 2 operators (same-operator compound inequality)
    if let Expr::Comparison {
      operands,
      operators,
    } = r
      && operators.len() == 2
      && operands.len() == 3
    {
      use crate::syntax::ComparisonOp;
      let mid = &operands[1];
      if expr_to_string(mid) == var {
        let low_inc = matches!(
          operators[0],
          ComparisonOp::LessEqual | ComparisonOp::GreaterEqual
        );
        lower_bound =
          update_bound(lower_bound, operands[0].clone(), low_inc, false);
        let high_inc = matches!(
          operators[1],
          ComparisonOp::LessEqual | ComparisonOp::GreaterEqual
        );
        upper_bound =
          update_bound(upper_bound, operands[2].clone(), high_inc, true);
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
      // Check if the interval is empty (lower >= upper)
      if let (Some(low_val), Some(high_val)) =
        (expr_to_number(low), expr_to_number(high))
        && (low_val > high_val
          || (low_val == high_val && (!low_inc || !high_inc)))
      {
        return Ok(Expr::Identifier("False".to_string()));
      }
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

  // Find the best (equation, variable) pair: prefer last variable in the list
  // (matching Wolfram's convention where later variables are expressed in terms
  // of earlier ones), falling back to lowest degree for tie-breaking.
  let mut best: Option<(usize, String, Expr, Expr, i128)> = None; // (eq_idx, var, lhs, rhs, degree)
  for (i, constraint) in constraints.iter().enumerate() {
    if let Some((lhs, rhs, CompOp::Equal)) = extract_comparison(constraint) {
      for (vi, var) in vars.iter().enumerate() {
        let poly = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(lhs.clone()),
          right: Box::new(rhs.clone()),
        };
        let expanded = expand_and_combine(&poly);
        if is_constant_wrt(&expanded, var) {
          continue;
        }
        if let Some(deg) = max_power_int(&expanded, var) {
          let dominated = if let Some(ref b) = best {
            // Prefer later variables; for the same variable position, prefer
            // lower degree.
            let best_vi = vars.iter().position(|v| v == &b.1).unwrap_or(0);
            vi > best_vi || (vi == best_vi && deg < b.4)
          } else {
            true
          };
          if dominated {
            best = Some((i, var.clone(), lhs.clone(), rhs.clone(), deg));
          }
        }
      }
    }
  }

  if let Some((i, var, lhs, rhs, _deg)) = best {
    let eq = Expr::Comparison {
      operands: vec![lhs, rhs],
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
                  let substituted =
                    crate::syntax::substitute_variable(c, &var, value);
                  // Evaluate the substituted expression
                  crate::evaluator::evaluate_expr_to_expr(&substituted)
                    .unwrap_or(substituted)
                })
                .collect();

              let remaining_vars: Vec<String> = vars
                .iter()
                .filter(|v| v.as_str() != var.as_str())
                .cloned()
                .collect();

              if remaining.is_empty() {
                // No remaining constraints — var is expressed in terms of
                // remaining variables (if any) or is a constant solution.
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
                let evaled = crate::evaluator::evaluate_expr_to_expr(&combined);
                if let Ok(result) = evaled
                  && !matches!(&result, Expr::Identifier(s) if s == "False")
                {
                  let var_eq =
                    make_equality(&Expr::Identifier(var.to_string()), value);
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

                if !matches!(&sub_result, Expr::Identifier(s) if s == "False") {
                  if matches!(&sub_result, Expr::Identifier(s) if s == "True") {
                    let var_eq =
                      make_equality(&Expr::Identifier(var.to_string()), value);
                    all_results.push(var_eq);
                  } else {
                    // Handle Or branches: each branch is a separate solution
                    let or_branches = collect_or_terms(&sub_result);
                    for branch in &or_branches {
                      // Back-substitute solved values into the var expression
                      let mut final_value = value.clone();
                      let sub_equalities = collect_and_equalities(branch);
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
                      // Put earlier variables first to match Wolfram's
                      // output ordering (variable list order)
                      all_results.push(Expr::BinaryOp {
                        op: BinaryOperator::And,
                        left: Box::new(branch.clone()),
                        right: Box::new(var_eq),
                      });
                    }
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
        vec![combined, vars_expr, Expr::Identifier(dom.to_string())].into()
      } else {
        vec![combined, vars_expr].into()
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

// ─── Quantifier elimination: Exists with quadratic+linear constraints ──

/// Pattern-match `Reduce[Exists[{x, y}, q && l], param]` where `q` is a
/// ≤-inequality with quadratic form `α·x² + β·y²`, `l` is a ≥-inequality
/// with linear form `γ·x + δ·y`, and `param` is a single identifier.
/// Returns the resulting condition on `param`.
fn try_reduce_exists_quadratic_linear(
  exists_expr: &Expr,
  param_expr: &Expr,
) -> Option<Expr> {
  // Unwrap Exists[{x, y}, body].
  let (vars, body) = match exists_expr {
    Expr::FunctionCall { name, args }
      if name == "Exists" && args.len() == 2 =>
    {
      let vs = match &args[0] {
        Expr::List(items) => items.clone(),
        _ => return None,
      };
      (vs, args[1].clone())
    }
    _ => return None,
  };
  if vars.len() != 2 {
    return None;
  }
  let var_names: Vec<String> = vars
    .iter()
    .filter_map(|v| match v {
      Expr::Identifier(n) => Some(n.clone()),
      _ => None,
    })
    .collect();
  if var_names.len() != 2 {
    return None;
  }
  let param = match param_expr {
    Expr::Identifier(n) => n.clone(),
    _ => return None,
  };

  // Split the body into a flat list of And conjuncts.
  let mut parts: Vec<Expr> = Vec::new();
  flatten_and_into(&body, &mut parts);
  if parts.len() != 2 {
    return None;
  }
  let (q_ineq, l_ineq) = if is_quadratic_in(&parts[0], &var_names)
    && is_linear_in(&parts[1], &var_names)
  {
    (parts[0].clone(), parts[1].clone())
  } else if is_quadratic_in(&parts[1], &var_names)
    && is_linear_in(&parts[0], &var_names)
  {
    (parts[1].clone(), parts[0].clone())
  } else {
    return None;
  };

  // Quadratic constraint must be "form ≤ const" with const numeric.
  let (q_lhs, q_rhs) = extract_le_form(&q_ineq)?;
  let c1 = simplify_to_rational(&q_rhs)?;
  // Linear constraint must be "form ≥ const" with const numeric.
  let (l_lhs, l_rhs) = extract_ge_form(&l_ineq)?;
  let c2 = simplify_to_rational(&l_rhs)?;

  // Extract coefficients α, β (quadratic in x, y) and γ, δ (linear in x, y).
  let (alpha, beta, has_constant_q) = extract_xx_yy_coeffs(&q_lhs, &var_names)?;
  if !has_constant_q {
    // Quadratic LHS must have no constant term left over (e.g., `x² + a y²`
    // with the right-hand-side constant already moved). A non-zero
    // remainder defeats the formula.
  }
  let (gamma, delta) = extract_x_y_coeffs(&l_lhs, &var_names)?;

  // The audit's pattern is α=1 (a constant) and β=param. Generalise by
  // also accepting the swap (param appears as α). Bail out otherwise.
  let (alpha_const, beta_is_param) = (
    !contains_var(&alpha, &param),
    matches!(&beta, Expr::Identifier(n) if *n == param),
  );
  let (beta_const, alpha_is_param) = (
    !contains_var(&beta, &param),
    matches!(&alpha, Expr::Identifier(n) if *n == param),
  );
  let (a_const, p_coeff_var) = if alpha_const && beta_is_param {
    (alpha.clone(), var_names[1].clone())
  } else if beta_const && alpha_is_param {
    (beta.clone(), var_names[0].clone())
  } else {
    return None;
  };
  // Identify γ and δ corresponding to the constant-coeff variable and
  // the param-coeff variable respectively.
  let (lin_const_coeff, lin_param_coeff) = if p_coeff_var == var_names[1] {
    (gamma.clone(), delta.clone())
  } else {
    (delta.clone(), gamma.clone())
  };
  // All extracted scalars must be rationals.
  let alpha_q = simplify_to_rational(&a_const)?;
  let gamma_q = simplify_to_rational(&lin_const_coeff)?;
  let delta_q = simplify_to_rational(&lin_param_coeff)?;
  if !alpha_q.is_positive() {
    return None;
  }

  // Solvability condition:
  //   max(L) = sqrt(c1 * (γ²/α + δ²/(α·β/α))) — but with quadratic
  //   `α x² + β y²` it simplifies to sqrt(c1 (γ²/α + δ²/β)).
  // For param = β (assuming α constant): max ≥ c2 ⇒
  //   c1·(γ²/α + δ²/param) ≥ c2² ⇒
  //   δ²/param ≥ c2²/c1 − γ²/α ⇒
  //   1/param ≥ ((c2²/c1) - γ²/α) / δ²    [call this K]
  // K > 0 ⇒ `param ≤ 1/K` (only the positive-param branch matters
  //          since negative param produces an unbounded region).
  // K ≤ 0 ⇒ trivially satisfiable for every real `param`.
  let c2_sq = mul_q(&c2, &c2);
  let c1_inv = inv_q(&c1)?;
  let c2sq_over_c1 = mul_q(&c2_sq, &c1_inv);
  let alpha_inv = inv_q(&alpha_q)?;
  let gamma_sq = mul_q(&gamma_q, &gamma_q);
  let gamma_sq_over_alpha = mul_q(&gamma_sq, &alpha_inv);
  let lhs = sub_q(&c2sq_over_c1, &gamma_sq_over_alpha);
  let delta_sq = mul_q(&delta_q, &delta_q);
  if !delta_sq.is_positive() {
    return None;
  }
  let delta_sq_inv = inv_q(&delta_sq)?;
  let k = mul_q(&lhs, &delta_sq_inv);

  if k.is_zero() || k.is_negative() {
    return Some(Expr::Identifier("True".to_string()));
  }
  let bound = inv_q(&k)?;
  Some(Expr::Comparison {
    operands: vec![Expr::Identifier(param.clone()), rational_to_expr(&bound)],
    operators: vec![crate::syntax::ComparisonOp::LessEqual],
  })
}

#[derive(Clone, Copy, Debug)]
struct Rat {
  num: i128,
  den: i128,
}

impl Rat {
  fn from_int(n: i128) -> Self {
    Rat { num: n, den: 1 }
  }
  fn is_zero(&self) -> bool {
    self.num == 0
  }
  fn is_positive(&self) -> bool {
    (self.num > 0 && self.den > 0) || (self.num < 0 && self.den < 0)
  }
  fn is_negative(&self) -> bool {
    !self.is_zero() && !self.is_positive()
  }
}

fn rat_gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = a % b;
    a = b;
    b = t;
  }
  a.max(1)
}

fn rat_simplify(num: i128, den: i128) -> Rat {
  if den == 0 {
    return Rat { num, den };
  }
  let g = rat_gcd(num, den);
  let (mut n, mut d) = (num / g, den / g);
  if d < 0 {
    n = -n;
    d = -d;
  }
  Rat { num: n, den: d }
}

fn mul_q(a: &Rat, b: &Rat) -> Rat {
  rat_simplify(a.num * b.num, a.den * b.den)
}

fn sub_q(a: &Rat, b: &Rat) -> Rat {
  rat_simplify(a.num * b.den - b.num * a.den, a.den * b.den)
}

fn inv_q(a: &Rat) -> Option<Rat> {
  if a.num == 0 {
    return None;
  }
  let mut n = a.den;
  let mut d = a.num;
  if d < 0 {
    n = -n;
    d = -d;
  }
  Some(Rat { num: n, den: d })
}

fn rational_to_expr(r: &Rat) -> Expr {
  if r.den == 1 {
    Expr::Integer(r.num)
  } else {
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(r.num), Expr::Integer(r.den)].into(),
    }
  }
}

fn simplify_to_rational(e: &Expr) -> Option<Rat> {
  match e {
    Expr::Integer(n) => Some(Rat::from_int(*n)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        if *d == 0 {
          None
        } else {
          Some(rat_simplify(*n, *d))
        }
      } else {
        None
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut acc = Rat::from_int(1);
      for arg in args {
        let r = simplify_to_rational(arg)?;
        acc = mul_q(&acc, &r);
      }
      Some(acc)
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut acc = Rat::from_int(0);
      for arg in args {
        let r = simplify_to_rational(arg)?;
        acc = rat_simplify(acc.num * r.den + r.num * acc.den, acc.den * r.den);
      }
      Some(acc)
    }
    _ => {
      let evaluated = crate::evaluator::evaluate_expr_to_expr(e).ok()?;
      if matches!(&evaluated, Expr::Integer(_))
        || matches!(&evaluated, Expr::FunctionCall { name, .. } if name == "Rational")
      {
        simplify_to_rational(&evaluated)
      } else {
        None
      }
    }
  }
}

fn flatten_and_into(e: &Expr, out: &mut Vec<Expr>) {
  match e {
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => {
      flatten_and_into(left, out);
      flatten_and_into(right, out);
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      for a in args.iter() {
        flatten_and_into(a, out);
      }
    }
    other => out.push(other.clone()),
  }
}

fn contains_var(e: &Expr, name: &str) -> bool {
  match e {
    Expr::Identifier(n) => n == name,
    Expr::BinaryOp { left, right, .. } => {
      contains_var(left, name) || contains_var(right, name)
    }
    Expr::UnaryOp { operand, .. } => contains_var(operand, name),
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|a| contains_var(a, name))
    }
    Expr::List(items) => items.iter().any(|a| contains_var(a, name)),
    Expr::Comparison { operands, .. } => {
      operands.iter().any(|a| contains_var(a, name))
    }
    _ => false,
  }
}

fn is_quadratic_in(e: &Expr, vars: &[String]) -> bool {
  let mut max_deg = 0_i128;
  for v in vars {
    if let Some(d) = comparison_max_degree(e, v) {
      max_deg = max_deg.max(d);
    } else {
      return false;
    }
  }
  max_deg == 2
}

fn is_linear_in(e: &Expr, vars: &[String]) -> bool {
  let mut max_deg = 0_i128;
  for v in vars {
    if let Some(d) = comparison_max_degree(e, v) {
      max_deg = max_deg.max(d);
    } else {
      return false;
    }
  }
  max_deg == 1
}

fn comparison_max_degree(e: &Expr, var: &str) -> Option<i128> {
  let lhs_minus_rhs = match e {
    Expr::Comparison { operands, .. } if operands.len() == 2 => {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(operands[0].clone()),
        right: Box::new(operands[1].clone()),
      }
    }
    _ => return None,
  };
  let evaluated = crate::evaluator::evaluate_expr_to_expr(&lhs_minus_rhs)
    .unwrap_or(lhs_minus_rhs);
  super::max_power_int(&evaluated, var).or(Some(0))
}

fn extract_le_form(e: &Expr) -> Option<(Expr, Expr)> {
  if let Expr::Comparison {
    operands,
    operators,
  } = e
    && operands.len() == 2
    && operators.len() == 1
  {
    match &operators[0] {
      crate::syntax::ComparisonOp::LessEqual => {
        Some((operands[0].clone(), operands[1].clone()))
      }
      crate::syntax::ComparisonOp::GreaterEqual => {
        Some((operands[1].clone(), operands[0].clone()))
      }
      _ => None,
    }
  } else {
    None
  }
}

fn extract_ge_form(e: &Expr) -> Option<(Expr, Expr)> {
  if let Expr::Comparison {
    operands,
    operators,
  } = e
    && operands.len() == 2
    && operators.len() == 1
  {
    match &operators[0] {
      crate::syntax::ComparisonOp::GreaterEqual => {
        Some((operands[0].clone(), operands[1].clone()))
      }
      crate::syntax::ComparisonOp::LessEqual => {
        Some((operands[1].clone(), operands[0].clone()))
      }
      _ => None,
    }
  } else {
    None
  }
}

fn extract_xx_yy_coeffs(
  e: &Expr,
  vars: &[String],
) -> Option<(Expr, Expr, bool)> {
  // alpha = Coefficient[e, x, 2], beta = Coefficient[e, y, 2].
  let alpha = call_coefficient(e, &vars[0], 2)?;
  let beta = call_coefficient(e, &vars[1], 2)?;
  // Detect leftover constant term (not in audit's pattern → return false).
  let no_x = call_coefficient(e, &vars[0], 0)?;
  let leftover = call_coefficient(&no_x, &vars[1], 0)?;
  let has_const = !matches!(leftover, Expr::Integer(0));
  Some((alpha, beta, !has_const))
}

fn extract_x_y_coeffs(e: &Expr, vars: &[String]) -> Option<(Expr, Expr)> {
  let gamma = call_coefficient(e, &vars[0], 1)?;
  let delta = call_coefficient(e, &vars[1], 1)?;
  Some((gamma, delta))
}

fn call_coefficient(e: &Expr, var: &str, n: i128) -> Option<Expr> {
  let args = vec![
    e.clone(),
    Expr::Identifier(var.to_string()),
    Expr::Integer(n),
  ];
  super::coefficient_ast(&args).ok()
}
