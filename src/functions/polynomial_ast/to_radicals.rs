use crate::InterpreterError;
use crate::syntax::Expr;

/// ToRadicals[expr] — convert Root objects to explicit radical expressions.
pub fn to_radicals_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ToRadicals expects 1 argument".into(),
    ));
  }
  to_radicals_inner(&args[0])
}

fn to_radicals_inner(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    // Root[f&, k] — evaluate to get the radical form
    Expr::FunctionCall { name, args } if name == "Root" && args.len() == 2 => {
      // First try the standard root_ast solver
      let result = super::root_ast(args)?;
      // If root_ast returned unevaluated Root[...], try our own radical extraction
      if let Expr::FunctionCall { name: rn, .. } = &result
        && rn == "Root"
      {
        return root_to_radical(&args[0], &args[1]);
      }
      Ok(result)
    }
    // Recurse into function calls
    Expr::FunctionCall { name, args } => {
      let new_args: Vec<Expr> = args
        .iter()
        .map(to_radicals_inner)
        .collect::<Result<Vec<_>, _>>()?;
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: name.clone(),
        args: new_args,
      })
    }
    // Recurse into lists
    Expr::List(items) => {
      let new_items: Vec<Expr> = items
        .iter()
        .map(to_radicals_inner)
        .collect::<Result<Vec<_>, _>>()?;
      Ok(Expr::List(new_items))
    }
    // Recurse into binary ops
    Expr::BinaryOp { op, left, right } => {
      let l = to_radicals_inner(left)?;
      let r = to_radicals_inner(right)?;
      crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: *op,
        left: Box::new(l),
        right: Box::new(r),
      })
    }
    // Recurse into unary ops
    Expr::UnaryOp { op, operand } => {
      let o = to_radicals_inner(operand)?;
      crate::evaluator::evaluate_expr_to_expr(&Expr::UnaryOp {
        op: *op,
        operand: Box::new(o),
      })
    }
    _ => Ok(expr.clone()),
  }
}

/// Helper: build Expr from common constructors.
fn mk_call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args,
  }
}

fn mk_int(n: i128) -> Expr {
  Expr::Integer(n)
}

fn mk_ratio(n: i128, d: i128) -> Expr {
  Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![Expr::Integer(n), Expr::Integer(d)],
  }
}

fn mk_times(a: Expr, b: Expr) -> Expr {
  mk_call("Times", vec![a, b])
}

fn mk_plus(args: Vec<Expr>) -> Expr {
  if args.len() == 1 {
    args.into_iter().next().unwrap()
  } else {
    mk_call("Plus", args)
  }
}

fn mk_power(base: Expr, exp: Expr) -> Expr {
  mk_call("Power", vec![base, exp])
}

/// Extract integer polynomial coefficients from a pure function body.
/// The body should be a polynomial in Slot[1]. Returns coefficients [a0, a1, a2, ...].
fn extract_poly_coefficients(body: &Expr) -> Option<Vec<Expr>> {
  // Collect terms into a map: degree -> coefficient
  let mut coeffs = std::collections::BTreeMap::new();

  let terms = match body {
    Expr::FunctionCall { name, args } if name == "Plus" => args.clone(),
    _ => vec![body.clone()],
  };

  for term in &terms {
    if let Some((deg, coeff)) = classify_term(term) {
      coeffs.insert(deg, coeff);
    } else {
      return None;
    }
  }

  if coeffs.is_empty() {
    return None;
  }

  let max_deg = *coeffs.keys().max()?;
  let mut result = Vec::new();
  for i in 0..=max_deg {
    result.push(coeffs.get(&i).cloned().unwrap_or(mk_int(0)));
  }
  Some(result)
}

/// Check if expr is Power[Slot[1], n] and return n.
fn get_slot_power_degree(expr: &Expr) -> Option<usize> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Expr::Slot(1) = &args[0]
        && let Expr::Integer(n) = &args[1]
      {
        return Some(*n as usize);
      }
      None
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Slot(1) = left.as_ref()
        && let Expr::Integer(n) = right.as_ref()
      {
        return Some(*n as usize);
      }
      None
    }
    _ => None,
  }
}

/// Classify a single term: return (degree, coefficient).
fn classify_term(term: &Expr) -> Option<(usize, Expr)> {
  // Constant (no slot): degree 0
  if !contains_slot(term) {
    return Some((0, term.clone()));
  }

  // Slot[1] alone: degree 1, coefficient 1
  if let Expr::Slot(1) = term {
    return Some((1, mk_int(1)));
  }

  // Power[Slot[1], n] (FunctionCall or BinaryOp form): degree n, coefficient 1
  if let Some(deg) = get_slot_power_degree(term) {
    return Some((deg, mk_int(1)));
  }

  // Times[coeff, slot_term] — handle both FunctionCall and BinaryOp forms
  let times_args: Option<(&Expr, &Expr)> = match term {
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      Some((&args[0], &args[1]))
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => Some((left.as_ref(), right.as_ref())),
    _ => None,
  };

  if let Some((a, b)) = times_args {
    let (slot_part, coeff_part) = if contains_slot(a) && !contains_slot(b) {
      (a, b)
    } else if contains_slot(b) && !contains_slot(a) {
      (b, a)
    } else {
      return None;
    };

    if let Expr::Slot(1) = slot_part {
      return Some((1, coeff_part.clone()));
    }
    if let Some(deg) = get_slot_power_degree(slot_part) {
      return Some((deg, coeff_part.clone()));
    }
  }

  // Times with more than 2 args: Times[c1, c2, ..., slot_term]
  if let Expr::FunctionCall { name, args } = term
    && name == "Times"
    && args.len() > 2
  {
    // Find which arg contains the slot
    let slot_idx = args.iter().position(contains_slot);
    if let Some(idx) = slot_idx {
      // All other args should be non-slot (coefficient)
      if args
        .iter()
        .enumerate()
        .all(|(i, a)| i == idx || !contains_slot(a))
      {
        let slot_part = &args[idx];
        let coeff_parts: Vec<Expr> = args
          .iter()
          .enumerate()
          .filter(|&(i, _)| i != idx)
          .map(|(_, a)| a.clone())
          .collect();
        let coeff = if coeff_parts.len() == 1 {
          coeff_parts.into_iter().next().unwrap()
        } else {
          mk_call("Times", coeff_parts)
        };

        if let Expr::Slot(1) = slot_part {
          return Some((1, coeff));
        }
        if let Some(deg) = get_slot_power_degree(slot_part) {
          return Some((deg, coeff));
        }
      }
    }
  }

  None
}

/// Check if an expression contains Slot[1].
fn contains_slot(expr: &Expr) -> bool {
  match expr {
    Expr::Slot(1) => true,
    Expr::FunctionCall { args, .. } => args.iter().any(contains_slot),
    Expr::BinaryOp { left, right, .. } => {
      contains_slot(left) || contains_slot(right)
    }
    Expr::UnaryOp { operand, .. } => contains_slot(operand),
    Expr::List(items) => items.iter().any(contains_slot),
    _ => false,
  }
}

/// Convert Root[f&, k] to radical form.
fn root_to_radical(
  func: &Expr,
  k_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let k = match k_expr {
    Expr::Integer(n) => *n,
    _ => {
      return Ok(mk_call("Root", vec![func.clone(), k_expr.clone()]));
    }
  };

  let body_raw = match func {
    Expr::Function { body } => body.as_ref(),
    _ => {
      return Ok(mk_call("Root", vec![func.clone(), k_expr.clone()]));
    }
  };
  // Evaluate the body to normalize it into FunctionCall form (Plus, Times, Power)
  let body_eval = crate::evaluator::evaluate_expr_to_expr(body_raw)
    .unwrap_or_else(|_| body_raw.clone());
  let body = &body_eval;

  let coeffs = match extract_poly_coefficients(body) {
    Some(c) => c,
    None => {
      return Ok(mk_call("Root", vec![func.clone(), k_expr.clone()]));
    }
  };

  let degree = coeffs.len() - 1;

  // Generate all roots and sort them, then pick the k-th one
  // Check if it's a pure polynomial x^n + c = 0 (only leading and constant terms)
  let is_pure = coeffs[1..degree]
    .iter()
    .all(|c| crate::syntax::expr_to_string(c) == "0");

  let roots = if is_pure && degree >= 1 {
    solve_pure_nth(&coeffs[degree], &coeffs[0], degree as i128)
  } else {
    match degree {
      1 => solve_linear(&coeffs),
      2 => solve_quadratic(&coeffs),
      3 => solve_cubic(&coeffs),
      4 => solve_quartic(&coeffs),
      _ => None,
    }
  };

  match roots {
    Some(root_exprs) => {
      // Evaluate and sort roots like root_ast does
      let mut evaluated_roots: Vec<Expr> = root_exprs
        .into_iter()
        .filter_map(|r| crate::evaluator::evaluate_expr_to_expr(&r).ok())
        .collect();

      evaluated_roots.sort_by(super::solve::root_order);

      let idx = (k as usize) - 1;
      if idx >= evaluated_roots.len() {
        Ok(mk_call("Root", vec![func.clone(), k_expr.clone()]))
      } else {
        Ok(evaluated_roots.remove(idx))
      }
    }
    None => Ok(mk_call("Root", vec![func.clone(), k_expr.clone()])),
  }
}

/// Solve linear: a0 + a1*x = 0 → x = -a0/a1
fn solve_linear(coeffs: &[Expr]) -> Option<Vec<Expr>> {
  Some(vec![mk_call(
    "Times",
    vec![
      mk_int(-1),
      coeffs[0].clone(),
      mk_power(coeffs[1].clone(), mk_int(-1)),
    ],
  )])
}

/// Solve quadratic: a0 + a1*x + a2*x^2 = 0
fn solve_quadratic(coeffs: &[Expr]) -> Option<Vec<Expr>> {
  let a = &coeffs[2];
  let b = &coeffs[1];
  let c = &coeffs[0];

  // discriminant = b^2 - 4*a*c
  let disc = mk_plus(vec![
    mk_power(b.clone(), mk_int(2)),
    mk_times(mk_int(-4), mk_times(a.clone(), c.clone())),
  ]);
  let sqrt_disc = mk_power(disc, mk_ratio(1, 2));
  let denom = mk_times(mk_int(2), a.clone());

  // (-b ± sqrt(disc)) / (2a)
  let root1 = mk_times(
    mk_plus(vec![
      mk_times(mk_int(-1), b.clone()),
      mk_times(mk_int(-1), sqrt_disc.clone()),
    ]),
    mk_power(denom.clone(), mk_int(-1)),
  );
  let root2 = mk_times(
    mk_plus(vec![mk_times(mk_int(-1), b.clone()), sqrt_disc]),
    mk_power(denom, mk_int(-1)),
  );

  Some(vec![root1, root2])
}

/// Solve depressed cubic x^3 + px + q = 0 using Cardano's formula.
/// Then shift for general cubic a0 + a1*x + a2*x^2 + a3*x^3 = 0.
fn solve_cubic(coeffs: &[Expr]) -> Option<Vec<Expr>> {
  // Normalize: divide by leading coefficient
  let a3 = &coeffs[3];
  let a2 = &coeffs[2];
  let a1 = &coeffs[1];
  let a0 = &coeffs[0];

  // For now, handle the common case: x^n - c = 0 (binomial)
  // Check if a2 == 0 and a1 == 0: pure cubic a3*x^3 + a0 = 0
  let a2_str = crate::syntax::expr_to_string(a2);
  let a1_str = crate::syntax::expr_to_string(a1);

  if a2_str == "0" && a1_str == "0" {
    return solve_pure_nth(a3, a0, 3);
  }

  // General cubic: use Cardano's formula
  // Substitute x = t - a2/(3*a3) to get depressed cubic t^3 + pt + q = 0
  // p = (3*a3*a1 - a2^2) / (3*a3^2)
  // q = (2*a2^3 - 9*a3*a2*a1 + 27*a3^2*a0) / (27*a3^3)
  let shift = mk_times(
    mk_times(mk_int(-1), a2.clone()),
    mk_power(mk_times(mk_int(3), a3.clone()), mk_int(-1)),
  );

  let p = mk_times(
    mk_plus(vec![
      mk_times(mk_int(3), mk_times(a3.clone(), a1.clone())),
      mk_times(mk_int(-1), mk_power(a2.clone(), mk_int(2))),
    ]),
    mk_power(
      mk_times(mk_int(3), mk_power(a3.clone(), mk_int(2))),
      mk_int(-1),
    ),
  );

  let q = mk_times(
    mk_plus(vec![
      mk_times(mk_int(2), mk_power(a2.clone(), mk_int(3))),
      mk_times(
        mk_int(-9),
        mk_call("Times", vec![a3.clone(), a2.clone(), a1.clone()]),
      ),
      mk_times(
        mk_int(27),
        mk_times(mk_power(a3.clone(), mk_int(2)), a0.clone()),
      ),
    ]),
    mk_power(
      mk_times(mk_int(27), mk_power(a3.clone(), mk_int(3))),
      mk_int(-1),
    ),
  );

  // Cardano: discriminant D = -(4p^3 + 27q^2)
  // Using cube root of unity: omega = (-1 + I*Sqrt[3])/2
  // The three roots of t^3 + pt + q = 0 are:
  // t_k = omega^k * cbrt(-q/2 + sqrt(q^2/4 + p^3/27))
  //      + omega^(-k) * cbrt(-q/2 - sqrt(q^2/4 + p^3/27))
  // for k = 0, 1, 2

  let omega = mk_times(
    mk_ratio(1, 2),
    mk_plus(vec![
      mk_int(-1),
      mk_times(
        Expr::Identifier("I".to_string()),
        mk_power(mk_int(3), mk_ratio(1, 2)),
      ),
    ]),
  );

  let inner = mk_plus(vec![
    mk_times(mk_ratio(1, 4), mk_power(q.clone(), mk_int(2))),
    mk_times(mk_ratio(1, 27), mk_power(p.clone(), mk_int(3))),
  ]);
  let sqrt_inner = mk_power(inner, mk_ratio(1, 2));

  let c_plus = mk_power(
    mk_plus(vec![
      mk_times(mk_ratio(-1, 2), q.clone()),
      sqrt_inner.clone(),
    ]),
    mk_ratio(1, 3),
  );
  let c_minus = mk_power(
    mk_plus(vec![
      mk_times(mk_ratio(-1, 2), q),
      mk_times(mk_int(-1), sqrt_inner),
    ]),
    mk_ratio(1, 3),
  );

  let mut roots = Vec::new();
  for k in 0..3 {
    let omega_k = if k == 0 {
      mk_int(1)
    } else {
      mk_power(omega.clone(), mk_int(k))
    };
    let omega_neg_k = if k == 0 {
      mk_int(1)
    } else {
      mk_power(omega.clone(), mk_int(-k))
    };

    let t = mk_plus(vec![
      mk_times(omega_k, c_plus.clone()),
      mk_times(omega_neg_k, c_minus.clone()),
    ]);

    let root = mk_plus(vec![t, shift.clone()]);
    roots.push(root);
  }

  Some(roots)
}

/// Solve pure nth degree: a_n * x^n + a_0 = 0 → x = (-a_0/a_n)^(1/n) * omega^k
fn solve_pure_nth(
  leading: &Expr,
  constant: &Expr,
  n: i128,
) -> Option<Vec<Expr>> {
  // x^n = -a0/an
  let base = mk_times(
    mk_times(mk_int(-1), constant.clone()),
    mk_power(leading.clone(), mk_int(-1)),
  );

  let root_base = mk_power(base, mk_ratio(1, n));

  // Build explicit roots of unity omega_k = exp(2*pi*i*k/n)
  let i_val = Expr::Identifier("I".to_string());
  let sqrt3 = mk_power(mk_int(3), mk_ratio(1, 2));

  let mut roots = Vec::new();
  for k in 0..n {
    let omega_k = nth_root_of_unity(k, n, &i_val, &sqrt3);
    if let Some(omega) = omega_k {
      roots.push(mk_times(omega, root_base.clone()));
    } else {
      roots.push(root_base.clone());
    }
  }

  Some(roots)
}

/// Compute the k-th n-th root of unity as an exact expression.
/// Returns None for k=0 (which is 1), Some(expr) otherwise.
fn nth_root_of_unity(
  k: i128,
  n: i128,
  i_val: &Expr,
  _sqrt3: &Expr,
) -> Option<Expr> {
  if k == 0 {
    return None; // 1, just use root_base directly
  }

  // Simplify common cases for exact representation
  // exp(2*pi*i*k/n)
  let (num, den) = simplify_fraction(2 * k, n);

  // cos(num*pi/den) + i*sin(num*pi/den)
  // Common exact values:
  match (num % (2 * den), den) {
    _ => {
      // General case: use cos + i*sin form which Wolfram can simplify
      let angle = if den == 1 {
        mk_times(mk_int(num), Expr::Identifier("Pi".to_string()))
      } else {
        mk_times(mk_ratio(num, den), Expr::Identifier("Pi".to_string()))
      };
      Some(mk_plus(vec![
        mk_call("Cos", vec![angle.clone()]),
        mk_times(i_val.clone(), mk_call("Sin", vec![angle])),
      ]))
    }
  }
}

/// Simplify a fraction n/d to lowest terms.
fn simplify_fraction(mut n: i128, mut d: i128) -> (i128, i128) {
  if d < 0 {
    n = -n;
    d = -d;
  }
  let g = gcd_abs(n.unsigned_abs(), d.unsigned_abs()) as i128;
  (n / g, d / g)
}

fn gcd_abs(mut a: u128, mut b: u128) -> u128 {
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Solve quartic using Ferrari's method.
/// For now, only handle pure quartic: a4*x^4 + a0 = 0
fn solve_quartic(coeffs: &[Expr]) -> Option<Vec<Expr>> {
  let a4 = &coeffs[4];
  let a3 = &coeffs[3];
  let a2 = &coeffs[2];
  let a1 = &coeffs[1];
  let a0 = &coeffs[0];

  let a3_str = crate::syntax::expr_to_string(a3);
  let a2_str = crate::syntax::expr_to_string(a2);
  let a1_str = crate::syntax::expr_to_string(a1);

  // Pure quartic: a4*x^4 + a0 = 0
  if a3_str == "0" && a2_str == "0" && a1_str == "0" {
    return solve_pure_nth(a4, a0, 4);
  }

  // General quartic is very complex; return None for now
  None
}
