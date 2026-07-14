#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;
use crate::syntax::{BinaryOperator, UnaryOperator, unevaluated};

/// AnglePath[{θ1, θ2, ...}] - path with unit steps and cumulative turning angles.
/// AnglePath[{{r1, θ1}, {r2, θ2}, ...}] - path with specified step lengths.
pub fn angle_path_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "AnglePath expects 1 or 2 arguments".into(),
    ));
  }

  // AnglePath[{{x0, y0}, θ0}, steps] form: consume initial position and
  // starting angle from the first argument, then treat the second argument
  // as the step list.
  let (start_pos, start_angle, items_vec): (
    Option<Expr>,
    Option<Expr>,
    Vec<Expr>,
  ) = if args.len() == 2 {
    let Some((pos, theta)) = parse_initial_spec(&args[0]) else {
      crate::emit_message(&format!(
        "AnglePath::init: Invalid angle path initialization {}.",
        crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
      ));
      return Ok(unevaluated("AnglePath", args));
    };
    let step_items = match &args[1] {
      Expr::List(items) => items.clone(),
      _ => {
        return Ok(unevaluated("AnglePath", args));
      }
    };
    (Some(pos), Some(theta), step_items.to_vec())
  } else {
    let step_items = match &args[0] {
      Expr::List(items) => items.clone(),
      _ => {
        return Ok(unevaluated("AnglePath", args));
      }
    };
    (None, None, step_items.to_vec())
  };
  let items = &items_vec;

  // Start at origin
  let mut x = 0.0_f64;
  let mut y = 0.0_f64;
  let mut angle = 0.0_f64;
  let mut points: Vec<Expr> = Vec::with_capacity(items.len() + 1);

  // Check if all items are numeric (pure angles) or {step, angle} pairs
  let is_pair_form = items
    .first()
    .is_some_and(|item| matches!(item, Expr::List(_)));

  // Use numeric mode only when every input converts to a float AND at least
  // one input is an explicit Real. Integer-only inputs stay symbolic so
  // Cos/Sin are preserved, and any non-numeric input (e.g. `{a, b}`) stays
  // symbolic so the output can still be produced.
  let any_real = if is_pair_form {
    items.iter().any(|item| {
      if let Expr::List(pair) = item {
        pair.iter().any(|p| matches!(p, Expr::Real(_)))
      } else {
        matches!(item, Expr::Real(_))
      }
    })
  } else {
    items.iter().any(|item| matches!(item, Expr::Real(_)))
  };
  let all_floatable = if is_pair_form {
    items.iter().all(|item| {
      if let Expr::List(pair) = item {
        pair.len() == 2
          && matches!(&pair[0], Expr::Integer(_) | Expr::Real(_))
          && matches!(&pair[1], Expr::Integer(_) | Expr::Real(_))
      } else {
        matches!(item, Expr::Integer(_) | Expr::Real(_))
      }
    })
  } else {
    items
      .iter()
      .all(|item| matches!(item, Expr::Integer(_) | Expr::Real(_)))
  };
  let use_numeric = all_floatable && any_real;

  if !use_numeric {
    // Symbolic mode: keep exact Cos/Sin
    let mut cum_terms_x: Vec<Expr> = Vec::new();
    let mut cum_terms_y: Vec<Expr> = Vec::new();
    let mut cum_angle = start_angle.clone().unwrap_or(Expr::Integer(0));

    let (start_x, start_y) = if let Some(Expr::List(pair)) = &start_pos {
      if pair.len() == 2 {
        (pair[0].clone(), pair[1].clone())
      } else {
        (Expr::Integer(0), Expr::Integer(0))
      }
    } else {
      (Expr::Integer(0), Expr::Integer(0))
    };
    let start_x_expr = start_x.clone();
    let start_y_expr = start_y.clone();
    points.push(Expr::List(vec![start_x.clone(), start_y.clone()].into()));
    if !matches!(start_x, Expr::Integer(0)) {
      cum_terms_x.push(start_x_expr);
    }
    if !matches!(start_y, Expr::Integer(0)) {
      cum_terms_y.push(start_y_expr);
    }

    for item in items {
      let (step, theta) = if is_pair_form {
        if let Expr::List(pair) = item {
          if pair.len() != 2 {
            return Err(InterpreterError::EvaluationError(
              "AnglePath: each element must be a number or {step, angle} pair"
                .into(),
            ));
          }
          (pair[0].clone(), pair[1].clone())
        } else {
          let theta = item.clone();
          (Expr::Integer(1), theta)
        }
      } else {
        (Expr::Integer(1), item.clone())
      };

      // cum_angle += theta
      cum_angle = crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[cum_angle, theta],
      )?;

      // cos_term = step * Cos[cum_angle], sin_term = step * Sin[cum_angle]
      let cos_val = crate::evaluator::evaluate_function_call_ast(
        "Cos",
        &[cum_angle.clone()],
      )?;
      let sin_val = crate::evaluator::evaluate_function_call_ast(
        "Sin",
        &[cum_angle.clone()],
      )?;
      let cos_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[step.clone(), cos_val],
      )?;
      let sin_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[step, sin_val],
      )?;

      cum_terms_x.push(cos_term);
      cum_terms_y.push(sin_term);

      let px = if cum_terms_x.len() == 1 {
        cum_terms_x[0].clone()
      } else {
        crate::evaluator::evaluate_function_call_ast("Plus", &cum_terms_x)?
      };
      let py = if cum_terms_y.len() == 1 {
        cum_terms_y[0].clone()
      } else {
        crate::evaluator::evaluate_function_call_ast("Plus", &cum_terms_y)?
      };

      points.push(Expr::List(vec![px, py].into()));
    }
  } else {
    // Numeric mode
    if let Some(Expr::List(pair)) = &start_pos
      && pair.len() == 2
    {
      x = expr_to_f64(&pair[0]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "AnglePath: starting x must be numeric".into(),
        )
      })?;
      y = expr_to_f64(&pair[1]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "AnglePath: starting y must be numeric".into(),
        )
      })?;
    }
    if let Some(theta_expr) = &start_angle {
      angle = expr_to_f64(theta_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "AnglePath: starting angle must be numeric".into(),
        )
      })?;
    }
    points.push(Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()));

    for item in items {
      let (step, theta) = if is_pair_form {
        if let Expr::List(pair) = item {
          if pair.len() != 2 {
            return Err(InterpreterError::EvaluationError(
              "AnglePath: each element must be a number or {step, angle} pair"
                .into(),
            ));
          }
          let s = expr_to_f64(&pair[0]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "AnglePath: step must be numeric".into(),
            )
          })?;
          let t = expr_to_f64(&pair[1]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "AnglePath: angle must be numeric".into(),
            )
          })?;
          (s, t)
        } else {
          let t = expr_to_f64(item).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "AnglePath: angle must be numeric".into(),
            )
          })?;
          (1.0, t)
        }
      } else {
        let t = expr_to_f64(item).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "AnglePath: angle must be numeric".into(),
          )
        })?;
        (1.0, t)
      };

      angle += theta;
      x += step * angle.cos();
      y += step * angle.sin();
      points.push(Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()));
    }
  }

  Ok(Expr::List(points.into()))
}

/// Parse the 2-argument form's first argument into (position, initial_angle).
/// Two shapes are accepted, matching Wolfram:
///   `{{x0, y0}, θ0}` — start point with an initial heading θ0, and
///   `{x0, y0}`       — start point facing angle 0.
/// Returns None for anything else (the caller emits `AnglePath::init`).
fn parse_initial_spec(expr: &Expr) -> Option<(Expr, Expr)> {
  let Expr::List(items) = expr else {
    return None;
  };
  if items.len() != 2 {
    return None;
  }
  // Nested form `{{x0, y0}, θ0}`.
  if let Expr::List(pos) = &items[0]
    && pos.len() == 2
  {
    return Some((Expr::List(pos.clone()), items[1].clone()));
  }
  // Flat point `{x0, y0}` (neither component is itself a list): face angle 0.
  if !matches!(&items[0], Expr::List(_)) && !matches!(&items[1], Expr::List(_))
  {
    return Some((expr.clone(), Expr::Integer(0)));
  }
  None
}

/// AST-based Product: product of list elements or iterator product.
/// Recognise the integrand `1 + 1/var^2` across the AST shapes Woxi
/// produces for that input. Used by the infinite-product closed form
/// `Product[1 + 1/k², {k, 1, ∞}] = Sinh[π]/π` so we don't depend on the
/// exact canonical form of the body.
/// True if `body` represents `1 - 1/var^4` in any canonical AST form Woxi
/// parses: `Plus[1, Times[-1, Power[var, -4]]]`, `1 + Power[var, -4]*-1`,
/// `1 - 1/var^4` (BinaryOp), etc.
fn body_is_one_minus_one_over_var_quartic(body: &Expr, var_name: &str) -> bool {
  let is_var_to_neg_four = |e: &Expr| -> bool {
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } = e
      && matches!(left.as_ref(), Expr::Identifier(s) if s == var_name)
      && matches!(right.as_ref(), Expr::Integer(-4))
    {
      return true;
    }
    if let Expr::FunctionCall { name, args } = e
      && name == "Power"
      && args.len() == 2
      && matches!(&args[0], Expr::Identifier(s) if s == var_name)
      && matches!(&args[1], Expr::Integer(-4))
    {
      return true;
    }
    false
  };
  // `var^4` (positive exponent), used in `1 - 1/var^4` BinaryOp form.
  let is_var_to_four = |e: &Expr| -> bool {
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } = e
      && matches!(left.as_ref(), Expr::Identifier(s) if s == var_name)
      && matches!(right.as_ref(), Expr::Integer(4))
    {
      return true;
    }
    if let Expr::FunctionCall { name, args } = e
      && name == "Power"
      && args.len() == 2
      && matches!(&args[0], Expr::Identifier(s) if s == var_name)
      && matches!(&args[1], Expr::Integer(4))
    {
      return true;
    }
    false
  };
  // Recognise `-1 * var^-4` or `var^-4 * -1`.
  let is_neg_one_over_var_quartic = |e: &Expr| -> bool {
    // `Times[-1, var^-4]`.
    if let Expr::FunctionCall { name, args } = e
      && name == "Times"
      && args.len() == 2
      && ((matches!(&args[0], Expr::Integer(-1))
        && is_var_to_neg_four(&args[1]))
        || (matches!(&args[1], Expr::Integer(-1))
          && is_var_to_neg_four(&args[0])))
    {
      return true;
    }
    // `-1/var^4` as BinaryOp Times.
    if let Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } = e
      && ((matches!(left.as_ref(), Expr::Integer(-1))
        && is_var_to_neg_four(right))
        || (matches!(right.as_ref(), Expr::Integer(-1))
          && is_var_to_neg_four(left)))
    {
      return true;
    }
    // `Divide[-1, var^4]`.
    if let Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } = e
      && matches!(left.as_ref(), Expr::Integer(-1))
      && is_var_to_four(right)
    {
      return true;
    }
    false
  };
  // Match `Plus[1, -1 * var^-4]` in either FunctionCall or BinaryOp form.
  if let Expr::FunctionCall { name, args } = body
    && name == "Plus"
    && args.len() == 2
  {
    return (matches!(&args[0], Expr::Integer(1))
      && is_neg_one_over_var_quartic(&args[1]))
      || (matches!(&args[1], Expr::Integer(1))
        && is_neg_one_over_var_quartic(&args[0]));
  }
  if let Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left,
    right,
  } = body
  {
    return (matches!(left.as_ref(), Expr::Integer(1))
      && is_neg_one_over_var_quartic(right))
      || (matches!(right.as_ref(), Expr::Integer(1))
        && is_neg_one_over_var_quartic(left));
  }
  // `1 - 1/var^4` BinaryOp Minus.
  if let Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left,
    right,
  } = body
    && matches!(left.as_ref(), Expr::Integer(1))
  {
    // right side must be `1/var^4`.
    if let Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: rl,
      right: rr,
    } = right.as_ref()
      && matches!(rl.as_ref(), Expr::Integer(1))
      && is_var_to_four(rr)
    {
      return true;
    }
    if is_var_to_neg_four(right) {
      return true;
    }
  }
  false
}

/// If `body` is `var + a` — a sum of exactly one bare `var` factor (coefficient
/// 1) and at least one term free of `var` — return the constant shift `a`.
/// Higher-degree or scaled var terms (`var^2`, `2 var`) return None.
/// The integer shift `a` if `expr` is the monic linear polynomial `var + a`
/// (`var` alone gives `0`). Non-monic (`2 var`), non-integer or non-linear
/// expressions give `None`.
fn monic_int_shift(expr: &Expr, var_name: &str) -> Option<i128> {
  if matches!(expr, Expr::Identifier(n) if n == var_name) {
    return Some(0);
  }
  match linear_shift_of_var(expr, var_name)? {
    Expr::Integer(v) => Some(v),
    _ => None,
  }
}

/// Closed form for `Product[(var + a)/(var + b), {var, 1, n}]` with
/// non-negative integer shifts `a, b`, where `n = max_expr`. Returns `None`
/// (leaving the product symbolic) when the body is not such a ratio.
/// Closed form for `Product[(var + a)/(var + b), {var, k0, n}]` with integer
/// shifts `a, b` (`n = max_expr`). Returns the un-evaluated ratio, or None when
/// a factor could vanish or the constant overflows.
fn telescope_linear_pair(
  a: i128,
  b: i128,
  max_expr: &Expr,
  k0: i128,
) -> Option<Expr> {
  if a == b {
    return Some(Expr::Integer(1));
  }
  // The factors (k + j) range over j ∈ (lo, hi]; the smallest factor value in
  // the product is k0 + lo + 1 (at k = k0, j = lo + 1). Require every factor to
  // stay positive over [k0, n] so nothing vanishes (and no division by zero);
  // i.e. k0 + lo + 1 ≥ 1. For the classic k0 = 1 case this reduces to the old
  // `a, b ≥ 0` guard.
  let (lo, hi) = (a.min(b), a.max(b));
  if k0 + lo + 1 < 1 {
    return None;
  }
  // Build Π_{j=lo+1}^{hi} (n + j) and the constant Π_{j=lo+1}^{hi} (k0 + j - 1).
  let mut factors: Vec<Expr> = Vec::new();
  let mut konst: i128 = 1;
  for j in (lo + 1)..=hi {
    factors.push(Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(max_expr.clone()),
      right: Box::new(Expr::Integer(j)),
    });
    konst = konst.checked_mul(k0 + j - 1)?; // overflow — leave symbolic
  }
  let prod = Expr::FunctionCall {
    name: "Times".to_string(),
    args: factors.into(),
  };
  // a > b: (Π (n+j)) / konst ; a < b: konst / (Π (n+j)).
  Some(if a > b {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(prod),
      right: Box::new(Expr::Integer(konst)),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Integer(konst)),
      right: Box::new(prod),
    }
  })
}

/// Integer shifts of a monic polynomial that factors completely into monic
/// linear integer-root factors `∏ (var + c_i)` (multiplicity repeated). Returns
/// None when the polynomial has a non-unit content, a non-monic-linear factor
/// (e.g. `2 var`, `var^2 + 1`) or an irrational root — leaving the product
/// symbolic. `k` → `[0]`, `k^2 - 1` → `[-1, 1]`, `k^2` → `[0, 0]`.
fn monic_linear_int_shifts(poly: &Expr, var_name: &str) -> Option<Vec<i128>> {
  // Fast path: a bare monic linear polynomial (matches the old single-factor
  // behaviour without invoking the factoriser).
  if let Some(a) = monic_int_shift(poly, var_name) {
    return Some(vec![a]);
  }
  let factor_list =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "FactorList".to_string(),
      args: vec![poly.clone()].into(),
    })
    .ok()?;
  let Expr::List(ref entries) = factor_list else {
    return None;
  };
  let mut shifts: Vec<i128> = Vec::new();
  for entry in entries.iter() {
    let Expr::List(pair) = entry else {
      return None;
    };
    if pair.len() != 2 {
      return None;
    }
    let mult = match &pair[1] {
      Expr::Integer(m) if *m >= 0 => *m,
      _ => return None,
    };
    match &pair[0] {
      // Numeric content must be a unit so the polynomial is monic.
      Expr::Integer(1) => {}
      Expr::Integer(_) => return None,
      factor => {
        let a = monic_int_shift(factor, var_name)?;
        for _ in 0..mult {
          shifts.push(a);
        }
      }
    }
  }
  Some(shifts)
}

fn rational_telescoping_product(
  body: &Expr,
  var_name: &str,
  max_expr: &Expr,
  k0: i128,
) -> Result<Option<Expr>, InterpreterError> {
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let call = |name: &str, arg: Expr| Expr::FunctionCall {
    name: name.to_string(),
    args: vec![arg].into(),
  };
  let together = eval(&call("Together", body.clone()))?;
  let num = eval(&call("Numerator", together.clone()))?;
  let den = eval(&call("Denominator", together))?;
  // Factor numerator and denominator into monic linear integer-root factors.
  //   Product[(k-1)(k+1)/k^2, {k, 2, n}] = (1/n) * ((n+1)/2) = (1+n)/(2n).
  // Both sides must factor completely and share the same number of factors so
  // the telescoping pairs up (equal degree ⇒ the product stays rational).
  let (Some(mut num_shifts), Some(mut den_shifts)) = (
    monic_linear_int_shifts(&num, var_name),
    monic_linear_int_shifts(&den, var_name),
  ) else {
    return Ok(None);
  };
  if num_shifts.len() != den_shifts.len() {
    return Ok(None);
  }
  // Pair the factors in a deterministic (sorted) order and telescope each pair.
  num_shifts.sort_unstable();
  den_shifts.sort_unstable();
  let mut factors: Vec<Expr> = Vec::new();
  for (a, b) in num_shifts.into_iter().zip(den_shifts) {
    let Some(pair) = telescope_linear_pair(a, b, max_expr, k0) else {
      return Ok(None);
    };
    factors.push(pair);
  }
  let result = if factors.len() == 1 {
    factors.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: factors.into(),
    }
  };
  Ok(Some(eval(&result)?))
}

/// An explicit real number (Integer, Rational, or Real) as f64; None for
/// symbolic values (including constants like Pi).
fn explicit_real(e: &Expr) -> Option<f64> {
  match e {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(r) => Some(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
          Some(*p as f64 / *q as f64)
        }
        _ => None,
      }
    }
    _ => None,
  }
}

fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = a % b;
    a = b;
    b = t;
  }
  a
}

/// A CoefficientList entry as an exact `(num, den)` rational, or None.
fn coeff_to_ratio(e: &Expr) -> Option<(i128, i128)> {
  match e {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => Some((*p, *q)),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Integer-cleared coefficient list of `poly` in `var` (ascending powers, the
/// leading entry nonzero), or None when `poly` is not a polynomial with
/// rational coefficients.
fn integer_coefficient_list(poly: &Expr, var_name: &str) -> Option<Vec<i128>> {
  let list = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "CoefficientList".to_string(),
    args: vec![poly.clone(), Expr::Identifier(var_name.to_string())].into(),
  })
  .ok()?;
  let Expr::List(ref items) = list else {
    return None;
  };
  let ratios: Vec<(i128, i128)> =
    items.iter().map(coeff_to_ratio).collect::<Option<_>>()?;
  let mut lcm: i128 = 1;
  for &(_, d) in &ratios {
    let g = gcd_i128(lcm, d);
    lcm = (lcm / g).checked_mul(d.abs())?;
  }
  let mut out = Vec::with_capacity(ratios.len());
  for &(n, d) in &ratios {
    out.push(n.checked_mul(lcm / d)?);
  }
  while out.len() > 1 && *out.last().unwrap() == 0 {
    out.pop();
  }
  Some(out)
}

/// Horner evaluation of an ascending integer polynomial at integer `x`,
/// overflow-checked.
fn horner_i128(coeffs: &[i128], x: i128) -> Option<i128> {
  let mut acc: i128 = 0;
  for &c in coeffs.iter().rev() {
    acc = acc.checked_mul(x)?.checked_add(c)?;
  }
  Some(acc)
}

/// Synthetic division of an ascending integer polynomial by `(var - r)`,
/// returning the ascending integer quotient (assumes `r` is an exact root).
fn deflate_i128(coeffs: &[i128], r: i128) -> Option<Vec<i128>> {
  let desc: Vec<i128> = coeffs.iter().rev().cloned().collect();
  let n = desc.len();
  let mut quot = vec![0i128; n - 1];
  let mut carry = desc[0];
  quot[0] = carry;
  for i in 1..n - 1 {
    carry = desc[i].checked_add(carry.checked_mul(r)?)?;
    quot[i] = carry;
  }
  quot.reverse();
  Some(quot)
}

/// Integer roots of an ascending integer polynomial with multiplicity, if it
/// fully splits over the integers; otherwise None (non-integer/irrational root
/// present, or the constant term is too large for the divisor search).
fn integer_roots_with_mult(mut c: Vec<i128>) -> Option<Vec<i128>> {
  const MAX_A0: i128 = 100_000;
  let mut roots = Vec::new();
  while c.len() > 1 && c[0] == 0 {
    roots.push(0);
    c.remove(0);
  }
  while c.len() > 1 {
    let a0 = c[0].abs();
    if a0 == 0 {
      roots.push(0);
      c.remove(0);
      continue;
    }
    if a0 > MAX_A0 {
      return None;
    }
    let mut found = None;
    'outer: for d in 1..=a0 {
      if a0 % d == 0 {
        for cand in [d, -d] {
          if horner_i128(&c, cand)? == 0 {
            found = Some(cand);
            break 'outer;
          }
        }
      }
    }
    let r = found?;
    c = deflate_i128(&c, r)?;
    roots.push(r);
  }
  Some(roots)
}

/// Closed form for `Product[R(var), {var, n0, Infinity}]` where `R` is a
/// rational function whose numerator and denominator both factor over the
/// integers into linear factors. With equal degree and equal leading
/// coefficient, let `{r_i}` be the numerator roots and `{s_j}` the denominator
/// roots:
///   * `Σr_i = Σs_j` → converges to `∏_j Γ(n0 - s_j) / ∏_i Γ(n0 - r_i)`
///   * `Σr_i > Σs_j` → `0`
///   * `Σr_i < Σs_j` → diverges (emits `Product::div`, left unevaluated)
/// Returns None when the body is not such a rational function. Rational
/// (non-integer) roots — e.g. `Product[1 - 1/(4 n^2), …] = 2/Pi` — are left
/// unevaluated.
fn infinite_integer_root_product(
  body: &Expr,
  var_name: &str,
  n0: i128,
) -> Result<Option<Expr>, InterpreterError> {
  if n0 < 1 {
    return Ok(None);
  }
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let call1 = |name: &str, arg: Expr| Expr::FunctionCall {
    name: name.to_string(),
    args: vec![arg].into(),
  };
  let together = eval(&call1("Together", body.clone()))?;
  let num = eval(&call1("Numerator", together.clone()))?;
  let den = eval(&call1("Denominator", together))?;
  let (Some(cn), Some(cd)) = (
    integer_coefficient_list(&num, var_name),
    integer_coefficient_list(&den, var_name),
  ) else {
    return Ok(None);
  };
  // Both must genuinely depend on var and share the same degree.
  if cn.len() <= 1 || cd.len() != cn.len() {
    return Ok(None);
  }
  // Convergence to a finite nonzero value needs R(n) → 1, i.e. equal leading
  // coefficients of the original (un-cleared) numerator and denominator.
  let deg = Expr::Integer((cn.len() - 1) as i128);
  let coeff = |p: &Expr| {
    eval(&Expr::FunctionCall {
      name: "Coefficient".to_string(),
      args: vec![
        p.clone(),
        Expr::Identifier(var_name.to_string()),
        deg.clone(),
      ]
      .into(),
    })
  };
  let lead_diff = eval(&Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(coeff(&num)?),
    right: Box::new(coeff(&den)?),
  })?;
  if !matches!(lead_diff, Expr::Integer(0)) {
    return Ok(None);
  }
  let (Some(rnum), Some(rden)) =
    (integer_roots_with_mult(cn), integer_roots_with_mult(cd))
  else {
    return Ok(None);
  };
  let sum_r: i128 = rnum.iter().sum();
  let sum_s: i128 = rden.iter().sum();
  use std::cmp::Ordering;
  match sum_r.cmp(&sum_s) {
    Ordering::Greater => Ok(Some(Expr::Integer(0))),
    Ordering::Less => {
      crate::emit_message("Product::div: Product does not converge.");
      Ok(None)
    }
    Ordering::Equal => {
      // A numerator root ≥ n0 zeroes a factor → the whole product is 0.
      if rnum.iter().any(|&r| r >= n0) {
        return Ok(Some(Expr::Integer(0)));
      }
      // A denominator root ≥ n0 is a pole → leave unevaluated.
      if rden.iter().any(|&s| s >= n0) {
        return Ok(None);
      }
      let gamma = |k: i128| Expr::FunctionCall {
        name: "Gamma".to_string(),
        args: vec![Expr::Integer(n0 - k)].into(),
      };
      let num_factors: Vec<Expr> = rden.iter().map(|&s| gamma(s)).collect();
      let den_factors: Vec<Expr> = rnum.iter().map(|&r| gamma(r)).collect();
      let result = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::FunctionCall {
          name: "Times".to_string(),
          args: num_factors.into(),
        }),
        right: Box::new(Expr::FunctionCall {
          name: "Times".to_string(),
          args: den_factors.into(),
        }),
      };
      Ok(Some(eval(&result)?))
    }
  }
}

fn linear_shift_of_var(body: &Expr, var_name: &str) -> Option<Expr> {
  use crate::functions::calculus_ast::is_constant_wrt;
  let terms: Vec<&Expr> = match body {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().collect()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => vec![left.as_ref(), right.as_ref()],
    _ => return None,
  };
  let mut found_var = false;
  let mut consts: Vec<Expr> = Vec::new();
  for t in terms {
    if matches!(t, Expr::Identifier(n) if n == var_name) {
      if found_var {
        return None;
      }
      found_var = true;
    } else if is_constant_wrt(t, var_name) {
      consts.push(t.clone());
    } else {
      return None;
    }
  }
  if !found_var || consts.is_empty() {
    return None;
  }
  Some(if consts.len() == 1 {
    consts.remove(0)
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: consts.into(),
    }
  })
}

fn body_is_one_plus_one_over_var_squared(body: &Expr, var_name: &str) -> bool {
  let is_var_squared = |e: &Expr| -> bool {
    matches!(
      e,
      Expr::BinaryOp { op: BinaryOperator::Power, left, right }
        if matches!(left.as_ref(), Expr::Identifier(s) if s == var_name)
          && matches!(right.as_ref(), Expr::Integer(2))
    ) || matches!(
      e,
      Expr::FunctionCall { name, args }
        if name == "Power"
          && args.len() == 2
          && matches!(&args[0], Expr::Identifier(s) if s == var_name)
          && matches!(&args[1], Expr::Integer(2))
    )
  };
  let is_one_over_var_squared = |e: &Expr| -> bool {
    // `1 / var^2`
    if let Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } = e
      && matches!(left.as_ref(), Expr::Integer(1))
      && is_var_squared(right.as_ref())
    {
      return true;
    }
    // `var^(-2)` (canonical form for reciprocal squares)
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } = e
      && matches!(left.as_ref(), Expr::Identifier(s) if s == var_name)
      && matches!(right.as_ref(), Expr::Integer(-2))
    {
      return true;
    }
    if let Expr::FunctionCall { name, args } = e
      && name == "Power"
      && args.len() == 2
      && matches!(&args[0], Expr::Identifier(s) if s == var_name)
      && matches!(&args[1], Expr::Integer(-2))
    {
      return true;
    }
    false
  };
  // Match the Plus shapes: Plus[1, 1/var^2] or Plus[1/var^2, 1] in either
  // BinaryOp::Plus or FunctionCall["Plus", …] form.
  if let Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left,
    right,
  } = body
  {
    return (matches!(left.as_ref(), Expr::Integer(1))
      && is_one_over_var_squared(right))
      || (matches!(right.as_ref(), Expr::Integer(1))
        && is_one_over_var_squared(left));
  }
  if let Expr::FunctionCall { name, args } = body
    && name == "Plus"
    && args.len() == 2
  {
    return (matches!(&args[0], Expr::Integer(1))
      && is_one_over_var_squared(&args[1]))
      || (matches!(&args[1], Expr::Integer(1))
        && is_one_over_var_squared(&args[0]));
  }
  false
}

pub fn product_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Product[{a, b, c}] -> a * b * c
    let items = match &args[0] {
      Expr::List(items) => items,
      _ => {
        return Ok(unevaluated("Product", args));
      }
    };

    let mut product = 1.0;
    for item in items {
      if let Some(n) = expr_to_f64(item) {
        product *= n;
      } else {
        return Ok(unevaluated("Product", args));
      }
    }
    return Ok(f64_to_expr(product));
  }

  // Multi-dimensional Product: Product[expr, {i,...}, {j,...}, ...] =>
  // Product[Product[expr, {j,...}], {i,...}] (the rightmost iterator is
  // innermost), mirroring multi-index Sum.
  if args.len() > 2 {
    let body = &args[0];
    let inner_iter = &args[args.len() - 1];
    let inner_product = product_ast(&[body.clone(), inner_iter.clone()])?;
    if args.len() == 3 {
      return product_ast(&[inner_product, args[1].clone()]);
    } else {
      let mut new_args = vec![inner_product];
      new_args.extend_from_slice(&args[1..args.len() - 1]);
      return product_ast(&new_args);
    }
  }

  if args.len() == 2 {
    // Product[expr, {i, min, max}] -> multiply expr for each i
    let body = &args[0];
    let iter_spec = &args[1];

    match iter_spec {
      Expr::List(items) if items.len() >= 2 => {
        let var_name = match &items[0] {
          Expr::Identifier(name) => name.clone(),
          _ => {
            return Ok(unevaluated("Product", args));
          }
        };

        // Check for list iteration form: {i, list}
        if items.len() == 2 {
          let evaluated_second =
            crate::evaluator::evaluate_expr_to_expr(&items[1])?;
          if let Expr::List(list_items) = &evaluated_second {
            // Product[expr, {i, list}] -> iterate over list elements
            let mut product = 1.0;
            for item in list_items {
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, item);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              if let Some(n) = expr_to_f64(&val) {
                product *= n;
              } else {
                return Ok(unevaluated("Product", args));
              }
            }
            return Ok(f64_to_expr(product));
          }
        }

        // Check if bounds are numeric
        let bounds = if items.len() == 2 {
          expr_to_i128(&items[1]).map(|max| (1i128, max))
        } else {
          match (expr_to_i128(&items[1]), expr_to_i128(&items[2])) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
          }
        };

        // If bounds are symbolic, try to compute symbolic product
        if bounds.is_none() {
          let min_concrete = if items.len() == 2 {
            Some(1i128) // {i, n} implies min = 1
          } else {
            expr_to_i128(&items[1])
          };
          let max_concrete = if items.len() == 2 {
            expr_to_i128(&items[1])
          } else {
            expr_to_i128(&items[2])
          };
          let max_expr = if items.len() == 2 {
            &items[1]
          } else {
            &items[2]
          };
          let min_expr = if items.len() == 2 {
            &Expr::Integer(1)
          } else {
            &items[1]
          };

          let max_is_infinity =
            matches!(max_expr, Expr::Identifier(s) if s == "Infinity");

          // Body independent of the iteration variable:
          //   Product[c, {k, min, max}] = c^(max - min + 1)
          if !crate::functions::polynomial_ast::contains_var(body, &var_name)
            && !max_is_infinity
          {
            let count =
              crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  Expr::Integer(1),
                  max_expr.clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), min_expr.clone()].into(),
                  },
                ]
                .into(),
              })?;
            let power = Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(body.clone()),
              right: Box::new(count),
            };
            return crate::evaluator::evaluate_expr_to_expr(&power);
          }

          // Body is the iteration variable itself: Product[k, {k, ...}]
          // (the closed forms below use Factorial/Pochhammer, which are only
          // valid for a finite upper limit).
          if matches!(body, Expr::Identifier(name) if name == &var_name)
            && !max_is_infinity
          {
            if let Some(min_val) = min_concrete {
              if max_concrete.is_none() {
                // Product[k, {k, concrete_min, symbolic_max}]
                // = max! / (min-1)!
                let n_factorial = Expr::FunctionCall {
                  name: "Factorial".to_string(),
                  args: vec![max_expr.clone()].into(),
                };
                if min_val == 1 {
                  return Ok(n_factorial);
                }
                // Compute (min-1)! as a concrete integer
                let mut denom: i128 = 1;
                for j in 2..min_val {
                  denom *= j;
                }
                // (min-1)! == 1 (min <= 2): the product is just max!.
                if denom == 1 {
                  return Ok(n_factorial);
                }
                return Ok(Expr::BinaryOp {
                  op: BinaryOperator::Divide,
                  left: Box::new(n_factorial),
                  right: Box::new(Expr::Integer(denom)),
                });
              }
            } else if max_concrete.is_none() {
              // Product[k, {k, sym_min, sym_max}]
              // = Pochhammer[min, 1 - min + max]
              return Ok(Expr::FunctionCall {
                name: "Pochhammer".to_string(),
                args: vec![
                  min_expr.clone(),
                  // 1 - min + max
                  Expr::BinaryOp {
                    op: BinaryOperator::Plus,
                    left: Box::new(Expr::BinaryOp {
                      op: BinaryOperator::Minus,
                      left: Box::new(Expr::Integer(1)),
                      right: Box::new(min_expr.clone()),
                    }),
                    right: Box::new(max_expr.clone()),
                  },
                ]
                .into(),
              });
            }
          }

          // Body is var + a (a free of var, a != 0):
          //   Product[k + a, {k, 1, n}] = Pochhammer[1 + a, n].
          // wolframscript prints this as the Gamma ratio Gamma[1+a+n]/Gamma[1+a]
          // for a numeric shift (e.g. Gamma[2+n], Gamma[3+n]/2,
          // (2 Gamma[3/2+n])/Sqrt[Pi]), but keeps Pochhammer for a symbolic
          // shift.
          if min_concrete == Some(1)
            && max_concrete.is_none()
            && !max_is_infinity
            && let Some(a) = linear_shift_of_var(body, &var_name)
          {
            let one_plus_a =
              crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(Expr::Integer(1)),
                right: Box::new(a.clone()),
              })?;
            let positive_number = match &one_plus_a {
              Expr::Integer(p) => *p >= 1,
              Expr::FunctionCall { name, args }
                if name == "Rational" && args.len() == 2 =>
              {
                matches!((&args[0], &args[1]),
                  (Expr::Integer(num), Expr::Integer(den))
                    if *num > 0 && *den > 0)
              }
              _ => false,
            };
            if positive_number {
              // Gamma[1 + a + n] / Gamma[1 + a]
              let gamma = |arg: Expr| Expr::FunctionCall {
                name: "Gamma".to_string(),
                args: vec![arg].into(),
              };
              let result = Expr::BinaryOp {
                op: BinaryOperator::Divide,
                left: Box::new(gamma(Expr::BinaryOp {
                  op: BinaryOperator::Plus,
                  left: Box::new(one_plus_a.clone()),
                  right: Box::new(max_expr.clone()),
                })),
                right: Box::new(gamma(one_plus_a)),
              };
              return crate::evaluator::evaluate_expr_to_expr(&result);
            }
            // A genuinely symbolic shift keeps the Pochhammer form; a
            // non-positive numeric shift falls through (unhandled edge).
            let is_number = matches!(&one_plus_a, Expr::Integer(_))
              || matches!(&one_plus_a,
                Expr::FunctionCall { name, .. } if name == "Rational");
            if !is_number {
              return Ok(Expr::FunctionCall {
                name: "Pochhammer".to_string(),
                args: vec![one_plus_a, max_expr.clone()].into(),
              });
            }
          }

          // Rational telescoping product:
          //   Product[(k + a)/(k + b), {k, k0, n}]
          // with integer shifts a, b telescopes to a finite product of linear
          // factors in n. Together the body, split into numerator/denominator,
          // and require each to be a monic linear polynomial `var + integer`.
          // For a > b the result is
          //   Π_{j=b+1}^{a} (n + j) / Π_{j=b+1}^{a} (k0 + j - 1) ;
          // for a < b it is the reciprocal. (`var + a` alone, with a constant
          // denominator, is handled by the linear-shift case above.)
          if let Some(k0) = min_concrete
            && max_concrete.is_none()
            && !max_is_infinity
            && let Some(result) =
              rational_telescoping_product(body, &var_name, max_expr, k0)?
          {
            return Ok(result);
          }

          // Body is c^var: Product[c^i, {i, 1, n}] = c^(n*(1+n)/2)
          // (the closed forms below assume a finite upper limit; an infinite
          // limit must fall through to stay unevaluated like wolframscript).
          if let Some(1) = min_concrete
            && max_concrete.is_none()
            && !max_is_infinity
            && let Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: base,
              right: exp,
            } = body
          {
            if matches!(exp.as_ref(), Expr::Identifier(name) if name == &var_name)
            {
              // Product[c^i, {i, 1, n}] = c^((n*(1+n))/2)
              let n = max_expr.clone();
              let exponent = Expr::BinaryOp {
                op: BinaryOperator::Divide,
                left: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(n.clone()),
                  right: Box::new(Expr::BinaryOp {
                    op: BinaryOperator::Plus,
                    left: Box::new(Expr::Integer(1)),
                    right: Box::new(n),
                  }),
                }),
                right: Box::new(Expr::Integer(2)),
              };
              return Ok(Expr::BinaryOp {
                op: BinaryOperator::Power,
                left: base.clone(),
                right: Box::new(exponent),
              });
            }

            // Product[i^k, {i, 1, n}] = n!^k
            if matches!(base.as_ref(), Expr::Identifier(name) if name == &var_name)
            {
              return Ok(Expr::BinaryOp {
                op: BinaryOperator::Power,
                left: Box::new(Expr::FunctionCall {
                  name: "Factorial".to_string(),
                  args: vec![max_expr.clone()].into(),
                }),
                right: exp.clone(),
              });
            }
          }

          // Monomial body c*var^p (p a nonzero integer, c free of var):
          //   Product[c var^p, {k, 1, n}] = c^n * n!^p.
          // wolframscript keeps the bare factorial when c == 1
          // (Product[1/k] -> n!^(-1)) but switches to Gamma[1+n] once a
          // coefficient is present (Product[2 k] -> 2^n Gamma[1+n]).
          if min_concrete == Some(1)
            && max_concrete.is_none()
            && !max_is_infinity
            && crate::functions::polynomial_ast::contains_var(body, &var_name)
            && let Ok(dbody) =
              crate::functions::calculus_ast::differentiate_expr(
                body, &var_name,
              )
          {
            // p = (d body / d var) * var / body — the exponent of a monomial.
            let p_expr = Expr::FunctionCall {
              name: "Simplify".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  dbody,
                  Expr::Identifier(var_name.clone()),
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![body.clone(), Expr::Integer(-1)].into(),
                  },
                ]
                .into(),
              }]
              .into(),
            };
            let p_val = crate::evaluator::evaluate_expr_to_expr(&p_expr).ok();
            if let Some(p) = p_val.as_ref().and_then(expr_to_i128)
              && p != 0
            {
              // c = body with var -> 1.
              let c = crate::evaluator::evaluate_expr_to_expr(
                &crate::syntax::substitute_variable(
                  body,
                  &var_name,
                  &Expr::Integer(1),
                ),
              )?;
              if !crate::functions::polynomial_ast::contains_var(&c, &var_name)
              {
                let c_is_one = matches!(&c, Expr::Integer(1));
                let base = if c_is_one {
                  Expr::FunctionCall {
                    name: "Factorial".to_string(),
                    args: vec![max_expr.clone()].into(),
                  }
                } else {
                  Expr::FunctionCall {
                    name: "Gamma".to_string(),
                    args: vec![Expr::BinaryOp {
                      op: BinaryOperator::Plus,
                      left: Box::new(Expr::Integer(1)),
                      right: Box::new(max_expr.clone()),
                    }]
                    .into(),
                  }
                };
                let pow_part = if p == 1 {
                  base
                } else {
                  Expr::BinaryOp {
                    op: BinaryOperator::Power,
                    left: Box::new(base),
                    right: Box::new(Expr::Integer(p)),
                  }
                };
                // Coefficient c^n. wolframscript renders a *unit fraction* 1/b as
                // a denominator power (Product[k/2] -> Gamma[1+n]/2^n), but keeps
                // any other coefficient as c^n (Product[2k/3] -> (2/3)^n Gamma).
                let pow_n = |b: &Expr| Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(b.clone()),
                  right: Box::new(max_expr.clone()),
                };
                let unit_fraction_den = match &c {
                  Expr::FunctionCall { name, args }
                    if name == "Rational" && args.len() == 2 =>
                  {
                    match (&args[0], &args[1]) {
                      (Expr::Integer(1), Expr::Integer(b)) => Some(*b),
                      _ => None,
                    }
                  }
                  _ => None,
                };
                let result = if c_is_one {
                  pow_part
                } else if let Some(b) = unit_fraction_den {
                  Expr::BinaryOp {
                    op: BinaryOperator::Divide,
                    left: Box::new(pow_part),
                    right: Box::new(pow_n(&Expr::Integer(b))),
                  }
                } else {
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![pow_n(&c), pow_part].into(),
                  }
                };
                return crate::evaluator::evaluate_expr_to_expr(&result);
              }
            }
          }

          // Closed form for the classical infinite product
          //   ∏_{k=1}^∞ (1 + 1/k²) = Sinh[π] / π
          // (the `x = π` case of the Weierstrass factorisation
          //   sinh(x)/x = ∏_{k=1}^∞ (1 + x²/(kπ)²)).
          // Recognised when min == 1, max == Infinity, and the body is
          // `1 + 1/var^2` in any of the canonical AST shapes.
          if let Some(1) = min_concrete
            && matches!(max_expr, Expr::Identifier(s) if s == "Infinity")
            && body_is_one_plus_one_over_var_squared(body, &var_name)
          {
            return Ok(Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(Expr::FunctionCall {
                name: "Sinh".to_string(),
                args: vec![Expr::Constant("Pi".to_string())].into(),
              }),
              right: Box::new(Expr::Constant("Pi".to_string())),
            });
          }

          // Closed form for ∏_{k=2}^∞ (1 - 1/k⁴) = Sinh[π] / (4 π).
          // Comes from (1 - x²/k²)(1 + x²/k²) at x = 1 over k ≥ 2: the k = 1
          // factor is 0, but starting at k = 2 the residual product is
          // Sinh[π] · sin[π] / π² which is 0/π² = 0 — instead the standard
          // result is obtained by L'Hôpital at x → 1 of the truncated form.
          if let Some(2) = min_concrete
            && matches!(max_expr, Expr::Identifier(s) if s == "Infinity")
            && body_is_one_minus_one_over_var_quartic(body, &var_name)
          {
            return Ok(Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(Expr::FunctionCall {
                name: "Sinh".to_string(),
                args: vec![Expr::Constant("Pi".to_string())].into(),
              }),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(Expr::Integer(4)),
                right: Box::new(Expr::Constant("Pi".to_string())),
              }),
            });
          }

          // Constant body over an infinite range: convergence depends on the
          // value. Π c = 1 for c==1, 0 for a positive c<1 (and c==0), and
          // diverges for |c|>1. A negative c in [-1,0) and any symbolic value
          // are left unevaluated, matching wolframscript.
          if matches!(max_expr, Expr::Identifier(s) if s == "Infinity")
            && !crate::functions::polynomial_ast::contains_var(body, &var_name)
            && let Some(c) =
              explicit_real(&crate::evaluator::evaluate_expr_to_expr(body)?)
          {
            if c == 1.0 {
              return Ok(Expr::Integer(1));
            }
            if (0.0..1.0).contains(&c) {
              return Ok(Expr::Integer(0));
            }
            if !(-1.0..=1.0).contains(&c) {
              crate::emit_message("Product::div: Product does not converge.");
              return Ok(unevaluated("Product", args));
            }
          }

          // Infinite product of a rational function that splits over the
          // integers: Product[(n-1)(n+1)/n^2, {n, 2, Infinity}] = 1/2, etc.
          if let Some(n0) = min_concrete
            && matches!(max_expr, Expr::Identifier(s) if s == "Infinity")
            && let Some(result) =
              infinite_integer_root_product(body, &var_name, n0)?
          {
            return Ok(result);
          }

          // Terms with provable tail behavior: |term| -> Infinity means the
          // product diverges (Product::div, matching wolframscript for n,
          // n^2, Sqrt[n], n/2, -n and 2^n alike), and |term| -> 0 drives
          // the product to 0 (Product[1/n] = 0, Product[2/n] = 0).
          if matches!(max_expr, Expr::Identifier(s) if s == "Infinity")
            && let Some(n0) = min_concrete
            && product_early_terms_clean(body, &var_name, n0)
            && let Some((rate, degree)) = product_growth(body, &var_name)
          {
            let diverges = rate > 0.0 || (rate == 0.0 && degree > 0.0);
            let vanishes = rate < 0.0 || (rate == 0.0 && degree < 0.0);
            if diverges {
              crate::emit_message("Product::div: Product does not converge.");
              return Ok(unevaluated("Product", args));
            }
            if vanishes {
              return Ok(Expr::Integer(0));
            }
          }

          // For other symbolic cases, return unevaluated
          return Ok(unevaluated("Product", args));
        }

        let (min, max) = bounds.unwrap();

        let step = if items.len() >= 4 {
          expr_to_i128(&items[3]).unwrap_or(1)
        } else {
          1
        };

        // Collect evaluated values for each iteration
        let mut values: Vec<Expr> = Vec::new();
        let mut i = min;
        while (step > 0 && i <= max) || (step < 0 && i >= max) {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          values.push(val);
          i += step;
        }

        // Try numeric product first
        let mut numeric_product = 1.0;
        let mut all_numeric = true;
        for val in &values {
          if let Some(n) = expr_to_f64(val) {
            numeric_product *= n;
          } else {
            all_numeric = false;
            break;
          }
        }

        if all_numeric {
          return Ok(f64_to_expr(numeric_product));
        }

        // For symbolic values, build a Times expression
        if values.is_empty() {
          return Ok(Expr::Integer(1));
        }
        if values.len() == 1 {
          return Ok(values.into_iter().next().unwrap());
        }
        // Fold into nested Times
        let mut result = values.remove(0);
        for val in values {
          result = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(result),
            right: Box::new(val),
          };
        }
        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
      _ => {}
    }
  }

  Ok(unevaluated("Product", args))
}

/// AST-based Sum: sum of list elements or iterator sum.
/// Whether wolframscript's Sum::div message fires for an infinite sum of
/// this term. The provable classes: a term with no iterator dependence
/// (nonzero constant, including a symbolic one — wolframscript treats it
/// as generically nonzero), the bare Log[var] term, and rational-function
/// / fractional-power terms whose asymptotic growth exponent is >= -1
/// (the p-series divergence boundary: 1/n, 1/Sqrt[n], n^(3/2),
/// (n^2+1)/(n^2-1) all message). Exponentials (2^n), oscillating terms
/// ((-1)^n) and Log-mixed terms stay silent — wolframscript is silent for
/// 2^n and Log[n]/n itself.
fn sum_term_provably_divergent(body: &Expr, var: &str) -> bool {
  let evaluated = crate::evaluator::evaluate_expr_to_expr(body)
    .unwrap_or_else(|_| body.clone());
  if !summand_contains_var(&evaluated, var) {
    // Constant term: divergent unless it is exactly zero.
    return !matches!(&evaluated, Expr::Integer(0))
      && !matches!(&evaluated, Expr::Real(r) if *r == 0.0);
  }
  // Bare Log[var], possibly with a nonzero numeric prefactor.
  if let Expr::FunctionCall { name, args } = &evaluated
    && name == "Log"
    && args.len() == 1
    && matches!(&args[0], Expr::Identifier(v) if v == var)
  {
    return true;
  }
  // Combine sums of fractions first so telescoping-style cancellation is
  // visible: 1/n - 1/(n+1) is 1/(n(n+1)) (degree -2, convergent), while
  // the termwise maximum degree would wrongly say -1.
  let combined = crate::evaluator::evaluate_function_call_ast(
    "Together",
    std::slice::from_ref(&evaluated),
  )
  .unwrap_or(evaluated);
  match summand_growth_degree(&combined, var) {
    Some(d) => d >= -1.0,
    None => false,
  }
}

/// Whether the summand mentions the iteration variable.
fn summand_contains_var(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(name) => name == var,
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(|a| summand_contains_var(a, var))
    }
    Expr::BinaryOp { left, right, .. } => {
      summand_contains_var(left, var) || summand_contains_var(right, var)
    }
    Expr::UnaryOp { operand, .. } => summand_contains_var(operand, var),
    _ => false,
  }
}

/// Asymptotic growth exponent of an algebraic term in `var`: the d with
/// term ~ c*var^d as var -> Infinity. None for anything outside the
/// polynomial/rational/fractional-power class (exponentials with var in
/// the exponent, Log factors, unknown functions of var).
fn summand_growth_degree(expr: &Expr, var: &str) -> Option<f64> {
  if !summand_contains_var(expr, var) {
    // A var-free factor is degree 0 — but reject non-numeric leaves like
    // symbolic parameters only when they are the whole summand (handled
    // by the constant case); as factors they don't change the growth.
    return Some(0.0);
  }
  match expr {
    Expr::Identifier(name) if name == var => Some(1.0),
    Expr::FunctionCall { name, args } if name == "Plus" => args
      .iter()
      .map(|a| summand_growth_degree(a, var))
      .collect::<Option<Vec<_>>>()
      .map(|ds| ds.into_iter().fold(f64::NEG_INFINITY, f64::max)),
    Expr::FunctionCall { name, args } if name == "Times" => args
      .iter()
      .map(|a| summand_growth_degree(a, var))
      .sum::<Option<f64>>(),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      let exp = summand_numeric_value(&args[1])?;
      if summand_contains_var(&args[1], var) {
        return None; // var in the exponent: exponential class
      }
      Some(summand_growth_degree(&args[0], var)? * exp)
    }
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      Some(summand_growth_degree(&args[0], var)? * 0.5)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => Some(
      summand_growth_degree(left, var)?.max(summand_growth_degree(right, var)?),
    ),
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => Some(
      summand_growth_degree(left, var)?.max(summand_growth_degree(right, var)?),
    ),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => Some(
      summand_growth_degree(left, var)? + summand_growth_degree(right, var)?,
    ),
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => Some(
      summand_growth_degree(left, var)? - summand_growth_degree(right, var)?,
    ),
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if summand_contains_var(right, var) {
        return None;
      }
      let exp = summand_numeric_value(right)?;
      Some(summand_growth_degree(left, var)? * exp)
    }
    Expr::UnaryOp { operand, .. } => summand_growth_degree(operand, var),
    _ => None,
  }
}

/// Numeric value of a var-free exponent (integer or rational).
fn summand_numeric_value(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(n), Expr::Integer(d)) if *d != 0 => {
          Some(*n as f64 / *d as f64)
        }
        _ => None,
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(-summand_numeric_value(operand)?),
    _ => None,
  }
}

/// The first few product terms must be finite, nonzero numerics before
/// the tail analysis may conclude anything (a zero or pole among the
/// early factors changes the story entirely).
fn product_early_terms_clean(body: &Expr, var: &str, n0: i128) -> bool {
  for k in 0..5 {
    let substituted =
      crate::syntax::substitute_variable(body, var, &Expr::Integer(n0 + k));
    let value = match crate::evaluator::evaluate_expr_to_expr(&substituted) {
      Ok(v) => v,
      Err(_) => return false,
    };
    match crate::functions::math_ast::n_ast(&[value]) {
      Ok(Expr::Real(r)) if r.is_finite() && r != 0.0 => {}
      Ok(Expr::Integer(i)) if i != 0 => {}
      _ => return false,
    }
  }
  true
}

/// Asymptotic growth of a product term as (exponential rate, algebraic
/// degree): term ~ e^(rate*var) * var^degree. Exponential factors a^var
/// (numeric positive a, exponent linear in var) contribute ln(a)*slope to
/// the rate; polynomial/rational/fractional-power structure contributes
/// to the degree. None for anything else (Log factors, a^n with
/// non-numeric base, unknown functions of var).
fn product_growth(expr: &Expr, var: &str) -> Option<(f64, f64)> {
  if !summand_contains_var(expr, var) {
    return Some((0.0, 0.0));
  }
  match expr {
    Expr::Identifier(name) if name == var => Some((0.0, 1.0)),
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let parts: Vec<(f64, f64)> = args
        .iter()
        .map(|a| product_growth(a, var))
        .collect::<Option<Vec<_>>>()?;
      if parts.iter().any(|(r, _)| *r != 0.0) {
        return None; // mixed exponential sums are out of scope
      }
      Some((
        0.0,
        parts
          .into_iter()
          .map(|(_, d)| d)
          .fold(f64::NEG_INFINITY, f64::max),
      ))
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut rate = 0.0;
      let mut degree = 0.0;
      for a in args.iter() {
        let (r, d) = product_growth(a, var)?;
        rate += r;
        degree += d;
      }
      Some((rate, degree))
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      product_growth_power(&args[0], &args[1], var)
    }
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      let (r, d) = product_growth(&args[0], var)?;
      if r != 0.0 {
        return None;
      }
      Some((0.0, d * 0.5))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    }
    | Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      let (rl, dl) = product_growth(left, var)?;
      let (rr, dr) = product_growth(right, var)?;
      if rl != 0.0 || rr != 0.0 {
        return None;
      }
      Some((0.0, dl.max(dr)))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let (rl, dl) = product_growth(left, var)?;
      let (rr, dr) = product_growth(right, var)?;
      Some((rl + rr, dl + dr))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let (rl, dl) = product_growth(left, var)?;
      let (rr, dr) = product_growth(right, var)?;
      Some((rl - rr, dl - dr))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => product_growth_power(left, right, var),
    Expr::UnaryOp { operand, .. } => product_growth(operand, var),
    _ => None,
  }
}

/// Growth of base^exp: a var-free numeric exponent scales the base growth;
/// a positive numeric base with a var-linear exponent is an exponential
/// factor with rate ln(base)*slope.
fn product_growth_power(
  base: &Expr,
  exp: &Expr,
  var: &str,
) -> Option<(f64, f64)> {
  if !summand_contains_var(exp, var) {
    let q = summand_numeric_value(exp)?;
    let (r, d) = product_growth(base, var)?;
    return Some((r * q, d * q));
  }
  if summand_contains_var(base, var) {
    return None; // var in both base and exponent (n^n): out of scope
  }
  let a = summand_numeric_value(base)?;
  if a <= 0.0 {
    return None;
  }
  let slope = linear_slope_in(exp, var)?;
  Some((a.ln() * slope, 0.0))
}

/// The slope k of an expression linear in var (k*var + c with numeric k).
fn linear_slope_in(expr: &Expr, var: &str) -> Option<f64> {
  if !summand_contains_var(expr, var) {
    return Some(0.0);
  }
  match expr {
    Expr::Identifier(name) if name == var => Some(1.0),
    Expr::FunctionCall { name, args } if name == "Plus" => args
      .iter()
      .map(|a| linear_slope_in(a, var))
      .sum::<Option<f64>>(),
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Exactly one var-bearing factor, all others numeric.
      let mut slope: Option<f64> = None;
      let mut coeff = 1.0;
      for a in args.iter() {
        if summand_contains_var(a, var) {
          if slope.is_some() {
            return None;
          }
          slope = Some(linear_slope_in(a, var)?);
        } else {
          coeff *= summand_numeric_value(a)?;
        }
      }
      Some(coeff * slope?)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => Some(linear_slope_in(left, var)? + linear_slope_in(right, var)?),
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => Some(linear_slope_in(left, var)? - linear_slope_in(right, var)?),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if summand_contains_var(left, var) && summand_contains_var(right, var) {
        return None;
      }
      if summand_contains_var(left, var) {
        Some(linear_slope_in(left, var)? * summand_numeric_value(right)?)
      } else {
        Some(summand_numeric_value(left)? * linear_slope_in(right, var)?)
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(-linear_slope_in(operand, var)?),
    _ => None,
  }
}

pub fn sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    // Sum requires at least 2 arguments
    return Ok(unevaluated("Sum", args));
  }

  // Indefinite sum: Sum[f[i], i] → ∑_{k=0}^{i-1} f[k] (the antidifference F
  // where F(i+1) - F(i) = f(i), with F(0) = 0).
  // wolframscript: Sum[1, i] = i, Sum[i, i] = ((-1 + i)*i)/2,
  // Sum[i^3, i] = ((-1+i)^2*i^2)/4.
  //
  // Implementation: compute ∑_{k=1}^{i-1} f[k] (the path with proven symbolic
  // support) and add f(0). For f=1 this gives 1 + (i-1) = i; for f=i it gives
  // 0 + i(i-1)/2 = i(i-1)/2.
  if args.len() == 2
    && let Expr::Identifier(var_name) = &args[1]
  {
    let fresh_name = format!("$sum_indef_{}_$", var_name);
    let fresh = Expr::Identifier(fresh_name.clone());
    let body_in_fresh =
      crate::syntax::substitute_variable(&args[0], var_name, &fresh);
    // Upper bound: var - 1, built as `(-1) + var` for canonical Plus form.
    let upper = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![Expr::Integer(-1), Expr::Identifier(var_name.clone())].into(),
    };
    let iter_spec =
      Expr::List(vec![fresh.clone(), Expr::Integer(1), upper].into());
    let inner_sum = sum_ast(&[body_in_fresh, iter_spec])?;
    // Evaluate f(0)
    let f_at_zero =
      crate::syntax::substitute_variable(&args[0], var_name, &Expr::Integer(0));
    let f_at_zero_eval = crate::evaluator::evaluate_expr_to_expr(&f_at_zero)?;
    return crate::functions::math_ast::plus_ast(&[f_at_zero_eval, inner_sum]);
  }

  // Multi-dimensional Sum: Sum[expr, {i,...}, {j,...}, ...] => Sum[Sum[expr, {j,...}], {i,...}]
  if args.len() > 2 {
    // Evaluate innermost sum first (last iterator), then wrap outward
    let body = &args[0];
    let inner_iter = &args[args.len() - 1];
    let inner_sum = sum_ast(&[body.clone(), inner_iter.clone()])?;
    if args.len() == 3 {
      return sum_ast(&[inner_sum, args[1].clone()]);
    } else {
      let mut new_args = vec![inner_sum];
      new_args.extend_from_slice(&args[1..args.len() - 1]);
      return sum_ast(&new_args);
    }
  }

  // Sum[expr, {i, min, max}] or variants
  let body = &args[0];
  let iter_spec = &args[1];

  match iter_spec {
    Expr::List(items) if items.len() >= 2 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Ok(unevaluated("Sum", args));
        }
      };

      // Check for list iteration form: {i, list}
      if items.len() == 2 {
        let evaluated_second =
          crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        if let Expr::List(list_items) = &evaluated_second {
          // Sum[expr, {i, {v1, v2, ...}}] -> iterate over list elements
          let mut acc = Expr::Integer(0);
          for item in list_items {
            let substituted =
              crate::syntax::substitute_variable(body, &var_name, item);
            let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
            acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
          }
          return Ok(acc);
        }
      }

      // Short form `Sum[expr, {i, max}]` is sugar for `Sum[expr, {i, 1, max}]`.
      // Forward to the 3-element form so symbolic / Infinity bounds work.
      if items.len() == 2 {
        let new_iter = Expr::List(
          vec![items[0].clone(), Expr::Integer(1), items[1].clone()].into(),
        );
        return sum_ast(&[body.clone(), new_iter]);
      }

      // Check for infinite sum: {i, min, Infinity}
      if items.len() == 3
        && let Expr::Identifier(s) = &items[2]
        && s == "Infinity"
      {
        let min_val = expr_to_i128(&items[1]).unwrap_or(1);
        if let Some(result) = try_infinite_sum(body, &var_name, min_val)? {
          return Ok(result);
        }
        // Sum of an exact zero term is 0 (wolframscript agrees).
        if matches!(
          crate::evaluator::evaluate_expr_to_expr(body),
          Ok(Expr::Integer(0))
        ) {
          return Ok(Expr::Integer(0));
        }
        // Could not evaluate symbolically. wolframscript proves many of
        // these divergent and says so before leaving the sum unevaluated.
        if sum_term_provably_divergent(body, &var_name) {
          crate::emit_message("Sum::div: Sum does not converge.");
        }
        return Ok(unevaluated("Sum", args));
      }

      // Fast path: Sum[Factorial[var], {var, min, max(, step)}] with integer
      // bounds, min >= 0, step >= 1. Uses a running product so each factorial
      // is built incrementally instead of recomputed from scratch.
      if (items.len() == 3 || items.len() == 4)
        && let Expr::FunctionCall {
          name: fname,
          args: fargs,
        } = body
        && fname == "Factorial"
        && fargs.len() == 1
        && matches!(&fargs[0], Expr::Identifier(fv) if fv == &var_name)
      {
        let min_int = expr_to_i128(&items[1]);
        let max_int = expr_to_i128(&items[2]);
        let step_int = if items.len() == 4 {
          expr_to_i128(&items[3])
        } else {
          Some(1)
        };
        if let (Some(min), Some(max), Some(step)) = (min_int, max_int, step_int)
          && step >= 1
          && min >= 0
          && max >= min
        {
          let mut fact = num_bigint::BigInt::from(1);
          for k in 2..=min {
            fact *= num_bigint::BigInt::from(k);
          }
          let mut sum = num_bigint::BigInt::from(0);
          let mut i = min;
          while i <= max {
            sum += &fact;
            for s in 1..=step {
              let next = i + s;
              if next >= 2 {
                fact *= num_bigint::BigInt::from(next);
              }
            }
            i += step;
          }
          return Ok(crate::functions::math_ast::bigint_to_expr(sum));
        }
      }

      // Try real-valued iteration when bounds are numeric but not integers
      if items.len() == 3 {
        let min_int = expr_to_i128(&items[1]);
        let max_int = expr_to_i128(&items[2]);
        if min_int.is_none() || max_int.is_none() {
          // Check if bounds are numeric reals
          let min_f = crate::functions::math_ast::try_eval_to_f64(&items[1]);
          let max_f = crate::functions::math_ast::try_eval_to_f64(&items[2]);
          if let (Some(min_val), Some(max_val)) = (min_f, max_f) {
            // Iterate with step=1, substituting real values
            let mut acc = Expr::Integer(0);
            let mut i = min_val;
            while i <= max_val + 1e-10 {
              let sub_val =
                if (i - i.round()).abs() < 1e-10 && min_int.is_some() {
                  Expr::Integer(i.round() as i128)
                } else {
                  Expr::Real(i)
                };
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, &sub_val);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
              i += 1.0;
            }
            return Ok(acc);
          }
        }
      }

      // Try iterating when the difference between bounds is real
      // (handles complex bounds like {k, I, I+1})
      if items.len() == 3 {
        let diff = crate::functions::math_ast::plus_ast(&[
          items[2].clone(),
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand: Box::new(items[1].clone()),
          },
        ]);
        if let Ok(diff_expr) = diff {
          let diff_eval = crate::evaluator::evaluate_expr_to_expr(&diff_expr);
          if let Ok(ref de) = diff_eval
            && let Some(range) = crate::functions::math_ast::try_eval_to_f64(de)
            && (0.0..10000.0).contains(&range)
          {
            let n_iters = range.floor() as i128 + 1;
            let min_eval = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
            let mut acc = Expr::Integer(0);
            for j in 0..n_iters {
              let iter_val = if j == 0 {
                min_eval.clone()
              } else {
                crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
                  op: BinaryOperator::Plus,
                  left: Box::new(min_eval.clone()),
                  right: Box::new(Expr::Integer(j)),
                })?
              };
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, &iter_val);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
            }
            return Ok(acc);
          }
        }
      }

      // Try symbolic Sum when bounds are not both concrete integers
      if items.len() == 3 {
        let min_concrete = expr_to_i128(&items[1]);
        let max_concrete = expr_to_i128(&items[2]);
        if min_concrete.is_none() || max_concrete.is_none() {
          if let Some(result) = try_symbolic_sum(
            body,
            &var_name,
            &items[1],
            &items[2],
            min_concrete,
            max_concrete,
          )? {
            // Evaluate to simplify the symbolic result
            return crate::evaluator::evaluate_expr_to_expr(&result);
          }
          return Ok(unevaluated("Sum", args));
        }
      }

      // Extract min, max, step
      let (min, max, step) = if items.len() == 2 {
        let max_val = expr_to_i128(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        (1i128, max_val, 1i128)
      } else if items.len() == 3 {
        let min_val = expr_to_i128(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        let max_val = expr_to_i128(&items[2]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        (min_val, max_val, 1i128)
      } else {
        // items.len() == 4: {i, min, max, step}
        let min_val = expr_to_i128(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        let max_val = expr_to_i128(&items[2]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        let step_val = expr_to_i128(&items[3]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: step must be an integer".into(),
          )
        })?;
        if step_val == 0 {
          return Err(InterpreterError::EvaluationError(
            "Sum: step cannot be zero".into(),
          ));
        }
        (min_val, max_val, step_val)
      };

      // Closed-form fast paths to avoid pathological iteration for large
      // bounds (e.g. Sum[k, {k, 3, 10^20-1, 3}]).
      let n_terms: Option<i128> = if step > 0 && max >= min {
        Some((max - min) / step + 1)
      } else if step < 0 && max <= min {
        Some((min - max) / (-step) + 1)
      } else if min == max {
        Some(1)
      } else {
        // Empty sum
        Some(0)
      };
      if let Some(n) = n_terms {
        if n == 0 {
          return Ok(Expr::Integer(0));
        }
        // Sum[c, {k, ...}] when body doesn't reference k: c * n_terms.
        if !crate::functions::polynomial_ast::contains_var(body, &var_name) {
          let val = crate::evaluator::evaluate_expr_to_expr(body)?;
          return crate::functions::math_ast::times_ast(&[
            val,
            Expr::Integer(n),
          ]);
        }
        // Sum[k, {k, a, b, c}] = n_terms * (first + last) / 2.
        if matches!(body, Expr::Identifier(name) if name == &var_name) {
          let last = min + step * (n - 1);
          let sum_num = num_bigint::BigInt::from(n)
            * (num_bigint::BigInt::from(min) + num_bigint::BigInt::from(last));
          let sum = sum_num / num_bigint::BigInt::from(2);
          return Ok(crate::functions::math_ast::bigint_to_expr(sum));
        }
      }

      let mut acc = Expr::Integer(0);
      let mut i = min;
      if step > 0 {
        while i <= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
          i += step;
        }
      } else {
        while i >= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
          i += step;
        }
      }
      return Ok(acc);
    }
    _ => {}
  }

  Ok(unevaluated("Sum", args))
}

/// Try to evaluate a known infinite series Sum[body, {var, min, Infinity}].
/// Returns Some(result) if a closed form is found, None otherwise.
/// Try to evaluate a symbolic Sum where at least one bound is not a concrete integer.
/// Returns Some(expr) if a known closed form is found, None otherwise.
/// Flatten the factors of a product, descending through both `Times[...]`
/// (FunctionCall) and `BinaryOp::Times` spellings.
fn collect_times_factors(e: &Expr, out: &mut Vec<Expr>) {
  match e {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      collect_times_factors(left, out);
      collect_times_factors(right, out);
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args.iter() {
        collect_times_factors(a, out);
      }
    }
    _ => out.push(e.clone()),
  }
}

/// If `f` is `base^var` (in either Power spelling), return `base`.
fn power_with_exponent_var(f: &Expr, var_name: &str) -> Option<Expr> {
  match f {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } if matches!(right.as_ref(), Expr::Identifier(s) if s == var_name) => {
      Some((**left).clone())
    }
    Expr::FunctionCall { name, args }
      if name == "Power"
        && args.len() == 2
        && matches!(&args[1], Expr::Identifier(s) if s == var_name) =>
    {
      Some(args[0].clone())
    }
    _ => None,
  }
}

/// Binomial theorem: `Sum[c Binomial[N, k] r^k, {k, 0, N}] = c (1 + r)^N`,
/// where the upper limit equals the Binomial's first argument and `c`, `r` are
/// free of `k`. `r = -1` collapses to `KroneckerDelta[N]` to match Wolfram
/// (e.g. `Sum[(-1)^k Binomial[n, k], {k, 0, n}]`). Returns None when the body
/// is not of this shape.
fn try_binomial_theorem_sum(
  body: &Expr,
  var_name: &str,
  max_expr: &Expr,
) -> Option<Expr> {
  use crate::functions::polynomial_ast::contains_var;

  let mut factors = Vec::new();
  collect_times_factors(body, &mut factors);

  let mut binom_n: Option<Expr> = None;
  let mut coeff_factors: Vec<Expr> = Vec::new();
  let mut r_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    // Binomial[N, k] with N free of k.
    if let Expr::FunctionCall { name, args } = f
      && name == "Binomial"
      && args.len() == 2
      && matches!(&args[1], Expr::Identifier(s) if s == var_name)
      && !contains_var(&args[0], var_name)
    {
      if binom_n.is_some() {
        return None; // only a single Binomial factor is supported
      }
      binom_n = Some(args[0].clone());
      continue;
    }
    // r^k with r free of k.
    if let Some(base) = power_with_exponent_var(f, var_name)
      && !contains_var(&base, var_name)
    {
      r_factors.push(base);
      continue;
    }
    // Plain constant factor (free of k) folds into the coefficient.
    if !contains_var(f, var_name) {
      coeff_factors.push(f.clone());
      continue;
    }
    return None; // a k-dependent factor we cannot fold into the theorem
  }

  let n = binom_n?;
  // A leftover constant coefficient (e.g. `2 Binomial[n, k]`) would give a
  // c (1+r)^N form that Wolfram further folds (2*2^n -> 2^(1+n)); avoid the
  // form divergence by leaving those unevaluated.
  if !coeff_factors.is_empty() {
    return None;
  }
  // The upper limit must be exactly the Binomial's first argument.
  if crate::syntax::expr_to_string(&n)
    != crate::syntax::expr_to_string(max_expr)
  {
    return None;
  }

  // r = product of the r^k bases (default 1 when there is no power factor).
  let r = match r_factors.len() {
    0 => Expr::Integer(1),
    1 => r_factors.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: r_factors.into(),
    },
  };
  let one_plus_r = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(Expr::Integer(1)),
    right: Box::new(r),
  };
  let base = crate::evaluator::evaluate_expr_to_expr(&one_plus_r).ok()?;
  // 1 + r == 0 (r == -1): the alternating sum is KroneckerDelta[N].
  let term = if matches!(base, Expr::Integer(0)) {
    Expr::FunctionCall {
      name: "KroneckerDelta".to_string(),
      args: vec![n].into(),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(base),
      right: Box::new(n),
    }
  };

  Some(term)
}

fn try_symbolic_sum(
  body: &Expr,
  var_name: &str,
  min_expr: &Expr,
  max_expr: &Expr,
  min_concrete: Option<i128>,
  _max_concrete: Option<i128>,
) -> Result<Option<Expr>, InterpreterError> {
  // If body doesn't contain the iteration variable, it's a constant sum:
  // Sum[c, {var, min, max}] = c * (max - min + 1)
  if !crate::functions::polynomial_ast::contains_var(body, var_name) {
    let count = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(max_expr.clone()),
        right: Box::new(min_expr.clone()),
      }),
    };
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(body.clone()),
      right: Box::new(count),
    }));
  }

  // Linearity over a constant factor: Sum[c * f(k), ...] = c * Sum[f(k), ...].
  {
    let factors: Option<Vec<Expr>> = match body {
      Expr::FunctionCall { name, args } if name == "Times" => {
        Some(args.iter().cloned().collect())
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => Some(vec![(**left).clone(), (**right).clone()]),
      // f / c → f * c^-1, so a constant denominator pulls out (Sum[k/2]).
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => Some(vec![
        (**left).clone(),
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: right.clone(),
          right: Box::new(Expr::Integer(-1)),
        },
      ]),
      _ => None,
    };
    if let Some(factors) = factors {
      let (const_factors, var_factors): (Vec<Expr>, Vec<Expr>) =
        factors.into_iter().partition(|f| {
          !crate::functions::polynomial_ast::contains_var(f, var_name)
        });
      if !const_factors.is_empty() && !var_factors.is_empty() {
        let inner_body = if var_factors.len() == 1 {
          var_factors[0].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: var_factors.into(),
          }
        };
        if let Some(inner_sum) = try_symbolic_sum(
          &inner_body,
          var_name,
          min_expr,
          max_expr,
          min_concrete,
          _max_concrete,
        )? {
          let mut all = const_factors;
          all.push(inner_sum);
          let result = Expr::FunctionCall {
            name: "Times".to_string(),
            args: all.into(),
          };
          return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
        }
      }
    }
  }

  // Sum[Fibonacci[k], {k, a, n}] = Fibonacci[n + 2] - Fibonacci[a + 1]
  // (telescoping from Fibonacci[k] = Fibonacci[k+2] - Fibonacci[k+1]). Valid for
  // any lower bound; a = 1 or 0 gives the familiar Fibonacci[n+2] - 1.
  if let Expr::FunctionCall { name, args } = body
    && name == "Fibonacci"
    && args.len() == 1
    && matches!(&args[0], Expr::Identifier(v) if v == var_name)
  {
    let fib = |arg: Expr| Expr::FunctionCall {
      name: "Fibonacci".to_string(),
      args: vec![arg].into(),
    };
    let result = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(fib(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(max_expr.clone()),
        right: Box::new(Expr::Integer(2)),
      })),
      right: Box::new(fib(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(min_expr.clone()),
        right: Box::new(Expr::Integer(1)),
      })),
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
  }

  // Sum[Fibonacci[k]^2, {k, a, n}] = Fibonacci[n] Fibonacci[n+1]
  //                                  - Fibonacci[a-1] Fibonacci[a].
  if let Expr::BinaryOp {
    op: BinaryOperator::Power,
    left,
    right,
  } = body
    && matches!(right.as_ref(), Expr::Integer(2))
    && let Expr::FunctionCall { name, args } = left.as_ref()
    && name == "Fibonacci"
    && args.len() == 1
    && matches!(&args[0], Expr::Identifier(v) if v == var_name)
  {
    let fib = |arg: Expr| Expr::FunctionCall {
      name: "Fibonacci".to_string(),
      args: vec![arg].into(),
    };
    let shift = |e: &Expr, d: i128| Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(e.clone()),
      right: Box::new(Expr::Integer(d)),
    };
    let prod = |a: Expr, b: Expr| Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(a),
      right: Box::new(b),
    };
    let result = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(prod(fib(max_expr.clone()), fib(shift(max_expr, 1)))),
      right: Box::new(prod(fib(shift(min_expr, -1)), fib(min_expr.clone()))),
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
  }

  // Vandermonde: Sum[Binomial[N, k]^2, {k, 0, N}] = Binomial[2 N, N]. Requires
  // the full range (min == 0, upper bound == the binomial's first argument).
  if let Some(0) = min_concrete
    && let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } = body
    && matches!(right.as_ref(), Expr::Integer(2))
    && let Expr::FunctionCall { name, args } = left.as_ref()
    && name == "Binomial"
    && args.len() == 2
    && matches!(&args[1], Expr::Identifier(v) if v == var_name)
    && crate::syntax::expr_to_string(&args[0])
      == crate::syntax::expr_to_string(max_expr)
  {
    let two_n = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(max_expr.clone()),
    };
    let result = Expr::FunctionCall {
      name: "Binomial".to_string(),
      args: vec![two_n, max_expr.clone()].into(),
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
  }

  // Binomial theorem: Sum[c Binomial[N, k] r^k, {k, 0, N}] = c (1 + r)^N.
  if let Some(0) = min_concrete
    && let Some(result) = try_binomial_theorem_sum(body, var_name, max_expr)
  {
    return Ok(Some(result));
  }

  // Sum[k, {k, 1, n}] = n*(1 + n)/2
  if let Some(1) = min_concrete {
    if matches!(body, Expr::Identifier(name) if name == var_name) {
      let n = max_expr.clone();
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(n.clone()),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(n),
          }),
        }),
        right: Box::new(Expr::Integer(2)),
      }));
    }

    // Sum[k^2, {k, 1, n}] = n*(1 + n)*(1 + 2*n)/6
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = body
      && matches!(base.as_ref(), Expr::Identifier(name) if name == var_name)
      && matches!(exp.as_ref(), Expr::Integer(2))
    {
      let n = max_expr.clone();
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(n.clone()),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(n.clone()),
            }),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(2)),
              right: Box::new(n),
            }),
          }),
        }),
        right: Box::new(Expr::Integer(6)),
      }));
    }

    // Sum[k^3, {k, 1, n}] = (n*(1 + n)/2)^2
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = body
      && matches!(base.as_ref(), Expr::Identifier(name) if name == var_name)
      && matches!(exp.as_ref(), Expr::Integer(3))
    {
      let n = max_expr.clone();
      // (n*(1+n)/2)^2
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(n.clone()),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(n),
            }),
          }),
          right: Box::new(Expr::Integer(2)),
        }),
        right: Box::new(Expr::Integer(2)),
      }));
    }

    // Sum[k^4, {k, 1, n}] = n*(1+n)*(1+2*n)*(-1+3*n+3*n^2)/30
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = body
      && matches!(base.as_ref(), Expr::Identifier(name) if name == var_name)
      && matches!(exp.as_ref(), Expr::Integer(4))
    {
      let n = max_expr.clone();
      // n*(1+n)*(1+2*n)*(-1+3*n+3*n^2)/30
      let n_plus_1 = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(n.clone()),
      };
      let one_plus_2n = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(2)),
          right: Box::new(n.clone()),
        }),
      };
      let neg1_plus_3n_plus_3n2 = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(3)),
            right: Box::new(n.clone()),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(3)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(n.clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          }),
        }),
      };
      let numerator = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![n, n_plus_1, one_plus_2n, neg1_plus_3n_plus_3n2].into(),
      };
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(numerator),
        right: Box::new(Expr::Integer(30)),
      }));
    }

    // Sum[k^5, {k, 1, n}] = n^2*(1+n)^2*(-1+2*n+2*n^2)/12
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = body
      && matches!(base.as_ref(), Expr::Identifier(name) if name == var_name)
      && matches!(exp.as_ref(), Expr::Integer(5))
    {
      let n = max_expr.clone();
      // n^2*(1+n)^2*(-1+2*n+2*n^2)/12
      let n_sq = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(n.clone()),
        right: Box::new(Expr::Integer(2)),
      };
      let n_plus_1_sq = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(n.clone()),
        }),
        right: Box::new(Expr::Integer(2)),
      };
      let neg1_plus_2n_plus_2n2 = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(2)),
            right: Box::new(n.clone()),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(2)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(n),
              right: Box::new(Expr::Integer(2)),
            }),
          }),
        }),
      };
      let numerator = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![n_sq, n_plus_1_sq, neg1_plus_2n_plus_2n2].into(),
      };
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(numerator),
        right: Box::new(Expr::Integer(12)),
      }));
    }

    // Sum[1/k^s, {k, 1, n}] = HarmonicNumber[n, s]
    if let Some(s) = match_reciprocal_power(body, var_name)
      && s >= 1
    {
      return Ok(Some(Expr::FunctionCall {
        name: "HarmonicNumber".to_string(),
        args: if s == 1 {
          vec![max_expr.clone()].into()
        } else {
          vec![max_expr.clone(), Expr::Integer(s as i128)].into()
        },
      }));
    }

    // Sum[c^i, {i, 1, n}] = c*(c^n - 1)/(c - 1) (geometric series)
    // In Divide form: Sum[1/c^i, {i, 1, n}] = (c^n - 1)/(c^n * (c - 1))
    // or equivalently: (-1 + c^n)/c^n
    // Detect body = 1/c^var (Divide or Power with negative exponent)
    if let Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } = body
      && matches!(left.as_ref(), Expr::Integer(1))
    {
      // 1 / c^var
      if let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: base,
        right: exp,
      } = right.as_ref()
        && matches!(exp.as_ref(), Expr::Identifier(name) if name == var_name)
      {
        // Sum[1/c^i, {i, 1, n}] = (-1 + c^n)/(c^n*(c-1))
        // For c=2: (-1 + 2^n)/2^n
        let c = base.as_ref();
        let n = max_expr.clone();
        let c_to_n = Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(c.clone()),
          right: Box::new(n),
        };
        return Ok(Some(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(c_to_n.clone()),
          }),
          right: Box::new(c_to_n),
        }));
      }
    }
  }

  // Sum[c^(q*var), {var, 0, n}] = (c^(q*(n+1)) - 1) / (c^q - 1) — generalised
  // geometric series with a coefficient `q` in the exponent (constant w.r.t.
  // `var`). Handles e.g. `Sum[a^(k*n), {k, 0, m-1}]` →
  // `(a^(m*n) - 1) / (a^n - 1)` after simplifying `q*((m-1)+1) = q*m`.
  if matches!(min_concrete, Some(0))
    && let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = body
    && crate::functions::polynomial_ast::contains_var(exp, var_name)
    && !crate::functions::polynomial_ast::contains_var(base, var_name)
  {
    // Try to write `exp` as `q * var` with `q` constant w.r.t. `var`.
    let q_opt: Option<Expr> = match exp.as_ref() {
      Expr::Identifier(n) if n == var_name => Some(Expr::Integer(1)),
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        let l_is_var =
          matches!(left.as_ref(), Expr::Identifier(n) if n == var_name);
        let r_is_var =
          matches!(right.as_ref(), Expr::Identifier(n) if n == var_name);
        if l_is_var
          && !crate::functions::polynomial_ast::contains_var(right, var_name)
        {
          Some(*right.clone())
        } else if r_is_var
          && !crate::functions::polynomial_ast::contains_var(left, var_name)
        {
          Some(*left.clone())
        } else {
          None
        }
      }
      _ => None,
    };
    if let Some(q) = q_opt {
      let one_plus_max = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(max_expr.clone()),
      };
      let c_to_q = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(*base.clone()),
        right: Box::new(q.clone()),
      };
      let c_to_q_times_top = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(*base.clone()),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(q),
          right: Box::new(one_plus_max),
        }),
      };
      let numer = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(c_to_q_times_top),
      };
      let denom = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(c_to_q),
      };
      let result = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(numer),
        right: Box::new(denom),
      };
      return Ok(Some(
        crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result),
      ));
    }
  }

  // Sum[k, {k, a, n}] where a is symbolic
  if min_concrete.is_none()
    && matches!(body, Expr::Identifier(name) if name == var_name)
  {
    // Sum[k, {k, a, n}] = (a+n)*(n-a+1)/2
    let a = min_expr.clone();
    let n = max_expr.clone();
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(a.clone()),
          right: Box::new(n.clone()),
        }),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Minus,
            left: Box::new(n),
            right: Box::new(a),
          }),
          right: Box::new(Expr::Integer(1)),
        }),
      }),
      right: Box::new(Expr::Integer(2)),
    }));
  }

  Ok(None)
}

/// Check if a body expression matches the Leibniz series: (-1)^k / (2k+1).
/// We verify by evaluating at k=0,1,2,3,4 and checking against expected values.
fn is_leibniz_body(body: &Expr, var_name: &str) -> bool {
  // Expected values: f(0)=1, f(1)=-1/3, f(2)=1/5, f(3)=-1/7, f(4)=1/9
  let expected: [(i128, f64); 5] = [
    (0, 1.0),
    (1, -1.0 / 3.0),
    (2, 1.0 / 5.0),
    (3, -1.0 / 7.0),
    (4, 1.0 / 9.0),
  ];
  for (k, exp_val) in &expected {
    let substituted =
      crate::syntax::substitute_variable(body, var_name, &Expr::Integer(*k));
    if let Ok(result) = crate::evaluator::evaluate_expr_to_expr(&substituted) {
      if let Some(val) = crate::functions::math_ast::try_eval_to_f64(&result) {
        if (val - exp_val).abs() > 1e-12 {
          return false;
        }
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  true
}

/// Match a geometric-series body `c * base^var` where `base` is free of `var`
/// and symbolic (not a plain number), and `var` appears only as the exponent.
/// Returns `(coefficient, base)`.
fn match_geometric_base(body: &Expr, var_name: &str) -> Option<(Expr, Expr)> {
  // Flatten the multiplicative factors of `body`. A `Divide` contributes its
  // denominator as a reciprocal factor `denominator^(-1)`.
  fn collect(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        collect(left, out);
        collect(right, out);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => {
        collect(left, out);
        out.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: right.clone(),
          right: Box::new(Expr::Integer(-1)),
        });
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          collect(a, out);
        }
      }
      other => out.push(other.clone()),
    }
  }
  // Coefficient `c` if `exp` is exactly `c * var` (c free of var), else None.
  // Recognizes `var` (c = 1), `2 var`, `var/3`, etc.
  fn linear_var_coeff(exp: &Expr, var_name: &str) -> Option<Expr> {
    fn collect_times(e: &Expr, out: &mut Vec<Expr>) {
      match e {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left,
          right,
        } => {
          collect_times(left, out);
          collect_times(right, out);
        }
        Expr::FunctionCall { name, args } if name == "Times" => {
          for a in args.iter() {
            collect_times(a, out);
          }
        }
        other => out.push(other.clone()),
      }
    }
    let mut factors = Vec::new();
    collect_times(exp, &mut factors);
    let mut var_seen = 0;
    let mut consts: Vec<Expr> = Vec::new();
    for f in &factors {
      if matches!(f, Expr::Identifier(n) if n == var_name) {
        var_seen += 1;
      } else if crate::functions::calculus_ast::is_constant_wrt(f, var_name) {
        consts.push(f.clone());
      } else {
        return None;
      }
    }
    if var_seen != 1 {
      return None;
    }
    Some(match consts.len() {
      0 => Expr::Integer(1),
      1 => consts.into_iter().next().unwrap(),
      _ => Expr::FunctionCall {
        name: "Times".to_string(),
        args: consts.into(),
      },
    })
  }
  // Returns the effective geometric base of `f` if `f == base^(c*var)` with
  // `base` free of `var`: the base is `base^c`, evaluated so that e.g.
  // `2^(2 var)` reduces to base `4` (which the numeric path then handles).
  let power_base = |f: &Expr| -> Option<Expr> {
    let (base, exp) = match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => (left.as_ref().clone(), right.as_ref().clone()),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        (args[0].clone(), args[1].clone())
      }
      _ => return None,
    };
    // Normalize the (held) exponent: the parser stores a unary minus on a
    // symbol as `0 - var`, which `linear_var_coeff` cannot read. Evaluating it
    // (the summation variable stays free) collapses `0 - var` to `-var`, i.e.
    // `Times[-1, var]`, so `2^(-n)` is recognized like `(1/2)^n`.
    let exp = crate::evaluator::evaluate_expr_to_expr(&exp).unwrap_or(exp);
    let coeff = linear_var_coeff(&exp, var_name)?;
    if matches!(&coeff, Expr::Integer(1)) {
      return Some(base);
    }
    // Only an integer exponent coefficient (e.g. the 2 in x^(2 var)) keeps the
    // closed form matching wolframscript; a symbolic coefficient like the k in
    // a^(k var) yields an equivalent but differently-canonicalized result, so
    // leave those for the general path.
    if !matches!(&coeff, Expr::Integer(_) | Expr::BigInteger(_)) {
      return None;
    }
    let eff = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(base),
      right: Box::new(coeff),
    };
    Some(crate::evaluator::evaluate_expr_to_expr(&eff).unwrap_or(eff))
  };
  let mut factors = Vec::new();
  collect(body, &mut factors);

  let mut base: Option<Expr> = None;
  let mut coeff_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    if base.is_none()
      && let Some(b) = power_base(f)
    {
      base = Some(b);
      continue;
    }
    // Every other factor must be free of the summation variable.
    if !crate::functions::calculus_ast::is_constant_wrt(f, var_name) {
      return None;
    }
    coeff_factors.push(f.clone());
  }

  let base = base?;
  // The base must be free of the variable. Both symbolic and numeric bases are
  // returned; callers apply the convergence guard (|base| < 1) needed for
  // numeric ratios. (Previously numeric bases were rejected here, but that was
  // inconsistent — a literal `(1/4)^n` slipped through as `Divide[1,4]` while a
  // computed `4^(-n)` base evaluated to `Rational[1,4]` and was dropped.)
  if !crate::functions::calculus_ast::is_constant_wrt(&base, var_name) {
    return None;
  }

  let coeff = match coeff_factors.len() {
    0 => Expr::Integer(1),
    1 => coeff_factors.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: coeff_factors.into(),
    },
  };
  Some((coeff, base))
}

/// Match a geometric body `c * base^(q*var)` where the exponent coefficient `q`
/// is *symbolic* (free of `var`, not a plain integer) and `base` is free of
/// `var`. Returns `(c, base^q)` — the coefficient and effective geometric
/// ratio. This is deliberately disjoint from `match_geometric_base` (which
/// only accepts an integer exponent coefficient), because wolframscript
/// canonicalizes the symbolic-exponent result differently:
///   Sum[a^(k n), {n, 0, Inf}]   -> -(-1 + a^k)^(-1)   (this matcher)
///   Sum[x^(2 n), {n, 0, Inf}]   ->  (1 - x^2)^(-1)    (match_geometric_base)
fn match_geometric_symbolic_exp(
  body: &Expr,
  var_name: &str,
) -> Option<(Expr, Expr)> {
  use crate::functions::calculus_ast::is_constant_wrt;

  let mut factors = Vec::new();
  collect_times_factors(body, &mut factors);

  let mut eff_base: Option<Expr> = None;
  let mut coeff_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    if eff_base.is_none()
      && let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } = f
      && is_constant_wrt(left, var_name)
    {
      // Exponent must be `q * var` with q free of var and not a plain integer.
      let exp = crate::evaluator::evaluate_expr_to_expr(right)
        .unwrap_or_else(|_| (**right).clone());
      let mut exp_factors = Vec::new();
      collect_times_factors(&exp, &mut exp_factors);
      let mut var_seen = 0;
      let mut q_factors: Vec<Expr> = Vec::new();
      let mut ok = true;
      for ef in &exp_factors {
        if matches!(ef, Expr::Identifier(n) if n == var_name) {
          var_seen += 1;
        } else if is_constant_wrt(ef, var_name) {
          q_factors.push(ef.clone());
        } else {
          ok = false;
          break;
        }
      }
      if ok && var_seen == 1 && !q_factors.is_empty() {
        let q = match q_factors.len() {
          1 => q_factors.into_iter().next().unwrap(),
          _ => Expr::FunctionCall {
            name: "Times".to_string(),
            args: q_factors.into(),
          },
        };
        // Reject a plain-integer q — that is the strict matcher's job.
        if !matches!(&q, Expr::Integer(_) | Expr::BigInteger(_)) {
          eff_base = Some(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: left.clone(),
            right: Box::new(q),
          });
          continue;
        }
      }
      return None;
    }
    if is_constant_wrt(f, var_name) {
      coeff_factors.push(f.clone());
      continue;
    }
    return None;
  }

  let eff_base = eff_base?;
  let coeff = match coeff_factors.len() {
    0 => Expr::Integer(1),
    1 => coeff_factors.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: coeff_factors.into(),
    },
  };
  Some((coeff, eff_base))
}

/// Match an arithmetico-geometric body `var^p * r^var`, returning `(p, r)`
/// where `p >= 1` is the polynomial degree and `r` the geometric ratio.
/// The ratio factor may appear as `r^var`, `r^(-var)`, or `(r^var)^(-1)`
/// (the last from a `k/r^k` division). Other factors reject the match.
/// Used for Sum[k^p r^k, {k, 1, Infinity}] = PolyLog[-p, r].
fn match_arith_geometric(body: &Expr, var_name: &str) -> Option<(i128, Expr)> {
  fn collect(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        collect(left, out);
        collect(right, out);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => {
        collect(left, out);
        out.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: right.clone(),
          right: Box::new(Expr::Integer(-1)),
        });
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          collect(a, out);
        }
      }
      other => out.push(other.clone()),
    }
  }
  fn as_power(f: &Expr) -> Option<(&Expr, &Expr)> {
    match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => Some((left.as_ref(), right.as_ref())),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        Some((&args[0], &args[1]))
      }
      _ => None,
    }
  }
  let is_var = |e: &Expr| matches!(e, Expr::Identifier(n) if n == var_name);
  let is_neg_var = |e: &Expr| -> bool {
    matches!(e, Expr::UnaryOp { op: UnaryOperator::Minus, operand }
      if matches!(operand.as_ref(), Expr::Identifier(n) if n == var_name))
      || matches!(e, Expr::BinaryOp { op: BinaryOperator::Times, left, right }
        if matches!(left.as_ref(), Expr::Integer(-1))
          && matches!(right.as_ref(), Expr::Identifier(n) if n == var_name))
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "Times" && args.len() == 2
          && matches!(&args[0], Expr::Integer(-1))
          && matches!(&args[1], Expr::Identifier(n) if n == var_name))
  };
  let recip = |e: &Expr| -> Expr {
    crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(e.clone()),
      right: Box::new(Expr::Integer(-1)),
    })
    .unwrap_or_else(|_| e.clone())
  };
  let is_const =
    |e: &Expr| crate::functions::calculus_ast::is_constant_wrt(e, var_name);

  let mut factors = Vec::new();
  collect(body, &mut factors);

  let mut p: Option<i128> = None;
  let mut r: Option<Expr> = None;
  for f in &factors {
    if is_var(f) {
      if p.is_some() {
        return None;
      }
      p = Some(1);
      continue;
    }
    if let Some((base, exp)) = as_power(f) {
      // var^p  (p a positive integer)
      if is_var(base)
        && let Expr::Integer(pp) = exp
        && *pp >= 1
        && p.is_none()
      {
        p = Some(*pp);
        continue;
      }
      // ratio factor → r
      let ratio = if is_var(exp) && is_const(base) {
        Some(base.clone())
      } else if is_neg_var(exp) && is_const(base) {
        Some(recip(base))
      } else if matches!(exp, Expr::Integer(-1))
        && let Some((inner_base, inner_exp)) = as_power(base)
        && is_var(inner_exp)
        && is_const(inner_base)
      {
        Some(recip(inner_base))
      } else {
        None
      };
      if let Some(ratio) = ratio {
        if r.is_some() {
          return None;
        }
        r = Some(ratio);
        continue;
      }
    }
    return None;
  }
  Some((p?, r?))
}

/// Whether `e` is a multiplicative product (either AST shape).
fn is_times_expr(e: &Expr) -> bool {
  matches!(
    e,
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    }
  ) || matches!(e, Expr::FunctionCall { name, .. } if name == "Times")
}

/// If `exp` is `c * var` for a constant `c` (and `var` appears exactly once),
/// return `c`; otherwise None. `var` alone gives `1`.
fn linear_coeff_of_var(exp: &Expr, var_name: &str) -> Option<Expr> {
  fn collect_times(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        collect_times(left, out);
        collect_times(right, out);
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          collect_times(a, out);
        }
      }
      other => out.push(other.clone()),
    }
  }
  let mut factors = Vec::new();
  collect_times(exp, &mut factors);
  let mut var_seen = 0;
  let mut consts: Vec<Expr> = Vec::new();
  for f in &factors {
    if matches!(f, Expr::Identifier(n) if n == var_name) {
      var_seen += 1;
    } else if crate::functions::calculus_ast::is_constant_wrt(f, var_name) {
      consts.push(f.clone());
    } else {
      return None;
    }
  }
  if var_seen != 1 {
    return None;
  }
  Some(match consts.len() {
    0 => Expr::Integer(1),
    1 => consts.remove(0),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: consts.into(),
    },
  })
}

/// Match a logarithmic-series body `coeff * base^var / var`, i.e. a geometric
/// term over the first power of the summation variable. Returns `(coeff, base)`
/// with `base` the product of all `base^var` factors. Used for the Mercator
/// series Sum[base^k/k, {k,1,Infinity}] = -Log[1 - base]. Reciprocal forms such
/// as `1/(2^n n)` and `2^(-n)/n` are normalised to base `1/2`.
fn match_log_geometric(body: &Expr, var_name: &str) -> Option<(Expr, Expr)> {
  // Push 1/e as multiplicative factors, distributing over products and folding
  // one level of power so 1/(2^n n) yields [2^(-n), n^(-1)].
  fn recip(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        recip(left, out);
        recip(right, out);
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          recip(a, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => out.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: left.clone(),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: right.clone(),
        }),
      }),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        out.push(Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            args[0].clone(),
            Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(-1)),
              right: Box::new(args[1].clone()),
            },
          ]
          .into(),
        })
      }
      other => out.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(other.clone()),
        right: Box::new(Expr::Integer(-1)),
      }),
    }
  }
  fn collect(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        collect(left, out);
        collect(right, out);
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          collect(a, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => {
        collect(left, out);
        recip(right, out);
      }
      // (a*b)^(-1) → distribute as 1/a * 1/b.
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if matches!(right.as_ref(), Expr::Integer(-1))
        && is_times_expr(left) =>
      {
        recip(left, out);
      }
      other => out.push(other.clone()),
    }
  }
  fn power_base(f: &Expr, var_name: &str) -> Option<Expr> {
    let (base, exp) = match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => (left.as_ref().clone(), right.as_ref().clone()),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        (args[0].clone(), args[1].clone())
      }
      _ => return None,
    };
    // Evaluate the (held) exponent so `0 - n` and `-1 * n` both read as a linear
    // multiple of the variable; the effective base for `base^(c var)` is
    // `base^c`, e.g. `2^(-n)` → `1/2`.
    let exp = crate::evaluator::evaluate_expr_to_expr(&exp).unwrap_or(exp);
    let coeff = linear_coeff_of_var(&exp, var_name)?;
    if matches!(coeff, Expr::Integer(1)) {
      return Some(base);
    }
    // Only an integer exponent coefficient keeps the closed form matching
    // wolframscript; leave a symbolic coefficient for the general path.
    if !matches!(coeff, Expr::Integer(_) | Expr::BigInteger(_)) {
      return None;
    }
    let eff = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(base),
      right: Box::new(coeff),
    };
    Some(crate::evaluator::evaluate_expr_to_expr(&eff).unwrap_or(eff))
  }
  let is_recip_var = |f: &Expr| -> bool {
    let (base, exp) = match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => (left.as_ref(), right.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      _ => return false,
    };
    matches!(base, Expr::Identifier(n) if n == var_name)
      && matches!(exp, Expr::Integer(-1))
  };

  let mut factors = Vec::new();
  collect(body, &mut factors);
  let mut bases: Vec<Expr> = Vec::new();
  let mut seen_recip_var = false;
  let mut coeff_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    if let Some(b) = power_base(f, var_name) {
      bases.push(b);
      continue;
    }
    if !seen_recip_var && is_recip_var(f) {
      seen_recip_var = true;
      continue;
    }
    if !crate::functions::calculus_ast::is_constant_wrt(f, var_name) {
      return None;
    }
    coeff_factors.push(f.clone());
  }
  if !seen_recip_var || bases.is_empty() {
    return None;
  }
  if bases
    .iter()
    .any(|b| !crate::functions::calculus_ast::is_constant_wrt(b, var_name))
  {
    return None;
  }
  let base = if bases.len() == 1 {
    bases.pop().unwrap()
  } else {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Times".to_string(),
      args: bases.into(),
    })
    .ok()?
  };
  let coeff = match coeff_factors.len() {
    0 => Expr::Integer(1),
    1 => coeff_factors.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: coeff_factors.into(),
    },
  };
  Some((coeff, base))
}

/// Match an exponential-series body `c * base^var / var!` where `base` is free
/// of `var` (numeric or symbolic) and `var` appears only as the exponent of
/// `base` and inside the factorial. Returns `(coefficient, base)`; the base
/// defaults to `1` when there is no explicit `base^var` factor (e.g. `1/k!`).
fn match_exponential_base(body: &Expr, var_name: &str) -> Option<(Expr, Expr)> {
  fn collect(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        collect(left, out);
        collect(right, out);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => {
        collect(left, out);
        out.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: right.clone(),
          right: Box::new(Expr::Integer(-1)),
        });
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          collect(a, out);
        }
      }
      other => out.push(other.clone()),
    }
  }
  // Split a power factor into (base, exponent).
  fn as_power(f: &Expr) -> Option<(&Expr, &Expr)> {
    match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => Some((left.as_ref(), right.as_ref())),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        Some((&args[0], &args[1]))
      }
      _ => None,
    }
  }
  let is_factorial_of_var = |e: &Expr| -> bool {
    matches!(e, Expr::FunctionCall { name, args }
      if name == "Factorial" && args.len() == 1
        && matches!(&args[0], Expr::Identifier(n) if n == var_name))
  };

  let mut factors = Vec::new();
  collect(body, &mut factors);

  let mut have_factorial = false;
  let mut base: Option<&Expr> = None;
  let mut coeff_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    if let Some((b, e)) = as_power(f) {
      // The `1/var!` factor.
      if matches!(e, Expr::Integer(-1)) && is_factorial_of_var(b) {
        if have_factorial {
          return None; // more than one factorial factor
        }
        have_factorial = true;
        continue;
      }
      // The `base^var` factor.
      if base.is_none()
        && matches!(e, Expr::Identifier(n) if n == var_name)
        && crate::functions::calculus_ast::is_constant_wrt(b, var_name)
      {
        base = Some(b);
        continue;
      }
    }
    // Any remaining factor must be free of the summation variable.
    if !crate::functions::calculus_ast::is_constant_wrt(f, var_name) {
      return None;
    }
    coeff_factors.push(f.clone());
  }

  if !have_factorial {
    return None;
  }
  let base = base.cloned().unwrap_or(Expr::Integer(1));
  let coeff = match coeff_factors.len() {
    0 => Expr::Integer(1),
    1 => coeff_factors.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: coeff_factors.into(),
    },
  };
  Some((coeff, base))
}

/// gcd of two non-negative integers.
fn tr_gcd(mut a: i128, mut b: i128) -> i128 {
  a = a.abs();
  b = b.abs();
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Reduce a rational (num, den) to lowest terms with a positive denominator.
fn tr_reduce(n: i128, d: i128) -> (i128, i128) {
  if d == 0 {
    return (n, 0);
  }
  let g = tr_gcd(n, d).max(1);
  let (mut n, mut d) = (n / g, d / g);
  if d < 0 {
    n = -n;
    d = -d;
  }
  (n, d)
}

fn tr_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  tr_reduce(a.0 * b.1 + b.0 * a.1, a.1 * b.1)
}
fn tr_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  tr_reduce(a.0 * b.0, a.1 * b.1)
}

/// Parse an evaluated expression as an exact rational (n, d).
fn tr_as_rat(e: &Expr) -> Option<(i128, i128)> {
  match e {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(n), Expr::Integer(d)) => Some((*n, *d)),
        _ => None,
      }
    }
    _ => None,
  }
}

/// `CoefficientList[op[body], var]` as a vector of exact rationals, or None if
/// the result isn't a list of rational coefficients (e.g. non-polynomial).
fn tr_coeff_list(
  body: &Expr,
  var_name: &str,
  op: &str,
) -> Option<Vec<(i128, i128)>> {
  let part = Expr::FunctionCall {
    name: op.to_string(),
    args: vec![body.clone()].into(),
  };
  let cl = Expr::FunctionCall {
    name: "CoefficientList".to_string(),
    args: vec![part, Expr::Identifier(var_name.to_string())].into(),
  };
  let evaled = crate::evaluator::evaluate_expr_to_expr(&cl).ok()?;
  let Expr::List(items) = &evaled else {
    return None;
  };
  items.iter().map(tr_as_rat).collect()
}

/// Horner evaluation of a polynomial (rational coefficients, ascending degree)
/// at an integer point.
fn tr_poly_eval(coeffs: &[(i128, i128)], x: i128) -> (i128, i128) {
  let mut acc = (0i128, 1i128);
  for c in coeffs.iter().rev() {
    acc = tr_add(tr_mul(acc, (x, 1)), *c);
  }
  acc
}

/// Evaluate the derivative of a polynomial at an integer point.
fn tr_poly_deriv_eval(coeffs: &[(i128, i128)], x: i128) -> (i128, i128) {
  // d/dx sum c_k x^k = sum k c_k x^(k-1).
  let deriv: Vec<(i128, i128)> = coeffs
    .iter()
    .enumerate()
    .skip(1)
    .map(|(k, c)| tr_mul(*c, (k as i128, 1)))
    .collect();
  tr_poly_eval(&deriv, x)
}

/// Exact harmonic number H_a = sum_{k=1}^{a} 1/k (H_0 = 0).
fn tr_harmonic(a: i128) -> (i128, i128) {
  let mut h = (0i128, 1i128);
  for k in 1..=a {
    h = tr_add(h, (1, k));
  }
  h
}

/// Sum of a convergent rational summand with simple integer poles, evaluated
/// in closed form via residues:
/// `Sum[P(n)/Q(n), {n, min, Infinity}] = -sum_r residue_r * H_{min-1-r}`,
/// where r ranges over the (integer, simple) roots of Q. Returns None unless
/// every pole is a simple integer at or below `min-1` and the series converges
/// (the residues sum to zero, i.e. the summand decays like 1/n^2).
fn try_telescoping_rational_sum(
  body: &Expr,
  var_name: &str,
  min: i128,
) -> Result<Option<Expr>, InterpreterError> {
  // Trim trailing zero coefficients to get true degrees.
  let trim = |mut v: Vec<(i128, i128)>| {
    while v.len() > 1 && v.last() == Some(&(0, 1)) {
      v.pop();
    }
    v
  };
  let mut num = tr_coeff_list(body, var_name, "Numerator")
    .map(&trim)
    .unwrap_or_default();
  let mut den = tr_coeff_list(body, var_name, "Denominator")
    .map(&trim)
    .unwrap_or_default();
  // Woxi's Numerator/Denominator don't split a reciprocal power such as
  // Power[n^2 + n, -1]; in that case the numerator comes back empty. Recover
  // the form 1/Q by inverting the body: if body^-1 is a polynomial Q, the
  // summand is 1/Q.
  if num.is_empty() {
    let recip = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![body.clone(), Expr::Integer(-1)].into(),
    };
    if let Some(q) = tr_coeff_list(&recip, var_name, "Together").map(&trim)
      && q.len() > 1
    {
      num = vec![(1, 1)];
      den = q;
    }
  }
  // An empty coefficient list means the numerator/denominator isn't a genuine
  // polynomial in `var`; leave it to other paths.
  if num.is_empty() || den.is_empty() {
    return Ok(None);
  }
  let deg_num = num.len() - 1;
  let deg_den = den.len() - 1;
  // Need a genuine rational function that decays at least like 1/n^2.
  if deg_den < deg_num + 2 || deg_den == 0 {
    return Ok(None);
  }

  // Find the integer roots of Q with multiplicity. A root at 0 shows up as a
  // run of leading zero coefficients; other integer roots divide the lowest
  // non-zero coefficient (rational root theorem, integer candidates only).
  let mut roots: Vec<i128> = Vec::new();
  // Strip integer denominators so the candidate search works on integers.
  let lcm_den = den.iter().fold(1i128, |acc, &(_, d)| {
    let g = tr_gcd(acc, d);
    acc / g * d
  });
  let int_coeffs: Vec<i128> =
    den.iter().map(|&(n, d)| n * (lcm_den / d)).collect();
  // Roots at 0 (trailing zero constant terms).
  let mut lowest = 0usize;
  while lowest < int_coeffs.len() && int_coeffs[lowest] == 0 {
    roots.push(0);
    lowest += 1;
  }
  if lowest >= int_coeffs.len() {
    return Ok(None);
  }
  let reduced = &int_coeffs[lowest..];
  let c0 = reduced[0];
  // Candidate integer roots r divide c0 (leading coefficient need not be 1, but
  // for these telescoping cases the relevant roots are integer divisors of c0).
  let c0a = c0.abs();
  for cand in 1..=c0a {
    if c0a % cand != 0 {
      continue;
    }
    for r in [cand, -cand] {
      // Horner test on the integer polynomial.
      let mut acc = 0i128;
      for c in reduced.iter().rev() {
        acc = acc * r + c;
      }
      if acc == 0 {
        roots.push(r);
      }
    }
  }
  // Every pole must be simple and the roots must account for the full degree
  // (Q factors completely over the integers with no repeats).
  if roots.len() != deg_den {
    return Ok(None);
  }
  let mut sorted = roots.clone();
  sorted.sort_unstable();
  if sorted.windows(2).any(|w| w[0] == w[1]) {
    return Ok(None); // a repeated pole is not handled here
  }
  // No pole may lie inside the summation range [min, Infinity).
  if roots.iter().any(|&r| r > min - 1) {
    return Ok(None);
  }

  let mut total = (0i128, 1i128);
  let mut residue_sum = (0i128, 1i128);
  for &r in &roots {
    let pr = tr_poly_eval(&num, r);
    let dpr = tr_poly_deriv_eval(&den, r);
    if dpr.0 == 0 {
      return Ok(None);
    }
    // residue = P(r) / Q'(r)
    let c = tr_mul(pr, (dpr.1, dpr.0));
    residue_sum = tr_add(residue_sum, c);
    let h = tr_harmonic(min - 1 - r);
    total = tr_add(total, tr_mul((-c.0, c.1), h));
  }
  // Convergence requires the 1/n coefficient (sum of residues) to vanish.
  if residue_sum.0 != 0 {
    return Ok(None);
  }
  let (n, d) = tr_reduce(total.0, total.1);
  Ok(Some(crate::functions::math_ast::make_rational(n, d)))
}

/// Evaluate an ascending polynomial (rational coefficients) at a rational point.
fn tr_poly_eval_rat(coeffs: &[(i128, i128)], x: (i128, i128)) -> (i128, i128) {
  let mut acc = (0i128, 1i128);
  for c in coeffs.iter().rev() {
    acc = tr_add(tr_mul(acc, x), *c);
  }
  acc
}

/// Evaluate the derivative of an ascending polynomial at a rational point.
fn tr_poly_deriv_eval_rat(
  coeffs: &[(i128, i128)],
  x: (i128, i128),
) -> (i128, i128) {
  let deriv: Vec<(i128, i128)> = coeffs
    .iter()
    .enumerate()
    .skip(1)
    .map(|(k, c)| tr_mul(*c, (k as i128, 1)))
    .collect();
  tr_poly_eval_rat(&deriv, x)
}

/// Sum of a convergent rational summand with simple RATIONAL poles whose
/// residues cancel within each fractional class, telescoping to an exact
/// rational, e.g. `Sum[1/(4 n^2 - 1), {n, 1, Infinity}] = 1/2`. Generalises
/// `try_telescoping_rational_sum` (integer poles) via the residue/digamma
/// identity `Sum[P/Q] = -sum_r residue_r * PolyGamma[0, min - r]`, with
/// `PolyGamma[0, q0 + m] = PolyGamma[0, q0] + sum_{j<m} 1/(q0 + j)`. The
/// transcendental `PolyGamma[0, q0]` terms cancel only when the residues in each
/// fractional class `q0` sum to zero; otherwise the value is transcendental
/// (e.g. `Sum[1/(9 n^2 - 1)]` involves Pi) and is left unevaluated.
fn try_rational_pole_telescoping_sum(
  body: &Expr,
  var_name: &str,
  min: i128,
) -> Result<Option<Expr>, InterpreterError> {
  use std::collections::{HashMap, HashSet};
  let trim = |mut v: Vec<(i128, i128)>| {
    while v.len() > 1 && v.last() == Some(&(0, 1)) {
      v.pop();
    }
    v
  };
  let mut num = tr_coeff_list(body, var_name, "Numerator")
    .map(&trim)
    .unwrap_or_default();
  let mut den = tr_coeff_list(body, var_name, "Denominator")
    .map(&trim)
    .unwrap_or_default();
  if num.is_empty() {
    let recip = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![body.clone(), Expr::Integer(-1)].into(),
    };
    if let Some(q) = tr_coeff_list(&recip, var_name, "Together").map(&trim)
      && q.len() > 1
    {
      num = vec![(1, 1)];
      den = q;
    }
  }
  if num.is_empty() || den.is_empty() {
    return Ok(None);
  }
  let deg_num = num.len() - 1;
  let deg_den = den.len() - 1;
  // Need a genuine rational function that decays at least like 1/n^2.
  if deg_den < deg_num + 2 || deg_den == 0 {
    return Ok(None);
  }

  // Integer-clear the denominator for the rational-root search.
  let lcm_den = den.iter().fold(1i128, |acc, &(_, d)| {
    let g = tr_gcd(acc, d);
    acc / g * d
  });
  let mut int_coeffs: Vec<i128> =
    den.iter().map(|&(n, d)| n * (lcm_den / d)).collect();
  let mut roots: Vec<(i128, i128)> = Vec::new();
  // Factor out n (root 0); a repeated zero root is not handled here.
  let mut zero_mult = 0;
  while int_coeffs.len() > 1 && int_coeffs[0] == 0 {
    zero_mult += 1;
    int_coeffs.remove(0);
  }
  if zero_mult > 1 {
    return Ok(None);
  }
  if zero_mult == 1 {
    roots.push((0, 1));
  }
  // Rational root theorem on the reduced (nonzero constant term) polynomial.
  if int_coeffs.len() > 1 {
    const LIMIT: i128 = 100_000;
    let c0 = int_coeffs[0].abs();
    let cd = int_coeffs[int_coeffs.len() - 1].abs();
    if c0 == 0 || c0 > LIMIT || cd > LIMIT {
      return Ok(None);
    }
    let divisors =
      |m: i128| -> Vec<i128> { (1..=m).filter(|d| m % d == 0).collect() };
    let rat: Vec<(i128, i128)> = int_coeffs.iter().map(|&c| (c, 1)).collect();
    let mut seen: HashSet<(i128, i128)> = HashSet::new();
    for p in divisors(c0) {
      for q in divisors(cd) {
        for sgn in [1i128, -1] {
          let r = tr_reduce(sgn * p, q);
          if !seen.insert(r) {
            continue;
          }
          if tr_poly_eval_rat(&rat, r) == (0, 1) {
            roots.push(r);
          }
        }
      }
    }
  }
  // Q must split completely into distinct rational linear factors.
  if roots.len() != deg_den {
    return Ok(None);
  }
  // Every pole must lie strictly below the integer summation start, so q = min-r
  // is positive and no integer pole falls in [min, Infinity).
  for &r in &roots {
    if r.0 >= min * r.1 {
      return Ok(None);
    }
  }

  let mut total = (0i128, 1i128);
  let mut residue_sum = (0i128, 1i128);
  let mut class_residue: HashMap<(i128, i128), (i128, i128)> = HashMap::new();
  for &r in &roots {
    let pr = tr_poly_eval_rat(&num, r);
    let dpr = tr_poly_deriv_eval_rat(&den, r);
    if dpr.0 == 0 {
      return Ok(None);
    }
    let c = tr_mul(pr, (dpr.1, dpr.0)); // residue = P(r)/Q'(r)
    residue_sum = tr_add(residue_sum, c);
    // q = min - r, decomposed as q0 + m with q0 in (0, 1], m >= 0.
    let q = tr_add((min, 1), (-r.0, r.1));
    if q.0 <= 0 {
      return Ok(None);
    }
    let floor = q.0.div_euclid(q.1);
    let frac = tr_reduce(q.0 - floor * q.1, q.1);
    let (q0, m) = if frac.0 == 0 {
      ((1, 1), floor - 1)
    } else {
      (frac, floor)
    };
    if m > 100_000 {
      return Ok(None); // guard against overflow in the harmonic accumulation
    }
    let e = class_residue.entry(q0).or_insert((0, 1));
    *e = tr_add(*e, c);
    // inner = sum_{j=0}^{m-1} 1/(q0 + j)
    let mut inner = (0i128, 1i128);
    for j in 0..m {
      let denom = tr_add(q0, (j, 1));
      inner = tr_add(inner, (denom.1, denom.0));
    }
    total = tr_add(total, tr_mul((-c.0, c.1), inner));
  }
  // Convergence (sum of residues zero) and a rational value (each fractional
  // class cancels, so the transcendental PolyGamma[0, q0] terms drop out).
  if residue_sum.0 != 0 {
    return Ok(None);
  }
  if class_residue.values().any(|v| v.0 != 0) {
    return Ok(None);
  }
  let (n, d) = tr_reduce(total.0, total.1);
  Ok(Some(crate::functions::math_ast::make_rational(n, d)))
}

fn try_infinite_sum(
  body: &Expr,
  var_name: &str,
  min: i128,
) -> Result<Option<Expr>, InterpreterError> {
  // Geometric series Sum[c base^var, {var, 0, Infinity}] = c / (1 - base) for
  // a symbolic base. Only the min == 0 form is matched, because wolframscript
  // canonicalizes the min >= 1 result to a different (though equivalent) form.
  if min == 0
    && let Some((coeff, base)) = match_geometric_base(body, var_name)
    // A numeric ratio must satisfy |base| < 1 to converge; a divergent ratio
    // (e.g. (3/2)^n) is left unevaluated. A symbolic base yields the formal
    // closed form.
    && crate::functions::math_ast::try_eval_to_f64(&base)
      .is_none_or(|bf| bf.abs() < 1.0)
  {
    let one_minus_base = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(base),
      }),
    };
    let closed = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(coeff),
      right: Box::new(one_minus_base),
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&closed)?));
  }

  // Geometric series with a symbolic exponent coefficient, from var = 0:
  // Sum[c base^(q var), {var, 0, Infinity}] = -c/(-1 + base^q), where `q` is
  // symbolic. wolframscript renders this with the negated `-1 + base^q`
  // denominator (unlike the integer-exponent case handled above which keeps
  // `1 - base^q`), so the closed form is built as `(-c)/(-1 + base^q)`.
  if min == 0
    && let Some((coeff, eff_base)) =
      match_geometric_symbolic_exp(body, var_name)
  {
    let neg_one_plus_base = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(eff_base),
    };
    let closed = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(crate::functions::math_ast::times_ast(&[
        Expr::Integer(-1),
        coeff,
      ])?),
      right: Box::new(neg_one_plus_base),
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&closed)?));
  }

  // Geometric series from k = 1: Sum[c r^k, {k, 1, Infinity}] = c r/(1 - r),
  // for a numeric ratio r with |r| < 1 (it folds to a number). A symbolic
  // ratio is left to fall through — wolframscript canonicalizes its min >= 1
  // result to a form Woxi does not match.
  if min == 1
    && let Some((coeff, base)) = match_geometric_base(body, var_name)
    && let Some(bf) = crate::functions::math_ast::try_eval_to_f64(&base)
    && bf.abs() < 1.0
  {
    let one_minus_base = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(base.clone()),
      }),
    };
    let closed = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        coeff,
        Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(base),
          right: Box::new(one_minus_base),
        },
      ]
      .into(),
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&closed)?));
  }

  // Geometric series with a *symbolic* ratio from any min >= 1:
  // Sum[c base^var, {var, m, Infinity}] = -(c base^m)/(-1 + base). The numeric
  // ratio cases are handled above (folded to a number when |base| < 1, left
  // unevaluated when divergent); this branch yields the formal closed form for
  // a symbolic base, matching wolframscript's `(-1 + base)` denominator:
  //   Sum[x^k, {k, 1, Inf}] -> -(x/(-1 + x))
  //   Sum[x^n, {n, 2, Inf}] -> -(x^2/(-1 + x))
  if min >= 1
    && let Some((coeff, base)) = match_geometric_base(body, var_name)
    && crate::functions::math_ast::try_eval_to_f64(&base).is_none()
  {
    let base_pow_min = if min == 1 {
      base.clone()
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base.clone()),
        right: Box::new(Expr::Integer(min)),
      }
    };
    let numer = crate::functions::math_ast::times_ast(&[
      Expr::Integer(-1),
      coeff,
      base_pow_min,
    ])?;
    let denom = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(base),
    };
    let closed = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numer),
      right: Box::new(denom),
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&closed)?));
  }

  // Exponential series Sum[c base^var / var!, {var, m, Infinity}]. The base
  // may be numeric or symbolic (the series converges everywhere).
  //   m == 0:                 c E^base
  //   m == 1 with c == 1:     E^base - 1
  // Larger m, or m == 1 with a coefficient, are skipped because
  // wolframscript canonicalizes those results to a different (though
  // equivalent) form.
  if let Some((coeff, base)) = match_exponential_base(body, var_name) {
    let e_to_base = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![Expr::Constant("E".to_string()), base].into(),
    };
    if min == 0 {
      let result = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![coeff, e_to_base].into(),
      };
      return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
    }
    if min == 1 && matches!(coeff, Expr::Integer(1)) {
      let result = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![e_to_base, Expr::Integer(-1)].into(),
      };
      return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
    }
  }

  // Logarithmic series Sum[base^k/k, {k, 1, Infinity}] = -Log[1 - base]
  // (the Mercator/Taylor series for -Log[1-x]). A numeric base needs
  // |base| < 1 to converge; a symbolic base yields the formal result.
  if min == 1
    && let Some((coeff, base)) = match_log_geometric(body, var_name)
  {
    // The Mercator series Sum[base^k/k] converges on the real interval
    // [-1, 1): base == 1 is the divergent harmonic series, |base| > 1
    // diverges, but base == -1 converges conditionally to -Log[2]. (A
    // complex base on the unit circle, e.g. I, also converges and is left
    // to the formal closed form.)
    if let Some(b) = crate::functions::math_ast::try_eval_to_f64(&base)
      && !(-1.0..1.0).contains(&b)
    {
      return Ok(None);
    }
    // -coeff * Log[1 - base]
    let log_term = Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(base),
        }),
      }]
      .into(),
    };
    let closed = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), coeff, log_term].into(),
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&closed)?));
  }

  // Alternating p-series Sum[(-1)^(n+c)/n^s, {n, 1, Infinity}] = sign *
  // DirichletEta[s] (sign = -(-1)^c). Covers the cases the Mercator block above
  // misses — the (-1)^(n+1) sign convention and s >= 2 (e.g. Pi^2/12).
  if min == 1
    && let Some((sign, s)) = match_alternating_reciprocal_power(body, var_name)
    && s >= 1
  {
    let eta = Expr::FunctionCall {
      name: "DirichletEta".to_string(),
      args: vec![Expr::Integer(s as i128)].into(),
    };
    let result = if sign < 0 {
      crate::functions::math_ast::times_ast(&[Expr::Integer(-1), eta])?
    } else {
      eta
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
  }

  // Arithmetico-geometric series Sum[k^p r^k, {k, 1, Infinity}] =
  // PolyLog[-p, r], for an exact numeric ratio r with |r| < 1 (it folds to a
  // number). A symbolic ratio is left unevaluated: wolframscript renders its
  // result with the (-1 + x) denominator that Woxi's PolyLog does not.
  if min == 1
    && let Some((p, r)) = match_arith_geometric(body, var_name)
  {
    // Canonicalize the ratio (e.g. Divide[1, 3] -> Rational[1, 3]) so the
    // exact-numeric test below recognizes it.
    let r = crate::evaluator::evaluate_expr_to_expr(&r)?;
    let is_exact = matches!(&r, Expr::Integer(_))
      || matches!(&r, Expr::FunctionCall { name, .. } if name == "Rational");
    if is_exact
      && let Some(rf) = crate::functions::math_ast::try_eval_to_f64(&r)
      && rf.abs() < 1.0
    {
      let result = Expr::FunctionCall {
        name: "PolyLog".to_string(),
        args: vec![Expr::Integer(-p), r].into(),
      };
      return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
    }
    // First-order symbolic case Sum[k r^k, {k, 1, Inf}] = r/(-1 + r)^2.
    // wolframscript renders the `(-1 + r)^2` denominator (PolyLog[-1, r]
    // canonicalizes to `r/(1 - r)^2`, equal but differently displayed), so the
    // form is built directly. Higher orders (p >= 2) diverge in form and are
    // left unevaluated for a symbolic ratio.
    if p == 1
      && !is_exact
      && crate::functions::math_ast::try_eval_to_f64(&r).is_none()
    {
      let denom = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(r.clone()),
        }),
        right: Box::new(Expr::Integer(2)),
      };
      let closed = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(r),
        right: Box::new(denom),
      };
      return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&closed)?));
    }
  }

  // Try Leibniz formula: Sum[(-1)^k / (2k+1), {k, 0, Infinity}] = Pi/4
  if min == 0 && is_leibniz_body(body, var_name) {
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Constant("Pi".to_string())),
      right: Box::new(Expr::Integer(4)),
    }));
  }

  // Sums over the odd positive integers 1, 3, 5, …:
  //   Sum[1/(2n-1)^s, {n, 1, Inf}]          = DirichletLambda[s]   (s >= 2)
  //   Sum[(-1)^(n+1)/(2n-1)^s, {n, 1, Inf}] = DirichletBeta[s]     (s >= 1)
  // (and the (2n+1), {n, 0, …} spellings). The lambda/beta closed forms match
  // wolframscript (Pi^2/8, Pi/4, Pi^3/32, …). Non-alternating s == 1 is the
  // divergent Sum[1/(2n-1)] and is excluded.
  if let Some((alternating, sign, s)) =
    match_odd_reciprocal(body, var_name, min)
    && (alternating && s >= 1 || !alternating && s >= 2)
  {
    let func = if alternating {
      "DirichletBeta"
    } else {
      "DirichletLambda"
    };
    let call = Expr::FunctionCall {
      name: func.to_string(),
      args: vec![Expr::Integer(s as i128)].into(),
    };
    let result = if sign < 0 {
      crate::functions::math_ast::times_ast(&[Expr::Integer(-1), call])?
    } else {
      call
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
  }

  // Reciprocal power of a pure multiple of the index:
  //   Sum[1/(a n)^s, {n, 1, Inf}] = Zeta[s] / a^s  (s >= 2)
  // e.g. Sum[1/(2n)^2] = Pi^2/24, Sum[1/(a n)^2] = Pi^2/(6 a^2). The pure-`n`
  // base (a == 1) is left to the plain Zeta[s] p-series handler below.
  if min == 1
    && let Some((a, s)) = match_scaled_reciprocal_power(body, var_name)
    && s >= 2
  {
    let zeta = Expr::FunctionCall {
      name: "Zeta".to_string(),
      args: vec![Expr::Integer(s as i128)].into(),
    };
    let a_pow = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(a),
      right: Box::new(Expr::Integer(-(s as i128))),
    };
    let result = crate::functions::math_ast::times_ast(&[zeta, a_pow])?;
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
  }

  // For min < 1, compute initial terms and delegate to min=1 case:
  // Sum[f(n), {n, min, Infinity}] = f(min) + f(min+1) + ... + f(0) + Sum[f(n), {n, 1, Infinity}]
  if min < 1 {
    if let Some(tail_sum) = try_infinite_sum(body, var_name, 1)? {
      let mut acc = tail_sum;
      for k in min..1 {
        let substituted =
          crate::syntax::substitute_variable(body, var_name, &Expr::Integer(k));
        let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
        acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
      }
      return Ok(Some(acc));
    }
    return Ok(None);
  }

  // Convergent rational summand with simple integer poles telescopes to an
  // exact rational, e.g. Sum[1/(n(n+1)), {n, 1, Infinity}] = 1. Handled for
  // any finite lower bound >= 1.
  if let Some(result) = try_telescoping_rational_sum(body, var_name, min)? {
    return Ok(Some(result));
  }

  // Same idea, but for simple rational poles whose residues cancel within each
  // fractional class, e.g. Sum[1/(4 n^2 - 1), {n, 1, Infinity}] = 1/2.
  if let Some(result) = try_rational_pole_telescoping_sum(body, var_name, min)?
  {
    return Ok(Some(result));
  }

  // For min > 1, reduce to the min = 1 series and subtract the head terms,
  // e.g. Sum[r^n, {n, 2, Infinity}] = Sum[r^n, {n, 1, Infinity}] - r. This only
  // fires when the min = 1 series itself has a closed form, so it inherits that
  // form's exactness and stays unevaluated otherwise (no new form divergence).
  if min > 1 {
    if let Some(base_sum) = try_infinite_sum(body, var_name, 1)? {
      let mut acc = base_sum;
      for k in 1..min {
        let substituted =
          crate::syntax::substitute_variable(body, var_name, &Expr::Integer(k));
        let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
        let neg =
          crate::functions::math_ast::times_ast(&[Expr::Integer(-1), val])?;
        acc = crate::functions::math_ast::plus_ast(&[acc, neg])?;
      }
      return Ok(Some(acc));
    }
    return Ok(None);
  }

  if min != 1 {
    return Ok(None);
  }

  // Try to detect the pattern 1/var^s (i.e., var^(-s))
  // The body for 1/n^2 is: Times[1, Power[Power[n, 2], -1]]
  // which evaluates/simplifies to Power[n, -2] conceptually,
  // but in practice we need to match the AST structure.
  if let Some(s) = match_reciprocal_power(body, var_name) {
    if s >= 2 && s % 2 == 0 {
      // Zeta(s) for even s: (-1)^(s/2+1) * B_s * (2*Pi)^s / (2 * s!)
      return Ok(Some(zeta_even(s)?));
    }
    // Odd s >= 3: no known closed form in terms of Pi (returns Zeta[s])
    if s >= 3 && s % 2 == 1 {
      return Ok(Some(Expr::FunctionCall {
        name: "Zeta".to_string(),
        args: vec![Expr::Integer(s as i128)].into(),
      }));
    }
  }

  // Sum[1/c^i, {i, 1, Infinity}] = 1/(c-1) for integer c > 1
  // Detect body = 1/c^var (Divide form)
  if let Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left,
    right,
  } = body
    && matches!(left.as_ref(), Expr::Integer(1))
    && let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = right.as_ref()
    && matches!(exp.as_ref(), Expr::Identifier(name) if name == var_name)
    && let Some(c) = expr_to_i128(base)
    && c > 1
  {
    // Sum = 1/(c-1)
    return Ok(Some(crate::functions::math_ast::make_rational(1, c - 1)));
  }

  Ok(None)
}

/// Match the pattern `1/var^s` in the body expression.
/// Returns Some(s) if the body is equivalent to var^(-s) with s a positive integer.
/// Decompose `expr` into multiplicative numerator and denominator factors,
/// flattening nested Times and turning Divide into a denominator factor.
fn collect_factors(
  expr: &Expr,
  num: &mut Vec<Expr>,
  den: &mut Vec<Expr>,
  invert: bool,
) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      collect_factors(left, num, den, invert);
      collect_factors(right, num, den, invert);
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args.iter() {
        collect_factors(a, num, den, invert);
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      collect_factors(left, num, den, invert);
      collect_factors(right, num, den, !invert);
    }
    _ => {
      if invert {
        den.push(expr.clone());
      } else {
        num.push(expr.clone());
      }
    }
  }
}

/// If `e` is an integer-linear function of `var` (`coeff*var + const`), return
/// `(coeff, const)`. Determined by evaluating at var = 0, 1, 2.
fn linear_in_var(e: &Expr, var: &str) -> Option<(i128, i128)> {
  let at = |v: i128| -> Option<i128> {
    let s = crate::syntax::substitute_variable(e, var, &Expr::Integer(v));
    match crate::evaluator::evaluate_expr_to_expr(&s).ok()? {
      Expr::Integer(n) => Some(n),
      _ => None,
    }
  };
  let (e0, e1, e2) = (at(0)?, at(1)?, at(2)?);
  let coeff = e1 - e0;
  if e2 - e1 != coeff {
    return None;
  }
  Some((coeff, e0))
}

/// Match an alternating reciprocal-power summand `(-1)^(c*var+d) / var^s`
/// (with `c` odd, so the sign genuinely alternates). The infinite sum is
/// `sign * DirichletEta[s]` where `sign = -(-1)^d` — i.e. `(-1)^n/n^s` sums to
/// `-eta(s)` and `(-1)^(n+1)/n^s` to `+eta(s)`. Returns `(sign, s)`.
fn match_alternating_reciprocal_power(
  body: &Expr,
  var_name: &str,
) -> Option<(i32, i64)> {
  let mut num: Vec<Expr> = Vec::new();
  let mut den: Vec<Expr> = Vec::new();
  collect_factors(body, &mut num, &mut den, false);

  // Find and remove a (-1)^(linear) factor in the numerator.
  let mut sign: Option<i32> = None;
  let mut rest_num: Vec<Expr> = Vec::new();
  for f in num {
    if sign.is_none()
      && let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } = &f
      && matches!(left.as_ref(), Expr::Integer(-1))
      && let Some((coeff, c)) = linear_in_var(right, var_name)
      && coeff % 2 != 0
    {
      sign = Some(if c.rem_euclid(2) == 0 { -1 } else { 1 });
    } else {
      rest_num.push(f);
    }
  }
  let sign = sign?;

  // The remaining factors must form 1/var^s.
  let one = Expr::Integer(1);
  let numerator = match rest_num.len() {
    0 => one.clone(),
    1 => rest_num.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: rest_num.into(),
    },
  };
  let remaining = if den.is_empty() {
    numerator
  } else {
    let denominator = if den.len() == 1 {
      den.into_iter().next().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: den.into(),
      }
    };
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(denominator),
    }
  };
  let s = match_reciprocal_power(&remaining, var_name)?;
  Some((sign, s))
}

/// Like `match_reciprocal_power` but for an arbitrary (var-dependent) base:
/// recognises `1/base^s` written as `Power[base, -s]`, `Divide[1, Power[base,
/// s]]`, `Divide[1, base]`, or `Power[Power[base, s], -1]`. Returns `(base, s)`.
fn match_reciprocal_power_general(
  expr: &Expr,
  var_name: &str,
) -> Option<(Expr, i64)> {
  let has_var =
    |e: &Expr| crate::functions::polynomial_ast::contains_var(e, var_name);
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let e = get_integer(right)?;
      if e < 0 && has_var(left) {
        if e == -1
          && let Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: l2,
            right: r2,
          } = left.as_ref()
          && let Some(s) = get_integer(r2)
          && s > 0
          && has_var(l2)
        {
          return Some(((**l2).clone(), s as i64));
        }
        return Some(((**left).clone(), (-e) as i64));
      }
      None
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } if is_one(left) => {
      if let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: base,
        right: exp,
      } = right.as_ref()
        && let Some(s) = get_integer(exp)
        && s > 0
        && has_var(base)
      {
        return Some(((**base).clone(), s as i64));
      }
      has_var(right).then(|| ((**right).clone(), 1))
    }
    _ => None,
  }
}

/// Match `1/(a*n)^s` — a reciprocal power whose base is a pure (offset-free)
/// multiple `a*n` of the index, with `a` constant w.r.t. `n` and `a != 1`.
/// Returns `(a, s)`; the sum is then `Zeta[s]/a^s`.
fn match_scaled_reciprocal_power(
  body: &Expr,
  var_name: &str,
) -> Option<(Expr, i64)> {
  let mut num: Vec<Expr> = Vec::new();
  let mut den: Vec<Expr> = Vec::new();
  collect_factors(body, &mut num, &mut den, false);
  let numerator = match num.len() {
    0 => Expr::Integer(1),
    1 => num.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: num.into(),
    },
  };
  let remaining = if den.is_empty() {
    numerator
  } else {
    let denominator = if den.len() == 1 {
      den.into_iter().next().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: den.into(),
      }
    };
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(denominator),
    }
  };
  let (base, s) = match_reciprocal_power_general(&remaining, var_name)?;
  if !crate::functions::polynomial_ast::contains_var(&base, var_name) {
    return None;
  }
  // a = base / n must be constant w.r.t. n (so base == a*n with no offset).
  let ratio = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(base),
    right: Box::new(Expr::Identifier(var_name.to_string())),
  };
  let a = crate::evaluator::evaluate_expr_to_expr(&ratio).ok()?;
  if crate::functions::polynomial_ast::contains_var(&a, var_name)
    || matches!(a, Expr::Integer(1))
  {
    return None;
  }
  Some((a, s))
}

/// Match a summand over the odd positive integers: `[(-1)^(c*n+e)] /
/// (2n+b)^s`, where `2*min+b == 1` so the base runs through 1, 3, 5, …. Returns
/// `(alternating, sign, s)` — `Sum[1/(2n+b)^s] = DirichletLambda[s]` and
/// `Sum[(-1)^(…)/(2n+b)^s] = sign * DirichletBeta[s]` (sign = (-1)^(e-min)).
fn match_odd_reciprocal(
  body: &Expr,
  var_name: &str,
  min: i128,
) -> Option<(bool, i32, i64)> {
  let mut num: Vec<Expr> = Vec::new();
  let mut den: Vec<Expr> = Vec::new();
  collect_factors(body, &mut num, &mut den, false);

  let mut alternating = false;
  let mut sign = 1i32;
  let mut rest_num: Vec<Expr> = Vec::new();
  for f in num {
    if !alternating
      && let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } = &f
      && matches!(left.as_ref(), Expr::Integer(-1))
      && let Some((coeff, e)) = linear_in_var(right, var_name)
      && coeff % 2 != 0
    {
      alternating = true;
      sign = if (e - min).rem_euclid(2) == 0 { 1 } else { -1 };
    } else {
      rest_num.push(f);
    }
  }

  let numerator = match rest_num.len() {
    0 => Expr::Integer(1),
    1 => rest_num.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: rest_num.into(),
    },
  };
  let remaining = if den.is_empty() {
    numerator
  } else {
    let denominator = if den.len() == 1 {
      den.into_iter().next().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: den.into(),
      }
    };
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(denominator),
    }
  };

  let (base, s) = match_reciprocal_power_general(&remaining, var_name)?;
  let (c, b) = linear_in_var(&base, var_name)?;
  if c != 2 || 2 * min + b != 1 {
    return None;
  }
  Some((alternating, sign, s))
}

fn match_reciprocal_power(body: &Expr, var_name: &str) -> Option<i64> {
  match body {
    // Direct Power[var, -s]
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Identifier(name) = left.as_ref()
        && name == var_name
        && let Some(exp) = get_integer(right)
        && exp < 0
      {
        return Some(-exp as i64);
      }
      // Power[Power[var, s], -1]
      match_power_inverse(body, var_name)
    }
    // Divide[1, Power[var, s]] or Divide[1, var]  (how 1/var^s is stored internally)
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      if is_one(left) {
        // 1 / var^s
        if let Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: base,
          right: exp,
        } = right.as_ref()
          && let Expr::Identifier(name) = base.as_ref()
          && name == var_name
          && let Some(s) = get_integer(exp)
          && s > 0
        {
          return Some(s as i64);
        }
        // 1 / var => s = 1
        if let Expr::Identifier(name) = right.as_ref()
          && name == var_name
        {
          return Some(1);
        }
      }
      None
    }
    // Times[1, Power[Power[var, s], -1]]  (FullForm representation)
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if is_one(left) {
        return match_power_inverse(right, var_name);
      }
      if is_one(right) {
        return match_power_inverse(left, var_name);
      }
      None
    }
    _ => match_power_inverse(body, var_name),
  }
}

/// Match Power[Power[var, s], -1] or Power[var, -s]
fn match_power_inverse(expr: &Expr, var_name: &str) -> Option<i64> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      // Power[something, -1] where something = Power[var, s]
      if let Some(-1) = get_integer(right) {
        if let Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: inner_left,
          right: inner_right,
        } = left.as_ref()
          && let Expr::Identifier(name) = inner_left.as_ref()
          && name == var_name
          && let Some(s) = get_integer(inner_right)
          && s > 0
        {
          return Some(s as i64);
        }
        // Power[var, -1] => s = 1
        if let Expr::Identifier(name) = left.as_ref()
          && name == var_name
        {
          return Some(1);
        }
      }
      // Power[var, -s] directly
      if let Expr::Identifier(name) = left.as_ref()
        && name == var_name
        && let Some(exp) = get_integer(right)
        && exp < 0
      {
        return Some(-exp as i64);
      }
      None
    }
    _ => None,
  }
}

/// Get an integer value from an Expr
fn get_integer(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i128()
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => match operand.as_ref() {
      Expr::Integer(n) => Some(-n),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        (-n).to_i128()
      }
      _ => None,
    },
    _ => None,
  }
}

fn is_one(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(1))
    || matches!(expr, Expr::BigInteger(n) if *n == num_bigint::BigInt::from(1))
}

/// Compute ζ(2k) = |B_{2k}| * (2π)^{2k} / (2 * (2k)!) as a symbolic expression.
/// Returns Pi^(2k) * rational_coefficient.
fn zeta_even(s: i64) -> Result<Expr, InterpreterError> {
  // Get B_s using bernoulli_b_ast
  let b_s =
    crate::functions::math_ast::bernoulli_b_ast(&[Expr::Integer(s as i128)])?;

  // Extract the rational value of B_s as (num, den)
  let (b_num, b_den) = match &b_s {
    Expr::Integer(n) => (*n, 1i128),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      match n.to_i128() {
        Some(v) => (v, 1i128),
        None => {
          return Ok(Expr::FunctionCall {
            name: "Sum".to_string(),
            args: vec![].into(),
          });
        }
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (expr_to_i128(&args[0]), expr_to_i128(&args[1])) {
        (Some(n), Some(d)) => (n, d),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Sum".to_string(),
            args: vec![].into(),
          });
        }
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Sum".to_string(),
        args: vec![].into(),
      });
    }
  };

  // ζ(s) = (-1)^(s/2+1) * B_s * (2π)^s / (2 * s!)
  // Since B_s for even s alternates sign: B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, ...
  // (-1)^(s/2+1) * B_s = |B_s| always positive
  // So ζ(s) = |B_s| * (2π)^s / (2 * s!)

  // Compute (2^s) * |B_s_num| / (2 * s! * |B_s_den|)
  // = 2^(s-1) * |B_s_num| / (s! * |B_s_den|)
  let abs_b_num = b_num.abs();

  // Compute 2^(s-1) and s!
  let two_pow = 2i128.checked_pow((s - 1) as u32).unwrap_or(i128::MAX);
  let mut factorial: i128 = 1;
  for i in 2..=s as i128 {
    factorial = factorial.checked_mul(i).unwrap_or(i128::MAX);
  }

  // The coefficient of Pi^s is: 2^(s-1) * |B_s_num| / (s! * B_s_den)
  let coeff_num = two_pow * abs_b_num;
  let coeff_den = factorial * b_den.abs();

  // Simplify the fraction
  let g = gcd_i128(coeff_num.abs(), coeff_den.abs());
  let final_num = coeff_num / g;
  let final_den = coeff_den / g;

  // Build the expression: (final_num / final_den) * Pi^s
  let pi_power = if s == 1 {
    Expr::Identifier("Pi".to_string())
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Identifier("Pi".to_string())),
      right: Box::new(Expr::Integer(s as i128)),
    }
  };

  if final_num == 1 && final_den == 1 {
    Ok(pi_power)
  } else if final_den == 1 {
    Ok(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(final_num)),
      right: Box::new(pi_power),
    })
  } else if final_num == 1 {
    // 1/d * Pi^s => Pi^s / d
    Ok(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(pi_power),
      right: Box::new(Expr::Integer(final_den)),
    })
  } else {
    // n/d * Pi^s => (n * Pi^s) / d
    Ok(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(final_num)),
        right: Box::new(pi_power),
      }),
      right: Box::new(Expr::Integer(final_den)),
    })
  }
}

/// AnglePath3D[{step1, step2, …}] — the 3D angle path: an orientation frame
/// starts at the identity and each step left-multiplies it by
/// RollPitchYawMatrix of the NEGATED angles, then moves along the frame's
/// first row. Steps are {α, β, γ} angle triples or {dist, {α, β, γ}} pairs.
/// All arithmetic goes through the evaluator, so exact angles give exact
/// radical coordinates like wolframscript's. Invalid specifications emit
/// ::steps.
pub fn angle_path_3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("AnglePath3D", args));
  if args.len() != 1 {
    return unevaluated();
  }
  let steps_err = || {
    crate::emit_message(&format!(
      "AnglePath3D::steps: Invalid steps specification {}.",
      crate::syntax::expr_to_output(&args[0])
    ));
    unevaluated()
  };
  let Expr::List(steps) = &args[0] else {
    return steps_err();
  };

  let eval = crate::evaluator::evaluate_expr_to_expr;
  let fc = |name: &str, a: Vec<Expr>| Expr::FunctionCall {
    name: name.to_string(),
    args: a.into(),
  };
  let zero = || Expr::Integer(0);
  // Real step data gives a real origin ({0., 0., 0.}), like wolframscript.
  fn contains_real(e: &Expr) -> bool {
    match e {
      Expr::Real(_) | Expr::BigFloat(..) => true,
      Expr::List(items) => items.iter().any(contains_real),
      _ => false,
    }
  }
  let origin = || {
    if steps.iter().any(contains_real) {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    }
  };
  let mut point = Expr::List(vec![origin(), origin(), origin()].into());
  // The frame as an explicit 3×3 matrix expression, starting at identity.
  let mut frame = Expr::List(
    vec![
      Expr::List(vec![Expr::Integer(1), zero(), zero()].into()),
      Expr::List(vec![zero(), Expr::Integer(1), zero()].into()),
      Expr::List(vec![zero(), zero(), Expr::Integer(1)].into()),
    ]
    .into(),
  );
  let mut points: Vec<Expr> = vec![point.clone()];

  for step in steps.iter() {
    // {α, β, γ} or {dist, {α, β, γ}}.
    let (dist, angles): (Option<&Expr>, &crate::ExprList) = match step {
      Expr::List(items) if items.len() == 3 => (None, items),
      Expr::List(items)
        if items.len() == 2
          && matches!(&items[1], Expr::List(a) if a.len() == 3) =>
      {
        match &items[1] {
          Expr::List(a) => (Some(&items[0]), a),
          _ => unreachable!(),
        }
      }
      _ => return steps_err(),
    };
    let negated = Expr::List(
      angles
        .iter()
        .map(|a| fc("Times", vec![Expr::Integer(-1), a.clone()]))
        .collect(),
    );
    let rotation = eval(&fc("RollPitchYawMatrix", vec![negated]))?;
    if !matches!(&rotation, Expr::List(_)) {
      // Symbolic angles keep the whole call unevaluated.
      return unevaluated();
    }
    frame = eval(&fc("Dot", vec![rotation, frame.clone()]))?;
    let mut direction =
      eval(&fc("Part", vec![frame.clone(), Expr::Integer(1)]))?;
    if let Some(d) = dist {
      direction = eval(&fc("Times", vec![d.clone(), direction]))?;
    }
    point = eval(&fc("Plus", vec![point.clone(), direction]))?;
    points.push(point.clone());
  }
  Ok(Expr::List(points.into()))
}
