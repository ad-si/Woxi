//! TrigFactor[expr] — single-step trigonometric factoring for the
//! same-argument families, in wolframscript's printed forms:
//! Sin[u] + Cos[u] -> Sqrt[2]*Sin[u + Pi/4], 1 +- Cos[u] -> half-angle
//! squares, Sin/Cos double angles, and the Sin^2 - Cos^2 products.
//! Each Sin[v +- Pi/4] factor follows Wolfram's canonical term order:
//! v leads when its first symbol sorts before "Pi" (case-insensitive),
//! otherwise Pi/4 leads and a negative leading term pulls -1 out of
//! the (odd) sine. Anything else passes through unchanged, as
//! wolframscript does for already-factored or non-trig input.

use crate::InterpreterError;
use crate::syntax::Expr;

pub fn trig_factor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "TrigFactor".to_string(),
      args: args.to_vec().into(),
    });
  }
  Ok(factor(&args[0]).unwrap_or_else(|| args[0].clone()))
}

fn trig_call(head: &str, arg: Expr) -> Expr {
  Expr::FunctionCall {
    name: head.to_string(),
    args: vec![arg].into(),
  }
}

fn times(fs: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: fs.into(),
  }
}

fn plus(ts: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: ts.into(),
  }
}

fn neg(e: Expr) -> Expr {
  Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand: Box::new(e),
  }
}

fn div(a: Expr, b: i128) -> Expr {
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(a),
    right: Box::new(Expr::Integer(b)),
  }
}

/// Half-angle term `e/2`, printed as `e/2` (Wolfram's distributed form,
/// not the collapsed `(...)/2`).
fn half(e: &Expr) -> Expr {
  div(e.clone(), 2)
}

/// `p/2 + q/2`.
fn half_sum(p: &Expr, q: &Expr) -> Expr {
  plus(vec![half(p), half(q)])
}

/// `p/2 - q/2`.
fn half_diff(p: &Expr, q: &Expr) -> Expr {
  plus(vec![half(p), neg(half(q))])
}

fn pow2(e: Expr) -> Expr {
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: Box::new(e),
    right: Box::new(Expr::Integer(2)),
  }
}

fn sqrt2() -> Expr {
  Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![Expr::Integer(2)].into(),
  }
}

fn pi_quarter() -> Expr {
  div(Expr::Constant("Pi".to_string()), 4)
}

/// First symbol of `v` in canonical term order, lowercased; decides
/// whether v sorts before Pi inside the printed sum.
fn leading_symbol(v: &Expr) -> Option<String> {
  match v {
    Expr::Identifier(s) | Expr::Constant(s) => Some(s.to_lowercase()),
    Expr::FunctionCall { name, args }
      if (name == "Plus" || name == "Times") && !args.is_empty() =>
    {
      args.iter().find_map(leading_symbol)
    }
    Expr::BinaryOp { left, right, .. } => {
      leading_symbol(left).or_else(|| leading_symbol(right))
    }
    Expr::UnaryOp { operand, .. } => leading_symbol(operand),
    _ => None,
  }
}

fn v_sorts_first(v: &Expr) -> bool {
  matches!(leading_symbol(v), Some(s) if s.as_str() < "pi")
}

/// Build the canonical Sin[v +- Pi/4] factor; returns (factor, sign)
/// where sign is -1 when -1 was pulled out of the sine.
fn sin_pm(v: &Expr, plus_quarter: bool) -> (Expr, i32) {
  if v_sorts_first(v) {
    let mut terms: Vec<Expr> = match v {
      Expr::FunctionCall { name, args } if name == "Plus" => args.to_vec(),
      _ => vec![v.clone()],
    };
    terms.push(if plus_quarter {
      pi_quarter()
    } else {
      neg(pi_quarter())
    });
    (trig_call("Sin", plus(terms)), 1)
  } else if plus_quarter {
    // Sin[v + Pi/4] == Sin[Pi/4 + v]
    (trig_call("Sin", plus(vec![pi_quarter(), v.clone()])), 1)
  } else {
    // Sin[v - Pi/4] == -Sin[Pi/4 - v]
    (
      trig_call("Sin", plus(vec![pi_quarter(), neg(v.clone())])),
      -1,
    )
  }
}

/// Sqrt[2]*Sin[u +- Pi/4], with an overall sign applied per Wolfram's
/// pulled-out-minus print form.
fn sqrt2_sin(u: &Expr, plus_quarter: bool, sign: i32) -> Expr {
  let (factor, fsign) = sin_pm(u, plus_quarter);
  let product = times(vec![sqrt2(), factor]);
  if sign * fsign < 0 {
    neg(product)
  } else {
    product
  }
}

/// 2*Sin[h +- Pi/4]^2 (the square absorbs any pulled-out sign).
fn two_sin_sq(h: &Expr, plus_quarter: bool) -> Expr {
  let (factor, _) = sin_pm(h, plus_quarter);
  times(vec![Expr::Integer(2), pow2(factor)])
}

/// sign*2*Sin[u - Pi/4]*Sin[u + Pi/4] in canonical form (the Cos[2u]
/// product family).
fn double_angle_product(u: &Expr, sign: i32) -> Expr {
  let (f1, s1) = sin_pm(u, false);
  let (f2, s2) = sin_pm(u, true);
  let coeff = sign * s1 * s2;
  times(vec![Expr::Integer(2 * coeff as i128), f1, f2])
}

/// Match Sin[u] / Cos[u] and return (is_sin, u).
fn trig_of(e: &Expr) -> Option<(bool, Expr)> {
  match e {
    Expr::FunctionCall { name, args } if args.len() == 1 => match name.as_str()
    {
      "Sin" => Some((true, args[0].clone())),
      "Cos" => Some((false, args[0].clone())),
      _ => None,
    },
    _ => None,
  }
}

/// Match Times[-1, inner] / UnaryMinus(inner).
fn negated(e: &Expr) -> Option<&Expr> {
  match e {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => Some(operand),
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(-1)) =>
    {
      Some(&args[1])
    }
    _ => None,
  }
}

fn same(a: &Expr, b: &Expr) -> bool {
  crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
}

fn factor(expr: &Expr) -> Option<Expr> {
  // Double angles: Sin[2 u] and Cos[2 u]
  if let Some((is_sin, arg)) = trig_of(expr)
    && let Some(u) = halved(&arg)
  {
    return Some(if is_sin {
      // 2*Cos[u]*Sin[u]
      times(vec![
        Expr::Integer(2),
        trig_call("Cos", u.clone()),
        trig_call("Sin", u),
      ])
    } else {
      // Cos[2u] = 2*Sin[Pi/4 - u]*Sin[Pi/4 + u]
      //         = -2*Sin[u - Pi/4]*Sin[u + Pi/4]
      double_angle_product(&u, -1)
    });
  }

  let terms: Vec<&Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() == 2 => {
      args.iter().collect()
    }
    _ => return None,
  };
  let (a, b) = (terms[0], terms[1]);

  // 1 +- Cos[u] / 1 +- Sin[u]
  if matches!(a, Expr::Integer(1)) {
    let half = |u: &Expr| halved(u).unwrap_or_else(|| div(u.clone(), 2));
    if let Some((is_sin, u)) = trig_of(b) {
      let h = half(&u);
      return Some(if is_sin {
        // 2*Sin[h + Pi/4]^2
        two_sin_sq(&h, true)
      } else {
        // 2*Cos[h]^2
        times(vec![Expr::Integer(2), pow2(trig_call("Cos", h))])
      });
    }
    if let Some(inner) = negated(b)
      && let Some((is_sin, u)) = trig_of(inner)
    {
      let h = half(&u);
      return Some(if is_sin {
        // 2*Sin[h - Pi/4]^2
        two_sin_sq(&h, false)
      } else {
        // 2*Sin[h]^2
        times(vec![Expr::Integer(2), pow2(trig_call("Sin", h))])
      });
    }
  }

  // Sin[u] +- Cos[u] (either Plus order, either negation)
  let classify = |e: &Expr| -> Option<(bool, bool, Expr)> {
    // (is_sin, negated, u)
    if let Some((is_sin, u)) = trig_of(e) {
      return Some((is_sin, false, u));
    }
    if let Some(inner) = negated(e)
      && let Some((is_sin, u)) = trig_of(inner)
    {
      return Some((is_sin, true, u));
    }
    None
  };
  if let (Some((sa, na, ua)), Some((sb, nb, ub))) = (classify(a), classify(b))
    && sa != sb
    && same(&ua, &ub)
  {
    let (sin_neg, cos_neg) = if sa { (na, nb) } else { (nb, na) };
    let u = ua;
    return match (sin_neg, cos_neg) {
      // Sin + Cos = Sqrt[2]*Sin[u + Pi/4]
      (false, false) => Some(sqrt2_sin(&u, true, 1)),
      // Sin - Cos = Sqrt[2]*Sin[u - Pi/4]
      (false, true) => Some(sqrt2_sin(&u, false, 1)),
      // Cos - Sin = -Sqrt[2]*Sin[u - Pi/4]
      (true, false) => Some(sqrt2_sin(&u, false, -1)),
      _ => None,
    };
  }

  // Sin[p] +- Sin[q] sum-to-product, restricted to distinct atomic
  // arguments where Wolfram yields the plain single-step factoring
  // (e.g. Sin[2 x] + Sin[4 x] factors further, so non-symbols are
  // deliberately excluded). The two factors carry different heads
  // (Cos and Sin), so Woxi's Times ordering already matches Wolfram.
  //   Sin[p] + Sin[q] = 2 Cos[p/2 - q/2] Sin[p/2 + q/2]
  //   Sin[p] - Sin[q] = 2 Cos[p/2 + q/2] Sin[p/2 - q/2]
  if let (Some((true, na, ua)), Some((true, nb, ub))) = (classify(a), classify(b))
    && matches!(ua, Expr::Identifier(_))
    && matches!(ub, Expr::Identifier(_))
    && !same(&ua, &ub)
  {
    let hd = half_diff(&ua, &ub);
    let hs = half_sum(&ua, &ub);
    // Fold the overall sign into the leading coefficient (-2, not -(2 ...))
    // to match Wolfram's printed form.
    let cos_sin = |k: i128, c: Expr, s: Expr| {
      times(vec![Expr::Integer(k), trig_call("Cos", c), trig_call("Sin", s)])
    };
    return Some(match (na, nb) {
      (false, false) => cos_sin(2, hd, hs),   // +Sin[p] +Sin[q]
      (false, true) => cos_sin(2, hs, hd),    // +Sin[p] -Sin[q]
      (true, false) => cos_sin(-2, hs, hd),   // -Sin[p] +Sin[q]
      (true, true) => cos_sin(-2, hd, hs),    // -Sin[p] -Sin[q]
    });
  }

  // Sin[u]^2 - Cos[u]^2 and Cos[u]^2 - Sin[u]^2
  let squared_trig = |e: &Expr| -> Option<(bool, bool, Expr)> {
    let (inner, negd) = match negated(e) {
      Some(inner) => (inner, true),
      None => (e, false),
    };
    if let Expr::FunctionCall { name, args } = inner
      && name == "Power"
      && args.len() == 2
      && matches!(&args[1], Expr::Integer(2))
      && let Some((is_sin, u)) = trig_of(&args[0])
    {
      return Some((is_sin, negd, u));
    }
    if let Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } = inner
      && matches!(right.as_ref(), Expr::Integer(2))
      && let Some((is_sin, u)) = trig_of(left)
    {
      return Some((is_sin, negd, u));
    }
    None
  };
  if let (Some((sa, na, ua)), Some((sb, nb, ub))) =
    (squared_trig(a), squared_trig(b))
    && sa != sb
    && na != nb
    && same(&ua, &ub)
  {
    let sin_negated = if sa { na } else { nb };
    // Cos^2 - Sin^2 = Cos[2u]; Sin^2 - Cos^2 = -Cos[2u]
    return Some(double_angle_product(&ua, if sin_negated { -1 } else { 1 }));
  }

  None
}

/// Match an argument with an exact leading factor of 2 and return the
/// other half (2*x -> x, 2*(a + b) -> a + b).
fn halved(arg: &Expr) -> Option<Expr> {
  match arg {
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() >= 2
        && matches!(&args[0], Expr::Integer(2)) =>
    {
      let rest: Vec<Expr> = args[1..].to_vec();
      Some(if rest.len() == 1 {
        rest.into_iter().next().unwrap()
      } else {
        times(rest)
      })
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(2)) => Some((**right).clone()),
    _ => None,
  }
}
