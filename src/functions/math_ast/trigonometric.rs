#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::functions::math_ast::gcd as gcd_i128;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, unevaluated};

/// Try to express a symbolic expression as a rational multiple of Pi: k*Pi/n.
/// Returns Some((k, n)) in lowest terms, None if not recognized.
/// Handles patterns: Pi, n*Pi, Pi/d, n*Pi/d, n*Degree, Degree, and FunctionCall variants.
fn try_symbolic_pi_fraction(expr: &Expr) -> Option<(i64, i64)> {
  // Helper to extract an integer value
  fn as_int(e: &Expr) -> Option<i64> {
    match e {
      Expr::Integer(n) => Some(*n as i64),
      _ => None,
    }
  }

  // Helper to extract a rational value (k, n) from Rational[k, n]
  fn as_rational(e: &Expr) -> Option<(i64, i64)> {
    if let Expr::FunctionCall { name, args } = e
      && name == "Rational"
      && args.len() == 2
      && let (Some(k), Some(n)) = (as_int(&args[0]), as_int(&args[1]))
    {
      Some((k, n))
    } else {
      None
    }
  }

  // Helper to check if expr is Pi
  fn is_pi(e: &Expr) -> bool {
    matches!(e, Expr::Constant(name) | Expr::Identifier(name) if name == "Pi")
  }

  // Helper to check if expr is Degree
  fn is_degree(e: &Expr) -> bool {
    matches!(e, Expr::Constant(name) | Expr::Identifier(name) if name == "Degree")
  }

  // Helper to reduce fraction
  fn reduce(k: i64, n: i64) -> (i64, i64) {
    if k == 0 {
      return (0, 1);
    }
    let g = gcd_i128(k as i128, n as i128) as i64;
    let (k, n) = (k / g, n / g);
    if n < 0 { (-k, -n) } else { (k, n) }
  }

  match expr {
    // Pi => (1, 1)
    _ if is_pi(expr) => Some((1, 1)),
    // -Pi => (-1, 1)
    Expr::Constant(name) if name == "-Pi" => Some((-1, 1)),
    // Degree => (1, 180)
    _ if is_degree(expr) => Some((1, 180)),
    // -Degree => (-1, 180)
    Expr::Constant(name) if name == "-Degree" => Some((-1, 180)),
    // Integer(0) => (0, 1) — sin(0) etc.
    Expr::Integer(0) => Some((0, 1)),

    // n * Pi or Pi * n
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if is_pi(right)
        && let Some(n) = as_int(left)
      {
        return Some(reduce(n, 1));
      }
      if is_pi(left)
        && let Some(n) = as_int(right)
      {
        return Some(reduce(n, 1));
      }
      // n * Degree or Degree * n
      if is_degree(right)
        && let Some(n) = as_int(left)
      {
        return Some(reduce(n, 180));
      }
      if is_degree(left)
        && let Some(n) = as_int(right)
      {
        return Some(reduce(n, 180));
      }
      // n * (Pi / d) or (Pi / d) * n
      if let Some(n) = as_int(left)
        && let Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: num,
          right: den,
        } = right.as_ref()
        && is_pi(num)
        && let Some(d) = as_int(den)
      {
        return Some(reduce(n, d));
      }
      if let Some(n) = as_int(right)
        && let Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: num,
          right: den,
        } = left.as_ref()
        && is_pi(num)
        && let Some(d) = as_int(den)
      {
        return Some(reduce(n, d));
      }
      // Rational[k, d] * Pi or Pi * Rational[k, d]
      if is_pi(right)
        && let Some((k, d)) = as_rational(left)
      {
        return Some(reduce(k, d));
      }
      if is_pi(left)
        && let Some((k, d)) = as_rational(right)
      {
        return Some(reduce(k, d));
      }
      None
    }

    // Pi / d
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      if is_pi(left)
        && let Some(d) = as_int(right)
      {
        return Some(reduce(1, d));
      }
      // (n * Pi) / d
      if let Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: tl,
        right: tr,
      } = left.as_ref()
      {
        if is_pi(tr)
          && let (Some(n), Some(d)) = (as_int(tl), as_int(right))
        {
          return Some(reduce(n, d));
        }
        if is_pi(tl)
          && let (Some(n), Some(d)) = (as_int(tr), as_int(right))
        {
          return Some(reduce(n, d));
        }
      }
      None
    }

    // FunctionCall("Times", [n, Pi]) or FunctionCall("Times", [n, Degree])
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if is_pi(&args[1])
        && let Some(n) = as_int(&args[0])
      {
        return Some(reduce(n, 1));
      }
      if is_pi(&args[0])
        && let Some(n) = as_int(&args[1])
      {
        return Some(reduce(n, 1));
      }
      if is_degree(&args[1])
        && let Some(n) = as_int(&args[0])
      {
        return Some(reduce(n, 180));
      }
      if is_degree(&args[0])
        && let Some(n) = as_int(&args[1])
      {
        return Some(reduce(n, 180));
      }
      // Times[Rational[k, n], Pi]
      if let Expr::FunctionCall {
        name: rn,
        args: rargs,
      } = &args[0]
        && rn == "Rational"
        && rargs.len() == 2
        && is_pi(&args[1])
        && let (Some(k), Some(n)) = (as_int(&rargs[0]), as_int(&rargs[1]))
      {
        return Some(reduce(k, n));
      }
      // Times[n, BinaryOp::Divide(Pi, d)] or Times[n, BinaryOp::Divide(n2*Pi, d)]
      if let Some(n) = as_int(&args[0])
        && let Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: num,
          right: den,
        } = &args[1]
        && is_pi(num)
        && let Some(d) = as_int(den)
      {
        return Some(reduce(n, d));
      }
      None
    }

    // Rational[k, n] * Pi handled via BinaryOp(Times, Rational[k,n], Pi)
    // FunctionCall("Rational", [k, n]) as a standalone — not a Pi fraction
    _ => None,
  }
}

/// Collect the additive terms of a sum, handling both the `Plus[...]` and the
/// `BinaryOp` Plus/Minus representations.
fn collect_plus_terms(e: &Expr, out: &mut Vec<Expr>) {
  match e {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for a in args {
        collect_plus_terms(a, out);
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      collect_plus_terms(left, out);
      collect_plus_terms(right, out);
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      collect_plus_terms(left, out);
      out.push(negate_expr((**right).clone()));
    }
    _ => out.push(e.clone()),
  }
}

/// Whether a canonical Plus term carries a negative leading sign
/// (a negative number, `-x`, or a product with a negative coefficient).
fn term_is_negative(e: &Expr) -> bool {
  match e {
    Expr::Integer(n) => *n < 0,
    Expr::Real(v) => *v < 0.0,
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      ..
    } => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!(&args[0], Expr::Integer(n) if *n < 0)
    }
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      term_is_negative(&args[0])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times | BinaryOperator::Divide,
      left,
      ..
    } => term_is_negative(left),
    _ => None::<()>.is_some(),
  }
}

/// Decompose a trig argument `rest + r*Pi` (rational `r`, non-empty Pi-free
/// `rest`) the way Wolfram canonicalizes trig arguments:
///   * `rest` is sign-normalized — a negative leading term flips the whole
///     sum (`Sin[b - a] -> -Sin[a - b]`, `Sin[Pi/3 - a] -> Cos[a + Pi/6]`);
///   * the phase is reduced into `[-Pi/4, Pi/4]` by quarter-turn (`Pi/2`)
///     steps; the step count `k` rounds half-to-even, so `Sin[a + Pi/4]`
///     stays while `Sin[a + 3*Pi/4]` becomes `-Sin[a - Pi/4]`.
/// Returns `(k mod 4, flip, (pn, pd), rest_terms)` with the residual phase
/// `pn/pd * Pi`, or None when the argument is already canonical.
fn extract_pi_phase(arg: &Expr) -> Option<(i64, bool, (i64, i64), Vec<Expr>)> {
  let is_sum = matches!(arg, Expr::FunctionCall { name, .. } if name == "Plus")
    || matches!(
      arg,
      Expr::BinaryOp {
        op: BinaryOperator::Plus | BinaryOperator::Minus,
        ..
      }
    );
  if !is_sum {
    return None;
  }
  let mut terms = Vec::new();
  collect_plus_terms(arg, &mut terms);
  // Sum all rational-multiple-of-Pi terms into pn/pd; keep the others.
  let (mut pn, mut pd): (i64, i64) = (0, 1);
  let mut rest: Vec<Expr> = Vec::new();
  for t in terms {
    match try_symbolic_pi_fraction(&t) {
      Some((a, b)) if a != 0 => {
        pn = pn.checked_mul(b)?.checked_add(a.checked_mul(pd)?)?;
        pd = pd.checked_mul(b)?;
        let g = gcd_i128(pn as i128, pd as i128).max(1) as i64;
        pn /= g;
        pd /= g;
      }
      _ => rest.push(t),
    }
  }
  if rest.is_empty() {
    return None;
  }
  // Sign-normalize: Wolfram flips the whole argument when the leading term
  // of the WL-canonical sum is negative. Pi-multiples sort alphabetically
  // as the symbol "Pi" among the other monomials (numbers still lead), so
  // the phase term itself leads when every rest term sorts after "Pi"
  // (`Sin[x - Pi/8] -> -Sin[Pi/8 - x]` but `Sin[a - Pi/4]` stays).
  let rest0_is_number = matches!(
    &rest[0],
    Expr::Integer(_)
      | Expr::Real(_)
      | Expr::BigInteger(_)
      | Expr::BigFloat(_, _)
  ) || matches!(&rest[0], Expr::FunctionCall { name, .. } if name == "Rational");
  let pi_leads = pn != 0 && !rest0_is_number && {
    let key =
      crate::functions::list_helpers_ast::sorting::expr_sort_key(&rest[0]);
    crate::functions::list_helpers_ast::wolfram_string_order("Pi", &key) > 0
  };
  let flip = if pi_leads {
    pn < 0
  } else {
    term_is_negative(&rest[0])
  };
  if flip {
    for t in rest.iter_mut() {
      *t = negate_expr(t.clone());
    }
    pn = -pn;
  }
  // k = round-half-to-even(2 * pn/pd): the quarter-turn step count.
  let num = 2 * pn;
  let div = num.div_euclid(pd);
  let rem = num.rem_euclid(pd);
  let k = if 2 * rem > pd {
    div + 1
  } else if 2 * rem < pd {
    div
  } else if div % 2 == 0 {
    div
  } else {
    div + 1
  };
  if k == 0 && !flip {
    return None;
  }
  // Residual phase r' = pn/pd - k/2 = (2*pn - k*pd) / (2*pd).
  let mut rpn = 2 * pn - k * pd;
  let mut rpd = 2 * pd;
  if rpn != 0 {
    let g = gcd_i128(rpn as i128, rpd as i128) as i64;
    rpn /= g;
    rpd /= g;
  }
  Some((k.rem_euclid(4), flip, (rpn, rpd), rest))
}

/// Apply Wolfram's trig-argument canonicalization: quarter-turn phase
/// shifts (`Sin[x + Pi/2] -> Cos[x]`, `Sin[a + Pi/3] -> Cos[a - Pi/6]`,
/// `Tan[x + Pi/2] -> -Cot[x]`) and leading-sign normalization
/// (`Sin[b - a] -> -Sin[a - b]`). Returns None for an unrecognized head or
/// an already-canonical argument.
fn try_trig_pi_phase(
  head: &str,
  arg: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let (k, flip, (pn, pd), mut rest) = extract_pi_phase(arg)?;
  let (new_head, neg): (&str, bool) = match (head, k) {
    ("Sin", 0) => ("Sin", false),
    ("Sin", 1) => ("Cos", false),
    ("Sin", 2) => ("Sin", true),
    ("Sin", 3) => ("Cos", true),
    ("Cos", 0) => ("Cos", false),
    ("Cos", 1) => ("Sin", true),
    ("Cos", 2) => ("Cos", true),
    ("Cos", 3) => ("Sin", false),
    ("Tan", 0) | ("Tan", 2) => ("Tan", false),
    ("Tan", 1) | ("Tan", 3) => ("Cot", true),
    ("Cot", 0) | ("Cot", 2) => ("Cot", false),
    ("Cot", 1) | ("Cot", 3) => ("Tan", true),
    ("Sec", 0) => ("Sec", false),
    ("Sec", 1) => ("Csc", true),
    ("Sec", 2) => ("Sec", true),
    ("Sec", 3) => ("Csc", false),
    ("Csc", 0) => ("Csc", false),
    ("Csc", 1) => ("Sec", false),
    ("Csc", 2) => ("Csc", true),
    ("Csc", 3) => ("Sec", true),
    _ => return None,
  };
  // Odd functions pick up a sign from the leading-term flip.
  let neg = neg ^ (flip && matches!(head, "Sin" | "Tan" | "Cot" | "Csc"));
  if pn != 0 {
    rest.push(lit(&format!("({}*Pi)/{}", pn, pd)));
  }
  let rest_expr = if rest.len() == 1 {
    rest.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: rest.into(),
    }
  };
  // Re-evaluate the rebuilt argument so the sum re-canonicalizes before
  // the (possibly different) trig head sees it.
  let rest_expr = match crate::evaluator::evaluate_expr_to_expr(&rest_expr) {
    Ok(e) => e,
    Err(err) => return Some(Err(err)),
  };
  let inner =
    crate::evaluator::evaluate_function_call_ast(new_head, &[rest_expr]);
  Some(inner.map(|e| if neg { negate_expr(e) } else { e }))
}

/// Exact Sin value for k*Pi/n. Returns None if no simple exact form.
/// Parse a fixed trig closed-form literal into an `Expr`. The argument is a
/// constant Wolfram expression, so parsing never fails at runtime.
fn lit(s: &str) -> Expr {
  crate::syntax::string_to_expr(s).expect("valid trig literal")
}

/// Build `head[m*Pi/d]` (e.g. `Cos[Pi/14]`, `Sin[(3*Pi)/16]`).
fn build_trig_angle_call(head: &str, m: i64, d: i64) -> Expr {
  let angle = if m == 1 {
    format!("Pi/{}", d)
  } else {
    format!("({}*Pi)/{}", m, d)
  };
  lit(&format!("{}[{}]", head, angle))
}

/// Canonical first-octant fallback for a trig function of `k*Pi/n` when there
/// is no exact radical value. `self_head`/`cofunc_head` are the function and
/// its co-function (Sin/Cos, Tan/Cot, Sec/Csc). `(kr, nr)` is the
/// first-quadrant reference angle in `[0, Pi/2]` with sign `sign`; `k_orig`/`n`
/// is the original coprime input used to detect the already-canonical case.
///
/// Wolfram canonicalizes any such angle to the first octant `[0, Pi/4]`:
/// angles above `Pi/4` fold to the co-function of their complement
/// (`Sin[x] = Cos[Pi/2 - x]`, `Tan[x] = Cot[Pi/2 - x]`, …). Returns `None` only
/// when the input is already the canonical form, so the caller leaves it
/// unevaluated (and no re-evaluation loop occurs).
fn octant_fallback(
  self_head: &str,
  cofunc_head: &str,
  kr: i64,
  nr: i64,
  sign: i64,
  k_orig: i64,
  n: i64,
) -> Option<Expr> {
  // Already canonical: the original angle is in [0, Pi/4] and positive.
  if sign == 1 && k_orig >= 0 && 4 * k_orig <= n {
    return None;
  }
  let result = if 4 * kr > nr {
    // Reference angle exceeds Pi/4 → co-function of the complement.
    let cm = nr - 2 * kr;
    let cd = 2 * nr;
    let g = gcd_i128(cm as i128, cd as i128) as i64;
    build_trig_angle_call(cofunc_head, cm / g, cd / g)
  } else {
    build_trig_angle_call(self_head, kr, nr)
  };
  Some(if sign == -1 {
    negate_expr(result)
  } else {
    result
  })
}

fn exact_sin(k: i64, n: i64) -> Option<Expr> {
  let k_orig = k;
  // Normalize to [0, 2*Pi) i.e., k mod 2n, with k in [0, 2n)
  let period = 2 * n;
  let k = ((k % period) + period) % period;
  // Use symmetry: sin is periodic with period 2*Pi
  // Map to first quadrant and track sign
  let (k_ref, sign) = if k <= n / 2 {
    // First quadrant: [0, Pi/2]
    // k/n is in [0, 1/2], angle is in [0, Pi/2]
    // But we need to check: k*Pi/n in [0, Pi/2] means k/n <= 1/2
    (k, 1i64)
  } else if k <= n {
    // Second quadrant: (Pi/2, Pi]
    // sin(Pi - x) = sin(x)
    (n - k, 1)
  } else if k < 2 * n {
    // Third and fourth quadrants: (Pi, 2*Pi)
    // sin(Pi + x) = -sin(x)
    if k <= 3 * n / 2 {
      (k - n, -1)
    } else {
      (2 * n - k, -1)
    }
  } else {
    (0, 1) // k == 2n, sin(2*Pi) = 0
  };

  // Reduce k_ref/n to lowest terms for table lookup
  let g = gcd_i128(k_ref as i128, n as i128) as i64;
  let (kr, nr) = (k_ref / g, n / g);
  // Now compute sin(kr * Pi / nr) for first quadrant reference angle
  let val = match (kr, nr) {
    (0, _) => Expr::Integer(0),
    // sin(Pi/12) = (-1 + Sqrt[3]) / (2*Sqrt[2])
    (1, 12) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(make_sqrt(Expr::Integer(3))),
      }),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(make_sqrt(Expr::Integer(2))),
      }),
    },
    // sin(Pi/6) = 1/2
    (1, 6) => make_rational(1, 2),
    // sin(Pi/4) = 1/Sqrt[2]
    (1, 4) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(make_sqrt(Expr::Integer(2))),
    },
    // sin(Pi/3) = Sqrt[3]/2
    (1, 3) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(make_sqrt(Expr::Integer(3))),
      right: Box::new(Expr::Integer(2)),
    },
    // sin(5*Pi/12) = (1 + Sqrt[3]) / (2*Sqrt[2])
    (5, 12) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(make_sqrt(Expr::Integer(3))),
      }),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(make_sqrt(Expr::Integer(2))),
      }),
    },
    // sin(Pi/5) = Sqrt[5/8 - Sqrt[5]/8]
    (1, 5) => lit("Sqrt[5/8 - Sqrt[5]/8]"),
    // sin(2*Pi/5) = Sqrt[5/8 + Sqrt[5]/8]
    (2, 5) => lit("Sqrt[5/8 + Sqrt[5]/8]"),
    // sin(Pi/10) = (-1 + Sqrt[5])/4
    (1, 10) => lit("(-1 + Sqrt[5])/4"),
    // sin(3*Pi/10) = (1 + Sqrt[5])/4
    (3, 10) => lit("(1 + Sqrt[5])/4"),
    // sin(Pi/2) = 1
    (1, 2) => Expr::Integer(1),
    // No radical form: fold to the canonical first-octant Sin/Cos.
    _ => return octant_fallback("Sin", "Cos", kr, nr, sign, k_orig, n),
  };

  if sign == -1 {
    if matches!(val, Expr::Integer(0)) {
      Some(Expr::Integer(0))
    } else {
      Some(negate_expr(val))
    }
  } else {
    Some(val)
  }
}

/// Exact Cos value for k*Pi/n. Uses cos(x) = sin(Pi/2 - x).
fn exact_cos(k: i64, n: i64) -> Option<Expr> {
  let k_orig = k;
  // cos(k*Pi/n) = sin(Pi/2 - k*Pi/n) = sin((n - 2k)*Pi/(2n))
  // Simplify: cos(k*Pi/n) = sin((n/2 - k)*Pi/n) -- only works if n is even
  // Better: use direct table
  let period = 2 * n;
  let k = ((k % period) + period) % period;
  // Map to first quadrant
  let (k_ref, sign) = if k <= n / 2 {
    (k, 1i64)
  } else if k <= n {
    (n - k, -1)
  } else if k <= 3 * n / 2 {
    (k - n, -1)
  } else {
    (2 * n - k, 1)
  };

  // Reduce k_ref/n to lowest terms for table lookup
  let g = gcd_i128(k_ref as i128, n as i128) as i64;
  let (kr, nr) = (k_ref / g, n / g);
  let val = match (kr, nr) {
    (0, _) => Expr::Integer(1),
    // cos(Pi/12) = (1 + Sqrt[3]) / (2*Sqrt[2])
    (1, 12) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(make_sqrt(Expr::Integer(3))),
      }),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(make_sqrt(Expr::Integer(2))),
      }),
    },
    // cos(Pi/6) = Sqrt[3]/2
    (1, 6) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(make_sqrt(Expr::Integer(3))),
      right: Box::new(Expr::Integer(2)),
    },
    // cos(Pi/4) = 1/Sqrt[2]
    (1, 4) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(make_sqrt(Expr::Integer(2))),
    },
    // cos(Pi/3) = 1/2
    (1, 3) => make_rational(1, 2),
    // cos(5*Pi/12) = (-1 + Sqrt[3]) / (2*Sqrt[2])
    (5, 12) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(make_sqrt(Expr::Integer(3))),
      }),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(make_sqrt(Expr::Integer(2))),
      }),
    },
    // cos(Pi/5) = (1 + Sqrt[5])/4
    (1, 5) => lit("(1 + Sqrt[5])/4"),
    // cos(2*Pi/5) = (-1 + Sqrt[5])/4
    (2, 5) => lit("(-1 + Sqrt[5])/4"),
    // cos(Pi/10) = Sqrt[5/8 + Sqrt[5]/8]
    (1, 10) => lit("Sqrt[5/8 + Sqrt[5]/8]"),
    // cos(3*Pi/10) = Sqrt[5/8 - Sqrt[5]/8]
    (3, 10) => lit("Sqrt[5/8 - Sqrt[5]/8]"),
    // cos(Pi/2) = 0
    (1, 2) => Expr::Integer(0),
    // No radical form: fold to the canonical first-octant Sin/Cos.
    _ => return octant_fallback("Cos", "Sin", kr, nr, sign, k_orig, n),
  };

  if sign == -1 {
    if matches!(val, Expr::Integer(0)) {
      Some(Expr::Integer(0))
    } else {
      Some(negate_expr(val))
    }
  } else {
    Some(val)
  }
}

/// Exact Tan value for k*Pi/n.
/// Tan has period Pi, so normalize k mod n. Undefined when cos = 0.
fn exact_tan(k: i64, n: i64) -> Option<Expr> {
  let k_orig = k;
  // Tan has period Pi, so reduce k*Pi/n mod Pi => (k mod n)*Pi/n
  let k_mod = ((k % n) + n) % n; // in [0, n)
  // Use symmetry: tan(-x) = -tan(x), tan(Pi - x) = -tan(x)
  // Normalize to [0, Pi/2) i.e., k_mod/n in [0, 1/2)
  // Check for Pi/2: k_mod*2 == n means angle is Pi/2 → ComplexInfinity
  if k_mod * 2 == n {
    return Some(Expr::Identifier("ComplexInfinity".to_string()));
  }
  let (k_ref, n_ref, sign) = if k_mod * 2 < n {
    // First half [0, Pi/2): positive
    (k_mod, n, 1i64)
  } else {
    // Second half (Pi/2, Pi): tan(Pi - x) = -tan(x)
    (n - k_mod, n, -1)
  };
  // Reduce fraction k_ref/n_ref
  let g = gcd_i128(k_ref as i128, n_ref as i128) as i64;
  let (kr, nr) = (k_ref / g, n_ref / g);

  let val = match (kr, nr) {
    (0, _) => Expr::Integer(0),
    // tan(Pi/12) = 2 - Sqrt[3]
    (1, 12) => Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(make_sqrt(Expr::Integer(3))),
    },
    // tan(Pi/6) = 1/Sqrt[3]
    (1, 6) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(make_sqrt(Expr::Integer(3))),
    },
    // tan(Pi/4) = 1
    (1, 4) => Expr::Integer(1),
    // tan(Pi/3) = Sqrt[3]
    (1, 3) => make_sqrt(Expr::Integer(3)),
    // tan(5*Pi/12) = 2 + Sqrt[3]
    (5, 12) => Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(make_sqrt(Expr::Integer(3))),
    },
    // tan(Pi/5) = Sqrt[5 - 2*Sqrt[5]]
    (1, 5) => lit("Sqrt[5 - 2*Sqrt[5]]"),
    // tan(2*Pi/5) = Sqrt[5 + 2*Sqrt[5]]
    (2, 5) => lit("Sqrt[5 + 2*Sqrt[5]]"),
    // tan(Pi/10) = Sqrt[1 - 2/Sqrt[5]]
    (1, 10) => lit("Sqrt[1 - 2/Sqrt[5]]"),
    // tan(3*Pi/10) = Sqrt[1 + 2/Sqrt[5]]
    (3, 10) => lit("Sqrt[1 + 2/Sqrt[5]]"),
    // No radical form: fold to the canonical first-octant Tan/Cot.
    _ => return octant_fallback("Tan", "Cot", kr, nr, sign, k_orig, n),
  };

  if sign == -1 {
    if matches!(val, Expr::Integer(0)) {
      Some(Expr::Integer(0))
    } else {
      Some(negate_expr(val))
    }
  } else {
    Some(val)
  }
}

fn exact_sec(k: i64, n: i64) -> Option<Expr> {
  let k_orig = k;
  // Sec has period 2*Pi, and Sec(-x) = Sec(x), Sec(Pi-x) = -Sec(x)
  // Reduce to [0, 2*Pi)
  let k_mod = ((k % (2 * n)) + 2 * n) % (2 * n);
  // Use Sec(x) = Sec(2*Pi - x) symmetry to reduce to [0, Pi]
  let k2 = if k_mod > n { 2 * n - k_mod } else { k_mod };
  // In [0, Pi]: Sec(Pi - x) = -Sec(x), so reduce to [0, Pi/2]
  let (k_ref, sign) = if k2 * 2 > n {
    (n - k2, -1i64) // Pi - angle
  } else {
    (k2, 1)
  };
  // Check for Pi/2: k_ref*2 == n means Sec(Pi/2) = ComplexInfinity
  if k_ref * 2 == n {
    return Some(Expr::Identifier("ComplexInfinity".to_string()));
  }
  let g = gcd_i128(k_ref as i128, n as i128) as i64;
  let (kr, nr) = (k_ref / g, n / g);

  let val = match (kr, nr) {
    (0, _) => Expr::Integer(1),
    // Sec(Pi/6) = 2/Sqrt[3]
    (1, 6) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(make_sqrt(Expr::Integer(3))),
    },
    // Sec(Pi/4) = Sqrt[2]
    (1, 4) => make_sqrt(Expr::Integer(2)),
    // Sec(Pi/3) = 2
    (1, 3) => Expr::Integer(2),
    // Sec(Pi/5) = -1 + Sqrt[5]
    (1, 5) => lit("-1 + Sqrt[5]"),
    // Sec(2*Pi/5) = 1 + Sqrt[5]
    (2, 5) => lit("1 + Sqrt[5]"),
    // Sec(Pi/12) = Sqrt[2]*(-1 + Sqrt[3])
    (1, 12) => lit("Sqrt[2]*(-1 + Sqrt[3])"),
    // Sec(5*Pi/12) = Sqrt[2]*(1 + Sqrt[3])
    (5, 12) => lit("Sqrt[2]*(1 + Sqrt[3])"),
    // Sec(Pi/10) = 1/Sqrt[5/8 + Sqrt[5]/8]
    (1, 10) => lit("1/Sqrt[5/8 + Sqrt[5]/8]"),
    // Sec(3*Pi/10) = 1/Sqrt[5/8 - Sqrt[5]/8]
    (3, 10) => lit("1/Sqrt[5/8 - Sqrt[5]/8]"),
    // No radical form: fold to the canonical first-octant Sec/Csc.
    _ => return octant_fallback("Sec", "Csc", kr, nr, sign, k_orig, n),
  };

  if sign == -1 {
    Some(negate_expr(val))
  } else {
    Some(val)
  }
}

fn exact_csc(k: i64, n: i64) -> Option<Expr> {
  let k_orig = k;
  // Csc has period 2*Pi, Csc(-x) = -Csc(x), Csc(Pi-x) = Csc(x)
  // Reduce to [0, 2*Pi)
  let k_mod = ((k % (2 * n)) + 2 * n) % (2 * n);
  // Csc(0) and Csc(Pi) are ComplexInfinity
  if k_mod == 0 || k_mod == n {
    return Some(Expr::Identifier("ComplexInfinity".to_string()));
  }
  // Use Csc(2*Pi - x) = -Csc(x) to reduce to [0, Pi]
  let (k2, sign1) = if k_mod > n {
    (2 * n - k_mod, -1i64)
  } else {
    (k_mod, 1)
  };
  // In (0, Pi): Csc(Pi - x) = Csc(x), so reduce to (0, Pi/2]
  let k_ref = if k2 * 2 > n { n - k2 } else { k2 };

  let g = gcd_i128(k_ref as i128, n as i128) as i64;
  let (kr, nr) = (k_ref / g, n / g);

  let val = match (kr, nr) {
    // Csc(Pi/2) = 1
    (1, 2) => Expr::Integer(1),
    // Csc(Pi/6) = 2
    (1, 6) => Expr::Integer(2),
    // Csc(Pi/4) = Sqrt[2]
    (1, 4) => make_sqrt(Expr::Integer(2)),
    // Csc(Pi/3) = 2/Sqrt[3]
    (1, 3) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(make_sqrt(Expr::Integer(3))),
    },
    // Csc(Pi/5) = 1/Sqrt[5/8 - Sqrt[5]/8]
    (1, 5) => lit("1/Sqrt[5/8 - Sqrt[5]/8]"),
    // Csc(2*Pi/5) = 1/Sqrt[5/8 + Sqrt[5]/8]
    (2, 5) => lit("1/Sqrt[5/8 + Sqrt[5]/8]"),
    // Csc(Pi/12) = Sqrt[2]*(1 + Sqrt[3])
    (1, 12) => lit("Sqrt[2]*(1 + Sqrt[3])"),
    // Csc(5*Pi/12) = Sqrt[2]*(-1 + Sqrt[3])
    (5, 12) => lit("Sqrt[2]*(-1 + Sqrt[3])"),
    // Csc(Pi/10) = 1 + Sqrt[5]
    (1, 10) => lit("1 + Sqrt[5]"),
    // Csc(3*Pi/10) = -1 + Sqrt[5]
    (3, 10) => lit("-1 + Sqrt[5]"),
    // No radical form: fold to the canonical first-octant Csc/Sec.
    _ => return octant_fallback("Csc", "Sec", kr, nr, sign1, k_orig, n),
  };

  if sign1 == -1 {
    Some(negate_expr(val))
  } else {
    Some(val)
  }
}

fn exact_cot(k: i64, n: i64) -> Option<Expr> {
  let k_orig = k;
  // Cot has period Pi, Cot(-x) = -Cot(x)
  // Reduce k*Pi/n mod Pi => (k mod n)*Pi/n
  let k_mod = ((k % n) + n) % n;
  // Cot(0) = ComplexInfinity
  if k_mod == 0 {
    return Some(Expr::Identifier("ComplexInfinity".to_string()));
  }
  // Cot(Pi/2) = 0
  if k_mod * 2 == n {
    return Some(Expr::Integer(0));
  }
  // Use Cot(Pi - x) = -Cot(x) to reduce to (0, Pi/2)
  let (k_ref, sign) = if k_mod * 2 > n {
    (n - k_mod, -1i64)
  } else {
    (k_mod, 1)
  };

  let g = gcd_i128(k_ref as i128, n as i128) as i64;
  let (kr, nr) = (k_ref / g, n / g);

  let val = match (kr, nr) {
    // Cot(Pi/12) = 2 + Sqrt[3]
    (1, 12) => Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(make_sqrt(Expr::Integer(3))),
    },
    // Cot(Pi/6) = Sqrt[3]
    (1, 6) => make_sqrt(Expr::Integer(3)),
    // Cot(Pi/4) = 1
    (1, 4) => Expr::Integer(1),
    // Cot(Pi/3) = 1/Sqrt[3]
    (1, 3) => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(make_sqrt(Expr::Integer(3))),
    },
    // Cot(5*Pi/12) = 2 - Sqrt[3]
    (5, 12) => Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(make_sqrt(Expr::Integer(3))),
    },
    // Cot(Pi/5) = Sqrt[1 + 2/Sqrt[5]]
    (1, 5) => lit("Sqrt[1 + 2/Sqrt[5]]"),
    // Cot(2*Pi/5) = Sqrt[1 - 2/Sqrt[5]]
    (2, 5) => lit("Sqrt[1 - 2/Sqrt[5]]"),
    // Cot(Pi/10) = Sqrt[5 + 2*Sqrt[5]]
    (1, 10) => lit("Sqrt[5 + 2*Sqrt[5]]"),
    // Cot(3*Pi/10) = Sqrt[5 - 2*Sqrt[5]]
    (3, 10) => lit("Sqrt[5 - 2*Sqrt[5]]"),
    // No radical form: fold to the canonical first-octant Cot/Tan.
    _ => return octant_fallback("Cot", "Tan", kr, nr, sign, k_orig, n),
  };

  if sign == -1 {
    if matches!(val, Expr::Integer(0)) {
      Some(Expr::Integer(0))
    } else {
      Some(negate_expr(val))
    }
  } else {
    Some(val)
  }
}

/// Negate an Expr, simplifying integer, rational, and division cases

/// Canonicalize an exact-value table result with one evaluation pass. The
/// tables build raw BinaryOp trees (e.g. Divide[1, Sqrt[2]]) that would
/// otherwise nest instead of simplifying in downstream arithmetic like
/// 2/Sin[Pi/4]. A negated sum distributes its sign first, so Cot[-Pi/12]
/// keeps wolframscript's "-2 - Sqrt[3]" form rather than "-(2 + Sqrt[3])".
fn canonicalize_exact_trig_value(
  exact: Expr,
) -> Result<Expr, InterpreterError> {
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = &exact
    && let Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } = operand.as_ref()
  {
    let distributed = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(negate_expr((**left).clone())),
      right: Box::new(negate_expr((**right).clone())),
    };
    return crate::evaluator::evaluate_expr_to_expr(&distributed);
  }
  crate::evaluator::evaluate_expr_to_expr(&exact)
}

pub fn negate_expr(mut expr: Expr) -> Expr {
  match &mut expr {
    Expr::Integer(n) => return Expr::Integer(-*n),
    Expr::Real(f) => return Expr::Real(-*f),
    // --x => x (collapse instead of stacking another wrapper)
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      return *std::mem::replace(operand, Box::new(Expr::Integer(0)));
    }
    // Times with a leading numeric coefficient: negate the coefficient,
    // collapsing Times[-1, x] => x.
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() >= 2
        && matches!(&args[0], Expr::Integer(_))
        && !matches!(&args[0], Expr::Integer(0)) =>
    {
      let Expr::Integer(n) = args[0] else {
        unreachable!()
      };
      if n == -1 && args.len() == 2 {
        return args[1].clone();
      }
      if n == -1 {
        let rest: Vec<Expr> = args[1..].to_vec();
        return Expr::FunctionCall {
          name: "Times".to_string(),
          args: rest.into(),
        };
      }
      let mut new_args = std::mem::take(args);
      new_args[0] = Expr::Integer(-n);
      return Expr::FunctionCall {
        name: "Times".to_string(),
        args: new_args,
      };
    }
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() >= 2
        && matches!(&args[0], Expr::FunctionCall { name: rn, args: ra }
          if rn == "Rational" && ra.len() == 2 && matches!(&ra[0], Expr::Integer(_))) =>
    {
      let mut new_args = std::mem::take(args);
      new_args[0] = negate_expr(new_args[0].clone());
      return Expr::FunctionCall {
        name: "Times".to_string(),
        args: new_args,
      };
    }
    Expr::BinaryOp { op, left, right }
      if *op == BinaryOperator::Times
        && matches!(left.as_ref(), Expr::Integer(n) if *n != 0) =>
    {
      let Expr::Integer(n) = **left else {
        unreachable!()
      };
      if n == -1 {
        return *std::mem::replace(right, Box::new(Expr::Integer(0)));
      }
      let right = std::mem::replace(right, Box::new(Expr::Integer(0)));
      return Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-n)),
        right,
      };
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let Expr::Integer(n) = &args[0] {
        return Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-n), args[1].clone()].into(),
        };
      }
      let name = std::mem::take(name);
      let args = std::mem::take(args);
      return Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), Expr::FunctionCall { name, args }].into(),
      };
    }
    // Canonical (evaluated) sums distribute the sign over every term, just
    // like the BinaryOp arm below, so -Plus[2, Sqrt[3]] stays the additive
    // "-2 - Sqrt[3]" instead of the factored "-(2 + Sqrt[3])".
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let args = std::mem::take(args);
      return Expr::FunctionCall {
        name: "Plus".to_string(),
        args: args.iter().map(|a| negate_expr(a.clone())).collect(),
      };
    }
    // -(a + b) => (-a) + (-b): distribute over a sum so the result keeps the
    // flattened additive form Wolfram displays (e.g. -(-1 + Sqrt[5]) => 1 - Sqrt[5]).
    Expr::BinaryOp { op, left, right } if *op == BinaryOperator::Plus => {
      let left = std::mem::replace(left, Box::new(Expr::Integer(0)));
      let right = std::mem::replace(right, Box::new(Expr::Integer(0)));
      return Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(negate_expr(*left)),
        right: Box::new(negate_expr(*right)),
      };
    }
    // -(a - b) => (-a) + b: likewise distribute over a difference so the
    // result keeps the additive form (e.g. -(2 - Sqrt[3]) => -2 + Sqrt[3]).
    Expr::BinaryOp { op, left, right } if *op == BinaryOperator::Minus => {
      let left = std::mem::replace(left, Box::new(Expr::Integer(0)));
      let right = std::mem::replace(right, Box::new(Expr::Integer(0)));
      return Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(negate_expr(*left)),
        right: Box::new(*right),
      };
    }
    // -(a/b) => Times[-1, a/b] to match Wolfram output style
    Expr::BinaryOp { op, left, right } if *op == BinaryOperator::Divide => {
      // If numerator is an integer, negate it directly: -(n/b) => (-n)/b
      if let Expr::Integer(n) = &**left
        && *n > 1
      {
        let neg_left = Box::new(Expr::Integer(-*n));
        let right = std::mem::replace(right, Box::new(Expr::Integer(0)));
        return Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: neg_left,
          right,
        };
      }
      // Otherwise pull the denominator's integer factor into a -1/k
      // coefficient, matching Wolfram: -(Sqrt[3]/2) => -1/2*Sqrt[3] and
      // -((-1 + Sqrt[3])/(2 Sqrt[2])) => -1/2*(-1 + Sqrt[3])/Sqrt[2].
      if let Some((k, rest)) = denom_integer_factor(right) {
        let mut factors = vec![
          crate::functions::math_ast::make_rational(-1, k),
          (**left).clone(),
        ];
        if let Some(rest) = rest {
          factors.push(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(rest),
            right: Box::new(Expr::Integer(-1)),
          });
        }
        return Expr::FunctionCall {
          name: "Times".to_string(),
          args: factors.into(),
        };
      }
      // fall through to default
    }
    _ => {}
  }
  // Default: wrap in Times[-1, expr]
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(-1), expr].into(),
  }
}

/// Split a denominator into `(k, rest)` where `k > 1` is its leading positive
/// integer factor and `rest` is the remaining (non-integer) part, or `None`
/// when there is no such integer factor. `2` → `(2, None)`,
/// `2*Sqrt[2]` → `(2, Some(Sqrt[2]))`.
fn denom_integer_factor(denom: &Expr) -> Option<(i128, Option<Expr>)> {
  match denom {
    Expr::Integer(n) if *n > 1 => Some((*n, None)),
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      if let Expr::Integer(n) = &args[0]
        && *n > 1
      {
        let rest: Vec<Expr> = args[1..].to_vec();
        let rest_expr = if rest.len() == 1 {
          rest.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: rest.into(),
          }
        };
        Some((*n, Some(rest_expr)))
      } else {
        None
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if let Expr::Integer(n) = left.as_ref()
        && *n > 1
      {
        Some((*n, Some((**right).clone())))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Build a Complex float result expression from real and imaginary parts.
fn build_complex_float_result(
  re: f64,
  im: f64,
) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_function_call_ast(
    "Complex",
    &[Expr::Real(re), Expr::Real(im)],
  )
}

/// Check if an argument is Indeterminate or ComplexInfinity.
/// For periodic/trig functions, both should return Indeterminate.
fn is_indeterminate_or_complex_infinity(expr: &Expr) -> bool {
  matches!(expr, Expr::Identifier(s) if s == "Indeterminate" || s == "ComplexInfinity")
}

/// If `arg` has the imaginary unit `I` as a factor, return the product of the
/// remaining factors `z` (so `arg == I*z`). Handles `I` alone (z = 1) and the
/// `Times` form (`I x`, `2 I x`, `-I x`, …).
fn extract_imaginary_factor(arg: &Expr) -> Option<Expr> {
  if matches!(arg, Expr::Identifier(s) if s == "I") {
    return Some(Expr::Integer(1));
  }
  if let Expr::FunctionCall { name, args } = arg
    && name == "Times"
  {
    let mut i_count = 0;
    let mut other: Vec<Expr> = Vec::new();
    for a in args {
      if matches!(a, Expr::Identifier(s) if s == "I") {
        i_count += 1;
      } else {
        other.push(a.clone());
      }
    }
    if i_count == 1 {
      return Some(match other.len() {
        0 => Expr::Integer(1),
        1 => other.into_iter().next().unwrap(),
        _ => Expr::FunctionCall {
          name: "Times".to_string(),
          args: other.into(),
        },
      });
    }
  }
  None
}

/// Reduce a (hyperbolic) trig function of an imaginary argument `I*z` to its
/// counterpart, matching wolframscript:
///   Cos[I z]=Cosh[z], Sin[I z]=I Sinh[z], Tan[I z]=I Tanh[z],
///   Cot[I z]=-I Coth[z], Sec[I z]=Sech[z], Csc[I z]=-I Csch[z], and the
///   hyperbolic duals Cosh[I z]=Cos[z], Sinh[I z]=I Sin[z], etc.
/// Returns None when the argument is not of the form `I*z`.
fn imaginary_arg_reduction(
  func: &str,
  arg: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let z = extract_imaginary_factor(arg)?;
  // (counterpart, leading factor): 0 = none, 1 = I, -1 = -I.
  let (target, factor): (&str, i8) = match func {
    "Cos" => ("Cosh", 0),
    "Sin" => ("Sinh", 1),
    "Tan" => ("Tanh", 1),
    "Cot" => ("Coth", -1),
    "Sec" => ("Sech", 0),
    "Csc" => ("Csch", -1),
    "Cosh" => ("Cos", 0),
    "Sinh" => ("Sin", 1),
    "Tanh" => ("Tan", 1),
    "Coth" => ("Cot", -1),
    "Sech" => ("Sec", 0),
    "Csch" => ("Csc", -1),
    _ => return None,
  };
  let inner = match crate::evaluator::evaluate_function_call_ast(target, &[z]) {
    Ok(v) => v,
    Err(e) => return Some(Err(e)),
  };
  let result = match factor {
    0 => inner,
    1 => Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Identifier("I".to_string()), inner].into(),
    },
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), Expr::Identifier("I".to_string()), inner]
        .into(),
    },
  };
  Some(crate::evaluator::evaluate_expr_to_expr(&result))
}

/// Sin, Cos, Tan - Trigonometric functions (fully symbolic)
/// Only evaluate to float for Real arguments. For integer/symbolic args,
/// `Sqrt[1 +- x^2]` with the inner `1 +- x^2` evaluated so that a compound
/// argument's square expands (e.g. `(2 y)^2 -> 4 y^2`), matching wolframscript.
fn sqrt_one_pm_sq(x: &Expr, plus: bool) -> Expr {
  let x_sq = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(x.clone()),
    right: Box::new(Expr::Integer(2)),
  };
  let inner = Expr::BinaryOp {
    op: if plus {
      BinaryOperator::Plus
    } else {
      BinaryOperator::Minus
    },
    left: Box::new(Expr::Integer(1)),
    right: Box::new(x_sq),
  };
  let inner = crate::evaluator::evaluate_expr_to_expr(&inner).unwrap_or(inner);
  // Evaluate the Sqrt too, so a rational radicand collapses (e.g.
  // Cos[ArcSin[3/5]] = Sqrt[16/25] -> 4/5). A symbolic radicand such as
  // 1 - x^2 stays as Sqrt[1 - x^2].
  let sqrt = Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![inner].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&sqrt).unwrap_or(sqrt)
}

/// `Sqrt[1 - x^2]` (used for trig-of-inverse-trig identities).
fn sqrt_one_minus_sq(x: &Expr) -> Expr {
  sqrt_one_pm_sq(x, false)
}

/// `Sqrt[1 + x^2]`.
fn sqrt_one_plus_sq(x: &Expr) -> Expr {
  sqrt_one_pm_sq(x, true)
}

fn divide(num: Expr, den: Expr) -> Expr {
  let quotient = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(num),
    right: Box::new(den),
  };
  // Evaluate so a rational numerator/denominator collapses, e.g.
  // Tan[ArcSin[3/5]] = (3/5)/(4/5) -> 3/4; a symbolic quotient such as
  // x/Sqrt[1 - x^2] is left in its canonical printed form.
  crate::evaluator::evaluate_expr_to_expr(&quotient).unwrap_or(quotient)
}

/// The bare reciprocal `1/x`, canonicalized the way wolframscript prints it:
/// `x^(-1)` for an atom, `1/(2 x)` for a product, `x^(-2)` for `x^2`, …
fn power_neg_one(x: &Expr) -> Expr {
  let p = Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![x.clone(), Expr::Integer(-1)].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&p).unwrap_or(p)
}

/// Reciprocal trig (Sec/Csc/Cot) of an inverse-trig function, collapsed to its
/// algebraic form (the reciprocals of the Sin/Cos/Tan identities). Returns
/// `None` when the argument is not `ArcSin`/`ArcCos`/`ArcTan` of one argument.
fn reciprocal_trig_of_inverse(outer: &str, inner: &Expr) -> Option<Expr> {
  let (name, x) = match inner {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      (name.as_str(), &args[0])
    }
    _ => return None,
  };
  let result = match (outer, name) {
    // Inverse-function identities: F[ArcF[x]] = x.
    ("Sec", "ArcSec") => x.clone(),
    ("Csc", "ArcCsc") => x.clone(),
    ("Cot", "ArcCot") => x.clone(),
    ("Sec", "ArcSin") => divide(Expr::Integer(1), sqrt_one_minus_sq(x)),
    ("Sec", "ArcCos") => power_neg_one(x),
    ("Sec", "ArcTan") => sqrt_one_plus_sq(x),
    ("Csc", "ArcSin") => power_neg_one(x),
    ("Csc", "ArcCos") => divide(Expr::Integer(1), sqrt_one_minus_sq(x)),
    ("Csc", "ArcTan") => divide(sqrt_one_plus_sq(x), x.clone()),
    ("Cot", "ArcSin") => divide(sqrt_one_minus_sq(x), x.clone()),
    ("Cot", "ArcCos") => divide(x.clone(), sqrt_one_minus_sq(x)),
    ("Cot", "ArcTan") => power_neg_one(x),
    _ => return None,
  };
  Some(result)
}

/// `Sqrt[(-1 + x)/(1 + x)] * (1 + x)` — wolframscript's branch-cut form for
/// the hyperbolic-of-ArcCosh identities.
fn arccosh_branch_form(x: &Expr) -> Expr {
  let one_plus = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(Expr::Integer(1)),
    right: Box::new(x.clone()),
  };
  let minus_one_plus = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(x.clone()),
  };
  let sqrt = Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(minus_one_plus),
      right: Box::new(one_plus.clone()),
    }]
    .into(),
  };
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![sqrt, one_plus].into(),
  }
}

/// Hyperbolic (Sinh/Cosh/Tanh) of an inverse-hyperbolic function, collapsed to
/// its algebraic form. Returns `None` unless the argument is `ArcSinh`/
/// `ArcCosh`/`ArcTanh` of one argument.
fn hyperbolic_of_inverse(outer: &str, inner: &Expr) -> Option<Expr> {
  let (name, x) = match inner {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      (name.as_str(), &args[0])
    }
    _ => return None,
  };
  let result = match (outer, name) {
    ("Sinh", "ArcCosh") => arccosh_branch_form(x),
    ("Sinh", "ArcTanh") => divide(x.clone(), sqrt_one_minus_sq(x)),
    ("Cosh", "ArcSinh") => sqrt_one_plus_sq(x),
    ("Cosh", "ArcTanh") => divide(Expr::Integer(1), sqrt_one_minus_sq(x)),
    ("Tanh", "ArcSinh") => divide(x.clone(), sqrt_one_plus_sq(x)),
    ("Tanh", "ArcCosh") => divide(arccosh_branch_form(x), x.clone()),
    _ => return None,
  };
  Some(result)
}

/// try exact Pi-fraction lookup, otherwise return unevaluated.
pub fn sin_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sin expects 1 argument".into(),
    ));
  }
  if let Some(r) = imaginary_arg_reduction("Sin", &args[0]) {
    return r;
  }
  // Sin[Interval[...]] → range over the interval.
  if let Some(r) =
    crate::functions::interval_ast::trig_interval("Sin", &args[0])
  {
    return Ok(r);
  }
  // Sin[±Infinity] → Interval[{-1, 1}]
  if let Some(r) = circular_at_infinity("Sin", &args[0]) {
    return r;
  }
  // Sin[-x] → -Sin[x] (odd function)
  if let Some(neg) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Sin", &[neg])?;
    return Ok(negate_expr(inner));
  }
  // Sin[ArcSin[x]] = x (inverse function identity)
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcSin"
    && ia.len() == 1
  {
    return Ok(ia[0].clone());
  }
  // Sin[ArcCos[x]] = Sqrt[1 - x^2]
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcCos"
    && ia.len() == 1
  {
    return Ok(sqrt_one_minus_sq(&ia[0]));
  }
  // Sin[ArcTan[x]] = x / Sqrt[1 + x^2]
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcTan"
    && ia.len() == 1
  {
    return Ok(divide(ia[0].clone(), sqrt_one_plus_sq(&ia[0])));
  }
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // Real args: evaluate numerically
  if let Expr::Real(f) = &args[0] {
    // An inexact argument gives an inexact result: Sin[0.] is 0., not 0.
    // (num_to_expr would collapse a whole-number value to an exact Integer).
    return Ok(Expr::Real(f.sin()));
  }
  // Sin of a BigFloat: evaluate at the input's working precision and
  // propagate the precision tag using the relative condition number.
  // For Sin[x], `d log(sin x) / d log x = x*cos(x)/sin(x) = x/tan(x)`,
  // so output precision = `prec_in - log10(|x/tan(x)|)`.
  if let Expr::BigFloat(digits, prec) = &args[0] {
    let p_in = (*prec).max(1.0);
    let result = crate::functions::math_ast::n_eval_arbitrary(
      &unevaluated("Sin", args),
      p_in,
    )?;
    if let Expr::BigFloat(ref out_digits, _) = result {
      let x_f64 = digits.parse::<f64>().unwrap_or(0.0);
      let t = x_f64.tan();
      let cond = if t.abs() > 0.0 {
        (x_f64 / t).abs()
      } else {
        0.0
      };
      let prec_out = if cond > 0.0 && cond.is_finite() {
        prec - cond.log10()
      } else {
        *prec
      };
      return Ok(Expr::BigFloat(out_digits.clone(), prec_out));
    }
    return Ok(result);
  }
  // Exact complex: sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
  // For purely imaginary (a=0): sin(bi) = i*sinh(b)
  if let Some(((re_num, re_den), (im_num, im_den))) =
    try_extract_complex_exact(&args[0])
    && im_num != 0
  {
    if re_num == 0 {
      // Pure imaginary: Sin[b*I] → I*Sinh[b]
      let im_expr = make_rational(im_num, im_den);
      let sinh_val =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      // I * sinh_val
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), sinh_val],
      );
    }
    // General complex with exact Pi-fraction real part:
    // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
    let re_expr = make_rational(re_num, re_den);
    if let Some((k, n)) = try_symbolic_pi_fraction(&re_expr)
      && let (Some(sin_a), Some(cos_a)) = (exact_sin(k, n), exact_cos(k, n))
    {
      let im_expr = make_rational(im_num, im_den);
      let cosh_b = crate::evaluator::evaluate_function_call_ast(
        "Cosh",
        &[im_expr.clone()],
      )?;
      let sinh_b =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      // sin(a)*cosh(b)
      let real_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[sin_a, cosh_b],
      )?;
      // cos(a)*sinh(b)
      let im_coeff = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[cos_a, sinh_b],
      )?;
      // I * im_coeff
      let im_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), im_coeff],
      )?;
      // real_term + im_term
      return crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[real_term, im_term],
      );
    }
    // Non-Pi-fraction real part: leave unevaluated (matches Wolfram)
    return Ok(unevaluated("Sin", args));
  }
  // General complex with non-float components: try splitting into
  // real_part + im*I where real_part can be any expression (e.g., Pi, Pi/6)
  if !contains_float(&args[0])
    && let Some((real_part, (im_num, im_den))) = try_split_real_imag(&args[0])
    && im_num != 0
  {
    if matches!(&real_part, Expr::Integer(0)) {
      // Pure imaginary (shouldn't reach here normally, but handle it)
      let im_expr = make_rational(im_num, im_den);
      let sinh_val =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), sinh_val],
      );
    }
    // Check if the real part is a Pi-fraction
    if let Some((k, n)) = try_symbolic_pi_fraction(&real_part)
      && let (Some(sin_a), Some(cos_a)) = (exact_sin(k, n), exact_cos(k, n))
    {
      let im_expr = make_rational(im_num, im_den);
      let cosh_b = crate::evaluator::evaluate_function_call_ast(
        "Cosh",
        &[im_expr.clone()],
      )?;
      let sinh_b =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      let real_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[sin_a, cosh_b],
      )?;
      let im_coeff = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[cos_a, sinh_b],
      )?;
      let im_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), im_coeff],
      )?;
      return crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[real_term, im_term],
      );
    }
    // Non-Pi-fraction real part: leave unevaluated
    return Ok(unevaluated("Sin", args));
  }
  // Complex float: sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
  // Only use float path when the argument actually contains a float value
  if contains_float(&args[0])
    && let Some((re, im)) = try_extract_complex_float(&args[0])
    && im != 0.0
  {
    let sin_re = re.sin() * im.cosh();
    let cos_re = re.cos() * im.sinh();
    return build_complex_float_result(sin_re, cos_re);
  }
  // Try symbolic Pi-fraction: Sin[k*Pi/n]
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_sin(k, n)
  {
    return canonicalize_exact_trig_value(exact);
  }
  // Pi/2 shift: Sin[rest + k*Pi/2] -> +/-Sin/Cos[rest].
  if let Some(r) = try_trig_pi_phase("Sin", &args[0]) {
    return r;
  }
  // Return unevaluated
  Ok(unevaluated("Sin", args))
}

pub fn cos_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Cos expects 1 argument".into(),
    ));
  }
  if let Some(r) = imaginary_arg_reduction("Cos", &args[0]) {
    return r;
  }
  // Cos[Interval[...]] → range over the interval.
  if let Some(r) =
    crate::functions::interval_ast::trig_interval("Cos", &args[0])
  {
    return Ok(r);
  }
  // Cos[±Infinity] → Interval[{-1, 1}]
  if let Some(r) = circular_at_infinity("Cos", &args[0]) {
    return r;
  }
  // Cos[-x] → Cos[x] (even function)
  if let Some(neg) = try_extract_negated(&args[0]) {
    return crate::evaluator::evaluate_function_call_ast("Cos", &[neg]);
  }
  // Cos[ArcCos[x]] = x (inverse function identity)
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcCos"
    && ia.len() == 1
  {
    return Ok(ia[0].clone());
  }
  // Cos[ArcSin[x]] = Sqrt[1 - x^2]
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcSin"
    && ia.len() == 1
  {
    return Ok(sqrt_one_minus_sq(&ia[0]));
  }
  // Cos[ArcTan[x]] = 1 / Sqrt[1 + x^2]
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcTan"
    && ia.len() == 1
  {
    return Ok(divide(Expr::Integer(1), sqrt_one_plus_sq(&ia[0])));
  }
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Expr::Real(f) = &args[0] {
    // An inexact argument gives an inexact result: Cos[0.] is 1., not 1.
    return Ok(Expr::Real(f.cos()));
  }
  // Cos of a BigFloat: evaluate at the input's working precision and
  // propagate the precision tag using the relative condition number.
  // For Cos[x], `d log(cos x) / d log x = -x*tan(x)`, so the output
  // relative precision is `prec_in - log10(|x*tan(x)|)`.
  if let Expr::BigFloat(digits, prec) = &args[0] {
    let p_in = (*prec).max(1.0);
    let result = crate::functions::math_ast::n_eval_arbitrary(
      &unevaluated("Cos", args),
      p_in,
    )?;
    if let Expr::BigFloat(ref out_digits, _) = result {
      let x_f64 = digits.parse::<f64>().unwrap_or(0.0);
      let cond = (x_f64 * x_f64.tan()).abs();
      let prec_out = if cond > 0.0 && cond.is_finite() {
        prec - cond.log10()
      } else {
        *prec
      };
      return Ok(Expr::BigFloat(out_digits.clone(), prec_out));
    }
    return Ok(result);
  }
  // Exact complex: cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
  // For purely imaginary (a=0): cos(bi) = cosh(b)
  if let Some(((re_num, re_den), (im_num, im_den))) =
    try_extract_complex_exact(&args[0])
    && im_num != 0
  {
    if re_num == 0 {
      // Pure imaginary: Cos[b*I] → Cosh[b]
      let im_expr = make_rational(im_num, im_den);
      return crate::evaluator::evaluate_function_call_ast("Cosh", &[im_expr]);
    }
    // General complex with exact Pi-fraction real part:
    // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
    let re_expr = make_rational(re_num, re_den);
    if let Some((k, n)) = try_symbolic_pi_fraction(&re_expr)
      && let (Some(sin_a), Some(cos_a)) = (exact_sin(k, n), exact_cos(k, n))
    {
      let im_expr = make_rational(im_num, im_den);
      let cosh_b = crate::evaluator::evaluate_function_call_ast(
        "Cosh",
        &[im_expr.clone()],
      )?;
      let sinh_b =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      // cos(a)*cosh(b)
      let real_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[cos_a, cosh_b],
      )?;
      // -sin(a)*sinh(b) (note the minus sign)
      let im_coeff = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[negate_expr(sin_a), sinh_b],
      )?;
      // I * im_coeff
      let im_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), im_coeff],
      )?;
      // real_term + im_term
      return crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[real_term, im_term],
      );
    }
    // Non-Pi-fraction real part: leave unevaluated
    return Ok(unevaluated("Cos", args));
  }
  // General complex with non-float components: try splitting into
  // real_part + im*I
  if !contains_float(&args[0])
    && let Some((real_part, (im_num, im_den))) = try_split_real_imag(&args[0])
    && im_num != 0
  {
    if matches!(&real_part, Expr::Integer(0)) {
      let im_expr = make_rational(im_num, im_den);
      return crate::evaluator::evaluate_function_call_ast("Cosh", &[im_expr]);
    }
    if let Some((k, n)) = try_symbolic_pi_fraction(&real_part)
      && let (Some(sin_a), Some(cos_a)) = (exact_sin(k, n), exact_cos(k, n))
    {
      let im_expr = make_rational(im_num, im_den);
      let cosh_b = crate::evaluator::evaluate_function_call_ast(
        "Cosh",
        &[im_expr.clone()],
      )?;
      let sinh_b =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      let real_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[cos_a, cosh_b],
      )?;
      let im_coeff = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[negate_expr(sin_a), sinh_b],
      )?;
      let im_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), im_coeff],
      )?;
      return crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[real_term, im_term],
      );
    }
    return Ok(unevaluated("Cos", args));
  }
  // Complex float: cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
  if contains_float(&args[0])
    && let Some((re, im)) = try_extract_complex_float(&args[0])
    && im != 0.0
  {
    let cos_re = re.cos() * im.cosh();
    let sin_re = -(re.sin() * im.sinh());
    return build_complex_float_result(cos_re, sin_re);
  }
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_cos(k, n)
  {
    return canonicalize_exact_trig_value(exact);
  }
  if let Some(r) = try_trig_pi_phase("Cos", &args[0]) {
    return r;
  }
  Ok(unevaluated("Cos", args))
}

/// Evaluate a circular function ratio at an arbitrary-precision argument, e.g.
/// Tan[x] = Sin[x]/Cos[x] or Sec[x] = 1/Cos[x]. `num = None` means a constant 1
/// numerator (for Sec/Csc). Both Sin/Cos have BigFloat paths and the division
/// propagates the precision tag, matching wolframscript.
fn bigfloat_trig_ratio(
  x: &Expr,
  num: Option<&str>,
  den: &str,
) -> Result<Expr, InterpreterError> {
  let numerator = match num {
    Some(head) => crate::evaluator::evaluate_function_call_ast(
      head,
      std::slice::from_ref(x),
    )?,
    None => Expr::Integer(1),
  };
  let denominator =
    crate::evaluator::evaluate_function_call_ast(den, std::slice::from_ref(x))?;
  crate::evaluator::evaluate_function_call_ast(
    "Divide",
    &[numerator, denominator],
  )
}

pub fn tan_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Tan expects 1 argument".into(),
    ));
  }
  if let Some(r) = imaginary_arg_reduction("Tan", &args[0]) {
    return r;
  }
  // Tan[Interval[...]] — range over each span, accounting for poles.
  if let Some(r) =
    crate::functions::interval_ast::tan_cot_interval("Tan", &args[0])
  {
    return Ok(r);
  }
  // Tan[±Infinity] → Interval[{-Infinity, Infinity}]
  if let Some(r) = circular_at_infinity("Tan", &args[0]) {
    return r;
  }
  // Tan[-x] → -Tan[x] (odd function)
  if let Some(neg) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Tan", &[neg])?;
    return Ok(negate_expr(inner));
  }
  // Tan[ArcTan[x]] = x (inverse function identity)
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcTan"
    && ia.len() == 1
  {
    return Ok(ia[0].clone());
  }
  // Tan[ArcSin[x]] = x / Sqrt[1 - x^2]
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcSin"
    && ia.len() == 1
  {
    return Ok(divide(ia[0].clone(), sqrt_one_minus_sq(&ia[0])));
  }
  // Tan[ArcCos[x]] = Sqrt[1 - x^2] / x
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcCos"
    && ia.len() == 1
  {
    return Ok(divide(sqrt_one_minus_sq(&ia[0]), ia[0].clone()));
  }
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Expr::Real(f) = &args[0] {
    // Tan of a Real always returns a Real, even if the numeric value
    // happens to be a whole number (matching wolframscript).
    return Ok(Expr::Real(f.tan()));
  }
  // Tan of a BigFloat: Tan[x] = Sin[x]/Cos[x]. Both have arbitrary-precision
  // paths, and the division propagates the precision tag (matching WS),
  // instead of leaving `Tan[1.`30.]` unevaluated.
  if matches!(&args[0], Expr::BigFloat(_, _)) {
    return bigfloat_trig_ratio(&args[0], Some("Sin"), "Cos");
  }
  // Exact complex: for purely imaginary: tan(bi) = i*tanh(b)
  if let Some(((re_num, _re_den), (im_num, im_den))) =
    try_extract_complex_exact(&args[0])
    && im_num != 0
  {
    if re_num == 0 {
      // Pure imaginary: Tan[b*I] → I*Tanh[b]
      let im_expr = make_rational(im_num, im_den);
      let tanh_val =
        crate::evaluator::evaluate_function_call_ast("Tanh", &[im_expr])?;
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), tanh_val],
      );
    }
    // Non-zero real part: leave unevaluated (matches Wolfram)
    return Ok(unevaluated("Tan", args));
  }
  // Complex float: tan(z) = sin(z)/cos(z)
  if contains_float(&args[0])
    && let Some((re, im)) = try_extract_complex_float(&args[0])
    && im != 0.0
  {
    // sin(a+bi)
    let sin_re = re.sin() * im.cosh();
    let sin_im = re.cos() * im.sinh();
    // cos(a+bi)
    let cos_re = re.cos() * im.cosh();
    let cos_im = -(re.sin() * im.sinh());
    // (sin_re + sin_im*i) / (cos_re + cos_im*i)
    let denom = cos_re * cos_re + cos_im * cos_im;
    let res_re = (sin_re * cos_re + sin_im * cos_im) / denom;
    let res_im = (sin_im * cos_re - sin_re * cos_im) / denom;
    return build_complex_float_result(res_re, res_im);
  }
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_tan(k, n)
  {
    return canonicalize_exact_trig_value(exact);
  }
  if let Some(r) = try_trig_pi_phase("Tan", &args[0]) {
    return r;
  }
  Ok(unevaluated("Tan", args))
}

pub fn sec_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sec expects 1 argument".into(),
    ));
  }
  if let Some(r) = imaginary_arg_reduction("Sec", &args[0]) {
    return r;
  }
  // Sec[Interval[...]] — range over each span, accounting for poles and extrema.
  if let Some(r) =
    crate::functions::interval_ast::sec_csc_interval("Sec", &args[0])
  {
    return Ok(r);
  }
  // Sec[±Infinity] → Interval[{-Infinity, -1}, {1, Infinity}]
  if let Some(r) = circular_at_infinity("Sec", &args[0]) {
    return r;
  }
  // Sec[-x] → Sec[x] (even function)
  if let Some(neg) = try_extract_negated(&args[0]) {
    return crate::evaluator::evaluate_function_call_ast("Sec", &[neg]);
  }
  // Sec of an inverse trig function → algebraic form.
  if let Some(r) = reciprocal_trig_of_inverse("Sec", &args[0]) {
    return Ok(r);
  }
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Expr::Real(f) = &args[0] {
    let c = f.cos();
    if c == 0.0 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return Ok(Expr::Real(1.0 / c));
  }
  // Sec of a BigFloat: 1/Cos[x] at arbitrary precision.
  if matches!(&args[0], Expr::BigFloat(_, _)) {
    return bigfloat_trig_ratio(&args[0], None, "Cos");
  }
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_sec(k, n)
  {
    return canonicalize_exact_trig_value(exact);
  }
  if let Some(r) = try_trig_pi_phase("Sec", &args[0]) {
    return r;
  }
  Ok(unevaluated("Sec", args))
}

pub fn csc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Csc expects 1 argument".into(),
    ));
  }
  if let Some(r) = imaginary_arg_reduction("Csc", &args[0]) {
    return r;
  }
  // Csc[Interval[...]] — range over each span, accounting for poles and extrema.
  if let Some(r) =
    crate::functions::interval_ast::sec_csc_interval("Csc", &args[0])
  {
    return Ok(r);
  }
  // Csc[±Infinity] → Interval[{-Infinity, -1}, {1, Infinity}]
  if let Some(r) = circular_at_infinity("Csc", &args[0]) {
    return r;
  }
  // Csc[-x] → -Csc[x] (odd function)
  if let Some(neg) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Csc", &[neg])?;
    return Ok(negate_expr(inner));
  }
  // Csc of an inverse trig function → algebraic form.
  if let Some(r) = reciprocal_trig_of_inverse("Csc", &args[0]) {
    return Ok(r);
  }
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Expr::Real(f) = &args[0] {
    let s = f.sin();
    if s == 0.0 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return Ok(Expr::Real(1.0 / s));
  }
  // Csc of a BigFloat: 1/Sin[x] at arbitrary precision.
  if matches!(&args[0], Expr::BigFloat(_, _)) {
    return bigfloat_trig_ratio(&args[0], None, "Sin");
  }
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_csc(k, n)
  {
    return canonicalize_exact_trig_value(exact);
  }
  if let Some(r) = try_trig_pi_phase("Csc", &args[0]) {
    return r;
  }
  Ok(unevaluated("Csc", args))
}

pub fn cot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Cot expects 1 argument".into(),
    ));
  }
  if let Some(r) = imaginary_arg_reduction("Cot", &args[0]) {
    return r;
  }
  // Cot[Interval[...]] — range over each span, accounting for poles.
  if let Some(r) =
    crate::functions::interval_ast::tan_cot_interval("Cot", &args[0])
  {
    return Ok(r);
  }
  // Cot[±Infinity] → Interval[{-Infinity, Infinity}]
  if let Some(r) = circular_at_infinity("Cot", &args[0]) {
    return r;
  }
  // Cot[-x] → -Cot[x] (odd function)
  if let Some(neg) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Cot", &[neg])?;
    return Ok(negate_expr(inner));
  }
  // Cot of an inverse trig function → algebraic form.
  if let Some(r) = reciprocal_trig_of_inverse("Cot", &args[0]) {
    return Ok(r);
  }
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Expr::Real(f) = &args[0] {
    let s = f.sin();
    if s == 0.0 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return Ok(Expr::Real(f.cos() / s));
  }
  // Cot of a BigFloat: Cos[x]/Sin[x] at arbitrary precision.
  if matches!(&args[0], Expr::BigFloat(_, _)) {
    return bigfloat_trig_ratio(&args[0], Some("Cos"), "Sin");
  }
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_cot(k, n)
  {
    return canonicalize_exact_trig_value(exact);
  }
  if let Some(r) = try_trig_pi_phase("Cot", &args[0]) {
    return r;
  }
  Ok(unevaluated("Cot", args))
}

pub fn exp_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Exp expects 1 argument".into(),
    ));
  }
  // Exp is monotonic increasing on ℝ: map it over each interval span.
  if let Some(r) =
    crate::functions::interval_ast::map_monotonic_interval("Exp", &args[0])
  {
    return Ok(r);
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  match &args[0] {
    Expr::Integer(0) => Ok(Expr::Integer(1)),
    Expr::Integer(1) => Ok(Expr::Constant("E".to_string())),
    Expr::Real(f) => {
      // Wolfram only emits General::ovfl + Overflow[] for *truly* huge
      // arguments (around |x| >= 10^15) — its big-exponent reals can
      // represent things like Exp[10^10] just fine. Inside that band
      // we let f64 do its thing (returns Infinity past x ≈ 709), so
      // downstream formulas like 1/(1+Exp[x]) — common in logistic
      // models — still collapse to 0 instead of breaking on Overflow[].
      if f.abs() >= 1.0e15 {
        crate::emit_message("General::ovfl: Overflow occurred in computation.");
        return Ok(Expr::FunctionCall {
          name: "Overflow".to_string(),
          args: vec![].into(),
        });
      }
      Ok(Expr::Real(f.exp()))
    }
    // Arbitrary-precision argument: compute e^x at the tracked precision
    // instead of leaving `E^1.`30.` unevaluated.
    Expr::BigFloat(digits, prec) => {
      if let Some(result) =
        crate::functions::math_ast::numerical::bigfloat_exp(digits, *prec)
      {
        return result;
      }
      power_two(&Expr::Constant("E".to_string()), &args[0])
    }
    _ => power_two(&Expr::Constant("E".to_string()), &args[0]),
  }
}

pub fn erf_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Erf expects 1 or 2 arguments".into(),
    ));
  }

  // Two-argument generalized form Erf[z0, z1] = Erf[z1] - Erf[z0]. Wolfram
  // keeps this symbolic as `Erf[z0, z1]`; it does NOT auto-expand to a
  // difference of one-argument Erfs. Only a few special cases reduce.
  if args.len() == 2 {
    // Erf[0, z] = Erf[z]
    if matches!(&args[0], Expr::Integer(0)) {
      return erf_ast(&args[1..]);
    }
    // Erf[z, 0] = -Erf[z]
    if matches!(&args[1], Expr::Integer(0)) {
      let erf_z0 = erf_ast(&args[..1])?;
      return crate::evaluator::evaluate_expr_to_expr(&Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(erf_z0),
      });
    }
    // Erf[z, z] = 0
    if crate::syntax::expr_to_string(&args[0])
      == crate::syntax::expr_to_string(&args[1])
    {
      return Ok(Expr::Integer(0));
    }
    // Numeric: both are machine reals. Use the exact identity
    // Erf[b] - Erf[a] = Erfc[a] - Erfc[b]; when both arguments are large and
    // same-signed, Erf is near +/-1 there, so the direct difference loses
    // precision to cancellation — the complementary form avoids it.
    if let (Expr::Real(f0), Expr::Real(f1)) = (&args[0], &args[1]) {
      let (a, b) = (*f0, *f1);
      let result = if a >= 0.0 && b >= 0.0 {
        erfc_f64(a) - erfc_f64(b)
      } else if a <= 0.0 && b <= 0.0 {
        // Erf[b] - Erf[a] = Erf[-a] - Erf[-b] = Erfc[-b] - Erfc[-a].
        erfc_f64(-b) - erfc_f64(-a)
      } else {
        erf_f64(b) - erf_f64(a)
      };
      return Ok(Expr::Real(result));
    }
    // Otherwise keep the symbolic two-argument form.
    return Ok(unevaluated("Erf", args));
  }
  // Helper: negate the Erf of the positive part
  let negate_erf = |inner: Expr| -> Expr {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::FunctionCall {
        name: "Erf".to_string(),
        args: vec![inner].into(),
      }),
    }
  };
  match &args[0] {
    // Erf[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // Erf[Infinity] = 1, Erf[-Infinity] = -1
    Expr::Identifier(s) if s == "Infinity" => Ok(Expr::Integer(1)),
    Expr::FunctionCall { name, args: dargs }
      if name == "DirectedInfinity" && dargs.len() == 1 =>
    {
      match &dargs[0] {
        Expr::Integer(1) => Ok(Expr::Integer(1)),
        Expr::Integer(-1) => Ok(Expr::Integer(-1)),
        _ => Ok(unevaluated("Erf", args)),
      }
    }
    // Erf[-Infinity] = -1 (UnaryOp form)
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity") => {
      Ok(Expr::Integer(-1))
    }
    // Erf[-x] = -Erf[x] (UnaryOp form)
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Ok(negate_erf(*operand.clone())),
    // Erf[Times[-1, x]] = -Erf[x] (evaluated form of -x)
    Expr::FunctionCall { name, args: fargs }
      if name == "Times" && fargs.len() == 2 =>
    {
      if matches!(&fargs[0], Expr::Integer(-1)) {
        return Ok(negate_erf(fargs[1].clone()));
      }
      if matches!(&fargs[1], Expr::Integer(-1)) {
        return Ok(negate_erf(fargs[0].clone()));
      }
      // Negative integer coefficient: Times[-n, x] -> -Erf[Times[n, x]]
      if let Expr::Integer(n) = &fargs[0]
        && *n < 0
      {
        let pos_arg = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-*n), fargs[1].clone()].into(),
        };
        return Ok(negate_erf(pos_arg));
      }
      Ok(unevaluated("Erf", args))
    }
    // BinaryOp::Times form: -1 * x
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Integer(-1)) {
        return Ok(negate_erf(*right.clone()));
      }
      if matches!(right.as_ref(), Expr::Integer(-1)) {
        return Ok(negate_erf(*left.clone()));
      }
      if let Expr::Integer(n) = left.as_ref()
        && *n < 0
      {
        let pos_arg = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-*n)),
          right: right.clone(),
        };
        return Ok(negate_erf(pos_arg));
      }
      Ok(unevaluated("Erf", args))
    }
    // Erf[-n] for negative integer
    Expr::Integer(n) if *n < 0 => Ok(negate_erf(Expr::Integer(-*n))),
    // Numeric evaluation for Real arguments
    Expr::Real(f) => Ok(Expr::Real(erf_f64(*f))),
    // Otherwise symbolic
    _ => Ok(unevaluated("Erf", args)),
  }
}

pub fn erfc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Erfc expects 1 argument".into(),
    ));
  }
  match &args[0] {
    // Erfc[0] = 1
    Expr::Integer(0) => Ok(Expr::Integer(1)),
    // Erfc[Infinity] = 0, Erfc[-Infinity] = 2
    Expr::Identifier(s) if s == "Infinity" => Ok(Expr::Integer(0)),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity") => {
      Ok(Expr::Integer(2))
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "DirectedInfinity" && dargs.len() == 1 =>
    {
      match &dargs[0] {
        Expr::Integer(1) => Ok(Expr::Integer(0)),
        Expr::Integer(-1) => Ok(Expr::Integer(2)),
        _ => Ok(unevaluated("Erfc", args)),
      }
    }
    // Numeric evaluation for Real arguments
    // (Wolfram keeps Erfc[-x] and Erfc[-n] unevaluated — no symbolic
    // 2 - Erfc[x] rewrite.)
    Expr::Real(f) => Ok(Expr::Real(1.0 - erf_f64(*f))),
    // Otherwise symbolic
    _ => Ok(unevaluated("Erfc", args)),
  }
}

pub fn erfi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Erfi expects 1 argument".into(),
    ));
  }
  // Helper: compute -Erfi[inner] by evaluating Erfi[inner] first, then negating
  let negate_erfi = |inner: Expr| -> Result<Expr, InterpreterError> {
    let inner_result = erfi_ast(&[inner])?;
    Ok(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(inner_result),
    })
  };
  match &args[0] {
    // Erfi[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // Erfi[Infinity] = Infinity
    Expr::Identifier(s) if s == "Infinity" => {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    // Erfi[-x] = -Erfi[x] (UnaryOp form)
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => negate_erfi(*operand.clone()),
    // Erfi[Times[-1, x]] = -Erfi[x] (evaluated form of -x)
    Expr::FunctionCall { name, args: fargs }
      if name == "Times" && fargs.len() == 2 =>
    {
      if matches!(&fargs[0], Expr::Integer(-1)) {
        return negate_erfi(fargs[1].clone());
      }
      if matches!(&fargs[1], Expr::Integer(-1)) {
        return negate_erfi(fargs[0].clone());
      }
      // Negative integer coefficient: Times[-n, x] -> -Erfi[Times[n, x]]
      if let Expr::Integer(n) = &fargs[0]
        && *n < 0
      {
        let pos_arg = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-*n), fargs[1].clone()].into(),
        };
        return negate_erfi(pos_arg);
      }
      Ok(unevaluated("Erfi", args))
    }
    // BinaryOp::Times form: -1 * x
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Integer(-1)) {
        return negate_erfi(*right.clone());
      }
      if matches!(right.as_ref(), Expr::Integer(-1)) {
        return negate_erfi(*left.clone());
      }
      if let Expr::Integer(n) = left.as_ref()
        && *n < 0
      {
        let pos_arg = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-*n)),
          right: right.clone(),
        };
        return negate_erfi(pos_arg);
      }
      Ok(unevaluated("Erfi", args))
    }
    // Erfi[-n] for negative integer
    Expr::Integer(n) if *n < 0 => negate_erfi(Expr::Integer(-*n)),
    // Numeric evaluation for Real arguments
    Expr::Real(f) => Ok(Expr::Real(erfi_f64(*f))),
    // Otherwise symbolic
    _ => Ok(unevaluated("Erfi", args)),
  }
}

/// DawsonF[x] = exp(-x^2) * integral_0^x exp(t^2) dt
/// = (sqrt(pi)/2) exp(-x^2) erfi(x). Odd function; DawsonF[0] = 0,
/// DawsonF[Infinity] = 0.
pub fn dawson_f_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DawsonF expects 1 argument".into(),
    ));
  }
  // Helper: compute -DawsonF[inner] by evaluating then negating.
  let negate = |inner: Expr| -> Result<Expr, InterpreterError> {
    let inner_result = dawson_f_ast(&[inner])?;
    Ok(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(inner_result),
    })
  };
  match &args[0] {
    // DawsonF[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // DawsonF[Infinity] = 0
    Expr::Identifier(s) if s == "Infinity" => Ok(Expr::Integer(0)),
    // DawsonF[-x] = -DawsonF[x] (UnaryOp form)
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => negate(*operand.clone()),
    // DawsonF[Times[-1, x]] = -DawsonF[x]
    Expr::FunctionCall { name, args: fargs }
      if name == "Times" && fargs.len() == 2 =>
    {
      if matches!(&fargs[0], Expr::Integer(-1)) {
        return negate(fargs[1].clone());
      }
      if matches!(&fargs[1], Expr::Integer(-1)) {
        return negate(fargs[0].clone());
      }
      if let Expr::Integer(n) = &fargs[0]
        && *n < 0
      {
        let pos_arg = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-*n), fargs[1].clone()].into(),
        };
        return negate(pos_arg);
      }
      Ok(unevaluated("DawsonF", args))
    }
    // BinaryOp::Times form: -1 * x
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Integer(-1)) {
        return negate(*right.clone());
      }
      if matches!(right.as_ref(), Expr::Integer(-1)) {
        return negate(*left.clone());
      }
      if let Expr::Integer(n) = left.as_ref()
        && *n < 0
      {
        let pos_arg = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-*n)),
          right: right.clone(),
        };
        return negate(pos_arg);
      }
      Ok(unevaluated("DawsonF", args))
    }
    // DawsonF[-n] for negative integer
    Expr::Integer(n) if *n < 0 => negate(Expr::Integer(-*n)),
    // Numeric evaluation for Real arguments
    Expr::Real(f) => Ok(Expr::Real(dawson_f64(*f))),
    // Otherwise symbolic
    _ => Ok(unevaluated("DawsonF", args)),
  }
}

pub fn inverse_erf_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InverseErf expects 1 argument".into(),
    ));
  }
  match &args[0] {
    // InverseErf[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // InverseErf[1] = Infinity
    Expr::Integer(1) => Ok(Expr::Identifier("Infinity".to_string())),
    // InverseErf[-1] = -Infinity
    Expr::Integer(-1) => Ok(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    }),
    // Numeric evaluation for Real arguments
    Expr::Real(f) => {
      if *f > -1.0 && *f < 1.0 {
        Ok(Expr::Real(
          crate::functions::math_ast::numeric_utils::inverse_erf_f64(*f),
        ))
      } else if *f == 1.0 {
        Ok(Expr::Identifier("Infinity".to_string()))
      } else if *f == -1.0 {
        Ok(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(Expr::Identifier("Infinity".to_string())),
        })
      } else {
        Ok(unevaluated("InverseErf", args))
      }
    }
    // Otherwise symbolic — return unevaluated
    _ => Ok(unevaluated("InverseErf", args)),
  }
}

/// InverseErfc[x] — inverse complementary error function
/// InverseErfc[x] = InverseErf[1 - x]
pub fn inverse_erfc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InverseErfc expects 1 argument".into(),
    ));
  }
  match &args[0] {
    // InverseErfc[0] = Infinity
    Expr::Integer(0) => Ok(Expr::Identifier("Infinity".to_string())),
    // InverseErfc[1] = 0
    Expr::Integer(1) => Ok(Expr::Integer(0)),
    // InverseErfc[2] = -Infinity
    Expr::Integer(2) => Ok(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    }),
    // Reflection for an exact rational z with 1 < z < 2:
    // InverseErfc[z] = -InverseErfc[2 - z]  (keeps the argument in (0, 1)).
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational"
        && rargs.len() == 2
        && matches!((&rargs[0], &rargs[1]),
          (Expr::Integer(p), Expr::Integer(q))
            if *q > 0 && *p > *q && *p < 2 * *q) =>
    {
      let (Expr::Integer(p), Expr::Integer(q)) = (&rargs[0], &rargs[1]) else {
        unreachable!()
      };
      let inner = inverse_erfc_ast(&[make_rational(2 * q - p, *q)])?;
      Ok(Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(inner),
      })
    }
    // Numeric evaluation for Real arguments
    Expr::Real(f) => {
      if *f > 0.0 && *f < 2.0 {
        // InverseErfc[x] = InverseErf[1 - x]
        Ok(Expr::Real(
          crate::functions::math_ast::numeric_utils::inverse_erf_f64(1.0 - *f),
        ))
      } else if *f == 0.0 {
        Ok(Expr::Identifier("Infinity".to_string()))
      } else if *f == 2.0 {
        Ok(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(Expr::Identifier("Infinity".to_string())),
        })
      } else {
        Ok(unevaluated("InverseErfc", args))
      }
    }
    _ => Ok(unevaluated("InverseErfc", args)),
  }
}

/// Largest `e >= 0` with `g^e == n`, or `None` when `n` is not an exact power
/// of `g`. Requires `g >= 2` and `n >= 1`.
fn exact_integer_log(n: i128, g: i128) -> Option<i128> {
  if n < 1 || g < 2 {
    return None;
  }
  let mut val = n;
  let mut e = 0i128;
  while val % g == 0 {
    val /= g;
    e += 1;
  }
  if val == 1 { Some(e) } else { None }
}

/// Primitive root `(g, s)` of `base`: the unique `g >= 2` that is not itself a
/// perfect power, with `base == g^s` and `s` maximal. E.g. 64 -> (2, 6),
/// 4 -> (2, 2), 10 -> (10, 1).
fn primitive_root(base: i128) -> (i128, i128) {
  // base <= g^s with g >= 2 implies s <= log2(base); try the largest s first.
  let max_s = ((base as f64).log2().floor() as i128).max(1);
  let mut s = max_s;
  while s >= 1 {
    let g = (base as f64).powf(1.0 / s as f64).round() as i128;
    if g >= 2 && g.checked_pow(s as u32) == Some(base) {
      return (g, s);
    }
    s -= 1;
  }
  (base, 1)
}

/// `Log[base, p/q]` as an exact rational when both `p` and `q` are integer
/// powers of `base`'s primitive root; otherwise `None`. Covers `Log[2, 8] = 3`,
/// `Log[2, 1/8] = -3`, and `Log[4, 1/2] = -1/2`.
fn log_base_exact(base: i128, p: i128, q: i128) -> Option<Expr> {
  let (g, s) = primitive_root(base);
  let a = exact_integer_log(p, g)?;
  let c = exact_integer_log(q, g)?;
  // base^k = g^(a - c) and base = g^s, so k = (a - c) / s.
  Some(make_rational(a - c, s))
}

pub fn log_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if !args.is_empty()
    && matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate")
  {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // Log is monotonic increasing on (0, ∞): map it over each interval span.
  if args.len() == 1
    && let Some(r) =
      crate::functions::interval_ast::map_monotonic_interval("Log", &args[0])
  {
    return Ok(r);
  }
  // Log[Overflow[]] = Overflow[] (matches wolframscript)
  if !args.is_empty()
    && matches!(&args[0], Expr::FunctionCall { name, args } if name == "Overflow" && args.is_empty())
  {
    return Ok(Expr::FunctionCall {
      name: "Overflow".to_string(),
      args: vec![].into(),
    });
  }
  match args.len() {
    1 => {
      // Log[0] = -Infinity
      if matches!(&args[0], Expr::Integer(0)) {
        return Ok(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(Expr::Identifier("Infinity".to_string())),
        });
      }
      // Log[1] = 0
      if matches!(&args[0], Expr::Integer(1)) {
        return Ok(Expr::Integer(0));
      }
      // Log[E] = 1
      if matches!(&args[0], Expr::Constant(c) if c == "E") {
        return Ok(Expr::Integer(1));
      }
      // Log[E^x] = x only when x is a known numeric constant (not symbolic)
      // Wolfram does not simplify Log[E^x] or Log[E^n] for symbolic arguments
      {
        let is_numeric_exponent = |e: &Expr| -> bool {
          matches!(
            e,
            Expr::Integer(_)
              | Expr::Real(_)
              | Expr::BigInteger(_)
              | Expr::Constant(_)
          ) || matches!(
            e,
            Expr::FunctionCall { name, .. }
            if name == "Rational"
          )
        };
        if let Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } = &args[0]
          && matches!(left.as_ref(), Expr::Constant(c) if c == "E")
          && is_numeric_exponent(right)
        {
          return Ok(*right.clone());
        }
        if let Expr::FunctionCall {
          name,
          args: pow_args,
        } = &args[0]
          && name == "Power"
          && pow_args.len() == 2
          && matches!(&pow_args[0], Expr::Constant(c) if c == "E")
          && is_numeric_exponent(&pow_args[1])
        {
          return Ok(pow_args[1].clone());
        }
      }
      // Log[E^z] for a numeric complex exponent z: the result is z with its
      // imaginary part reduced into (-Pi, Pi]. Both Re[z] and Im[z] must be
      // concrete reals — a symbolic real part means E^Re need not be a positive
      // real, so the principal-branch identity can't be applied (wolframscript
      // leaves Log[E^(a + 3 I)] unevaluated, but reduces Log[E^(2 + 4 I)] to
      // (2 + 4 I) - 2 I Pi and Log[E^(3 I)] to 3 I).
      {
        let e_exp: Option<&Expr> = match &args[0] {
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } if matches!(left.as_ref(), Expr::Constant(c) if c == "E") => {
            Some(right.as_ref())
          }
          Expr::FunctionCall { name, args: pa }
            if name == "Power"
              && pa.len() == 2
              && matches!(&pa[0], Expr::Constant(c) if c == "E") =>
          {
            Some(&pa[1])
          }
          _ => None,
        };
        if let Some(z) = e_exp {
          let imz =
            crate::evaluator::evaluate_function_call_ast("Im", &[z.clone()])?;
          let rez =
            crate::evaluator::evaluate_function_call_ast("Re", &[z.clone()])?;
          if let (Some(im_f), Some(_re_f)) = (
            crate::functions::math_ast::try_eval_to_f64(&imz),
            crate::functions::math_ast::try_eval_to_f64(&rez),
          ) {
            use std::f64::consts::PI;
            let two_pi = 2.0 * PI;
            // k brings im into (-Pi, Pi]: im - 2*k*Pi ∈ (-Pi, Pi].
            let k = ((im_f - PI) / two_pi - 1e-9).ceil() as i128;
            if k == 0 {
              return Ok(z.clone());
            }
            let correction = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::Integer(-2 * k),
                Expr::Constant("Pi".to_string()),
                Expr::Identifier("I".to_string()),
              ]
              .into(),
            };
            let result = Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![z.clone(), correction].into(),
            };
            return crate::evaluator::evaluate_expr_to_expr(&result);
          }
        }
      }
      // Log[base^exp] = exp*Log[base] when base is a positive real (integer >= 2
      // or a positive real constant) and exp is a non-integer rational.
      // Matches wolframscript: Log[Sqrt[2]] -> Log[2]/2, Log[3^(2/5)] ->
      // (2 Log[3])/5, Log[5^(-1/2)] -> -1/2 Log[5], Log[Pi^(1/2)] -> Log[Pi]/2.
      // Integer bases with |exp| > 1 are pre-reduced to a product (e.g.
      // 2^(3/2) -> 2 Sqrt[2]), so they never reach Log as a pure Power.
      {
        let power_parts: Option<(&Expr, &Expr)> = match &args[0] {
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } => Some((left.as_ref(), right.as_ref())),
          Expr::FunctionCall { name, args: pa }
            if name == "Power" && pa.len() == 2 =>
          {
            Some((&pa[0], &pa[1]))
          }
          _ => None,
        };
        if let Some((base, exp)) = power_parts {
          let base_is_positive_real = match base {
            Expr::Integer(n) => *n >= 2,
            Expr::Constant(c) | Expr::Identifier(c) => matches!(
              c.as_str(),
              "Pi" | "E" | "EulerGamma" | "GoldenRatio" | "Catalan" | "Degree"
            ),
            _ => false,
          };
          let exp_is_noninteger_rational = matches!(
            exp,
            Expr::FunctionCall { name, args: ra }
              if name == "Rational" && ra.len() == 2
          );
          if base_is_positive_real && exp_is_noninteger_rational {
            let log_base = Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![base.clone()].into(),
            };
            return crate::evaluator::evaluate_function_call_ast(
              "Times",
              &[exp.clone(), log_base],
            );
          }
        }
      }
      // Log[I] = I*Pi/2
      if matches!(&args[0], Expr::Identifier(s) if s == "I") {
        return crate::evaluator::evaluate_function_call_ast(
          "Times",
          &[
            Expr::Identifier("I".to_string()),
            crate::functions::math_ast::make_rational(1, 2),
            Expr::Constant("Pi".to_string()),
          ],
        );
      }
      // Log[-I] = -I*Pi/2
      {
        let is_neg_i = match &args[0] {
          Expr::FunctionCall { name, args: targs }
            if name == "Times"
              && targs.len() == 2
              && matches!(&targs[0], Expr::Integer(-1))
              && matches!(&targs[1], Expr::Identifier(s) if s == "I") =>
          {
            true
          }
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } if matches!(operand.as_ref(), Expr::Identifier(s) if s == "I") => {
            true
          }
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left,
            right,
          } if matches!(left.as_ref(), Expr::Integer(-1))
            && matches!(right.as_ref(), Expr::Identifier(s) if s == "I") =>
          {
            true
          }
          _ => false,
        };
        if is_neg_i {
          return crate::evaluator::evaluate_function_call_ast(
            "Times",
            &[
              Expr::Integer(-1),
              Expr::Identifier("I".to_string()),
              crate::functions::math_ast::make_rational(1, 2),
              Expr::Constant("Pi".to_string()),
            ],
          );
        }
      }
      // Log[c*I] for an exact (integer/rational) real coefficient c:
      //   = Log[Abs[c]] + Sign[c]*I*Pi/2.
      // Real coefficients (e.g. 2.5 I) are handled by the numeric complex
      // path below; symbolic-real ones (Pi I, Sqrt[2] I) stay unevaluated,
      // matching wolframscript.
      if let Some(coeff) =
        crate::functions::math_ast::complex::extract_i_times_real(&args[0])
      {
        let is_exact_real = matches!(&coeff, Expr::Integer(_))
          || matches!(&coeff, Expr::FunctionCall { name, .. } if name == "Rational");
        if is_exact_real {
          // Log[Abs[c]] + Sign[c]*I*Pi/2, evaluated as a whole so Abs/Sign/Log
          // and the product all reduce (e.g. Log[1/2] → -Log[2], 1*… → …).
          let result = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Log".to_string(),
                args: vec![Expr::FunctionCall {
                  name: "Abs".to_string(),
                  args: vec![coeff.clone()].into(),
                }]
                .into(),
              },
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  Expr::FunctionCall {
                    name: "Sign".to_string(),
                    args: vec![coeff.clone()].into(),
                  },
                  crate::functions::math_ast::make_rational(1, 2),
                  Expr::Identifier("I".to_string()),
                  Expr::Constant("Pi".to_string()),
                ]
                .into(),
              },
            ]
            .into(),
          };
          return crate::evaluator::evaluate_expr_to_expr(&result);
        }
      }
      // Log[-r] for a negative rational r: = I*Pi + Log[-r].
      if let Expr::FunctionCall { name, args: ra } = &args[0]
        && name == "Rational"
        && ra.len() == 2
        && let (Expr::Integer(p), Expr::Integer(q)) = (&ra[0], &ra[1])
        && (*p < 0) ^ (*q < 0)
      {
        let result = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Identifier("I".to_string())),
              right: Box::new(Expr::Constant("Pi".to_string())),
            },
            Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![make_rational(p.abs(), q.abs())].into(),
            },
          ]
          .into(),
        };
        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
      // Log[-n] for negative integers: Log[-1] = I*Pi, Log[-n] = I*Pi + Log[n]
      if let Expr::Integer(n) = &args[0]
        && *n < 0
      {
        let abs_n = -*n;
        // I*Pi
        let i_pi = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Identifier("I".to_string())),
          right: Box::new(Expr::Constant("Pi".to_string())),
        };
        if abs_n == 1 {
          return Ok(i_pi);
        }
        // I*Pi + Log[abs_n]
        let log_n = Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![Expr::Integer(abs_n)].into(),
        };
        return crate::evaluator::evaluate_function_call_ast(
          "Plus",
          &[i_pi, log_n],
        );
      }
      // Log[-expr] for negated expressions: check for Times[-1, ...]
      {
        let inner = match &args[0] {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left,
            right,
          } if matches!(left.as_ref(), Expr::Integer(-1)) => {
            Some(*right.clone())
          }
          Expr::FunctionCall { name, args: targs }
            if name == "Times"
              && targs.len() == 2
              && matches!(&targs[0], Expr::Integer(-1)) =>
          {
            Some(targs[1].clone())
          }
          _ => None,
        };
        if let Some(inner_expr) = inner {
          let i_pi = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Identifier("I".to_string())),
            right: Box::new(Expr::Constant("Pi".to_string())),
          };
          let log_x =
            crate::evaluator::evaluate_function_call_ast("Log", &[inner_expr])?;
          return crate::evaluator::evaluate_function_call_ast(
            "Plus",
            &[i_pi, log_x],
          );
        }
      }
      // Log[Infinity] = Infinity
      if matches!(&args[0], Expr::Identifier(s) if s == "Infinity") {
        return Ok(Expr::Identifier("Infinity".to_string()));
      }
      // Log[ComplexInfinity] = Infinity
      if matches!(&args[0], Expr::Identifier(s) if s == "ComplexInfinity") {
        return Ok(Expr::Identifier("Infinity".to_string()));
      }
      // Log[-Infinity] = Infinity (principal value)
      if crate::functions::math_ast::is_neg_infinity(&args[0]) {
        return Ok(Expr::Identifier("Infinity".to_string()));
      }
      // Log[p/q] where 0 < p < q: return -Log[q/p]
      if let Expr::FunctionCall {
        name,
        args: rat_args,
      } = &args[0]
        && name == "Rational"
        && rat_args.len() == 2
        && matches!((&rat_args[0], &rat_args[1]), (Expr::Integer(p), Expr::Integer(q)) if *p > 0 && *q > 0 && *p < *q)
        && let (Expr::Integer(p), Expr::Integer(q)) =
          (&rat_args[0], &rat_args[1])
      {
        let inverted = make_rational(*q, *p);
        let log_inv = Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![inverted].into(),
        };
        return Ok(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(log_inv),
        });
      }
      // Arbitrary-precision argument: compute ln(x) at the tracked precision
      // instead of leaving `Log[2.`30.]` unevaluated.
      if let Expr::BigFloat(digits, prec) = &args[0]
        && let Some(result) =
          crate::functions::math_ast::numerical::bigfloat_log(digits, *prec)
      {
        return result;
      }
      if let Expr::Real(f) = &args[0] {
        if *f > 0.0 {
          return Ok(Expr::Real(f.ln()));
        } else if *f == 0.0 {
          return Ok(Expr::Identifier("Indeterminate".to_string()));
        } else {
          // Log of negative real: return complex result
          let re = f.abs().ln();
          let im = std::f64::consts::PI;
          return crate::evaluator::evaluate_function_call_ast(
            "Complex",
            &[Expr::Real(re), Expr::Real(im)],
          );
        }
      }
      // Log of a complex floating-point number: principal value
      // = 0.5*Log[a^2 + b^2] + I*atan2(b, a)
      if let Some((a, b)) =
        crate::functions::math_ast::try_extract_complex_float(&args[0])
        && b != 0.0
        && contains_inexact_real_log(&args[0])
      {
        let re = 0.5 * (a * a + b * b).ln();
        let im = b.atan2(a);
        return Ok(crate::functions::math_ast::build_complex_float_expr(
          re, im,
        ));
      }
      Ok(unevaluated("Log", args))
    }
    2 => {
      // Log[base, x] — integer base and a positive rational argument x = p/q.
      // Collapses to a rational exponent when both are exact powers of a common
      // primitive root (Log[2, 8] = 3, Log[2, 1/8] = -3, Log[4, 1/2] = -1/2).
      // Non-power arguments fall through to the Log[x]/Log[base] canonical form.
      if let (Some(base), Some((p, q))) =
        (expr_to_i128(&args[0]), expr_to_rational(&args[1]))
        && base > 1
        && p > 0
        && q > 0
        && let Some(result) = log_base_exact(base, p, q)
      {
        return Ok(result);
      }
      // Log[base, x] — numeric result whenever at least one operand is an
      // inexact (machine Real) plain number. Wolfram's two-argument Log is a
      // dedicated primitive that evaluates as the *direct* division
      // Log[x]/Log[base]; unlike a user-level Divide it does NOT round through
      // multiply-by-reciprocal, so Log[10, 100.0] == 2. exactly (whereas
      // Log[100.0]/Log[10] == 1.9999999999999998). Handle any positive plain
      // numeric base/argument here so the value never falls through to the
      // generic reciprocal-multiply division below.
      fn is_plain_number(e: &Expr) -> bool {
        matches!(e, Expr::Integer(_) | Expr::Real(_))
          || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
      }
      if (matches!(&args[0], Expr::Real(_))
        || matches!(&args[1], Expr::Real(_)))
        && is_plain_number(&args[0])
        && is_plain_number(&args[1])
        && let (Some(base), Some(x)) =
          (try_eval_to_f64(&args[0]), try_eval_to_f64(&args[1]))
        && base > 0.0
        && base != 1.0
        && x > 0.0
      {
        return Ok(Expr::Real(x.ln() / base.ln()));
      }
      // Log[base, x] with an inexact negative real argument gives the numeric
      // complex value Log[x]/Log[base] = (Log|x| + Pi I)/Log[base]. The base
      // may be exact (e.g. Log[10, -5.]) as long as it is a positive number;
      // without this the result kept an unevaluated Log[base] denominator.
      if let Expr::Real(x) = &args[1]
        && *x < 0.0
        && let Some(base_f) = try_eval_to_f64(&args[0])
        && base_f > 0.0
        && base_f != 1.0
      {
        let base_ln = base_f.ln();
        return build_complex_float_result(
          x.abs().ln() / base_ln,
          std::f64::consts::PI / base_ln,
        );
      }
      // Canonicalize Log[base, x] → Log[x]/Log[base] (evaluated so
      // sub-expressions like Log[0] collapse to -Infinity).
      let result = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![args[1].clone()].into(),
        }),
        right: Box::new(Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![args[0].clone()].into(),
        }),
      };
      crate::evaluator::evaluate_expr_to_expr(&result)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Log expects 1 or 2 arguments".into(),
    )),
  }
}

/// Log10[x] - Base-10 logarithm
pub fn log10_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Log10 expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) if *n > 0 => {
      // Check if n is an exact power of 10
      let mut val = *n;
      let mut exp = 0i128;
      while val > 1 && val % 10 == 0 {
        val /= 10;
        exp += 1;
      }
      if val == 1 {
        return Ok(Expr::Integer(exp));
      }
    }
    Expr::Real(f) if *f > 0.0 => {
      return Ok(Expr::Real(f.log10()));
    }
    // Negative real: Log10[x] = (Log|x| + Pi I)/Log[10] (numeric complex).
    Expr::Real(f) if *f < 0.0 => {
      let base_ln = 10f64.ln();
      return build_complex_float_result(
        f.abs().ln() / base_ln,
        std::f64::consts::PI / base_ln,
      );
    }
    _ => {}
  }
  // Symbolic fallback: Log10[x] = Log[x] / Log[10]
  let expr = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(unevaluated("Log", args)),
    right: Box::new(Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![Expr::Integer(10)].into(),
    }),
  };
  crate::evaluator::evaluate_expr_to_expr(&expr)
}

/// Log2[x] - Base-2 logarithm
pub fn log2_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Log2 expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) if *n > 0 => {
      // Check if n is an exact power of 2
      let val = *n as u128;
      if val.is_power_of_two() {
        return Ok(Expr::Integer(val.trailing_zeros() as i128));
      }
    }
    Expr::Real(f) if *f > 0.0 => {
      return Ok(Expr::Real(f.log2()));
    }
    // Negative real: Log2[x] = (Log|x| + Pi I)/Log[2] (numeric complex).
    Expr::Real(f) if *f < 0.0 => {
      let base_ln = 2f64.ln();
      return build_complex_float_result(
        f.abs().ln() / base_ln,
        std::f64::consts::PI / base_ln,
      );
    }
    _ => {}
  }
  // Symbolic fallback: Log2[x] = Log[x] / Log[2]
  let expr = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(unevaluated("Log", args)),
    right: Box::new(Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![Expr::Integer(2)].into(),
    }),
  };
  crate::evaluator::evaluate_expr_to_expr(&expr)
}

/// ArcSin[x] - Inverse sine (symbolic)
/// Numeric evaluation of an inverse trig / hyperbolic function at a complex
/// float argument, via its principal-branch closed form. Returns `None` for
/// non-complex or exact arguments (which stay symbolic, matching wolframscript).
fn try_complex_inverse_trig(
  name: &str,
  arg: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  if !contains_inexact_real(arg) {
    return None;
  }
  let (x, y) = crate::functions::math_ast::try_extract_complex_float(arg)?;
  if y == 0.0 {
    return None;
  }
  let cmul = |a: (f64, f64), b: (f64, f64)| {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
  };
  // Principal complex square root and logarithm.
  let csqrt = |a: f64, b: f64| {
    let r = a.hypot(b);
    let re = ((r + a) / 2.0).sqrt();
    let im = ((r - a) / 2.0).sqrt() * if b < 0.0 { -1.0 } else { 1.0 };
    (re, im)
  };
  let cln = |a: f64, b: f64| (a.hypot(b).ln(), b.atan2(a));
  let (rr, ri) = match name {
    // ArcSin[z] = -I Log[I z + Sqrt[1 - z^2]]
    "ArcSin" => {
      let z2 = cmul((x, y), (x, y));
      let s = csqrt(1.0 - z2.0, -z2.1);
      let l = cln(-y + s.0, x + s.1);
      (l.1, -l.0)
    }
    // ArcCos[z] = Pi/2 - ArcSin[z]
    "ArcCos" => {
      let z2 = cmul((x, y), (x, y));
      let s = csqrt(1.0 - z2.0, -z2.1);
      let l = cln(-y + s.0, x + s.1);
      (std::f64::consts::FRAC_PI_2 - l.1, l.0)
    }
    // ArcTan[z] = (I/2) (Log[1 - I z] - Log[1 + I z])
    "ArcTan" => {
      let l1 = cln(1.0 + y, -x);
      let l2 = cln(1.0 - y, x);
      (-(l1.1 - l2.1) / 2.0, (l1.0 - l2.0) / 2.0)
    }
    // ArcSinh[z] = Log[z + Sqrt[z^2 + 1]]
    "ArcSinh" => {
      let z2 = cmul((x, y), (x, y));
      let s = csqrt(z2.0 + 1.0, z2.1);
      cln(x + s.0, y + s.1)
    }
    _ => return None,
  };
  // A non-finite component means the argument hit a pole (e.g. ArcTan[±I]);
  // wolframscript returns Indeterminate there.
  if !rr.is_finite() || !ri.is_finite() {
    return Some(Ok(Expr::Identifier("Indeterminate".to_string())));
  }
  Some(build_complex_float_result(rr, ri))
}

pub fn arcsin_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcSin expects exactly 1 argument".into(),
    ));
  }
  if let Some(r) = try_complex_inverse_trig("ArcSin", &args[0]) {
    return r;
  }
  // ArcSin is monotonic increasing on [-1, 1]: map over interval spans.
  if let Some(r) =
    crate::functions::interval_ast::map_monotonic_interval("ArcSin", &args[0])
  {
    return Ok(r);
  }
  // ArcSin[-x] → -ArcSin[x] (odd function)
  // Only apply for symbolic (non-numeric) arguments; for numeric arguments
  // the existing Exact/Real paths below produce the canonical form.
  if !matches!(
    &args[0],
    Expr::Integer(_)
      | Expr::Real(_)
      | Expr::BigInteger(_)
      | Expr::BigFloat(_, _)
  ) && !matches!(
    &args[0],
    Expr::FunctionCall { name, .. } if name == "Rational"
  ) && let Some(neg) = try_extract_negated(&args[0])
  {
    let inner = crate::evaluator::evaluate_function_call_ast("ArcSin", &[neg])?;
    // Build -inner and let the evaluator simplify
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[Expr::Integer(-1), inner],
    );
  }
  // Exact values: ArcSin[0] = 0, ArcSin[1] = Pi/2, ArcSin[-1] = -Pi/2
  // ArcSin[1/2] = Pi/6, ArcSin[-1/2] = -Pi/6
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Integer(1) => {
      return Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      });
    }
    Expr::Integer(-1) => {
      // -1/2*Pi = Times[Rational[-1, 2], Pi]
      return Ok(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-1), Expr::Integer(2)].into(),
        }),
        right: Box::new(Expr::Constant("Pi".to_string())),
      });
    }
    Expr::Real(f) if (-1.0..=1.0).contains(f) => {
      return Ok(Expr::Real(f.asin()));
    }
    // Outside [-1, 1] the (inexact) real argument gives a complex result:
    // ArcSin[x] = sign(x)*Pi/2 - sign(x)*ArcCosh[|x|].
    Expr::Real(f) if f.abs() > 1.0 => {
      use std::f64::consts::PI;
      let x = *f;
      let s = x.signum();
      let arccosh = (x.abs() + (x * x - 1.0).sqrt()).ln();
      return build_complex_float_result(s * PI / 2.0, -s * arccosh);
    }
    _ => {}
  }
  // Check for special rational/irrational values via numeric comparison
  if let Some(v) = try_eval_to_f64(&args[0])
    && let Some(result) = arcsin_special_value(v)
  {
    return Ok(result);
  }
  Ok(unevaluated("ArcSin", args))
}

/// ArcCos[x] - Inverse cosine (symbolic)
/// Detect an expression of the form `±I·Infinity` (in either factor order,
/// with an optional leading `-` or a `Times[-1, …]` wrapper). Returns
/// `Some(+1)` for `I·Infinity`, `Some(-1)` for `-I·Infinity`, else `None`.
fn imaginary_infinity_sign(expr: &Expr) -> Option<i8> {
  let is_infinity =
    |e: &Expr| matches!(e, Expr::Identifier(s) if s == "Infinity");
  let is_i = |e: &Expr| matches!(e, Expr::Identifier(s) if s == "I");
  let match_pair = |a: &Expr, b: &Expr| -> bool {
    (is_i(a) && is_infinity(b)) || (is_infinity(a) && is_i(b))
  };
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if match_pair(left, right) => Some(1),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => imaginary_infinity_sign(operand).map(|s| -s),
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut sign: i8 = 1;
      let mut has_i = false;
      let mut has_inf = false;
      for a in args {
        if is_i(a) {
          has_i = true;
        } else if is_infinity(a) {
          has_inf = true;
        } else if matches!(a, Expr::Integer(-1)) {
          sign = -sign;
        } else if let Expr::FunctionCall {
          name: dn,
          args: dargs,
        } = a
          && dn == "DirectedInfinity"
          && dargs.len() == 1
        {
          // `Times[-1, DirectedInfinity[I]]` ≡ -I·Infinity. Treat
          // `DirectedInfinity[±I]` as the combined I·Infinity factor.
          match &dargs[0] {
            Expr::Identifier(s) if s == "I" => {
              has_i = true;
              has_inf = true;
            }
            Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand,
            } if matches!(operand.as_ref(), Expr::Identifier(s) if s == "I") => {
              has_i = true;
              has_inf = true;
              sign = -sign;
            }
            _ => return None,
          }
        } else {
          return None;
        }
      }
      if has_i && has_inf { Some(sign) } else { None }
    }
    // After Times canonicalisation, `I·Infinity` collapses to
    // `DirectedInfinity[I]` (and `-I·Infinity` to `DirectedInfinity[-I]`).
    // Recognise both forms here so trig handlers downstream still match.
    Expr::FunctionCall { name, args }
      if name == "DirectedInfinity" && args.len() == 1 =>
    {
      match &args[0] {
        Expr::Identifier(s) if s == "I" => Some(1),
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } if matches!(operand.as_ref(), Expr::Identifier(s) if s == "I") => {
          Some(-1)
        }
        _ => None,
      }
    }
    _ => None,
  }
}

pub fn arccos_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCos expects exactly 1 argument".into(),
    ));
  }
  if let Some(r) = try_complex_inverse_trig("ArcCos", &args[0]) {
    return r;
  }
  // ArcCos is monotonic decreasing on [-1, 1]: map over interval spans
  // (normalization re-sorts the swapped endpoints).
  if let Some(r) =
    crate::functions::interval_ast::map_monotonic_interval("ArcCos", &args[0])
  {
    return Ok(r);
  }
  // Exact values: ArcCos[0] = Pi/2, ArcCos[1] = 0, ArcCos[-1] = Pi
  match &args[0] {
    Expr::Integer(1) => return Ok(Expr::Integer(0)),
    Expr::Integer(0) => {
      return Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      });
    }
    Expr::Integer(-1) => return Ok(Expr::Constant("Pi".to_string())),
    Expr::Real(f) if (-1.0..=1.0).contains(f) => {
      return Ok(Expr::Real(f.acos()));
    }
    // Outside [-1, 1] the (inexact) real argument gives a complex result:
    // ArcCos[x] = Pi/2 - ArcSin[x] = (Pi/2 - sign(x)*Pi/2) + sign(x)*ArcCosh[|x|] I.
    Expr::Real(f) if f.abs() > 1.0 => {
      use std::f64::consts::PI;
      let x = *f;
      let s = x.signum();
      let arccosh = (x.abs() + (x * x - 1.0).sqrt()).ln();
      return build_complex_float_result(PI / 2.0 - s * PI / 2.0, s * arccosh);
    }
    _ => {}
  }
  // ArcCos[±I * Infinity] → DirectedInfinity[∓I]
  // (As z → ±I·∞, ArcSin[z] → ±I·∞, and ArcCos = π/2 − ArcSin diverges
  // with the opposite imaginary sign.)
  if let Some(sign) = imaginary_infinity_sign(&args[0]) {
    let direction = if sign > 0 {
      // Negate I: Times[-1, I]
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(Expr::Identifier("I".to_string())),
      }
    } else {
      Expr::Identifier("I".to_string())
    };
    return Ok(Expr::FunctionCall {
      name: "DirectedInfinity".to_string(),
      args: vec![direction].into(),
    });
  }
  // Check for special rational/irrational values via numeric comparison
  if let Some(v) = try_eval_to_f64(&args[0])
    && let Some(result) = arccos_special_value(v)
  {
    return Ok(result);
  }
  Ok(unevaluated("ArcCos", args))
}

/// Check if a float value matches a known ArcCos special angle
fn arccos_special_value(v: f64) -> Option<Expr> {
  let eps = 1e-12;

  // Helper to build n*Pi/d
  let pi_frac = |num: i128, den: i128| -> Expr {
    if den == 1 {
      if num == 1 {
        Expr::Constant("Pi".to_string())
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(num)),
          right: Box::new(Expr::Constant("Pi".to_string())),
        }
      }
    } else {
      let numerator = if num == 1 {
        Expr::Constant("Pi".to_string())
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(num)),
          right: Box::new(Expr::Constant("Pi".to_string())),
        }
      };
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(numerator),
        right: Box::new(Expr::Integer(den)),
      }
    }
  };

  // ArcCos special values (value -> result as n*Pi/d)
  let table: &[(f64, i128, i128)] = &[
    (0.5, 1, 3),                              // ArcCos[1/2] = Pi/3
    (-0.5, 2, 3),                             // ArcCos[-1/2] = 2*Pi/3
    (std::f64::consts::FRAC_1_SQRT_2, 1, 4),  // ArcCos[Sqrt[2]/2] = Pi/4
    (-std::f64::consts::FRAC_1_SQRT_2, 3, 4), // ArcCos[-Sqrt[2]/2] = 3*Pi/4
    (0.8660254037844386, 1, 6),               // ArcCos[Sqrt[3]/2] = Pi/6
    (-0.8660254037844386, 5, 6),              // ArcCos[-Sqrt[3]/2] = 5*Pi/6
    (0.9659258262890682, 1, 12), // ArcCos[(1+Sqrt[3])/(2*Sqrt[2])] = Pi/12
    (-0.9659258262890682, 11, 12), // ArcCos[-(1+Sqrt[3])/(2*Sqrt[2])] = 11*Pi/12
  ];

  for &(val, num, den) in table {
    if (v - val).abs() < eps {
      return Some(pi_frac(num, den));
    }
  }
  None
}

/// Check if a float value matches a known ArcSin special angle
fn arcsin_special_value(v: f64) -> Option<Expr> {
  let eps = 1e-12;

  // ArcSin special values (|value| -> Pi/d)
  let table: &[(f64, i128)] = &[
    (0.5, 6),                             // ArcSin[1/2] = Pi/6
    (std::f64::consts::FRAC_1_SQRT_2, 4), // ArcSin[Sqrt[2]/2] = Pi/4
    (0.8660254037844386, 3),              // ArcSin[Sqrt[3]/2] = Pi/3
  ];

  for &(val, den) in table {
    if (v.abs() - val).abs() < eps {
      if v < 0.0 {
        // Negative: build Times[Rational[-1, den], Pi] to display as -1/den*Pi
        return Some(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(-1), Expr::Integer(den)].into(),
          }),
          right: Box::new(Expr::Constant("Pi".to_string())),
        });
      } else if den == 1 {
        return Some(Expr::Constant("Pi".to_string()));
      } else {
        return Some(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::Constant("Pi".to_string())),
          right: Box::new(Expr::Integer(den)),
        });
      }
    }
  }
  None
}

/// ArcTan[x] - Inverse tangent (symbolic)
pub fn arctan_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcTan expects exactly 1 argument".into(),
    ));
  }
  if let Some(r) = try_complex_inverse_trig("ArcTan", &args[0]) {
    return r;
  }
  // ArcTan is monotonic increasing on ℝ: map over interval spans.
  if let Some(r) =
    crate::functions::interval_ast::map_monotonic_interval("ArcTan", &args[0])
  {
    return Ok(r);
  }
  // ArcTan[ComplexInfinity] = Indeterminate; ArcTan[Indeterminate] = Indeterminate
  if matches!(&args[0], Expr::Identifier(s) if s == "ComplexInfinity" || s == "Indeterminate")
  {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // ArcTan[-x] → -ArcTan[x] (odd function). Reals/BigFloats are excluded —
  // they evaluate to a numeric atan directly below — but negative integers
  // and rationals reduce to -ArcTan[|x|] (e.g. ArcTan[-4/3] = -ArcTan[4/3]).
  if !matches!(&args[0], Expr::Real(_) | Expr::BigFloat(_, _))
    && let Some(neg) = try_extract_negated(&args[0])
  {
    let inner = crate::evaluator::evaluate_function_call_ast("ArcTan", &[neg])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[Expr::Integer(-1), inner],
    );
  }
  // Exact values: ArcTan[0] = 0, ArcTan[1] = Pi/4, ArcTan[-1] = -Pi/4
  // ArcTan[Infinity] = Pi/2, ArcTan[-Infinity] = -Pi/2
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Integer(1) => {
      return Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(4)),
      });
    }
    Expr::Integer(-1) => {
      // -1/4*Pi = Times[Rational[-1, 4], Pi]
      return Ok(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-1), Expr::Integer(4)].into(),
        }),
        right: Box::new(Expr::Constant("Pi".to_string())),
      });
    }
    Expr::Identifier(s) if s == "Infinity" => {
      // ArcTan[Infinity] = Pi/2
      return Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      });
    }
    Expr::Real(f) => return Ok(Expr::Real(f.atan())),
    _ => {}
  }
  // ArcTan[-Infinity] = -Pi/2
  if crate::functions::math_ast::is_neg_infinity(&args[0]) {
    return Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-1), Expr::Integer(2)].into(),
        },
        Expr::Constant("Pi".to_string()),
      ]
      .into(),
    });
  }

  // Additional exact values: ArcTan[Sqrt[3]] = Pi/3, ArcTan[1/Sqrt[3]] = Pi/6
  // Use numerical comparison to detect these values robustly regardless of internal form.
  if let Some(val) = crate::functions::math_ast::try_eval_to_f64(&args[0]) {
    let sqrt3 = 3.0_f64.sqrt();
    let eps = 1e-12;
    if (val - sqrt3).abs() < eps {
      // ArcTan[Sqrt[3]] = Pi/3
      return Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(3)),
      });
    }
    if (val + sqrt3).abs() < eps {
      // ArcTan[-Sqrt[3]] = -Pi/3
      return Ok(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(-1), Expr::Integer(3)].into(),
          },
          Expr::Constant("Pi".to_string()),
        ]
        .into(),
      });
    }
    let inv_sqrt3 = 1.0 / sqrt3;
    if (val - inv_sqrt3).abs() < eps {
      // ArcTan[1/Sqrt[3]] = Pi/6
      return Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(6)),
      });
    }
    if (val + inv_sqrt3).abs() < eps {
      // ArcTan[-1/Sqrt[3]] = -Pi/6
      return Ok(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(-1), Expr::Integer(6)].into(),
          },
          Expr::Constant("Pi".to_string()),
        ]
        .into(),
      });
    }
    // Twelfth-angle values (inverse of Tan[Pi/12] = 2 - Sqrt[3] and
    // Tan[5 Pi/12] = 2 + Sqrt[3]). `k_over_12_pi(k)` builds k*Pi/12.
    let k_over_12_pi = |k: i128| -> Expr {
      if k == 1 {
        Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::Constant("Pi".to_string())),
          right: Box::new(Expr::Integer(12)),
        }
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(k), Expr::Integer(12)].into(),
            },
            Expr::Constant("Pi".to_string()),
          ]
          .into(),
        }
      }
    };
    let two_minus = 2.0 - sqrt3;
    let two_plus = 2.0 + sqrt3;
    if (val - two_minus).abs() < eps {
      return Ok(k_over_12_pi(1)); // ArcTan[2 - Sqrt[3]] = Pi/12
    }
    if (val + two_minus).abs() < eps {
      return Ok(k_over_12_pi(-1)); // ArcTan[-(2 - Sqrt[3])] = -Pi/12
    }
    if (val - two_plus).abs() < eps {
      return Ok(k_over_12_pi(5)); // ArcTan[2 + Sqrt[3]] = 5 Pi/12
    }
    if (val + two_plus).abs() < eps {
      return Ok(k_over_12_pi(-5)); // ArcTan[-(2 + Sqrt[3])] = -5 Pi/12
    }
  }

  Ok(unevaluated("ArcTan", args))
}

/// ArcTan[x, y] - Two-argument arctangent (atan2)
/// Returns the angle in radians of the point (x, y).
/// ArcTan[x, y] = ArcTan[y/x] with quadrant adjustment.
pub fn arctan2_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ArcTan expects exactly 2 arguments".into(),
    ));
  }
  let x = &args[0];
  let y = &args[1];

  // If either argument is an inexact machine real, ArcTan[x, y] is computed
  // numerically (atan2), matching wolframscript: ArcTan[0, 2.] is 1.5707…
  // (not Pi/2) and ArcTan[0, 0.] is 0. (not Indeterminate).
  fn has_real(e: &Expr) -> bool {
    match e {
      Expr::Real(_) | Expr::BigFloat(_, _) => true,
      Expr::BinaryOp { left, right, .. } => has_real(left) || has_real(right),
      Expr::UnaryOp { operand, .. } => has_real(operand),
      Expr::FunctionCall { args, .. } => args.iter().any(has_real),
      _ => false,
    }
  }
  if (has_real(x) || has_real(y))
    && let (Some(fx), Some(fy)) = (try_eval_to_f64(x), try_eval_to_f64(y))
  {
    return Ok(Expr::Real(fy.atan2(fx)));
  }

  // ArcTan[0, 0] = Indeterminate, with the ArcTan::indet message (matching
  // wolframscript). The inexact case ArcTan[0., 0.] above returns 0. instead.
  if matches!((x, y), (Expr::Integer(0), Expr::Integer(0))) {
    crate::emit_message(
      "ArcTan::indet: Indeterminate expression ArcTan[0, 0] encountered.",
    );
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  // Helper to build rational * Pi
  let rational_pi = |num: i128, den: i128| -> Expr {
    if den == 1 {
      if num == 1 {
        Expr::Constant("Pi".to_string())
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(num)),
          right: Box::new(Expr::Constant("Pi".to_string())),
        }
      }
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(num), Expr::Integer(den)].into(),
        }),
        right: Box::new(Expr::Constant("Pi".to_string())),
      }
    }
  };

  // Exact special values for common angles — nicer closed forms than the
  // general ArcTan[y/x] reduction below.
  if let (Some(xn), Some(yn)) = (expr_to_i128(x), expr_to_i128(y)) {
    let special = match (xn, yn) {
      (_, 0) if xn > 0 => Some(Expr::Integer(0)), // ArcTan[+x, 0] = 0
      (_, 0) if xn < 0 => Some(rational_pi(1, 1)), // ArcTan[-x, 0] = Pi
      (0, _) if yn > 0 => Some(rational_pi(1, 2)), // ArcTan[0, +y] = Pi/2
      (0, _) if yn < 0 => Some(rational_pi(-1, 2)), // ArcTan[0, -y] = -Pi/2
      _ if xn > 0 && yn == xn => Some(rational_pi(1, 4)), // Pi/4
      _ if xn > 0 && yn == -xn => Some(rational_pi(-1, 4)), // -Pi/4
      _ if xn < 0 && yn == -xn => Some(rational_pi(3, 4)), // 3*Pi/4
      _ if xn < 0 && yn == xn => Some(rational_pi(-3, 4)), // -3*Pi/4
      _ => None,
    };
    if let Some(v) = special {
      return Ok(v);
    }
  }

  // General numeric reduction: ArcTan[x, y] = ArcTan[y/x] adjusted by the
  // quadrant of (x, y). Wolfram keeps the single-argument ArcTan[y/x] even
  // when it does not simplify further, e.g. ArcTan[3, 4] = ArcTan[4/3] and
  // ArcTan[-3, 4] = Pi - ArcTan[4/3].
  if let (Some(xf), Some(yf)) = (try_eval_to_f64(x), try_eval_to_f64(y)) {
    if xf == 0.0 {
      return Ok(if yf > 0.0 {
        rational_pi(1, 2)
      } else if yf < 0.0 {
        rational_pi(-1, 2)
      } else {
        Expr::Identifier("Indeterminate".to_string())
      });
    }
    let y_over_x = crate::evaluator::evaluate_function_call_ast(
      "Divide",
      &[y.clone(), x.clone()],
    )?;
    let at =
      crate::evaluator::evaluate_function_call_ast("ArcTan", &[y_over_x])?;
    if xf > 0.0 {
      return Ok(at);
    }
    // x < 0: shift by +Pi (y >= 0) or -Pi (y < 0).
    let pi_term = if yf >= 0.0 {
      Expr::Constant("Pi".to_string())
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(Expr::Constant("Pi".to_string())),
      }
    };
    return crate::evaluator::evaluate_function_call_ast(
      "Plus",
      &[pi_term, at],
    );
  }

  // Return unevaluated for symbolic args
  Ok(unevaluated("ArcTan", args))
}

/// Try to split an expression into real part and exact imaginary coefficient.
/// Returns Some((real_part_expr, (im_num, im_den))) if the expression
/// has the form `real_part + (im_num/im_den)*I` where the imaginary
/// coefficient is exact (integer or rational).
/// This is more general than try_extract_complex_exact since real_part
/// can be any expression (e.g., Pi, Pi/6, symbolic).
fn try_split_real_imag(expr: &Expr) -> Option<(Expr, (i128, i128))> {
  // Helper: check if expr is purely I*rational (no real part)
  fn as_imag_only(e: &Expr) -> Option<(i128, i128)> {
    match e {
      Expr::Identifier(name) if name == "I" => Some((1, 1)),
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => {
        let (n, d) = as_imag_only(operand)?;
        Some((-n, d))
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        if matches!(&**right, Expr::Identifier(s) if s == "I")
          && let Some((n, d)) = expr_to_rational(left)
        {
          return Some((n, d));
        }
        if matches!(&**left, Expr::Identifier(s) if s == "I")
          && let Some((n, d)) = expr_to_rational(right)
        {
          return Some((n, d));
        }
        None
      }
      Expr::FunctionCall { name, args }
        if name == "Times" && args.len() >= 2 =>
      {
        // Check if I is one of the args
        let mut i_idx = None;
        for (idx, arg) in args.iter().enumerate() {
          if matches!(arg, Expr::Identifier(s) if s == "I") {
            i_idx = Some(idx);
            break;
          }
        }
        let i_idx = i_idx?;
        // Remaining args should form a rational
        let remaining: Vec<&Expr> = args
          .iter()
          .enumerate()
          .filter(|(i, _)| *i != i_idx)
          .map(|(_, a)| a)
          .collect();
        if remaining.len() == 1
          && let Some((n, d)) = expr_to_rational(remaining[0])
        {
          return Some((n, d));
        }
        None
      }
      _ => None,
    }
  }

  // Pure imaginary
  if let Some(im) = as_imag_only(expr) {
    return Some((Expr::Integer(0), im));
  }

  // a + b*I or a - b*I
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      if let Some(im) = as_imag_only(right) {
        return Some((*left.clone(), im));
      }
      if let Some(im) = as_imag_only(left) {
        return Some((*right.clone(), im));
      }
      None
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      if let Some((n, d)) = as_imag_only(right) {
        return Some((*left.clone(), (-n, d)));
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2 => {
      // Check if one of the args is purely imaginary
      for (idx, arg) in args.iter().enumerate() {
        if let Some(im) = as_imag_only(arg) {
          let remaining: Vec<Expr> = args
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, a)| a.clone())
            .collect();
          let real_part = if remaining.len() == 1 {
            remaining.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: remaining.into(),
            }
          };
          return Some((real_part, im));
        }
      }
      None
    }
    _ => None,
  }
}

// ─── Hyperbolic Trig Functions ────────────────────────────────────

/// Try to extract a "negated" form of an expression.
/// Returns Some(positive_expr) if the expression looks negative
/// (negative integer, negative rational, unary minus, Times[-1, x]).
fn try_extract_negated(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::Integer(n) if *n < 0 => Some(Expr::Integer(-n)),
    Expr::Real(f) if *f < 0.0 => Some(Expr::Real(-f)),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some((**operand).clone()),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        if *n < 0 {
          Some(make_rational(n.abs(), *d))
        } else {
          None
        }
      } else {
        None
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if let Some(negated) = try_extract_negated(left) {
        if matches!(&negated, Expr::Integer(1)) {
          Some((**right).clone())
        } else {
          Some(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(negated),
            right: right.clone(),
          })
        }
      } else {
        None
      }
    }
    // Handle evaluated Times[-1, x, ...] function call form
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      if let Some(negated_first) = try_extract_negated(&args[0]) {
        if args.len() == 1 {
          return Some(negated_first);
        }
        if matches!(&negated_first, Expr::Integer(1)) {
          if args.len() == 2 {
            return Some(args[1].clone());
          }
          return Some(unevaluated("Times", &args[1..]));
        }
        let mut new_args = vec![negated_first];
        new_args.extend_from_slice(&args[1..]);
        Some(Expr::FunctionCall {
          name: "Times".to_string(),
          args: new_args.into(),
        })
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Sinh[x] - Hyperbolic sine
/// Hyperbolic functions of a natural logarithm reduce to rational
/// functions of the logarithm's argument, since `Log[u]` makes
/// `E^Log[u] = u`. wolframscript:
///   Sinh[Log[u]] = (u^2 - 1)/(2 u)   Cosh[Log[u]] = (u^2 + 1)/(2 u)
///   Tanh[Log[u]] = (u^2 - 1)/(u^2 + 1)  Coth[Log[u]] = (u^2 + 1)/(u^2 - 1)
///   Sech[Log[u]] = (2 u)/(u^2 + 1)      Csch[Log[u]] = (2 u)/(u^2 - 1)
/// Only a bare single-argument `Log[u]` triggers this (matching
/// wolframscript, which leaves e.g. `Sinh[2 Log[2]]` unevaluated).
/// Returns the evaluated rational form, or `None` when the argument is
/// not a single-argument `Log`.
fn hyperbolic_of_log(
  name: &str,
  arg: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  use BinaryOperator as B;
  let u = match arg {
    Expr::FunctionCall { name: n, args } if n == "Log" && args.len() == 1 => {
      &args[0]
    }
    _ => return None,
  };
  // When `u` is a power with a non-integer exponent (e.g. Sqrt[2] =
  // 2^(1/2)), wolframscript first rewrites Log[u] = exponent*Log[base]
  // (so Sinh[Log[Sqrt[2]]] stays as Sinh[Log[2]/2]). Woxi does not perform
  // that Log-of-power extraction, so firing here would yield a value-correct
  // but differently-shaped result; leave such arguments unevaluated instead.
  let exponent = match u {
    Expr::BinaryOp {
      op: B::Power,
      right,
      ..
    } => Some(right.as_ref()),
    Expr::FunctionCall { name: n, args } if n == "Power" && args.len() == 2 => {
      Some(&args[1])
    }
    _ => None,
  };
  if let Some(exp) = exponent
    && !matches!(exp, Expr::Integer(_))
  {
    return None;
  }
  let bin = |op: B, a: Expr, b: Expr| Expr::BinaryOp {
    op,
    left: Box::new(a),
    right: Box::new(b),
  };
  let u2 = bin(B::Power, u.clone(), Expr::Integer(2));
  let u2_minus_1 = bin(B::Minus, u2.clone(), Expr::Integer(1));
  let u2_plus_1 = bin(B::Plus, u2, Expr::Integer(1));
  let two_u = bin(B::Times, Expr::Integer(2), u.clone());
  let result = match name {
    "Sinh" => bin(B::Divide, u2_minus_1, two_u),
    "Cosh" => bin(B::Divide, u2_plus_1, two_u),
    "Tanh" => bin(B::Divide, u2_minus_1, u2_plus_1),
    "Coth" => bin(B::Divide, u2_plus_1, u2_minus_1),
    "Sech" => bin(B::Divide, two_u, u2_plus_1),
    "Csch" => bin(B::Divide, two_u, u2_minus_1),
    _ => return None,
  };
  Some(crate::evaluator::evaluate_expr_to_expr(&result))
}

/// Special values of the six hyperbolic functions at (Complex)Infinity.
/// Handles positive real `Infinity` and `ComplexInfinity`; `-Infinity` flows
/// through each function's own odd/even reflection arm (e.g. `Tanh[-Infinity]`
/// → `-Tanh[Infinity]` → `-1`).
fn hyperbolic_at_infinity(name: &str, arg: &Expr) -> Option<Expr> {
  let Expr::Identifier(id) = arg else {
    return None;
  };
  match id.as_str() {
    "Infinity" => match name {
      "Sinh" | "Cosh" => Some(Expr::Identifier("Infinity".to_string())),
      "Tanh" | "Coth" => Some(Expr::Integer(1)),
      "Sech" | "Csch" => Some(Expr::Integer(0)),
      _ => None,
    },
    "ComplexInfinity" => Some(Expr::Identifier("Indeterminate".to_string())),
    _ => None,
  }
}

/// Special values of the six circular trig functions at real ±Infinity.
/// As `x → ±∞` along the real axis the value oscillates without limit, so
/// wolframscript returns the range as an `Interval`:
///   Sin, Cos → Interval[{-1, 1}]
///   Tan, Cot → Interval[{-Infinity, Infinity}]
///   Sec, Csc → Interval[{-Infinity, -1}, {1, Infinity}]
/// Every result is symmetric about 0, so `Infinity` and `-Infinity`
/// (`Times[-1, Infinity]`) map to the same interval. `ComplexInfinity` is left
/// for the caller (it yields Indeterminate).
fn circular_at_infinity(
  name: &str,
  arg: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let is_real_infinity = matches!(arg, Expr::Identifier(id) if id == "Infinity")
    || matches!(
      try_extract_negated(arg),
      Some(Expr::Identifier(ref id)) if id == "Infinity"
    );
  if !is_real_infinity {
    return None;
  }
  let inf = || Expr::Identifier("Infinity".to_string());
  let neg_inf = || Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(Expr::Identifier("Infinity".to_string())),
  };
  let span = |lo: Expr, hi: Expr| Expr::List(vec![lo, hi].into());
  let interval = |spans: Vec<Expr>| Expr::FunctionCall {
    name: "Interval".to_string(),
    args: spans.into(),
  };
  let result = match name {
    "Sin" | "Cos" => interval(vec![span(Expr::Integer(-1), Expr::Integer(1))]),
    "Tan" | "Cot" => interval(vec![span(neg_inf(), inf())]),
    "Sec" | "Csc" => interval(vec![
      span(neg_inf(), Expr::Integer(-1)),
      span(Expr::Integer(1), inf()),
    ]),
    _ => return None,
  };
  Some(crate::evaluator::evaluate_expr_to_expr(&result))
}

pub fn sinh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sinh expects 1 argument".into(),
    ));
  }
  if let Some(r) = imaginary_arg_reduction("Sinh", &args[0]) {
    return r;
  }
  // Sinh is monotonic increasing on ℝ: map over interval spans.
  if let Some(r) =
    crate::functions::interval_ast::map_monotonic_interval("Sinh", &args[0])
  {
    return Ok(r);
  }
  // Sinh[ArcSinh[x]] = x
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcSinh"
    && ia.len() == 1
  {
    return Ok(ia[0].clone());
  }
  // Sinh of an inverse hyperbolic function → algebraic form.
  if let Some(r) = hyperbolic_of_inverse("Sinh", &args[0]) {
    return Ok(r);
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Some(r) = hyperbolic_at_infinity("Sinh", &args[0]) {
    return Ok(r);
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => return Ok(Expr::Real(f.sinh())),
    _ => {}
  }
  // Sinh[Log[u]] = (u^2 - 1)/(2 u)
  if let Some(res) = hyperbolic_of_log("Sinh", &args[0]) {
    return res;
  }
  // Odd function: Sinh[-x] → -Sinh[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Sinh", &[pos])?;
    return Ok(negate_expr(inner));
  }
  Ok(unevaluated("Sinh", args))
}

/// Cosh[x] - Hyperbolic cosine
pub fn cosh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Cosh expects 1 argument".into(),
    ));
  }
  if let Some(r) = imaginary_arg_reduction("Cosh", &args[0]) {
    return r;
  }
  // Cosh[Interval[...]] — U-shaped (even, minimum Cosh[0] = 1 at the origin),
  // so a span containing 0 bottoms out at 1; otherwise it is monotonic.
  if let Some(r) = crate::functions::interval_ast::cosh_interval(&args[0]) {
    return Ok(r);
  }
  // Cosh[ArcCosh[x]] = x
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcCosh"
    && ia.len() == 1
  {
    return Ok(ia[0].clone());
  }
  // Cosh of an inverse hyperbolic function → algebraic form.
  if let Some(r) = hyperbolic_of_inverse("Cosh", &args[0]) {
    return Ok(r);
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Some(r) = hyperbolic_at_infinity("Cosh", &args[0]) {
    return Ok(r);
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(1)),
    Expr::Real(f) => return Ok(Expr::Real(f.cosh())),
    _ => {}
  }
  // Cosh[Log[u]] = (u^2 + 1)/(2 u)
  if let Some(res) = hyperbolic_of_log("Cosh", &args[0]) {
    return res;
  }
  // Even function: Cosh[-x] → Cosh[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    return crate::evaluator::evaluate_function_call_ast("Cosh", &[pos]);
  }
  Ok(unevaluated("Cosh", args))
}

/// Tanh[x] - Hyperbolic tangent
pub fn tanh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Tanh expects 1 argument".into(),
    ));
  }
  if let Some(r) = imaginary_arg_reduction("Tanh", &args[0]) {
    return r;
  }
  // Tanh is monotonic increasing on ℝ: map over interval spans.
  if let Some(r) =
    crate::functions::interval_ast::map_monotonic_interval("Tanh", &args[0])
  {
    return Ok(r);
  }
  // Tanh[ArcTanh[x]] = x
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcTanh"
    && ia.len() == 1
  {
    return Ok(ia[0].clone());
  }
  // Tanh of an inverse hyperbolic function → algebraic form.
  if let Some(r) = hyperbolic_of_inverse("Tanh", &args[0]) {
    return Ok(r);
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Some(r) = hyperbolic_at_infinity("Tanh", &args[0]) {
    return Ok(r);
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => return Ok(Expr::Real(f.tanh())),
    _ => {}
  }
  // Tanh[Log[u]] = (u^2 - 1)/(u^2 + 1)
  if let Some(res) = hyperbolic_of_log("Tanh", &args[0]) {
    return res;
  }
  // Odd function: Tanh[-x] → -Tanh[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Tanh", &[pos])?;
    return Ok(negate_expr(inner));
  }
  Ok(unevaluated("Tanh", args))
}

/// Coth[x] - Hyperbolic cotangent
pub fn coth_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Coth expects 1 argument".into(),
    ));
  }
  if let Some(r) = imaginary_arg_reduction("Coth", &args[0]) {
    return r;
  }
  // Coth[Interval[...]] — range over each span (single pole at 0, decreasing).
  if let Some(r) =
    crate::functions::interval_ast::coth_csch_interval("Coth", &args[0])
  {
    return Ok(r);
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Some(r) = hyperbolic_at_infinity("Coth", &args[0]) {
    return Ok(r);
  }
  // Coth[ArcCoth[x]] = x (inverse function identity)
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcCoth"
    && ia.len() == 1
  {
    return Ok(ia[0].clone());
  }
  match &args[0] {
    Expr::Integer(0) => {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    Expr::Real(f) => {
      let t = f.tanh();
      if t == 0.0 {
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      return Ok(Expr::Real(1.0 / t));
    }
    _ => {}
  }
  // Coth[Log[u]] = (u^2 + 1)/(u^2 - 1)
  if let Some(res) = hyperbolic_of_log("Coth", &args[0]) {
    return res;
  }
  // Odd function: Coth[-x] → -Coth[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Coth", &[pos])?;
    return Ok(negate_expr(inner));
  }
  Ok(unevaluated("Coth", args))
}

/// Sech[x] - Hyperbolic secant
pub fn sech_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sech expects 1 argument".into(),
    ));
  }
  if let Some(r) = imaginary_arg_reduction("Sech", &args[0]) {
    return r;
  }
  // Sech[Interval[...]] — even, max 1 at 0, no poles.
  if let Some(r) = crate::functions::interval_ast::sech_interval(&args[0]) {
    return Ok(r);
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Some(r) = hyperbolic_at_infinity("Sech", &args[0]) {
    return Ok(r);
  }
  // Sech[ArcSech[x]] = x (inverse function identity)
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcSech"
    && ia.len() == 1
  {
    return Ok(ia[0].clone());
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(1)),
    Expr::Real(f) => return Ok(Expr::Real(1.0 / f.cosh())),
    _ => {}
  }
  // Sech[Log[u]] = (2 u)/(u^2 + 1)
  if let Some(res) = hyperbolic_of_log("Sech", &args[0]) {
    return res;
  }
  // Even function: Sech[-x] → Sech[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    return crate::evaluator::evaluate_function_call_ast("Sech", &[pos]);
  }
  Ok(unevaluated("Sech", args))
}

/// Csch[x] - Hyperbolic cosecant
pub fn csch_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Csch expects 1 argument".into(),
    ));
  }
  // Csch[0] = ComplexInfinity (Sinh[0] = 0, so 1/Sinh[0] diverges).
  if matches!(&args[0], Expr::Integer(0)) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if let Some(r) = imaginary_arg_reduction("Csch", &args[0]) {
    return r;
  }
  // Csch[Interval[...]] — range over each span (single pole at 0, decreasing).
  if let Some(r) =
    crate::functions::interval_ast::coth_csch_interval("Csch", &args[0])
  {
    return Ok(r);
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Some(r) = hyperbolic_at_infinity("Csch", &args[0]) {
    return Ok(r);
  }
  // Csch[ArcCsch[x]] = x (inverse function identity)
  if let Expr::FunctionCall { name, args: ia } = &args[0]
    && name == "ArcCsch"
    && ia.len() == 1
  {
    return Ok(ia[0].clone());
  }
  if let Expr::Real(f) = &args[0] {
    let s = f.sinh();
    if s == 0.0 {
      // Csch has a pole at 0 (Sinh[0] = 0): Csch[0.] = ComplexInfinity, like
      // the exact Csch[0], rather than raising an error.
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return Ok(Expr::Real(1.0 / s));
  }
  // Csch[Log[u]] = (2 u)/(u^2 - 1)
  if let Some(res) = hyperbolic_of_log("Csch", &args[0]) {
    return res;
  }
  // Odd function: Csch[-x] → -Csch[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Csch", &[pos])?;
    return Ok(negate_expr(inner));
  }
  Ok(unevaluated("Csch", args))
}

/// ArcSinh[x] - Inverse hyperbolic sine
pub fn arcsinh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcSinh expects 1 argument".into(),
    ));
  }
  if let Some(r) = try_complex_inverse_trig("ArcSinh", &args[0]) {
    return r;
  }
  // ArcSinh is monotonic increasing on ℝ: map over interval spans.
  if let Some(r) =
    crate::functions::interval_ast::map_monotonic_interval("ArcSinh", &args[0])
  {
    return Ok(r);
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => return Ok(Expr::Real(f.asinh())),
    // ArcSinh is odd and unbounded: ArcSinh[±Infinity] = ±Infinity. An
    // undirected ComplexInfinity maps to ComplexInfinity.
    Expr::Identifier(s) if s == "Infinity" => {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    Expr::Identifier(s) if s == "ComplexInfinity" => {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity") => {
      return Ok(Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
    _ => {}
  }
  // Odd function: ArcSinh[-x] → -ArcSinh[x] (negative integers, rationals, and
  // negated symbolic arguments; reals are handled numerically above).
  if let Some(neg) = try_extract_negated(&args[0]) {
    let inner =
      crate::evaluator::evaluate_function_call_ast("ArcSinh", &[neg])?;
    return Ok(negate_expr(inner));
  }
  Ok(unevaluated("ArcSinh", args))
}

/// ArcCosh[x] - Inverse hyperbolic cosine
pub fn arccosh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCosh expects 1 argument".into(),
    ));
  }
  // ArcCosh is monotonic increasing on [1, ∞): map over interval spans.
  // (Out-of-domain endpoints give complex images and leave it unevaluated.)
  if let Some(r) =
    crate::functions::interval_ast::map_monotonic_interval("ArcCosh", &args[0])
  {
    return Ok(r);
  }
  match &args[0] {
    Expr::Integer(0) => {
      // ArcCosh[0] = (I/2)*Pi
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
          },
          Expr::Identifier("I".to_string()),
          Expr::Constant("Pi".to_string()),
        ],
      );
    }
    Expr::Integer(1) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => {
      let x = *f;
      if x >= 1.0 {
        return Ok(Expr::Real(x.acosh()));
      } else if x >= -1.0 {
        // ArcCosh[x] = I*ArcCos[x] for real x in [-1, 1].
        return build_complex_float_result(0.0, x.acos());
      } else {
        // x < -1: ArcCosh[x] = ArcCosh[|x|] + I*Pi. (x.acos() would be NaN.)
        return build_complex_float_result((-x).acosh(), std::f64::consts::PI);
      }
    }
    // ArcCosh[0``α] = (Pi/2 at α-digit accuracy) * I
    Expr::BigFloat(s, prec) if s == "0" => {
      let pi_half = crate::evaluator::evaluate_function_call_ast(
        "N",
        &[
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
              },
              Expr::Constant("Pi".to_string()),
            ]
            .into(),
          },
          Expr::Real(*prec),
        ],
      )?;
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[pi_half, Expr::Identifier("I".to_string())],
      );
    }
    // ArcCosh grows without bound in magnitude, so every infinite argument —
    // Infinity, -Infinity, and the undirected ComplexInfinity — maps to
    // Infinity (matching wolframscript).
    Expr::Identifier(s) if s == "Infinity" || s == "ComplexInfinity" => {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity") => {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    _ => {}
  }
  Ok(unevaluated("ArcCosh", args))
}

/// True if `expr` contains a Real or BigFloat anywhere — used to gate
/// numeric evaluation of otherwise-exact symbolic inputs.

/// ArcTanh[x] - Inverse hyperbolic tangent
pub fn arctanh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcTanh expects 1 argument".into(),
    ));
  }
  // ArcTanh is monotonic increasing on (-1, 1): map over interval spans.
  if let Some(r) =
    crate::functions::interval_ast::map_monotonic_interval("ArcTanh", &args[0])
  {
    return Ok(r);
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Integer(1) => return Ok(Expr::Identifier("Infinity".to_string())),
    Expr::Integer(-1) => {
      return Ok(Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
    Expr::Real(f) if f.abs() < 1.0 => return Ok(Expr::Real(f.atanh())),
    // Outside (-1, 1) the (inexact) real argument gives a complex result:
    // ArcTanh[x] = (1/2) Log|(1+x)/(1-x)| - sign(x)*Pi/2 I.
    Expr::Real(f) if f.abs() > 1.0 => {
      use std::f64::consts::PI;
      let x = *f;
      let re = 0.5 * ((1.0 + x).abs().ln() - (1.0 - x).abs().ln());
      return build_complex_float_result(re, -x.signum() * PI / 2.0);
    }
    _ => {}
  }
  // Complex float input: ArcTanh[x + I y] with careful precision.
  //   Re = (1/4) * log1p(4*x / ((1-x)^2 + y^2))
  //   Im = (1/2) * atan2(2*y, (1-x)*(1+x) - y^2)
  // The log1p form avoids catastrophic cancellation near z = 0.
  // Only fire for inexact inputs — wolframscript keeps purely exact
  // arguments like ArcTanh[2 + I] symbolic.
  if contains_inexact_real(&args[0])
    && let Some((re, im)) =
      crate::functions::math_ast::try_extract_complex_float(&args[0])
    && im != 0.0
  {
    let one_minus_re_sq = (1.0 - re) * (1.0 - re);
    let denom = one_minus_re_sq + im * im;
    let result_re = 0.25 * (4.0 * re / denom).ln_1p();
    let result_im = 0.5 * (2.0 * im).atan2((1.0 - re) * (1.0 + re) - im * im);
    return Ok(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Real(result_re),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Real(result_im), Expr::Identifier("I".to_string())]
            .into(),
        },
      ]
      .into(),
    });
  }
  // Odd function: ArcTanh[-x] → -ArcTanh[x] (negative integers/rationals and
  // negated symbolic arguments; reals are handled numerically above).
  if !matches!(&args[0], Expr::Real(_))
    && let Some(neg) = try_extract_negated(&args[0])
  {
    let inner =
      crate::evaluator::evaluate_function_call_ast("ArcTanh", &[neg])?;
    return Ok(negate_expr(inner));
  }
  Ok(unevaluated("ArcTanh", args))
}

/// ArcCoth[x] - Inverse hyperbolic cotangent
pub fn arccoth_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCoth expects 1 argument".into(),
    ));
  }
  // ArcCoth[±Infinity] = 0.
  if matches!(&args[0], Expr::Identifier(s) if s == "Infinity")
    || crate::functions::math_ast::is_neg_infinity(&args[0])
  {
    return Ok(Expr::Integer(0));
  }
  match &args[0] {
    Expr::Integer(0) => {
      // ArcCoth[0] = (I/2)*Pi
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
          },
          Expr::Identifier("I".to_string()),
          Expr::Constant("Pi".to_string()),
        ],
      );
    }
    Expr::Integer(1) => return Ok(Expr::Identifier("Infinity".to_string())),
    Expr::Integer(-1) => {
      return Ok(Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
    Expr::Real(f) => {
      // ArcCoth[x] = ArcTanh[1/x]
      // For |x| > 1: real result
      // For |x| < 1: complex result
      // For x == 0: I*Pi/2
      // For x == ±1: a singularity — Indeterminate.
      let x = *f;
      if x.abs() == 1.0 {
        return Ok(Expr::Identifier("Indeterminate".to_string()));
      }
      if x.abs() > 1.0 {
        return Ok(Expr::Real((1.0 / x).atanh()));
      } else if x == 0.0 {
        return crate::evaluator::evaluate_function_call_ast(
          "Complex",
          &[Expr::Real(0.0), Expr::Real(std::f64::consts::FRAC_PI_2)],
        );
      } else {
        // |x| <= 1 and x != 0: complex result
        // ArcCoth[x] = (1/2) * ln((x+1)/(x-1))
        // For |x| < 1: real part = (1/2)*ln(|(x+1)/(x-1)|), imag part = ±Pi/2
        let re = 0.5 * ((x + 1.0) / (x - 1.0)).abs().ln();
        let im = if x > 0.0 {
          -std::f64::consts::FRAC_PI_2
        } else {
          std::f64::consts::FRAC_PI_2
        };
        return crate::evaluator::evaluate_function_call_ast(
          "Complex",
          &[Expr::Real(re), Expr::Real(im)],
        );
      }
    }
    // ArcCoth[0``α] = (Pi/2 at α-digit accuracy) * I
    Expr::BigFloat(s, prec) if s == "0" => {
      let pi_half = crate::evaluator::evaluate_function_call_ast(
        "N",
        &[
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
              },
              Expr::Constant("Pi".to_string()),
            ]
            .into(),
          },
          Expr::Real(*prec),
        ],
      )?;
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[pi_half, Expr::Identifier("I".to_string())],
      );
    }
    _ => {}
  }
  // Odd function: ArcCoth[-x] = -ArcCoth[x] (exact negatives not handled
  // above, e.g. ArcCoth[-2] = -ArcCoth[2]).
  if let Some(neg) = try_extract_negated(&args[0]) {
    let inner =
      crate::evaluator::evaluate_function_call_ast("ArcCoth", &[neg])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[Expr::Integer(-1), inner],
    );
  }
  Ok(unevaluated("ArcCoth", args))
}

/// ArcSech[x] - Inverse hyperbolic secant
pub fn arcsech_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcSech expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Identifier("Infinity".to_string())),
    Expr::Integer(1) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => {
      // ArcSech[x] = ArcCosh[1/x].
      let x = *f;
      if x == 0.0 {
        // 1/0. is indeterminate for an inexact zero (ArcSech[0] exact stays
        // Infinity, handled above).
        return Ok(Expr::Identifier("Indeterminate".to_string()));
      }
      if x > 0.0 && x <= 1.0 {
        return Ok(Expr::Real((1.0 / x).acosh()));
      }
      // Out of domain (x < 0 or x > 1): delegate for the complex result.
      return crate::evaluator::evaluate_function_call_ast(
        "ArcCosh",
        &[Expr::Real(1.0 / x)],
      );
    }
    _ => {}
  }
  Ok(unevaluated("ArcSech", args))
}

pub fn arccot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCot expects 1 argument".into(),
    ));
  }
  let x = &args[0];
  let unevaluated = || Ok(unevaluated("ArcCot", args));

  // ArcCot[±Infinity] = 0; ArcCot[0] = Pi/2.
  if matches!(x, Expr::Identifier(s) if s == "Infinity")
    || crate::functions::math_ast::is_neg_infinity(x)
  {
    return Ok(Expr::Integer(0));
  }
  if matches!(x, Expr::Integer(0)) {
    return Ok(pi_over_n(2));
  }
  // Only inexact arguments evaluate numerically; exact ones stay symbolic
  // unless they reduce to a closed form below (e.g. ArcCot[2] = ArcCot[2]).
  if let Expr::Real(f) = x {
    return Ok(Expr::Real((1.0 / f).atan()));
  }
  // Odd function: ArcCot[-x] = -ArcCot[x].
  if let Some(neg) = try_extract_negated(x) {
    let inner = crate::evaluator::evaluate_function_call_ast("ArcCot", &[neg])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[Expr::Integer(-1), inner],
    );
  }
  // For positive exact x, ArcCot[x] = ArcTan[1/x]; keep that only when it
  // reduces to a closed form (Pi/4, Pi/6, …). Otherwise stay ArcCot[x],
  // matching Wolfram (which keeps ArcCot[2], not ArcTan[1/2]).
  if matches!(try_eval_to_f64(x), Some(v) if v > 0.0) {
    let recip = crate::evaluator::evaluate_function_call_ast(
      "Divide",
      &[Expr::Integer(1), x.clone()],
    )?;
    let at = crate::evaluator::evaluate_function_call_ast("ArcTan", &[recip])?;
    if !matches!(&at, Expr::FunctionCall { name, .. } if name == "ArcTan") {
      return Ok(at);
    }
  }
  unevaluated()
}

pub fn arccsc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCsc expects 1 argument".into(),
    ));
  }
  match &args[0] {
    // ArcCsc[0] = ArcSin[1/0] = ComplexInfinity.
    Expr::Integer(0) => {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    Expr::Integer(1) => return Ok(pi_over_n(2)), // Pi/2
    Expr::Integer(-1) => return Ok(negative_pi_over_2()), // -Pi/2
    _ => {}
  }
  // For exact (non-Real) numeric args, compute ArcSin[1/x]
  if !matches!(&args[0], Expr::Real(_)) {
    let reciprocal = crate::evaluator::evaluate_function_call_ast(
      "Power",
      &[args[0].clone(), Expr::Integer(-1)],
    )?;
    let result =
      crate::evaluator::evaluate_function_call_ast("ArcSin", &[reciprocal])?;
    // If ArcSin returned an exact result (not unevaluated), return it
    if !matches!(&result, Expr::FunctionCall { name, .. } if name == "ArcSin") {
      return Ok(result);
    }
  }
  // Inexact real argument: ArcCsc[x] = ArcSin[1/x]. Delegating covers the
  // in-domain (|x| >= 1, real) and out-of-domain (|x| < 1, complex) cases;
  // an exact argument that did not reduce above stays symbolic (not
  // numericized), matching wolframscript.
  if let Expr::Real(f) = &args[0] {
    if *f == 0.0 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return crate::evaluator::evaluate_function_call_ast(
      "ArcSin",
      &[Expr::Real(1.0 / *f)],
    );
  }
  Ok(unevaluated("ArcCsc", args))
}

pub fn arcsec_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcSec expects 1 argument".into(),
    ));
  }
  match &args[0] {
    // ArcSec[0] = ArcCos[1/0] = ComplexInfinity.
    Expr::Integer(0) => {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    Expr::Integer(1) => return Ok(Expr::Integer(0)),
    Expr::Integer(-1) => return Ok(Expr::Identifier("Pi".to_string())),
    _ => {}
  }
  // For exact (non-Real) numeric args, compute ArcCos[1/x]
  if !matches!(&args[0], Expr::Real(_)) {
    let reciprocal = crate::evaluator::evaluate_function_call_ast(
      "Power",
      &[args[0].clone(), Expr::Integer(-1)],
    )?;
    let result =
      crate::evaluator::evaluate_function_call_ast("ArcCos", &[reciprocal])?;
    // If ArcCos returned an exact result (not unevaluated), return it
    if !matches!(&result, Expr::FunctionCall { name, .. } if name == "ArcCos") {
      return Ok(result);
    }
  }
  // Inexact real argument: ArcSec[x] = ArcCos[1/x]. Delegating covers the
  // in-domain (|x| >= 1, real) and out-of-domain (|x| < 1, complex) cases;
  // an exact argument that did not reduce above stays symbolic (not
  // numericized), matching wolframscript.
  if let Expr::Real(f) = &args[0] {
    if *f == 0.0 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return crate::evaluator::evaluate_function_call_ast(
      "ArcCos",
      &[Expr::Real(1.0 / *f)],
    );
  }
  Ok(unevaluated("ArcSec", args))
}

pub fn arccsch_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCsch expects 1 argument".into(),
    ));
  }
  let x = &args[0];
  if let Expr::Integer(0) = x {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  // ArcCsch[±Infinity] = 0.
  if matches!(x, Expr::Identifier(s) if s == "Infinity")
    || crate::functions::math_ast::is_neg_infinity(x)
  {
    return Ok(Expr::Integer(0));
  }
  // Only inexact arguments evaluate numerically; exact ones (e.g. ArcCsch[2])
  // stay symbolic, matching Wolfram.
  if let Expr::Real(f) = x {
    if *f == 0.0 {
      // ArcCsch[0.] = ArcSinh[1/0.] diverges (like the exact ArcCsch[0]).
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return Ok(Expr::Real((1.0 / f).asinh()));
  }
  // Odd function: ArcCsch[-x] = -ArcCsch[x].
  if let Some(neg) = try_extract_negated(x) {
    let inner =
      crate::evaluator::evaluate_function_call_ast("ArcCsch", &[neg])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[Expr::Integer(-1), inner],
    );
  }
  Ok(unevaluated("ArcCsch", args))
}

/// Helper to construct -Pi/2 matching wolframscript output format
/// Construct Pi/n as an AST expression
fn pi_over_n(n: i128) -> Expr {
  Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::Constant("Pi".to_string())),
    right: Box::new(Expr::Integer(n)),
  }
}

fn negative_pi_over_2() -> Expr {
  Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(Expr::Integer(2)),
    }),
    right: Box::new(Expr::Constant("Pi".to_string())),
  }
}

/// Gudermannian[x] - the Gudermannian function: 2 ArcTan[Tanh[x/2]]
/// Gudermannian[0] = 0, Gudermannian[Infinity] = Pi/2, Gudermannian[-Infinity] = -Pi/2
/// If `e` is `Times[…]` whose flat factors are exactly:
/// a single Rational(k, 2) (k odd integer) together with one `I` and
/// one `Pi`, return `k`. Used by `gudermannian_ast` to detect arguments
/// at the simple poles `(2n+1) π i / 2`. Times trees may be nested
/// (e.g. `Times[Rational(5,2), Times[I, Pi]]`), so we flatten first.
fn extract_half_odd_pi_i_coefficient(e: &Expr) -> Option<i128> {
  fn flatten<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          flatten(a, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        flatten(left, out);
        flatten(right, out);
      }
      _ => out.push(e),
    }
  }
  let mut factors: Vec<&Expr> = Vec::new();
  flatten(e, &mut factors);
  if factors.len() < 3 {
    return None;
  }
  let mut k: Option<i128> = None;
  let mut has_i = false;
  let mut has_pi = false;
  for f in factors {
    match f {
      Expr::FunctionCall { name, args }
        if name == "Rational"
          && args.len() == 2
          && matches!(&args[1], Expr::Integer(2)) =>
      {
        if let Expr::Integer(n) = &args[0]
          && n % 2 != 0
        {
          if k.is_some() {
            return None;
          }
          k = Some(*n);
        } else {
          return None;
        }
      }
      Expr::Identifier(s) if s == "I" => {
        if has_i {
          return None;
        }
        has_i = true;
      }
      Expr::Constant(s) | Expr::Identifier(s) if s == "Pi" => {
        if has_pi {
          return None;
        }
        has_pi = true;
      }
      _ => return None,
    }
  }
  if has_i && has_pi { k } else { None }
}

pub fn gudermannian_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Gudermannian expects 1 argument".into(),
    ));
  }
  // Gudermannian has simple poles at z = (2n+1) π i / 2 with values
  // alternating between DirectedInfinity[I] and DirectedInfinity[-I].
  // Detect `Times[Rational(k, 2), I, Pi]` with odd `k`.
  if let Some(k) = extract_half_odd_pi_i_coefficient(&args[0]) {
    // k odd: k mod 4 ∈ {1, 3} (or {-1, -3}). +I for k ≡ 1 (mod 4),
    // -I for k ≡ -1 (mod 4).
    let m = ((k % 4) + 4) % 4;
    let direction = if m == 1 {
      Expr::Identifier("I".to_string())
    } else {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("I".to_string())),
      }
    };
    return Ok(Expr::FunctionCall {
      name: "DirectedInfinity".to_string(),
      args: vec![direction].into(),
    });
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) if *f == 0.0 => return Ok(Expr::Real(0.0)),
    Expr::Real(f) => {
      // Gudermannian[x] = 2 * atan(tanh(x/2))
      let result = 2.0 * (f / 2.0).tanh().atan();
      return Ok(Expr::Real(result));
    }
    Expr::Identifier(name) if name == "Infinity" => {
      // Gudermannian[Infinity] = Pi/2
      return Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      });
    }
    Expr::Identifier(name) if name == "ComplexInfinity" => {
      // Gudermannian[ComplexInfinity] is unevaluated in Wolfram
      return Ok(unevaluated("Gudermannian", args));
    }
    Expr::Identifier(name) if name == "Undefined" => {
      return Ok(Expr::Identifier("Undefined".to_string()));
    }
    Expr::Identifier(name) if name == "Indeterminate" => {
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // -Infinity (as UnaryOp)
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } if matches!(operand.as_ref(), Expr::Identifier(n) if n == "Infinity") => {
      return Ok(negative_pi_over_2());
    }
    // -Infinity (as BinaryOp)
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(-1))
      && matches!(right.as_ref(), Expr::Identifier(n) if n == "Infinity") =>
    {
      return Ok(negative_pi_over_2());
    }
    // -Infinity (as FunctionCall)
    Expr::FunctionCall { name, args: fargs }
      if name == "Times"
        && fargs.len() == 2
        && matches!(fargs[0], Expr::Integer(-1))
        && matches!(&fargs[1], Expr::Identifier(n) if n == "Infinity") =>
    {
      return Ok(negative_pi_over_2());
    }
    _ => {}
  }
  // Symbolic: return unevaluated (matching wolframscript behavior)
  Ok(unevaluated("Gudermannian", args))
}

/// InverseGudermannian[x] - the inverse Gudermannian function
pub fn inverse_gudermannian_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InverseGudermannian expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) if *f == 0.0 => return Ok(Expr::Real(0.0)),
    Expr::Real(f) => {
      // InverseGudermannian[x] = 2 * atanh(tan(x/2)).
      // f64 atanh is not exactly odd around 0, so the direct formula
      // yields slightly different bit patterns for +x and -x. Force
      // odd-symmetric output by always computing on the absolute value
      // and re-applying the sign.
      let abs_f = f.abs();
      let pos = 2.0 * (abs_f / 2.0).tan().atanh();
      let result = if *f < 0.0 { -pos } else { pos };
      return Ok(Expr::Real(result));
    }
    _ => {}
  }
  Ok(unevaluated("InverseGudermannian", args))
}

/// LogisticSigmoid[x] = 1 / (1 + Exp[-x])
pub fn logistic_sigmoid_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LogisticSigmoid expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => {
      // LogisticSigmoid[0] = 1/2
      return Ok(Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      });
    }
    Expr::Real(f) => {
      let result = 1.0 / (1.0 + (-f).exp());
      return Ok(Expr::Real(result));
    }
    // ±Infinity limits: LogisticSigmoid[Infinity] = 1,
    // LogisticSigmoid[-Infinity] = 0. Exact non-zero integers (and other
    // exact values like 1/2 or I) stay symbolic, matching wolframscript —
    // they are NOT numericized.
    Expr::Identifier(s) if s == "Infinity" => return Ok(Expr::Integer(1)),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity") => {
      return Ok(Expr::Integer(0));
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "DirectedInfinity" && dargs.len() == 1 =>
    {
      match &dargs[0] {
        Expr::Integer(1) => return Ok(Expr::Integer(1)),
        Expr::Integer(-1) => return Ok(Expr::Integer(0)),
        _ => {}
      }
    }
    _ => {}
  }
  // Complex float input: 1 / (1 + exp(-z)) using complex arithmetic. Only
  // fires for an inexact argument so an exact complex like I stays symbolic.
  if contains_inexact_real(&args[0])
    && let Some((re, im)) =
      crate::functions::math_ast::try_extract_complex_float(&args[0])
    && im != 0.0
  {
    // exp(-z) = exp(-re) * (cos(-im) + I*sin(-im))
    let exp_neg_re = (-re).exp();
    let (neg_im_sin, neg_im_cos) = (-im).sin_cos();
    let ex_re = exp_neg_re * neg_im_cos;
    let ex_im = exp_neg_re * neg_im_sin;
    // 1 + exp(-z)
    let denom_re = 1.0 + ex_re;
    let denom_im = ex_im;
    // 1 / (denom_re + I*denom_im) = (denom_re - I*denom_im) / (denom_re^2 + denom_im^2)
    let mag2 = denom_re * denom_re + denom_im * denom_im;
    let result_re = denom_re / mag2;
    let result_im = -denom_im / mag2;
    return Ok(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Real(result_re),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Real(result_im), Expr::Identifier("I".to_string())]
            .into(),
        },
      ]
      .into(),
    });
  }
  Ok(unevaluated("LogisticSigmoid", args))
}

// ─── Degree Trig Functions ──────────────────────────────────────────────

/// Helper: multiply argument by Degree (Pi/180) and evaluate trig function
fn trig_degrees_ast(
  func_name: &str,
  degrees_name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects 1 argument",
      degrees_name
    )));
  }
  // Convert degrees to radians: x * Degree
  let radians = crate::evaluator::evaluate_function_call_ast(
    "Times",
    &[args[0].clone(), Expr::Constant("Degree".to_string())],
  )?;
  let result =
    crate::evaluator::evaluate_function_call_ast(func_name, &[radians])?;
  // If the result is still unevaluated (symbolic), return unevaluated degree form
  if let Expr::FunctionCall { name, .. } = &result
    && name == func_name
  {
    return Ok(unevaluated(degrees_name, args));
  }
  Ok(result)
}

/// Helper: evaluate inverse trig function and convert result from radians to degrees
fn arc_trig_degrees_ast(
  func_name: &str,
  degrees_name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects 1 argument",
      degrees_name
    )));
  }
  let radians = crate::evaluator::evaluate_function_call_ast(
    func_name,
    &[args[0].clone()],
  )?;
  // If the result is still unevaluated (symbolic), return unevaluated
  if let Expr::FunctionCall { name, .. } = &radians
    && name == func_name
  {
    return Ok(unevaluated(degrees_name, args));
  }
  // For numeric results, convert radians to degrees (keep as Real)
  if let Expr::Real(f) = &radians {
    return Ok(Expr::Real(f.to_degrees()));
  }
  // For exact results, multiply by 180/Pi
  crate::evaluator::evaluate_function_call_ast(
    "Times",
    &[
      radians,
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(180), Expr::Integer(1)].into(),
      },
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![Expr::Constant("Pi".to_string()), Expr::Integer(-1)].into(),
      },
    ],
  )
}

pub fn sin_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  trig_degrees_ast("Sin", "SinDegrees", args)
}

pub fn cos_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  trig_degrees_ast("Cos", "CosDegrees", args)
}

pub fn tan_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  trig_degrees_ast("Tan", "TanDegrees", args)
}

pub fn cot_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  trig_degrees_ast("Cot", "CotDegrees", args)
}

pub fn sec_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  trig_degrees_ast("Sec", "SecDegrees", args)
}

pub fn csc_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  trig_degrees_ast("Csc", "CscDegrees", args)
}

pub fn arcsin_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  arc_trig_degrees_ast("ArcSin", "ArcSinDegrees", args)
}

pub fn arccos_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  arc_trig_degrees_ast("ArcCos", "ArcCosDegrees", args)
}

pub fn arctan_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  arc_trig_degrees_ast("ArcTan", "ArcTanDegrees", args)
}

pub fn arccot_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  arc_trig_degrees_ast("ArcCot", "ArcCotDegrees", args)
}

pub fn arcsec_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  arc_trig_degrees_ast("ArcSec", "ArcSecDegrees", args)
}

pub fn arccsc_degrees_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  arc_trig_degrees_ast("ArcCsc", "ArcCscDegrees", args)
}

// ─── TrigExpand ─────────────────────────────────────────────────────────

/// TrigExpand[expr] — expand trig functions of sums and integer multiples.
pub fn trig_expand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(unevaluated("TrigExpand", args));
  }
  let result = trig_expand_recursive(&args[0]);
  // TrigExpand distributes products over sums (and expands powers).
  let expanded = crate::functions::expand_and_combine(&result);
  crate::evaluator::evaluate_expr_to_expr(&expanded)
}

/// Recursively apply TrigExpand to an expression.
fn trig_expand_recursive(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      let trig_name = name.as_str();
      match trig_name {
        "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc" | "Sinh" | "Cosh"
        | "Tanh" | "Coth" | "Sech" | "Csch" => {
          // First recursively expand the argument
          let arg = trig_expand_recursive(&args[0]);
          expand_trig_function(trig_name, &arg)
        }
        _ => {
          // Recurse into other function calls
          Expr::FunctionCall {
            name: name.clone(),
            args: args.iter().map(trig_expand_recursive).collect(),
          }
        }
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(trig_expand_recursive).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(trig_expand_recursive(left)),
      right: Box::new(trig_expand_recursive(right)),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(trig_expand_recursive).collect())
    }
    _ => expr.clone(),
  }
}

/// Expand a single trig function application.
fn expand_trig_function(name: &str, arg: &Expr) -> Expr {
  // Try to decompose the argument as a sum of terms
  let terms = collect_additive_terms(arg);

  if terms.len() >= 2 {
    // It's a sum — apply addition formulas
    match name {
      "Sin" => expand_sin_sum(&terms),
      "Cos" => expand_cos_sum(&terms),
      "Tan" => {
        let sin_exp = expand_sin_sum(&terms);
        let cos_exp = expand_cos_sum(&terms);
        make_divide(&sin_exp, &cos_exp)
      }
      "Cot" => {
        let sin_exp = expand_sin_sum(&terms);
        let cos_exp = expand_cos_sum(&terms);
        make_divide(&cos_exp, &sin_exp)
      }
      "Sec" => {
        let cos_exp = expand_cos_sum(&terms);
        make_divide(&Expr::Integer(1), &cos_exp)
      }
      "Csc" => {
        let sin_exp = expand_sin_sum(&terms);
        make_divide(&Expr::Integer(1), &sin_exp)
      }
      "Sinh" => expand_sinh_sum(&terms),
      "Cosh" => expand_cosh_sum(&terms),
      "Tanh" => {
        let s = expand_sinh_sum(&terms);
        let c = expand_cosh_sum(&terms);
        make_divide(&s, &c)
      }
      "Coth" => {
        let s = expand_sinh_sum(&terms);
        let c = expand_cosh_sum(&terms);
        make_divide(&c, &s)
      }
      "Sech" => {
        let c = expand_cosh_sum(&terms);
        make_divide(&Expr::Integer(1), &c)
      }
      "Csch" => {
        let s = expand_sinh_sum(&terms);
        make_divide(&Expr::Integer(1), &s)
      }
      _ => make_fn(name, &[arg.clone()]),
    }
  } else {
    // Single term — check for integer multiple
    let (coeff, base) = extract_integer_factor(arg);
    if coeff >= 2 {
      match name {
        "Sin" => expand_sin_multiple(coeff as usize, &base),
        "Cos" => expand_cos_multiple(coeff as usize, &base),
        "Tan" => {
          let s = expand_sin_multiple(coeff as usize, &base);
          let c = expand_cos_multiple(coeff as usize, &base);
          make_divide(&s, &c)
        }
        "Cot" => {
          let s = expand_sin_multiple(coeff as usize, &base);
          let c = expand_cos_multiple(coeff as usize, &base);
          make_divide(&c, &s)
        }
        "Sec" => {
          let c = expand_cos_multiple(coeff as usize, &base);
          make_divide(&Expr::Integer(1), &c)
        }
        "Csc" => {
          let s = expand_sin_multiple(coeff as usize, &base);
          make_divide(&Expr::Integer(1), &s)
        }
        "Sinh" => expand_sinh_multiple(coeff as usize, &base),
        "Cosh" => expand_cosh_multiple(coeff as usize, &base),
        "Tanh" => {
          let s = expand_sinh_multiple(coeff as usize, &base);
          let c = expand_cosh_multiple(coeff as usize, &base);
          make_divide(&s, &c)
        }
        "Coth" => {
          let s = expand_sinh_multiple(coeff as usize, &base);
          let c = expand_cosh_multiple(coeff as usize, &base);
          make_divide(&c, &s)
        }
        "Sech" => {
          let c = expand_cosh_multiple(coeff as usize, &base);
          make_divide(&Expr::Integer(1), &c)
        }
        "Csch" => {
          let s = expand_sinh_multiple(coeff as usize, &base);
          make_divide(&Expr::Integer(1), &s)
        }
        _ => make_fn(name, &[arg.clone()]),
      }
    } else {
      // Nothing to expand
      make_fn(name, &[arg.clone()])
    }
  }
}

/// Collect additive terms from a Plus/Minus expression.
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
      // Negate the right side
      let neg = make_times(&Expr::Integer(-1), right);
      terms.push(neg);
      terms
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().flat_map(collect_additive_terms).collect()
    }
    _ => vec![expr.clone()],
  }
}

/// Extract integer factor from n*x. Returns (n, x).
fn extract_integer_factor(expr: &Expr) -> (i128, Expr) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if let Expr::Integer(n) = left.as_ref() {
        return (*n, *right.clone());
      }
      if let Expr::Integer(n) = right.as_ref() {
        return (*n, *left.clone());
      }
      (1, expr.clone())
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if let Expr::Integer(n) = &args[0] {
        return (*n, args[1].clone());
      }
      if let Expr::Integer(n) = &args[1] {
        return (*n, args[0].clone());
      }
      (1, expr.clone())
    }
    _ => (1, expr.clone()),
  }
}

// ─── Sin/Cos expansion helpers ──────────────────────────────────────

/// Expand Sin[a + b + ...] using recursive addition formula. Each
/// per-term `Sin[a]` / `Cos[a]` is built via `expand_trig_function` so
/// integer-multiple terms like `Sin[2 x y]` are themselves expanded
/// (matching wolframscript's full TrigExpand which decomposes both the
/// sum and any embedded multiple-angle subterms).
fn expand_sin_sum(terms: &[Expr]) -> Expr {
  if terms.len() == 1 {
    return expand_trig_function("Sin", &terms[0]);
  }
  // Sin[a + rest] = Sin[a]*Cos[rest] + Cos[a]*Sin[rest]
  let a = &terms[0];
  let rest = &terms[1..];
  let sin_a = expand_trig_function("Sin", a);
  let cos_a = expand_trig_function("Cos", a);
  let sin_rest = expand_sin_sum(rest);
  let cos_rest = expand_cos_sum(rest);
  make_plus(
    &make_times(&sin_a, &cos_rest),
    &make_times(&cos_a, &sin_rest),
  )
}

/// Expand Cos[a + b + ...] using recursive addition formula.
fn expand_cos_sum(terms: &[Expr]) -> Expr {
  if terms.len() == 1 {
    return expand_trig_function("Cos", &terms[0]);
  }
  // Cos[a + rest] = Cos[a]*Cos[rest] - Sin[a]*Sin[rest]
  let a = &terms[0];
  let rest = &terms[1..];
  let sin_a = expand_trig_function("Sin", a);
  let cos_a = expand_trig_function("Cos", a);
  let sin_rest = expand_sin_sum(rest);
  let cos_rest = expand_cos_sum(rest);
  make_minus(
    &make_times(&cos_a, &cos_rest),
    &make_times(&sin_a, &sin_rest),
  )
}

/// Expand Sin[n*x] using Chebyshev-like formula:
/// Sin[n*x] = Sum_{k} (-1)^k * C(n, 2k+1) * cos(x)^(n-2k-1) * sin(x)^(2k+1)
fn expand_sin_multiple(n: usize, x: &Expr) -> Expr {
  let sin_x = make_fn("Sin", &[x.clone()]);
  let cos_x = make_fn("Cos", &[x.clone()]);
  let mut terms = Vec::new();

  for k in 0..=(n - 1) / 2 {
    let sign = if k % 2 == 0 { 1i128 } else { -1 };
    let binom =
      crate::functions::binomial_coeff(n as i128, (2 * k + 1) as i128);
    let coeff = sign * binom;
    let cos_pow = n - 2 * k - 1;
    let sin_pow = 2 * k + 1;

    let mut term = Expr::Integer(coeff);
    if cos_pow > 0 {
      term = make_times(&term, &make_power(&cos_x, cos_pow as i128));
    }
    if sin_pow > 0 {
      term = make_times(&term, &make_power(&sin_x, sin_pow as i128));
    }
    terms.push(term);
  }

  build_sum(&terms)
}

/// Expand Cos[n*x] using Chebyshev-like formula:
/// Cos[n*x] = Sum_{k} (-1)^k * C(n, 2k) * cos(x)^(n-2k) * sin(x)^(2k)
fn expand_cos_multiple(n: usize, x: &Expr) -> Expr {
  let sin_x = make_fn("Sin", &[x.clone()]);
  let cos_x = make_fn("Cos", &[x.clone()]);
  let mut terms = Vec::new();

  for k in 0..=n / 2 {
    let sign = if k % 2 == 0 { 1i128 } else { -1 };
    let binom = crate::functions::binomial_coeff(n as i128, (2 * k) as i128);
    let coeff = sign * binom;
    let cos_pow = n - 2 * k;
    let sin_pow = 2 * k;

    let mut term = Expr::Integer(coeff);
    if cos_pow > 0 {
      term = make_times(&term, &make_power(&cos_x, cos_pow as i128));
    }
    if sin_pow > 0 {
      term = make_times(&term, &make_power(&sin_x, sin_pow as i128));
    }
    terms.push(term);
  }

  build_sum(&terms)
}

// ─── Hyperbolic expansion helpers ───────────────────────────────────

/// Expand Sinh[a + b + ...] recursively.
fn expand_sinh_sum(terms: &[Expr]) -> Expr {
  if terms.len() == 1 {
    return make_fn("Sinh", &[terms[0].clone()]);
  }
  let a = &terms[0];
  let rest = &terms[1..];
  // Sinh[a + rest] = Sinh[a]*Cosh[rest] + Cosh[a]*Sinh[rest]
  make_plus(
    &make_times(&make_fn("Sinh", &[a.clone()]), &expand_cosh_sum(rest)),
    &make_times(&make_fn("Cosh", &[a.clone()]), &expand_sinh_sum(rest)),
  )
}

/// Expand Cosh[a + b + ...] recursively.
fn expand_cosh_sum(terms: &[Expr]) -> Expr {
  if terms.len() == 1 {
    return make_fn("Cosh", &[terms[0].clone()]);
  }
  let a = &terms[0];
  let rest = &terms[1..];
  // Cosh[a + rest] = Cosh[a]*Cosh[rest] + Sinh[a]*Sinh[rest]
  make_plus(
    &make_times(&make_fn("Cosh", &[a.clone()]), &expand_cosh_sum(rest)),
    &make_times(&make_fn("Sinh", &[a.clone()]), &expand_sinh_sum(rest)),
  )
}

/// Expand Sinh[n*x] using the multiple-angle formula.
fn expand_sinh_multiple(n: usize, x: &Expr) -> Expr {
  let sinh_x = make_fn("Sinh", &[x.clone()]);
  let cosh_x = make_fn("Cosh", &[x.clone()]);
  let mut terms = Vec::new();

  // Sinh[n*x] = Sum_{k} C(n, 2k+1) * cosh(x)^(n-2k-1) * sinh(x)^(2k+1)
  // (same as sin but without (-1)^k)
  for k in 0..=(n - 1) / 2 {
    let binom =
      crate::functions::binomial_coeff(n as i128, (2 * k + 1) as i128);
    let cosh_pow = n - 2 * k - 1;
    let sinh_pow = 2 * k + 1;

    let mut term = Expr::Integer(binom);
    if cosh_pow > 0 {
      term = make_times(&term, &make_power(&cosh_x, cosh_pow as i128));
    }
    if sinh_pow > 0 {
      term = make_times(&term, &make_power(&sinh_x, sinh_pow as i128));
    }
    terms.push(term);
  }

  build_sum(&terms)
}

/// Expand Cosh[n*x] using the multiple-angle formula.
fn expand_cosh_multiple(n: usize, x: &Expr) -> Expr {
  let sinh_x = make_fn("Sinh", &[x.clone()]);
  let cosh_x = make_fn("Cosh", &[x.clone()]);
  let mut terms = Vec::new();

  // Cosh[n*x] = Sum_{k} C(n, 2k) * cosh(x)^(n-2k) * sinh(x)^(2k)
  // (same as cos but without (-1)^k)
  for k in 0..=n / 2 {
    let binom = crate::functions::binomial_coeff(n as i128, (2 * k) as i128);
    let cosh_pow = n - 2 * k;
    let sinh_pow = 2 * k;

    let mut term = Expr::Integer(binom);
    if cosh_pow > 0 {
      term = make_times(&term, &make_power(&cosh_x, cosh_pow as i128));
    }
    if sinh_pow > 0 {
      term = make_times(&term, &make_power(&sinh_x, sinh_pow as i128));
    }
    terms.push(term);
  }

  build_sum(&terms)
}

// ─── AST building helpers ───────────────────────────────────────────

fn make_fn(name: &str, args: &[Expr]) -> Expr {
  unevaluated(name, args)
}

fn make_times(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![a.clone(), b.clone()].into(),
  }
}

fn make_plus(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![a.clone(), b.clone()].into(),
  }
}

fn make_minus(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      a.clone(),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), b.clone()].into(),
      },
    ]
    .into(),
  }
}

fn make_divide(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      a.clone(),
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![b.clone(), Expr::Integer(-1)].into(),
      },
    ]
    .into(),
  }
}

fn make_power(base: &Expr, exp: i128) -> Expr {
  if exp == 1 {
    base.clone()
  } else {
    Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![base.clone(), Expr::Integer(exp)].into(),
    }
  }
}

fn build_sum(terms: &[Expr]) -> Expr {
  if terms.is_empty() {
    return Expr::Integer(0);
  }
  if terms.len() == 1 {
    return terms[0].clone();
  }
  unevaluated("Plus", terms)
}

// ─── TrigReduce ─────────────────────────────────────────────────────────

/// TrigReduce[expr] — rewrite products and powers of trig functions as
/// sums of trig functions with linear arguments.
pub fn trig_reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(unevaluated("TrigReduce", args));
  }
  // A single reduction pass can leave a residual product of trig functions:
  // reducing Cos[x]^2 to (1 + Cos[2x])/2 and distributing over Sin[x] yields
  // (Sin[x] + Cos[2x] Sin[x])/2, whose Cos[2x] Sin[x] must be reduced again.
  // Iterate to a fixed point so the result is fully linearized, e.g.
  // (Sin[x] + Sin[3x])/4.
  let mut current = args[0].clone();
  for _ in 0..8 {
    let reduced = trig_reduce_recursive(&current);
    let evaluated = crate::evaluator::evaluate_expr_to_expr(&reduced)?;
    // Expand so any product left factored by a power reduction — e.g.
    // (1 + Cos[2x])/2 * Sin[x] — is distributed into Cos[2x] Sin[x] + …, then
    // recombine into a single fraction whose numerator carries the product as
    // a clean term. The next pass's recursion reduces that Cos[2x] Sin[x].
    let expanded =
      crate::evaluator::evaluate_function_call_ast("Expand", &[evaluated])?;
    let combined =
      crate::evaluator::evaluate_function_call_ast("Together", &[expanded])?;
    if crate::syntax::expr_to_string(&combined)
      == crate::syntax::expr_to_string(&current)
    {
      break;
    }
    current = combined;
  }
  // Combine the per-term reductions: a sum like Sin[x]^2 + Cos[x]^2 reduces to
  // (1 + Cos[2x])/2 + (1 - Cos[2x])/2, which should collapse to 1. Together
  // merges the fractions and Cancel reduces the result, while an
  // already-combined single fraction (e.g. (Cos[x-y] - Cos[x+y])/2) is left
  // intact.
  let combined =
    crate::evaluator::evaluate_function_call_ast("Together", &[current])?;
  crate::evaluator::evaluate_function_call_ast("Cancel", &[combined])
}

/// Recursively apply TrigReduce.
fn trig_reduce_recursive(expr: &Expr) -> Expr {
  match expr {
    // Handle Times[...] — look for products of trig functions
    Expr::FunctionCall { name, args } if name == "Times" => {
      trig_reduce_product(args)
    }
    // Handle BinaryOp Times — two-factor product
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let l = trig_reduce_recursive(left);
      let r = trig_reduce_recursive(right);
      reduce_two_factor_product(&l, &r)
    }
    // Handle Power[Sin[x], n] or Power[Cos[x], n]
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let base = trig_reduce_recursive(left);
      if let Expr::Integer(n) = right.as_ref()
        && *n >= 2
        && let Some(result) = reduce_trig_power(&base, *n)
      {
        return result;
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(trig_reduce_recursive(right)),
      }
    }
    // Recurse into other structures
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(trig_reduce_recursive).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(trig_reduce_recursive(left)),
      right: Box::new(trig_reduce_recursive(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(trig_reduce_recursive(operand)),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(trig_reduce_recursive).collect())
    }
    _ => expr.clone(),
  }
}

/// Handle Times[a1, a2, ...] — find pairs of trig functions to reduce.
fn trig_reduce_product(factors: &[Expr]) -> Expr {
  let mut reduced: Vec<Expr> =
    factors.iter().map(trig_reduce_recursive).collect();

  // Repeatedly try to find and reduce pairs of trig functions
  let mut changed = true;
  while changed {
    changed = false;
    for i in 0..reduced.len() {
      for j in (i + 1)..reduced.len() {
        if let Some(result) = try_reduce_trig_pair(&reduced[i], &reduced[j]) {
          // Replace i with the result, remove j
          let mut new_factors = Vec::new();
          for (k, f) in reduced.iter().enumerate() {
            if k == i {
              new_factors.push(result.clone());
            } else if k != j {
              new_factors.push(f.clone());
            }
          }
          reduced = new_factors;
          changed = true;
          break;
        }
      }
      if changed {
        break;
      }
    }
  }

  if reduced.len() == 1 {
    reduced.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: reduced.into(),
    }
  }
}

/// Try to reduce product of two terms using product-to-sum identities.
fn reduce_two_factor_product(a: &Expr, b: &Expr) -> Expr {
  if let Some(result) = try_reduce_trig_pair(a, b) {
    return result;
  }
  // Check if either factor is a trig power
  if let Some(result) = try_reduce_with_power(a, b) {
    return result;
  }
  if let Some(result) = try_reduce_with_power(b, a) {
    return result;
  }
  Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(a.clone()),
    right: Box::new(b.clone()),
  }
}

/// Try product-to-sum for Sin[a]*Cos[b], Sin[a]*Sin[b], Cos[a]*Cos[b].
fn try_reduce_trig_pair(a: &Expr, b: &Expr) -> Option<Expr> {
  let (a_name, a_arg) = extract_trig(a)?;
  let (b_name, b_arg) = extract_trig(b)?;

  let sum = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(a_arg.clone()),
    right: Box::new(b_arg.clone()),
  };
  let diff = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(a_arg.clone()),
    right: Box::new(b_arg.clone()),
  };

  let half = |e: Expr| -> Expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(e),
      right: Box::new(Expr::Integer(2)),
    }
  };

  let sin = |e: Expr| -> Expr {
    Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![e].into(),
    }
  };
  let cos = |e: Expr| -> Expr {
    Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![e].into(),
    }
  };

  match (a_name, b_name) {
    // Sin[a]*Cos[b] = (Sin[a+b] + Sin[a-b])/2
    ("Sin", "Cos") | ("Cos", "Sin") => {
      let (sin_arg, cos_arg) = if a_name == "Sin" {
        (a_arg, b_arg)
      } else {
        (b_arg, a_arg)
      };
      let s = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(sin_arg.clone()),
        right: Box::new(cos_arg.clone()),
      };
      let d = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(sin_arg),
        right: Box::new(cos_arg),
      };
      let result = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(sin(s)),
        right: Box::new(sin(d)),
      };
      Some(half(result))
    }
    // Cos[a]*Cos[b] = (Cos[a-b] + Cos[a+b])/2
    ("Cos", "Cos") => {
      let result = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(cos(diff)),
        right: Box::new(cos(sum)),
      };
      Some(half(result))
    }
    // Sin[a]*Sin[b] = (Cos[a-b] - Cos[a+b])/2
    ("Sin", "Sin") => {
      let result = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(cos(diff)),
        right: Box::new(cos(sum)),
      };
      Some(half(result))
    }
    _ => None,
  }
}

/// Reduce trig power: Sin[x]^n or Cos[x]^n using direct formulas.
fn reduce_trig_power(base: &Expr, n: i128) -> Option<Expr> {
  let (trig_name, arg) = extract_trig(base)?;
  if !matches!(trig_name, "Sin" | "Cos") {
    return None;
  }

  // Use the power-reduction formulas directly:
  // For even n:
  //   cos^n(x) = 1/2^n * C(n,n/2) + 2/2^n * sum_{k=0}^{n/2-1} C(n,k) cos((n-2k)x)
  //   sin^n(x) = 1/2^n * C(n,n/2) + 2/2^n * sum_{k=0}^{n/2-1} (-1)^(n/2-k) C(n,k) cos((n-2k)x)
  // For odd n:
  //   cos^n(x) = 2/2^n * sum_{k=0}^{(n-1)/2} C(n,k) cos((n-2k)x)
  //   sin^n(x) = 2/2^n * sum_{k=0}^{(n-1)/2} (-1)^((n-1)/2-k) C(n,k) sin((n-2k)x)

  let is_cos = trig_name == "Cos";
  let nu = n as usize;
  let denom = 1i128 << n; // 2^n

  let make_int_times_arg = |coeff: i128, m: i128| -> Expr {
    if m == 1 {
      arg.clone()
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(coeff)),
        right: Box::new(arg.clone()),
      }
    }
  };

  // Collect (numerator_coeff, trig_expr_or_None_for_constant) pairs
  let mut num_terms: Vec<(i128, Option<Expr>)> = Vec::new();

  let half_n = nu / 2;
  let trig_fn = if nu.is_multiple_of(2) {
    // Even power
    // Constant term: C(n, n/2)
    let binom = crate::functions::binomial_coeff(n, half_n as i128);
    num_terms.push((binom, None));
    "Cos"
  } else {
    // Odd power
    if is_cos { "Cos" } else { "Sin" }
  };

  for k in 0..nu - half_n {
    let m = n - 2 * k as i128;
    let binom_val = crate::functions::binomial_coeff(n, k as i128);
    let coeff = 2 * binom_val;
    let sign = if is_cos || (half_n - k).is_multiple_of(2) {
      1
    } else {
      -1
    };
    let trig_arg = make_int_times_arg(m, m);
    let trig_call = Expr::FunctionCall {
      name: trig_fn.to_string(),
      args: vec![trig_arg].into(),
    };
    num_terms.push((sign * coeff, Some(trig_call)));
  }

  // Compute GCD of all numerator coefficients and denom to simplify the fraction
  let mut g = denom;
  for (c, _) in &num_terms {
    g = gcd_i128(g, *c);
  }
  let reduced_denom = denom / g;

  // Build numerator terms with reduced coefficients
  let mut terms: Vec<Expr> = Vec::new();
  for (c, trig_opt) in num_terms {
    let reduced_c = c / g;
    match trig_opt {
      None => {
        // Constant term
        terms.push(Expr::Integer(reduced_c));
      }
      Some(trig_call) => {
        if reduced_c == 1 {
          terms.push(trig_call);
        } else if reduced_c == -1 {
          terms.push(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(trig_call),
          });
        } else {
          terms.push(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(reduced_c)),
            right: Box::new(trig_call),
          });
        }
      }
    }
  }

  let numerator = if terms.len() == 1 {
    terms.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    }
  };

  if reduced_denom == 1 {
    Some(numerator)
  } else {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(Expr::Integer(reduced_denom)),
    })
  }
}

fn try_reduce_with_power(a: &Expr, b: &Expr) -> Option<Expr> {
  // Check if a is Sin[x]^n or Cos[x]^n
  if let Expr::BinaryOp {
    op: BinaryOperator::Power,
    left,
    right,
  } = a
    && let Expr::Integer(n) = right.as_ref()
    && *n >= 2
    && let Some(reduced_power) = reduce_trig_power(left, *n)
  {
    return Some(reduce_two_factor_product(&reduced_power, b));
  }
  None
}

/// Extract trig function name and argument.
fn extract_trig(expr: &Expr) -> Option<(&str, Expr)> {
  match expr {
    Expr::FunctionCall { name, args }
      if args.len() == 1
        && matches!(
          name.as_str(),
          "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc"
        ) =>
    {
      Some((name.as_str(), args[0].clone()))
    }
    _ => None,
  }
}

/// Returns true if the expression contains a Real (machine-precision)
/// number anywhere in the tree. Used to distinguish exact symbolic
/// arguments from inexact ones that should evaluate numerically.
fn contains_inexact_real_log(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::UnaryOp { operand, .. } => contains_inexact_real_log(operand),
    Expr::BinaryOp { left, right, .. } => {
      contains_inexact_real_log(left) || contains_inexact_real_log(right)
    }
    Expr::FunctionCall { args, .. } => {
      args.iter().any(contains_inexact_real_log)
    }
    _ => false,
  }
}
