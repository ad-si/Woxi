#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

/// Check if an expression is known to be non-negative without assumptions.
/// Used for simplifications like Sqrt[x^2] → x (only valid when x >= 0).
fn is_known_non_negative(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(n) => *n >= 0,
    Expr::Real(f) => *f >= 0.0,
    // Known positive constants
    Expr::Constant(name) => matches!(
      name.as_str(),
      "Pi"
        | "E"
        | "EulerGamma"
        | "GoldenRatio"
        | "Degree"
        | "Catalan"
        | "Glaisher"
        | "Khinchin"
    ),
    // Abs[anything] is always non-negative
    Expr::FunctionCall { name, args } if name == "Abs" && args.len() == 1 => {
      true
    }
    // Sqrt[anything] is always non-negative (for real results)
    expr if is_sqrt(expr).is_some() => true,
    _ => false,
  }
}

/// Abs[x] - Absolute value
pub fn abs_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Abs expects exactly 1 argument".into(),
    ));
  }
  // Abs[Undefined] = Undefined
  if matches!(&args[0], Expr::Identifier(s) if s == "Undefined") {
    return Ok(Expr::Identifier("Undefined".to_string()));
  }
  // Abs[Interval[{a, b}, ...]] → the interval of absolute values.
  if let Some(result) = crate::functions::interval_ast::abs_interval(&args[0]) {
    return Ok(result);
  }
  // Handle any expression containing Infinity → Infinity
  if contains_infinity(&args[0]) {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }
  // Handle integers and reals directly
  match &args[0] {
    Expr::Integer(n) => {
      // `checked_abs` returns None for `i128::MIN` (e.g. `Abs[-2^127]`),
      // whose magnitude `2^127` is not representable as an i128; promote
      // to a BigInteger in that case.
      return Ok(match n.checked_abs() {
        Some(a) => Expr::Integer(a),
        None => Expr::BigInteger(-num_bigint::BigInt::from(*n)),
      });
    }
    Expr::Real(f) => return Ok(Expr::Real(f.abs())),
    _ => {}
  }
  // Abs[known positive real] = x (symbolic, no numeric conversion):
  // Pi, E, GoldenRatio, Sqrt[n>0], positive^anything, etc.
  if crate::functions::math_ast::complex::is_strictly_positive_real(&args[0]) {
    return Ok(args[0].clone());
  }
  // Abs[Conjugate[x]] = Abs[x]
  if let Expr::FunctionCall { name, args: cargs } = &args[0]
    && name == "Conjugate"
    && cargs.len() == 1
  {
    return abs_ast(&[cargs[0].clone()]);
  }
  // Abs[Abs[x]] = Abs[x] (idempotent).
  if let Expr::FunctionCall { name, args: inner } = &args[0]
    && name == "Abs"
    && inner.len() == 1
  {
    return Ok(args[0].clone());
  }
  // Abs[Times[...]]: pull out real-constant factors as their magnitude and
  // drop the unit-modulus factor I (|I| = 1), keeping the remaining factors
  // under a single Abs. E.g. Abs[-2 x] = 2 Abs[x], Abs[I x] = Abs[x],
  // Abs[-Pi] = Pi. Matches wolframscript.
  {
    let mut factors: Vec<&Expr> = Vec::new();
    let is_times = collect_times_factors(&args[0], &mut factors);
    if is_times && factors.len() >= 2 {
      let mut pulled: Vec<Expr> = Vec::new();
      let mut kept: Vec<Expr> = Vec::new();
      let mut simplified = false;
      for f in &factors {
        if crate::functions::math_ast::complex::is_strictly_positive_real(f) {
          pulled.push((*f).clone());
          simplified = true;
        } else if let Some(absval) = negative_literal_abs(f) {
          pulled.push(absval);
          simplified = true;
        } else if is_imaginary_unit(f) {
          simplified = true; // |I| = 1, drop it
        } else {
          kept.push((*f).clone());
        }
      }
      if simplified {
        let mut all = pulled;
        if !kept.is_empty() {
          let kept_prod = if kept.len() == 1 {
            kept.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: kept.into(),
            }
          };
          all.push(Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![kept_prod].into(),
          });
        }
        let result = match all.len() {
          0 => Expr::Integer(1),
          1 => all.into_iter().next().unwrap(),
          _ => Expr::FunctionCall {
            name: "Times".to_string(),
            args: all.into(),
          },
        };
        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
    }
  }
  // Abs[b^z] = b^Re[z] for a strictly-positive real base b. Since
  // b^z = E^(z Log b) with Log b real, the magnitude is b^Re(z). This covers
  // complex/symbolic exponents the real-exponent rule below can't reach,
  // e.g. Abs[E^(2 I)] = 1, Abs[E^(2 + 3 I)] = E^2, Abs[2^(I x)] = 2^(-Im[x]).
  if let Some((base, exp)) = as_power(&args[0])
    && crate::functions::math_ast::complex::is_strictly_positive_real(base)
  {
    let re_exp = Expr::FunctionCall {
      name: "Re".to_string(),
      args: vec![exp.clone()].into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![base.clone(), re_exp].into(),
    });
  }
  // Abs[base^exp] = Abs[base]^exp for a real numeric exponent (|z^n| = |z|^n).
  if let Some((base, exp)) = power_with_real_exponent(&args[0]) {
    let abs_base = abs_ast(&[base.clone()])?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![abs_base, exp.clone()].into(),
    });
  }
  // Handle exact complex numbers and rationals: Abs[a + b*I] = Sqrt[a^2 + b^2]
  if let Some(((rn, rd), (in_, id))) = try_extract_complex_exact(&args[0]) {
    let g_r = gcd(rn, rd);
    let (rn, rd) = if rd < 0 {
      (-rn / g_r, -rd / g_r)
    } else {
      (rn / g_r, rd / g_r)
    };
    let g_i = gcd(in_, id);
    let (in_, id) = if id < 0 {
      (-in_ / g_i, -id / g_i)
    } else {
      (in_ / g_i, id / g_i)
    };
    if in_ == 0 {
      // Pure real
      return Ok(make_rational(rn.abs(), rd));
    }
    // Abs = Sqrt[(rn/rd)^2 + (in_/id)^2]
    // = Sqrt[(rn^2 * id^2 + in_^2 * rd^2) / (rd^2 * id^2)]
    if let (Some(num2), Some(den2)) = (
      rn.checked_mul(rn)
        .and_then(|a| id.checked_mul(id).and_then(|b| a.checked_mul(b)))
        .and_then(|a| {
          in_
            .checked_mul(in_)
            .and_then(|c| rd.checked_mul(rd).and_then(|d| c.checked_mul(d)))
            .and_then(|b| a.checked_add(b))
        }),
      rd.checked_mul(rd)
        .and_then(|a| id.checked_mul(id).and_then(|b| a.checked_mul(b))),
    ) {
      // Check if num2 is a perfect square
      let num_sqrt = (num2 as f64).sqrt().round() as i128;
      let den_sqrt = (den2 as f64).sqrt().round() as i128;
      if num_sqrt * num_sqrt == num2 && den_sqrt * den_sqrt == den2 {
        return Ok(make_rational(num_sqrt, den_sqrt));
      }
      // Sqrt[num2/den2], simplified so perfect-square factors come out of the
      // radical (e.g. Abs[2 + 2 I] = Sqrt[8] = 2 Sqrt[2]), matching wolframscript.
      return sqrt_ast(&[make_rational(num2, den2)]);
    }
  }
  // Handle floating-point complex numbers: Abs[3.0 + I] = sqrt(10)
  if let Some((re, im)) = try_extract_complex_f64(&args[0])
    && im != 0.0
  {
    return Ok(num_to_expr((re * re + im * im).sqrt()));
  }
  // Fallback for a real-valued numeric expression that wasn't simplified
  // above (e.g. a sum like Sqrt[2] - 3): |x| exactly. Negative values are
  // negated (Abs[Sqrt[2] - 3] -> 3 - Sqrt[2]); non-negative values are
  // returned unchanged (Abs[Sqrt[2] + 1] -> 1 + Sqrt[2]). Floatifying here
  // would lose exactness. A Real-leaf expression still collapses to a Real
  // because the negation/identity re-evaluates the original arithmetic.
  if let Some(v) = try_eval_to_f64(&args[0]) {
    if v < 0.0 {
      let neg = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), args[0].clone()].into(),
      };
      return crate::evaluator::evaluate_expr_to_expr(&neg);
    }
    return Ok(args[0].clone());
  }
  Ok(Expr::FunctionCall {
    name: "Abs".to_string(),
    args: args.to_vec().into(),
  })
}

/// True iff `expr` mentions the imaginary unit `I` (either as the symbol
/// `I` or the literal `Complex[0, 1]`). Used to recognise expressions that
/// are syntactically real (no I anywhere) so unit-magnitude shortcuts in
/// `Sign` can fire safely.
fn mentions_imaginary_unit(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(s) | Expr::Constant(s) => s == "I",
    Expr::FunctionCall { name, args } => {
      (name == "Complex"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(0))
        && matches!(&args[1], Expr::Integer(1)))
        || args.iter().any(mentions_imaginary_unit)
    }
    Expr::List(items) => items.iter().any(mentions_imaginary_unit),
    Expr::BinaryOp { left, right, .. } => {
      mentions_imaginary_unit(left) || mentions_imaginary_unit(right)
    }
    Expr::UnaryOp { operand, .. } => mentions_imaginary_unit(operand),
    _ => false,
  }
}

/// If `f` is a negative numeric literal (Integer, Real, or Rational), return
/// its absolute value as an exact Expr; otherwise None. Used by `Abs` to pull
/// the magnitude of a negative coefficient out of a product.
fn negative_literal_abs(f: &Expr) -> Option<Expr> {
  match f {
    Expr::Integer(n) if *n < 0 => Some(Expr::Integer(-*n)),
    Expr::Real(x) if *x < 0.0 => Some(Expr::Real(-*x)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&args[0], &args[1])
        && ((*p < 0) ^ (*q < 0))
      {
        Some(make_rational(p.abs(), q.abs()))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// If `expr` is a power `base^exp` (in either Power form) whose exponent is a
/// real numeric literal (Integer, Real, or Rational — not symbolic, not
/// complex), return `(base, exp)`. Used by `Abs`/`Sign` for the rule
/// `f[base^exp] = f[base]^exp`, valid for any base when exp is real.
/// Extract `(base, exponent)` from a power expression in either the BinaryOp
/// or FunctionCall["Power", …] representation, for any exponent.
fn as_power(expr: &Expr) -> Option<(&Expr, &Expr)> {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => Some((left.as_ref(), right.as_ref())),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      Some((&args[0], &args[1]))
    }
    _ => None,
  }
}

fn power_with_real_exponent(expr: &Expr) -> Option<(&Expr, &Expr)> {
  let (base, exp) = match expr {
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
  let exp_is_real_numeric = match exp {
    Expr::Integer(_) | Expr::Real(_) => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!((&args[0], &args[1]), (Expr::Integer(_), Expr::Integer(_)))
    }
    _ => false,
  };
  if exp_is_real_numeric {
    Some((base, exp))
  } else {
    None
  }
}

/// True iff `expr` evaluates to the imaginary unit `I` itself.
fn is_imaginary_unit(expr: &Expr) -> bool {
  matches!(expr, Expr::Identifier(s) | Expr::Constant(s) if s == "I")
    || matches!(
      expr,
      Expr::FunctionCall { name, args }
        if name == "Complex"
          && args.len() == 2
          && matches!(&args[0], Expr::Integer(0))
          && matches!(&args[1], Expr::Integer(1))
    )
}

/// True iff `expr` is a positive real literal (Integer ≥ 1, BigInteger,
/// positive Real, positive Rational). These have a real-valued logarithm,
/// so `posreal^(b I)` lies on the unit circle whenever `b` is real.
fn is_positive_real_literal(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(n) => *n > 0,
    Expr::BigInteger(n) => n > &num_bigint::BigInt::from(0),
    Expr::Real(f) => *f > 0.0,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      (matches!(&args[0], Expr::Integer(n) if *n > 0)
        && matches!(&args[1], Expr::Integer(n) if *n > 0))
    }
    _ => false,
  }
}

/// Collect a Times product into a flat list of factors. Handles both the
/// `FunctionCall { name: "Times", args }` form and the binary-operator
/// tree form `BinaryOp { op: Times, left, right }`.
fn collect_times_factors<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) -> bool {
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args {
        collect_times_factors(a, out);
      }
      true
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      collect_times_factors(left, out);
      collect_times_factors(right, out);
      true
    }
    _ => {
      out.push(expr);
      false
    }
  }
}

/// True iff `expr` is a value with zero real part and a non-zero
/// imaginary part — i.e. it lies on the imaginary axis. Recognised forms:
///   - `I` itself
///   - `Complex[0, n]` (the canonical pure-imaginary literal)
///   - `Times[..., I, ...]` where every non-`I` factor is syntactically
///     real (mentions no `I`).
fn is_purely_imaginary_product(expr: &Expr) -> bool {
  if is_imaginary_unit(expr) {
    return true;
  }
  if let Expr::FunctionCall { name, args } = expr
    && name == "Complex"
    && args.len() == 2
    && matches!(&args[0], Expr::Integer(0))
    && !matches!(&args[1], Expr::Integer(0))
  {
    return true;
  }
  // Try to flatten a Times product (FunctionCall or BinaryOp). If the
  // expression isn't a product, give up.
  let mut factors: Vec<&Expr> = Vec::new();
  let is_times = collect_times_factors(expr, &mut factors);
  if !is_times {
    return false;
  }
  let mut i_count = 0;
  for f in &factors {
    if is_imaginary_unit(f) {
      i_count += 1;
    } else if mentions_imaginary_unit(f) {
      return false;
    }
  }
  i_count == 1
}

/// Sign[x] - Sign of a number (-1, 0, or 1)
/// For complex z: Sign[z] = z / Abs[z]
pub fn sign_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sign expects exactly 1 argument".into(),
    ));
  }
  // Sign of a unit-magnitude power is the power itself:
  //   Sign[I^realexp]    → I^realexp     (|I^realexp| = 1)
  //   Sign[posreal^(b I)] → posreal^(b I) (|a^(b I)| = 1 for a>0, b real)
  // Detected structurally: I^x where x mentions no I, or a^(c*I) where a
  // is a positive real literal and c is real (i.e. the exponent is a Times
  // product whose only I-mentioning factor is I itself).
  let power_parts: Option<(&Expr, &Expr)> = match &args[0] {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => Some((left.as_ref(), right.as_ref())),
    Expr::FunctionCall { name, args: pargs }
      if name == "Power" && pargs.len() == 2 =>
    {
      Some((&pargs[0], &pargs[1]))
    }
    _ => None,
  };
  if let Some((base, exp)) = power_parts
    && ((is_imaginary_unit(base) && !mentions_imaginary_unit(exp))
      || (is_positive_real_literal(base) && is_purely_imaginary_product(exp)))
  {
    return Ok(args[0].clone());
  }
  // Handle Infinity, -Infinity, ComplexInfinity, Indeterminate
  if matches!(&args[0], Expr::Identifier(s) if s == "Infinity") {
    return Ok(Expr::Integer(1));
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "ComplexInfinity") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // Check for -Infinity (UnaryOp::Minus applied to Infinity)
  if let Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand,
  } = &args[0]
    && matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity")
  {
    return Ok(Expr::Integer(-1));
  }
  // Check for negative infinity: Times[n, Infinity] where n < 0
  if let Expr::FunctionCall { name, args: fargs } = &args[0]
    && name == "Times"
  {
    let has_infinity = fargs
      .iter()
      .any(|a| matches!(a, Expr::Identifier(s) if s == "Infinity"));
    if has_infinity {
      let coeff = fargs.iter().find_map(|a| {
        if let Expr::Integer(n) = a {
          Some(*n as f64)
        } else {
          try_eval_to_f64(a)
        }
      });
      if let Some(c) = coeff {
        if c < 0.0 {
          return Ok(Expr::Integer(-1));
        } else if c > 0.0 {
          return Ok(Expr::Integer(1));
        }
      }
    }
  }
  if let Some(n) = try_eval_to_f64(&args[0]) {
    return Ok(Expr::Integer(if n > 0.0 {
      1
    } else if n < 0.0 {
      -1
    } else {
      0
    }));
  }
  // `Sign[Times[r1, r2, ..., z]]` where every `ri` is a positive
  // *exact* real factor and `z` is an exact complex literal: pull the
  // reals out and recurse on the lone complex factor.
  // |Times[r..., z]| = (r1...rk) |z|, so the product divided by its
  // magnitude equals `z / |z| = Sign[z]`. Inexact factors (Real /
  // BigFloat) are intentionally excluded so the inexact-complex
  // numerical fallback can still produce a Real-coefficient answer.
  fn is_exact_positive(f: &Expr) -> bool {
    fn inner(f: &Expr) -> bool {
      match f {
        Expr::Integer(n) => *n > 0,
        Expr::FunctionCall { name, args }
          if name == "Rational" && args.len() == 2 =>
        {
          matches!(&args[0], Expr::Integer(n) if *n > 0)
            && matches!(&args[1], Expr::Integer(d) if *d > 0)
        }
        // `Power[positive_exact, exact_real]` — `2^(-1/2)` etc. The base
        // must be strictly positive; no non-real exponent (which would
        // introduce an imaginary phase). We don't recurse into the
        // exponent's positivity since real-valued exponents on a
        // positive base always give a positive result.
        Expr::FunctionCall { name, args }
          if name == "Power" && args.len() == 2 =>
        {
          inner(&args[0]) && {
            // Exponent must not contain `I` to keep the result real.
            !mentions_imaginary_unit(&args[1])
          }
        }
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left,
          right,
        } => inner(left) && !mentions_imaginary_unit(right),
        _ => is_positive_real_literal(f) && !matches!(f, Expr::Real(_)),
      }
    }
    inner(f)
  }
  if let Expr::FunctionCall {
    name,
    args: factors,
  } = &args[0]
    && name == "Times"
    && factors.len() >= 2
  {
    let mut complex_factor: Option<Expr> = None;
    let mut all_positive_reals_or_one_complex = true;
    for f in factors {
      if try_extract_complex_exact(f).is_some()
        && !matches!(f, Expr::Integer(_))
        && !matches!(f, Expr::FunctionCall { name, .. } if name == "Rational")
      {
        if complex_factor.is_some() {
          all_positive_reals_or_one_complex = false;
          break;
        }
        complex_factor = Some(f.clone());
      } else if !is_exact_positive(f) {
        all_positive_reals_or_one_complex = false;
        break;
      }
    }
    if all_positive_reals_or_one_complex && let Some(z) = complex_factor {
      return sign_ast(&[z]);
    }
  }
  // Handle complex numbers: Sign[a + b*I] = (a + b*I) / Abs[a + b*I]
  if let Some(((rn, rd), (in_, id))) = try_extract_complex_exact(&args[0]) {
    let g_r = gcd(rn, rd);
    let (rn, rd) = if rd < 0 {
      (-rn / g_r, -rd / g_r)
    } else {
      (rn / g_r, rd / g_r)
    };
    let g_i = gcd(in_, id);
    let (in_, id) = if id < 0 {
      (-in_ / g_i, -id / g_i)
    } else {
      (in_ / g_i, id / g_i)
    };
    if in_ == 0 {
      // Pure real - already handled above by try_eval_to_f64
      return Ok(Expr::Integer(if rn > 0 {
        1
      } else if rn < 0 {
        -1
      } else {
        0
      }));
    }
    // Compute |z|^2 = (rn/rd)^2 + (in_/id)^2 = (rn^2*id^2 + in_^2*rd^2) / (rd^2*id^2)
    if let (Some(abs2_num), Some(abs2_den)) = (
      rn.checked_mul(rn)
        .and_then(|a| id.checked_mul(id).and_then(|b| a.checked_mul(b)))
        .and_then(|a| {
          in_
            .checked_mul(in_)
            .and_then(|c| rd.checked_mul(rd).and_then(|d| c.checked_mul(d)))
            .and_then(|b| a.checked_add(b))
        }),
      rd.checked_mul(rd)
        .and_then(|a| id.checked_mul(id).and_then(|b| a.checked_mul(b))),
    ) {
      // Simplify |z|^2 fraction
      let g_abs = gcd(abs2_num.abs(), abs2_den.abs());
      let (abs2_n, abs2_d) = (abs2_num / g_abs, abs2_den / g_abs);

      // Check if |z|^2 is a perfect square (so |z| is rational)
      let abs2_sqrt = (abs2_n as f64).sqrt().round() as i128;
      let den_sqrt = (abs2_d as f64).sqrt().round() as i128;
      if abs2_sqrt * abs2_sqrt == abs2_n
        && den_sqrt * den_sqrt == abs2_d
        && abs2_sqrt != 0
      {
        // |z| = abs2_sqrt / den_sqrt
        // Sign = z / |z| = (rn/rd + (in_/id)*I) * (den_sqrt/abs2_sqrt)
        // Real part: (rn * den_sqrt) / (rd * abs2_sqrt)
        // Imag part: (in_ * den_sqrt) / (id * abs2_sqrt)
        let sign_re_num = rn.checked_mul(den_sqrt);
        let sign_re_den = rd.checked_mul(abs2_sqrt);
        let sign_im_num = in_.checked_mul(den_sqrt);
        let sign_im_den = id.checked_mul(abs2_sqrt);
        if let (Some(srn), Some(srd), Some(sin), Some(sid)) =
          (sign_re_num, sign_re_den, sign_im_num, sign_im_den)
        {
          return Ok(build_complex_expr(srn, srd, sin, sid));
        }
      } else {
        // |z| is irrational: Sign[z] = z / Sqrt[|z|^2]
        // = (a + b*I) / Sqrt[|z|^2]
        // Build (a + b*I) * (1/Sqrt[|z|^2])
        let z_expr = build_complex_expr(rn, rd, in_, id);
        let abs_expr = make_sqrt(make_rational(abs2_n, abs2_d));
        return Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(z_expr),
          right: Box::new(abs_expr),
        });
      }
    }
  }
  // Inexact complex (e.g. 1. + I, 2. + 3 I): produce a numerical Sign.
  // We trigger this only when the expression actually contains a Real or
  // BigFloat literal — otherwise leave symbolic forms like Sign[Pi + I]
  // alone for a future symbolic implementation.
  fn has_real_or_bigfloat(expr: &Expr) -> bool {
    match expr {
      Expr::Real(_) | Expr::BigFloat(_, _) => true,
      Expr::BinaryOp { left, right, .. } => {
        has_real_or_bigfloat(left) || has_real_or_bigfloat(right)
      }
      Expr::UnaryOp { operand, .. } => has_real_or_bigfloat(operand),
      Expr::FunctionCall { args, .. } | Expr::List(args) => {
        args.iter().any(has_real_or_bigfloat)
      }
      _ => false,
    }
  }
  if has_real_or_bigfloat(&args[0])
    && let Some((re, im)) = try_extract_complex_float(&args[0])
    && im != 0.0
  {
    let abs = (re * re + im * im).sqrt();
    if abs > 0.0 {
      return Ok(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Real(re / abs),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Real(im / abs), Expr::Identifier("I".to_string())]
              .into(),
          },
        ]
        .into(),
      });
    }
  }
  // Sign[Sign[x]] = Sign[x] (idempotent).
  if let Expr::FunctionCall { name, args: inner } = &args[0]
    && name == "Sign"
    && inner.len() == 1
  {
    return Ok(args[0].clone());
  }
  // Sign[Times[...]]: Sign is multiplicative. Pull the sign of each
  // real-constant factor out (positive → 1, negative → -1) and I (Sign[I] = I),
  // keeping the remaining factors under a single Sign. E.g. Sign[-2 x] =
  // -Sign[x], Sign[I x] = I Sign[x]. Matches wolframscript.
  {
    let mut factors: Vec<&Expr> = Vec::new();
    let is_times = collect_times_factors(&args[0], &mut factors);
    if is_times && factors.len() >= 2 {
      let mut pulled: Vec<Expr> = Vec::new();
      let mut kept: Vec<Expr> = Vec::new();
      let mut simplified = false;
      for f in &factors {
        if crate::functions::math_ast::complex::is_strictly_positive_real(f) {
          simplified = true; // Sign = 1, drop it
        } else if negative_literal_abs(f).is_some() {
          pulled.push(Expr::Integer(-1));
          simplified = true;
        } else if is_imaginary_unit(f) {
          pulled.push(Expr::Identifier("I".to_string()));
          simplified = true;
        } else {
          kept.push((*f).clone());
        }
      }
      if simplified {
        let mut all = pulled;
        if !kept.is_empty() {
          let kept_prod = if kept.len() == 1 {
            kept.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: kept.into(),
            }
          };
          all.push(Expr::FunctionCall {
            name: "Sign".to_string(),
            args: vec![kept_prod].into(),
          });
        }
        let result = match all.len() {
          0 => Expr::Integer(1),
          1 => all.into_iter().next().unwrap(),
          _ => Expr::FunctionCall {
            name: "Times".to_string(),
            args: all.into(),
          },
        };
        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
    }
  }
  // Sign[base^exp] = Sign[base]^exp for a real numeric exponent
  // (z^n / |z^n| = (z/|z|)^n).
  if let Some((base, exp)) = power_with_real_exponent(&args[0]) {
    let sign_base = sign_ast(&[base.clone()])?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![sign_base, exp.clone()].into(),
    });
  }
  Ok(Expr::FunctionCall {
    name: "Sign".to_string(),
    args: args.to_vec().into(),
  })
}

/// Extract the largest easily-found perfect-square factor from a non-negative
/// BigInt: returns `(outside, inside)` with `n == outside^2 * inside`, so
/// `Sqrt[n] = outside * Sqrt[inside]`. Square factors from primes below a
/// bound are pulled out by trial division; a leftover whole-square cofactor
/// (e.g. a large prime squared) is caught by a final `isqrt` check. A cofactor
/// that is square-free over the bound but not itself a perfect square stays
/// in `inside` (matching wolframscript, which also can't factor large semiprimes
/// cheaply).
fn extract_square_factor_big(
  n: &num_bigint::BigInt,
) -> (num_bigint::BigInt, num_bigint::BigInt) {
  use num_bigint::BigInt;
  let zero = BigInt::from(0);
  let one = BigInt::from(1);
  let mut outside = BigInt::from(1);
  let mut inside = n.clone();
  const BOUND: u64 = 100_000;
  let mut factor: u64 = 2;
  while factor <= BOUND {
    let fbig = BigInt::from(factor);
    if &fbig * &fbig > inside {
      break;
    }
    let mut count: u64 = 0;
    while &inside % &fbig == zero {
      inside /= &fbig;
      count += 1;
    }
    if count >= 2 {
      outside *= fbig.pow((count / 2) as u32);
    }
    if count % 2 == 1 {
      inside *= &fbig;
    }
    factor += 1;
  }
  if inside > one {
    let r = inside.sqrt();
    if &r * &r == inside {
      outside *= &r;
      inside = BigInt::from(1);
    }
  }
  (outside, inside)
}

/// Sqrt[x] - Square root
pub fn sqrt_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sqrt expects exactly 1 argument".into(),
    ));
  }
  // Sqrt is monotonic increasing on [0, ∞): map it over each interval span.
  if let Some(r) =
    crate::functions::interval_ast::map_monotonic_interval("Sqrt", &args[0])
  {
    return Ok(r);
  }
  // Strip a top-level Unevaluated wrapper before computing.
  if let Expr::FunctionCall { name, args: u_args } = &args[0]
    && name == "Unevaluated"
    && u_args.len() == 1
  {
    return sqrt_ast(&[u_args[0].clone()]);
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // Sqrt[Infinity] = Infinity
  if matches!(&args[0], Expr::Identifier(s) if s == "Infinity") {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }
  // Sqrt[ComplexInfinity] = ComplexInfinity
  if matches!(&args[0], Expr::Identifier(s) if s == "ComplexInfinity") {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  // Sqrt[I] / Sqrt[-I]: delegate to Power[base, 1/2] so the imaginary-unit
  // canonicalisation (Sqrt[I] = (-1)^(1/4), Sqrt[-I] = -(-1)^(3/4)) applies.
  if matches!(&args[0], Expr::Identifier(s) if s == "I")
    || matches!(&args[0],
      Expr::UnaryOp { op: crate::syntax::UnaryOperator::Minus, operand }
        if matches!(operand.as_ref(), Expr::Identifier(s) if s == "I"))
  {
    return crate::functions::math_ast::power_two(
      &args[0],
      &make_rational(1, 2),
    );
  }
  // Handle Sqrt[Quantity[mag, unit]] by delegating to Power[quantity, 1/2]
  if crate::functions::quantity_ast::is_quantity(&args[0]).is_some() {
    let half = make_rational(1, 2);
    if let Some(result) =
      crate::functions::quantity_ast::try_quantity_power(&args[0], &half)
    {
      return result;
    }
  }
  match &args[0] {
    // Perfect squares: Sqrt[0]=0, Sqrt[1]=1, Sqrt[4]=2, etc.
    Expr::Integer(n) if *n >= 0 => {
      // Exact perfect square via BigInt (f64 sqrt is imprecise near i128 max).
      let bn = num_bigint::BigInt::from(*n);
      let r = bn.sqrt();
      if &r * &r == bn {
        return Ok(crate::functions::math_ast::bigint_to_expr(r));
      }
      // Partial extraction (e.g. Sqrt[12] = 2*Sqrt[3]) runs in u64, so only
      // when n fits u64 — casting a larger i128 to u64 would truncate.
      if *n <= u64::MAX as i128 {
        let mut outside = 1u64;
        let mut inside = *n as u64;
        let mut factor = 2u64;
        while factor * factor <= inside {
          while inside.is_multiple_of(factor * factor) {
            outside *= factor;
            inside /= factor * factor;
          }
          factor += 1;
        }
        if outside > 1 && inside > 1 {
          return Ok(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(outside as i128),
              make_sqrt(Expr::Integer(inside as i128)),
            ]
            .into(),
          });
        }
      } else {
        // Larger than u64: extract square factors in BigInt.
        let (outside, inside) = extract_square_factor_big(&bn);
        if outside > num_bigint::BigInt::from(1) {
          let sqrt_part =
            make_sqrt(crate::functions::math_ast::bigint_to_expr(inside));
          return times_ast(&[
            crate::functions::math_ast::bigint_to_expr(outside),
            sqrt_part,
          ]);
        }
      }
      // Not a perfect square, return symbolic
      Ok(make_sqrt(args[0].clone()))
    }
    // Sqrt of a BigInteger: extract the largest perfect-square factor, e.g.
    // Sqrt[10^41] -> 10^20 Sqrt[10].
    Expr::BigInteger(n) if *n >= num_bigint::BigInt::from(0) => {
      let r = n.sqrt();
      if &r * &r == *n {
        return Ok(crate::functions::math_ast::bigint_to_expr(r));
      }
      let (outside, inside) = extract_square_factor_big(n);
      if outside > num_bigint::BigInt::from(1) {
        let sqrt_part =
          make_sqrt(crate::functions::math_ast::bigint_to_expr(inside));
        return times_ast(&[
          crate::functions::math_ast::bigint_to_expr(outside),
          sqrt_part,
        ]);
      }
      Ok(make_sqrt(args[0].clone()))
    }
    // Sqrt[Rational[a, b]] — simplify by extracting perfect square factors
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      // Only the u64 fast path here; a numerator/denominator that fits i128 but
      // exceeds u64 (e.g. 7*10^30) would truncate on `as u64`, so it falls
      // through to the BigInt path below.
      if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1])
        && *n >= 0
        && *d > 0
        && *n <= u64::MAX as i128
        && *d <= u64::MAX as i128
      {
        // Extract perfect square factors from numerator and denominator
        let extract_square_factor = |val: u64| -> (u64, u64) {
          let mut outside = 1u64;
          let mut inside = val;
          let mut factor = 2u64;
          while factor * factor <= inside {
            while inside.is_multiple_of(factor * factor) {
              outside *= factor;
              inside /= factor * factor;
            }
            factor += 1;
          }
          (outside, inside)
        };
        let (n_out, n_in) = extract_square_factor(*n as u64);
        let (d_out, d_in) = extract_square_factor(*d as u64);
        // Result: (n_out / d_out) * Sqrt[n_in / d_in]
        if n_in == 1 && d_in == 1 {
          // Perfect square: just a rational
          return Ok(make_rational(n_out as i128, d_out as i128));
        }
        if d_in == 1 {
          // Result: (n_out / d_out) * Sqrt[n_in]
          let sqrt_part = make_sqrt(Expr::Integer(n_in as i128));
          if n_out == 1 && d_out == 1 {
            return Ok(sqrt_part);
          }
          let coeff = make_rational(n_out as i128, d_out as i128);
          return times_ast(&[coeff, sqrt_part]);
        } else if n_in == 1 {
          // Result: (n_out / d_out) * Power[d_in, -1/2].
          // Use Power[d, -1/2] (Wolfram's normal form for Sqrt[1/d])
          // rather than BinaryOp Divide, so structural equality with
          // 1/Sqrt[d] (which evaluates to Power[d, -1/2]) holds.
          let power =
            power_ast(&[Expr::Integer(d_in as i128), make_rational(-1, 2)])?;
          if n_out == 1 && d_out == 1 {
            return Ok(power);
          }
          let coeff = make_rational(n_out as i128, d_out as i128);
          return times_ast(&[coeff, power]);
        } else {
          // General case: (n_out / d_out) * Sqrt[n_in / d_in]
          let sqrt_part = make_sqrt(make_rational(n_in as i128, d_in as i128));
          if n_out == 1 && d_out == 1 {
            return Ok(sqrt_part);
          }
          let coeff = make_rational(n_out as i128, d_out as i128);
          return times_ast(&[coeff, sqrt_part]);
        }
      }
      // BigInt fallback for rational arguments whose numerator/denominator is a
      // BigInteger or exceeds u64 (e.g. Sqrt[2744*10^90/729] = 14*10^45/27 *
      // Sqrt[14], reached via Skewness/Correlation of large-integer lists).
      let to_big = |e: &Expr| -> Option<num_bigint::BigInt> {
        match e {
          Expr::Integer(v) if *v >= 0 => Some(num_bigint::BigInt::from(*v)),
          Expr::BigInteger(v) if *v >= num_bigint::BigInt::from(0) => {
            Some(v.clone())
          }
          _ => None,
        }
      };
      if let (Some(nb), Some(db)) = (to_big(&rargs[0]), to_big(&rargs[1]))
        && db > num_bigint::BigInt::from(0)
      {
        use crate::functions::math_ast::{bigint_to_expr, make_rational_expr};
        let one = num_bigint::BigInt::from(1);
        let (n_out, n_in) = extract_square_factor_big(&nb);
        let (d_out, d_in) = extract_square_factor_big(&db);
        let coeff = make_rational_expr(n_out.clone(), d_out.clone());
        if n_in == one && d_in == one {
          return Ok(coeff);
        }
        let radicand = if d_in == one {
          make_sqrt(bigint_to_expr(n_in))
        } else if n_in == one {
          power_ast(&[bigint_to_expr(d_in), make_rational(-1, 2)])?
        } else {
          make_sqrt(make_rational_expr(n_in, d_in))
        };
        if matches!(&coeff, Expr::Integer(1)) {
          return Ok(radicand);
        }
        return times_ast(&[coeff, radicand]);
      }
      Ok(make_sqrt(args[0].clone()))
    }
    // Sqrt[-n] for negative integer → I * Sqrt[n]
    Expr::Integer(n) if *n < 0 => {
      let pos = -*n;
      let sqrt_pos = sqrt_ast(&[Expr::Integer(pos)])?;
      times_ast(&[Expr::Identifier("I".to_string()), sqrt_pos])
    }
    Expr::Real(f) if *f >= 0.0 => Ok(Expr::Real(f.sqrt())),
    // Sqrt[base^(2n)] → base^n only when base is known non-negative
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: base,
      right: exp,
    } if matches!(exp.as_ref(), Expr::Integer(n) if *n > 0 && n % 2 == 0)
      && is_known_non_negative(base) =>
    {
      if let Expr::Integer(n) = exp.as_ref() {
        let half = n / 2;
        if half == 1 {
          return Ok(*base.clone());
        } else {
          return Ok(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: base.clone(),
            right: Box::new(Expr::Integer(half)),
          });
        }
      }
      Ok(make_sqrt(args[0].clone()))
    }
    // Sqrt of a product: Sqrt[n * expr^2 * ...] → simplify factors
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      ..
    }
    | Expr::FunctionCall { name: _, args: _ }
      if matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Times") =>
    {
      let factors =
        crate::functions::polynomial_ast::collect_multiplicative_factors(
          &args[0],
        );
      // Separate into: integer part, squared symbolic factors, remainder
      let mut int_product: i128 = 1;
      let mut outside: Vec<Expr> = Vec::new(); // factors to move outside sqrt
      let mut inside: Vec<Expr> = Vec::new(); // factors to keep inside sqrt

      for f in &factors {
        match f {
          Expr::Integer(n) => {
            int_product *= n;
          }
          // expr^2 → move expr outside only if expr is known non-negative
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: base,
            right: exp,
          } if matches!(exp.as_ref(), Expr::Integer(2))
            && is_known_non_negative(base) =>
          {
            outside.push(*base.clone());
          }
          // expr^(2n) → move expr^n outside (for any even n, including negative)
          // only if expr is known non-negative
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: base,
            right: exp,
          } if matches!(exp.as_ref(), Expr::Integer(n) if n % 2 == 0 && *n != 0)
            && is_known_non_negative(base) =>
          {
            if let Expr::Integer(n) = exp.as_ref() {
              let half = n / 2;
              if half == 1 {
                outside.push(*base.clone());
              } else {
                outside.push(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Power,
                  left: base.clone(),
                  right: Box::new(Expr::Integer(half)),
                });
              }
            }
          }
          // Power[base, even_n] (FunctionCall form) → move base^(n/2) outside
          // only if base is known non-negative
          Expr::FunctionCall {
            name: pname,
            args: pargs,
          } if pname == "Power"
            && pargs.len() == 2
            && matches!(&pargs[1], Expr::Integer(n) if n % 2 == 0 && *n != 0)
            && is_known_non_negative(&pargs[0]) =>
          {
            if let Expr::Integer(n) = &pargs[1] {
              let half = n / 2;
              if half == 1 {
                outside.push(pargs[0].clone());
              } else {
                outside.push(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Power,
                  left: Box::new(pargs[0].clone()),
                  right: Box::new(Expr::Integer(half)),
                });
              }
            }
          }
          _ => {
            inside.push(f.clone());
          }
        }
      }

      // Handle the integer part: extract perfect square factors
      let sign = if int_product < 0 { -1i128 } else { 1i128 };
      let abs_int = (int_product * sign) as u64;
      let mut int_outside = 1u64;
      let mut int_inside = abs_int;
      let mut factor = 2u64;
      while factor * factor <= int_inside {
        while int_inside.is_multiple_of(factor * factor) {
          int_outside *= factor;
          int_inside /= factor * factor;
        }
        factor += 1;
      }

      if int_outside <= 1 && outside.is_empty() {
        // No simplification possible
        if let Some(result) = try_sqrt_gaussian(&args[0]) {
          return Ok(result);
        }
        return Ok(make_sqrt(args[0].clone()));
      }

      // Build outside factors
      if int_outside > 1 {
        outside.insert(0, Expr::Integer(int_outside as i128));
      }

      // Build inside factors
      if int_inside > 1 {
        inside.insert(0, Expr::Integer(int_inside as i128));
      }
      if sign < 0 {
        inside.insert(0, Expr::Integer(-1));
      }

      let outside_expr = if outside.len() == 1 {
        outside.remove(0)
      } else {
        crate::functions::polynomial_ast::build_product(outside)
      };

      if inside.is_empty() {
        Ok(outside_expr)
      } else {
        let inside_expr = if inside.len() == 1 {
          inside.remove(0)
        } else {
          crate::functions::polynomial_ast::build_product(inside)
        };
        let sqrt_part = make_sqrt(inside_expr);
        times_ast(&[outside_expr, sqrt_part])
      }
    }
    // Sqrt[Plus[...]] — extract perfect square GCD from integer coefficients.
    // E.g. Sqrt[4 + 36*t^2 + 36*t^4] → 2*Sqrt[1 + 9*t^2 + 9*t^4]
    _ if matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Plus") =>
    {
      if let Some(result) = try_sqrt_plus_gcd(&args[0]) {
        return Ok(result);
      }
      if let Some(result) = try_sqrt_gaussian(&args[0]) {
        return Ok(result);
      }
      Ok(make_sqrt(args[0].clone()))
    }
    // Sqrt[a + b*I] for Gaussian integers — try to find exact Gaussian integer sqrt
    _ => {
      if let Some(result) = try_sqrt_gaussian(&args[0]) {
        return Ok(result);
      }
      Ok(make_sqrt(args[0].clone()))
    }
  }
}

/// Extract the integer coefficient from a Plus term.
/// Returns (coefficient, base) where term = coefficient * base.
/// For Integer(n), returns (n, Integer(1)).
/// For Times[n, rest], returns (n, rest).
fn extract_int_coeff(term: &Expr) -> Option<(i128, Expr)> {
  match term {
    Expr::Integer(n) => Some((*n, Expr::Integer(1))),
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() >= 2
        && matches!(&args[0], Expr::Integer(_)) =>
    {
      if let Expr::Integer(n) = &args[0] {
        let base = if args.len() == 2 {
          args[1].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: args[1..].to_vec().into(),
          }
        };
        Some((*n, base))
      } else {
        None
      }
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let (c, base) = extract_int_coeff(operand)?;
      Some((-c, base))
    }
    _ => None,
  }
}

/// Try to simplify Sqrt[Plus[...]] by extracting the GCD of integer coefficients.
/// If the GCD is a perfect square, factor it out.
/// E.g. Sqrt[4 + 36*t^2 + 36*t^4] → 2*Sqrt[1 + 9*t^2 + 9*t^4]
fn try_sqrt_plus_gcd(expr: &Expr) -> Option<Expr> {
  use crate::functions::math_ast::numeric_utils::gcd;

  let terms = match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => args,
    _ => return None,
  };

  if terms.is_empty() {
    return None;
  }

  // Extract integer coefficients from each Plus term
  let mut pairs: Vec<(i128, Expr)> = Vec::new();
  for term in terms {
    match extract_int_coeff(term) {
      Some(pair) => pairs.push(pair),
      None => return None,
    }
  }

  // Compute GCD of absolute values of all coefficients
  let mut g = pairs[0].0.abs();
  for &(c, _) in &pairs[1..] {
    g = gcd(g, c.abs()).unsigned_abs() as i128;
    if g <= 1 {
      return None;
    }
  }

  if g <= 1 {
    return None;
  }

  // Find the largest perfect square factor of the GCD
  let mut sqrt_factor = 1i128;
  let mut remaining_g = g;
  let mut f = 2i128;
  while f * f <= remaining_g {
    while remaining_g % (f * f) == 0 {
      sqrt_factor *= f;
      remaining_g /= f * f;
    }
    f += 1;
  }

  if sqrt_factor <= 1 {
    return None;
  }

  // Factor to divide out from under the sqrt: sqrt_factor^2
  let factor_out = sqrt_factor * sqrt_factor;

  // Build new terms with coefficients divided by factor_out
  let mut new_terms: Vec<Expr> = Vec::new();
  for (coeff, base) in &pairs {
    let new_coeff = coeff / factor_out;
    if matches!(base, Expr::Integer(1)) {
      // Bare integer term
      new_terms.push(Expr::Integer(new_coeff));
    } else if new_coeff == 1 {
      new_terms.push(base.clone());
    } else if new_coeff == -1 {
      new_terms.push(super::trigonometric::negate_expr(base.clone()));
    } else {
      new_terms.push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(new_coeff), base.clone()].into(),
      });
    }
  }

  let new_sum = if new_terms.len() == 1 {
    new_terms.remove(0)
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: new_terms.into(),
    }
  };

  let sqrt_part = make_sqrt(new_sum);

  if sqrt_factor == 1 {
    Some(sqrt_part)
  } else {
    times_ast(&[Expr::Integer(sqrt_factor), sqrt_part]).ok()
  }
}

/// Try to compute an exact Gaussian integer square root of a complex number.
///
/// For `z = a + b*I` (Gaussian integer), finds `p + q*I` such that `(p+qI)^2 = z`,
/// choosing the principal root with `p >= 0` (or `p == 0, q > 0`).
/// Returns `None` if no such integer solution exists.
fn try_sqrt_gaussian(expr: &Expr) -> Option<Expr> {
  use crate::functions::math_ast::numeric_utils::try_extract_complex_exact;
  let ((rn, rd), (in_, id)) = try_extract_complex_exact(expr)?;
  // Normalize to integers a and b where z = a + b*I
  // Require both parts to be integers (denominator 1 after reducing)
  let g_r = crate::functions::math_ast::numeric_utils::gcd(rn, rd).abs();
  let (rn, rd) = (rn / g_r, rd / g_r);
  let g_i = crate::functions::math_ast::numeric_utils::gcd(in_, id).abs();
  let (in_, id) = (in_ / g_i, id / g_i);
  // Pure real cases are handled by existing code; skip to avoid duplication
  if in_ == 0 {
    return None;
  }
  if rd != 1 || id != 1 {
    return None; // Only handle Gaussian integers, not Gaussian rationals
  }
  let a = rn; // real part
  let b = in_; // imaginary part
  // |z|^2 = a^2 + b^2; needs to be a perfect square n^2
  let m = a.checked_mul(a)?.checked_add(b.checked_mul(b)?)?;
  let n = (m as f64).sqrt() as i128;
  if n * n != m {
    return None; // |z| is not an integer
  }
  // p^2 = (n + a) / 2; needs n + a to be even and non-negative
  let n_plus_a = n.checked_add(a)?;
  if n_plus_a < 0 || n_plus_a % 2 != 0 {
    return None;
  }
  let p_sq = n_plus_a / 2;
  let p = (p_sq as f64).sqrt() as i128;
  if p * p != p_sq {
    return None; // p is not an integer
  }
  // Determine q: q = b / (2*p)
  if p == 0 {
    // Handle p == 0: then a == -n <= 0 and b != 0 → q^2 = n, pick q > 0
    let q_sq = n;
    let q = (q_sq as f64).sqrt() as i128;
    if q * q != q_sq {
      return None;
    }
    // Choose q with same sign as b (principal root convention)
    let q = if b < 0 { -q } else { q };
    return Some(build_complex_expr(0, 1, q, 1));
  }
  let two_p = p.checked_mul(2)?;
  if b % two_p != 0 {
    return None;
  }
  let q = b / two_p;
  // Verify: (p + q*I)^2 = p^2 - q^2 + 2pq*I should equal a + b*I
  let check_re = p.checked_mul(p)?.checked_sub(q.checked_mul(q)?)?;
  let check_im = p.checked_mul(q)?.checked_mul(2)?;
  if check_re != a || check_im != b {
    return None;
  }
  Some(build_complex_expr(p, 1, q, 1))
}

/// Surd[x, n] - Real-valued nth root
pub fn surd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Surd expects exactly 2 arguments".into(),
    ));
  }
  let base = &args[0];
  let degree = &args[1];

  // Surd[x, 0] → Surd::indet, Indeterminate (for any x, including symbolic).
  if is_literal_zero(degree) {
    crate::emit_message(&format!(
      "Surd::indet: Indeterminate expression Surd[{}, 0] encountered.",
      crate::syntax::expr_to_string(base)
    ));
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  // Exact path: an integer degree n with an exact (non-machine-Real) base.
  // Surd is the real n-th root, so delegate to Power, which returns the exact
  // symbolic form (Surd[2, 2] = Sqrt[2], Surd[8, -3] = 1/2, Surd[12, 2] =
  // 2 Sqrt[3], ...). A negative base with an odd n uses the real negative
  // root -(|b|^(1/n)); with an even n it is undefined.
  let base_is_exact = matches!(base, Expr::Integer(_) | Expr::BigInteger(_))
    || matches!(base, Expr::FunctionCall { name, .. } if name == "Rational");
  if let Expr::Integer(n) = degree
    && base_is_exact
    && let Some(x) = expr_to_num(base)
  {
    let n = *n;
    let power = |b: Expr| -> Result<Expr, InterpreterError> {
      let expr = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![b, make_rational(1, n)].into(),
      };
      crate::evaluator::evaluate_expr_to_expr(&expr)
    };
    let negate = |e: Expr| -> Result<Expr, InterpreterError> {
      let expr = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), e].into(),
      };
      crate::evaluator::evaluate_expr_to_expr(&expr)
    };
    if x < 0.0 {
      if n.rem_euclid(2) == 0 {
        crate::emit_message(
          "Surd::noneg: Surd is not defined for even roots of negative values.",
        );
        return Ok(Expr::Identifier("Indeterminate".to_string()));
      }
      // Odd root of a negative value: -(|b|^(1/n)).
      return negate(power(negate(base.clone())?)?);
    }
    return power(base.clone());
  }

  // Numeric fallback: a machine-Real base (or non-integer degree).
  match (expr_to_num(base), expr_to_num(degree)) {
    (Some(x), Some(n)) => {
      // Real-valued nth root: sign(x) * |x|^(1/n)
      let result = if x < 0.0 && n.fract() == 0.0 && (n as i128) % 2 != 0 {
        // Odd integer root of negative number
        -((-x).powf(1.0 / n))
      } else if x < 0.0 {
        // Even root of negative number - return symbolic
        return Ok(Expr::FunctionCall {
          name: "Surd".to_string(),
          args: args.to_vec().into(),
        });
      } else {
        x.powf(1.0 / n)
      };
      Ok(num_to_expr(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Surd".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Compute Floor or Ceiling via arbitrary-precision BigFloat when the value
/// exceeds what i128 can represent. `approx` is the f64 approximation used
/// only to pick a target precision. Returns an `Expr::BigInteger` on
/// success, or `None` if BigFloat evaluation of the input fails.
fn floor_via_bigfloat(
  expr: &Expr,
  approx: f64,
  is_floor: bool,
) -> Option<Expr> {
  use astro_float::{Consts, RoundingMode};
  if !approx.is_finite() {
    return None;
  }
  // Target precision: at least as many decimal digits as the integer part,
  // plus a safety margin.
  let mag_digits = approx.abs().log10().ceil() as i64;
  let precision = (mag_digits.max(20) as usize) + 10;
  let bits = crate::functions::math_ast::numerical::nominal_bits(precision);
  let mut cc = Consts::new().ok()?;
  let rm = RoundingMode::ToEven;
  let bf = crate::functions::math_ast::numerical::expr_to_bigfloat(
    expr, bits, rm, &mut cc,
  )
  .ok()?;
  let decimal = crate::functions::math_ast::numerical::bigfloat_to_string(
    &bf, None, rm, &mut cc,
  )
  .ok()?;
  // decimal looks like "[-]DIGITS.FRAC" (or just "[-]DIGITS.").
  let (sign, rest) = if let Some(s) = decimal.strip_prefix('-') {
    ("-", s)
  } else {
    ("", decimal.as_str())
  };
  let dot_pos = rest.find('.')?;
  let int_part = &rest[..dot_pos];
  let frac_part = &rest[dot_pos + 1..];
  // Parse the integer part as a BigInt.
  let int_str = format!("{}{}", sign, int_part);
  let int_bi = int_str.parse::<num_bigint::BigInt>().ok()?;
  // For Floor on a negative number with a non-zero fractional part, subtract 1.
  // For Ceiling on a positive number with a non-zero fractional part, add 1.
  let has_frac = !frac_part.chars().all(|c| c == '0');
  use num_bigint::BigInt;
  let adjusted = if is_floor {
    if sign == "-" && has_frac {
      int_bi - BigInt::from(1)
    } else {
      int_bi
    }
  } else if sign != "-" && has_frac {
    int_bi + BigInt::from(1)
  } else {
    int_bi
  };
  // Downshift to Integer if it fits i128.
  if let Ok(small) = adjusted.to_string().parse::<i128>() {
    Some(Expr::Integer(small))
  } else {
    Some(Expr::BigInteger(adjusted))
  }
}

/// Floor[x] - Floor function
/// Exact integer Floor/Ceiling of an `Integer`, `BigInteger`, or
/// `Rational[…]` whose parts are Integer/BigInteger. Computed with BigInt
/// arithmetic so it stays correct for magnitudes beyond f64 range — the f64
/// path would convert such rationals to ±inf and then saturate `as i128` to
/// i128::MAX (e.g. Egyptian-fraction denominators around 1e300). Returns None
/// for non-exact arguments (Real, symbolic, complex), which fall through.
fn exact_floor_ceil(arg: &Expr, is_floor: bool) -> Option<Expr> {
  use crate::functions::math_ast::{bigint_to_expr, expr_to_bigint};
  use num_bigint::Sign;
  use num_traits::Zero;
  // Plain integers are already integral.
  if let Some(n) = expr_to_bigint(arg) {
    return Some(bigint_to_expr(n));
  }
  if let Expr::FunctionCall { name, args } = arg
    && name == "Rational"
    && args.len() == 2
  {
    let num = expr_to_bigint(&args[0])?;
    let den = expr_to_bigint(&args[1])?;
    if den.is_zero() {
      return None;
    }
    // Normalize so the denominator is positive.
    let (num, den) = if den.sign() == Sign::Minus {
      (-num, -den)
    } else {
      (num, den)
    };
    // BigInt `/` truncates toward zero; correct toward -inf (floor) or
    // +inf (ceiling) when the division has a remainder.
    let q = &num / &den;
    let r = &num - &q * &den;
    let adjusted = if r.is_zero() {
      q
    } else if is_floor {
      if num.sign() == Sign::Minus { q - 1 } else { q }
    } else if num.sign() == Sign::Minus {
      q
    } else {
      q + 1
    };
    return Some(bigint_to_expr(adjusted));
  }
  None
}

/// Floor/Ceiling/Round/IntegerPart leave an (un)signed infinity or
/// `Indeterminate` unchanged: `Floor[Infinity] == Infinity`,
/// `Floor[-Infinity] == -Infinity`, `Floor[ComplexInfinity] ==
/// ComplexInfinity`, `Floor[Indeterminate] == Indeterminate`. Returns the
/// canonical expression to emit, or `None` when the argument is finite. The
/// two-argument forms (e.g. `Floor[Infinity, 2]`) also pass these through, so
/// callers check only the first argument.
fn infinity_passthrough(arg: &Expr) -> Option<Expr> {
  if matches!(arg, Expr::Identifier(s) if s == "Infinity" || s == "ComplexInfinity" || s == "Indeterminate")
  {
    return Some(arg.clone());
  }
  if crate::functions::math_ast::is_neg_infinity(arg) {
    return Some(arg.clone());
  }
  None
}

/// True if `e` is already integer-valued because it is a single-argument
/// Floor/Ceiling/Round/IntegerPart. An outer Floor/Ceiling/Round/IntegerPart is
/// then idempotent, matching wolframscript (Floor[Floor[x]] -> Floor[x],
/// Floor[Ceiling[x]] -> Ceiling[x], ...). The two-argument forms (e.g.
/// Floor[x, 2]) are not integer-valued, so they are excluded.
fn is_integer_valued_rounding(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, args }
    if args.len() == 1
    && matches!(name.as_str(), "Floor" | "Ceiling" | "Round" | "IntegerPart"))
}

pub fn floor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Floor expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 1 && is_integer_valued_rounding(&args[0]) {
    return Ok(args[0].clone());
  }
  if let Some(inf) = infinity_passthrough(&args[0]) {
    return Ok(inf);
  }
  if args.len() == 1
    && let Some(r) =
      crate::functions::interval_ast::map_monotonic_interval("Floor", &args[0])
  {
    return Ok(r);
  }
  if args.len() == 2 {
    return floor_ceil_two_arg(&args[0], &args[1], true);
  }
  if let Some(result) = exact_floor_ceil(&args[0], true) {
    return Ok(result);
  }
  if let Some(n) = try_eval_to_f64(&args[0]) {
    // i128 fits ~38 decimal digits; beyond that, `.floor() as i128` saturates
    // at i128::MAX, mis-computing Floor for large symbolic products like
    // `Pi * 10^100`. Fall through to the arbitrary-precision path.
    // f64 has ~15-16 digits of integer precision, so any magnitude beyond
    // that needs BigFloat to preserve the integer part exactly.
    if n.is_finite() && n.abs() < 1e15 {
      return Ok(Expr::Integer(n.floor() as i128));
    }
    // BigFloat fallback: compute the expression to enough precision to
    // resolve the integer part exactly, then parse the leading digits as
    // a BigInt.
    if let Some(result) = floor_via_bigfloat(&args[0], n, true) {
      return Ok(result);
    }
    Ok(Expr::Integer(n.floor() as i128))
  } else if let Some((re, im)) = try_extract_complex_float(&args[0]) {
    if im == 0.0 {
      Ok(Expr::Integer(re.floor() as i128))
    } else {
      // Floor[a + b*I] = Floor[a] + Floor[b]*I
      let floor_re = re.floor() as i128;
      let floor_im = im.floor() as i128;
      crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[
          Expr::Integer(floor_re),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(floor_im),
              Expr::Identifier("I".to_string()),
            ]
            .into(),
          },
        ],
      )
    }
  } else if let Some(r) = try_symbolic_complex_rounding(&args[0], floor_ast) {
    Ok(r)
  } else {
    Ok(Expr::FunctionCall {
      name: "Floor".to_string(),
      args: args.to_vec().into(),
    })
  }
}

/// Ceiling[x] or Ceiling[x, a] - Ceiling function
pub fn ceiling_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Ceiling expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 1 && is_integer_valued_rounding(&args[0]) {
    return Ok(args[0].clone());
  }
  if let Some(inf) = infinity_passthrough(&args[0]) {
    return Ok(inf);
  }
  if args.len() == 1
    && let Some(r) = crate::functions::interval_ast::map_monotonic_interval(
      "Ceiling", &args[0],
    )
  {
    return Ok(r);
  }
  if args.len() == 2 {
    return floor_ceil_two_arg(&args[0], &args[1], false);
  }
  if let Some(result) = exact_floor_ceil(&args[0], false) {
    return Ok(result);
  }
  if let Some(n) = try_eval_to_f64(&args[0]) {
    // f64 has ~15-16 digits of integer precision, so any magnitude beyond
    // that needs BigFloat to preserve the integer part exactly.
    if n.is_finite() && n.abs() < 1e15 {
      return Ok(Expr::Integer(n.ceil() as i128));
    }
    if let Some(result) = floor_via_bigfloat(&args[0], n, false) {
      return Ok(result);
    }
    Ok(Expr::Integer(n.ceil() as i128))
  } else if let Some((re, im)) = try_extract_complex_float(&args[0]) {
    if im == 0.0 {
      Ok(Expr::Integer(re.ceil() as i128))
    } else {
      // Ceiling[a + b*I] = Ceiling[a] + Ceiling[b]*I
      let ceil_re = re.ceil() as i128;
      let ceil_im = im.ceil() as i128;
      crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[
          Expr::Integer(ceil_re),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(ceil_im),
              Expr::Identifier("I".to_string()),
            ]
            .into(),
          },
        ],
      )
    }
  } else if let Some(r) = try_symbolic_complex_rounding(&args[0], ceiling_ast) {
    Ok(r)
  } else {
    Ok(Expr::FunctionCall {
      name: "Ceiling".to_string(),
      args: args.to_vec().into(),
    })
  }
}

/// Helper for Floor[x, a] and Ceiling[x, a]
/// Floor[x, a] = a * Floor[x/a], Ceiling[x, a] = a * Ceiling[x/a]
pub fn floor_ceil_two_arg(
  x: &Expr,
  a: &Expr,
  is_floor: bool,
) -> Result<Expr, InterpreterError> {
  // Complex x: apply componentwise to the real and imaginary parts, e.g.
  // Floor[a + b I, c] = Floor[a, c] + Floor[b, c] I.
  if let Some((re, im)) = complex_parts_for_rounding(x) {
    let re_r = floor_ceil_two_arg(&re, a, is_floor)?;
    let im_r = floor_ceil_two_arg(&im, a, is_floor)?;
    return build_complex_result(re_r, im_r);
  }
  // Try rational arithmetic first for exact results
  if let (Some(xn), Some(xd), Some(an), Some(ad)) = (
    expr_numerator(x),
    expr_denominator(x),
    expr_numerator(a),
    expr_denominator(a),
  ) {
    if an == 0 {
      // Floor[x, 0] or Ceiling[x, 0] → Indeterminate
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // x/a = (xn * ad) / (xd * an)
    let num = xn * ad;
    let den = xd * an;
    // Floor of rational num/den
    let floored = if is_floor {
      rational_floor(num, den)
    } else {
      rational_ceil(num, den)
    };
    // Result = a * floored = (an/ad) * floored = (an * floored) / ad
    let res_num = an * floored;
    if ad == 1 {
      return Ok(Expr::Integer(res_num));
    }
    let g = gcd_i128(res_num.abs(), ad.abs());
    let rn = res_num / g;
    let rd = ad / g;
    if rd == 1 {
      return Ok(Expr::Integer(rn));
    }
    return Ok(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(rn), Expr::Integer(rd)].into(),
    });
  }
  // Exact step `a` (rational): the result follows the type of `a`, so it stays
  // exact even when `x` is a symbolic constant (Floor[Pi, 1/10]) or a machine
  // float (Floor[2.7, 1/10] -> 27/10), matching wolframscript. Compute
  // a * Floor[x/a] via the single-argument Floor/Ceiling, which resolves both
  // exact transcendentals and floats to an integer. Falls through when the
  // quotient does not reduce to an integer (e.g. a symbolic x).
  if let (Some(an), Some(ad)) = (expr_numerator(a), expr_denominator(a))
    && an != 0
  {
    // q = x / a = x * (ad / an)
    let q = crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[x.clone(), make_rational(ad, an)],
    )?;
    let fq = if is_floor {
      floor_ast(std::slice::from_ref(&q))?
    } else {
      ceiling_ast(std::slice::from_ref(&q))?
    };
    if let Expr::Integer(k) = fq {
      // result = a * k = (an * k) / ad
      let res_num = an * k;
      if ad == 1 {
        return Ok(Expr::Integer(res_num));
      }
      let g = gcd_i128(res_num.abs(), ad.abs());
      let rn = res_num / g;
      let rd = ad / g;
      if rd == 1 {
        return Ok(Expr::Integer(rn));
      }
      return Ok(Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(rn), Expr::Integer(rd)].into(),
      });
    }
  }
  // Fall back to floating point
  if let (Some(xf), Some(af)) = (try_eval_to_f64(x), try_eval_to_f64(a)) {
    if af == 0.0 {
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    let result = if is_floor {
      (xf / af).floor() * af
    } else {
      (xf / af).ceil() * af
    };
    // Result type follows `a`: if `a` is Real, result is Real; otherwise Integer
    let a_is_float = matches!(a, Expr::Real(_));
    if a_is_float {
      return Ok(Expr::Real(result));
    }
    if result.fract() == 0.0 && result.abs() < i128::MAX as f64 {
      return Ok(Expr::Integer(result as i128));
    }
    return Ok(Expr::Real(result));
  }
  let name = if is_floor { "Floor" } else { "Ceiling" };
  Ok(Expr::FunctionCall {
    name: name.to_string(),
    args: vec![x.clone(), a.clone()].into(),
  })
}

/// Banker's rounding: round half to even
fn bankers_round(n: f64) -> f64 {
  if n.fract().abs() == 0.5 {
    let floor = n.floor();
    if floor as i128 % 2 == 0 {
      floor
    } else {
      n.ceil()
    }
  } else {
    n.round()
  }
}

/// Round[x] - Round to nearest integer using banker's rounding (round half to even)
/// Convert an integer-valued `f64` to an `Integer`, falling back to an exact
/// `BigInteger` when the magnitude exceeds the `i128` range (a plain `as i128`
/// cast would saturate to `i128::MAX`). `{:.0}` renders the float's exact
/// integer value without scientific notation.
pub(crate) fn f64_to_int_expr(v: f64) -> Expr {
  if v.abs() < i128::MAX as f64 {
    Expr::Integer(v as i128)
  } else if v.is_finite() {
    match format!("{v:.0}").parse::<num_bigint::BigInt>() {
      Ok(b) => Expr::BigInteger(b),
      Err(_) => Expr::Real(v),
    }
  } else {
    Expr::Real(v)
  }
}

pub fn round_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Round expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 1 && is_integer_valued_rounding(&args[0]) {
    return Ok(args[0].clone());
  }
  if let Some(inf) = infinity_passthrough(&args[0]) {
    return Ok(inf);
  }
  if args.len() == 1
    && let Some(r) =
      crate::functions::interval_ast::map_monotonic_interval("Round", &args[0])
  {
    return Ok(r);
  }
  if args.len() == 2 {
    // Round[x, a] - round x to nearest multiple of a
    let eval_a = crate::evaluator::evaluate_expr_to_expr(&args[1])?;
    let eval_x = crate::evaluator::evaluate_expr_to_expr(&args[0])?;

    // Complex x with a real step: round the real and imaginary parts
    // componentwise (Round[a + b I, c] = Round[a, c] + Round[b, c] I).
    if !is_complex_number(&eval_a)
      && let Some((re, im)) = complex_parts_for_rounding(&eval_x)
    {
      let re_r = round_ast(&[re, eval_a.clone()])?;
      let im_r = round_ast(&[im, eval_a])?;
      return build_complex_result(re_r, im_r);
    }

    // Complex step: Round[x, a] = a * Round[x/a]
    if is_complex_number(&eval_a) || is_complex_number(&eval_x) {
      let quotient = crate::evaluator::evaluate_function_call_ast(
        "Divide",
        &[eval_x.clone(), eval_a.clone()],
      )?;
      let rounded = round_ast(&[quotient])?;
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[eval_a, rounded],
      );
    }

    // Check if a is a Rational (n/d)
    if let Expr::FunctionCall { name, args: rargs } = &eval_a
      && name == "Rational"
      && rargs.len() == 2
      && let (Some(x_val), Some(a_val)) =
        (try_eval_to_f64(&eval_x), try_eval_to_f64(&eval_a))
      && a_val != 0.0
    {
      let n = bankers_round(x_val / a_val) as i128;
      // Return n * (num/den) as a rational
      if let (Some(num), Some(den)) =
        (expr_to_i128(&rargs[0]), expr_to_i128(&rargs[1]))
      {
        return Ok(make_rational_pub(n * num, den));
      }
    }

    // Check if a is symbolic (like Pi) — return n * a
    if let (Some(x_val), Some(a_val)) =
      (try_eval_to_f64(&eval_x), try_eval_to_f64(&eval_a))
    {
      if a_val == 0.0 {
        return Ok(eval_x);
      }
      let quotient_f = bankers_round(x_val / a_val);
      let a_is_real = matches!(&eval_a, Expr::Real(_));
      let a_is_int = matches!(&eval_a, Expr::Integer(_));
      // Big quotient: build the result without the saturating `as i128` cast.
      if quotient_f.is_finite() && quotient_f.abs() >= i128::MAX as f64 {
        let rounded = quotient_f * a_val;
        if a_is_int
          && rounded.fract() == 0.0
          && let Ok(b) = format!("{rounded:.0}").parse::<num_bigint::BigInt>()
        {
          return Ok(Expr::BigInteger(b));
        }
        return Ok(Expr::Real(rounded));
      }
      let n = quotient_f as i128;
      // If a is not a plain number, return n * a symbolically
      if !a_is_real && !a_is_int {
        // Symbolic: return n * a
        return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(n)),
          right: Box::new(eval_a),
        });
      }
      let rounded = n as f64 * a_val;
      // When the step a is Real, result should be Real;
      // when a is Integer, result should be Integer (if whole)
      if a_is_int && rounded.fract() == 0.0 && rounded.abs() < i128::MAX as f64
      {
        return Ok(Expr::Integer(rounded as i128));
      }
      if a_is_real || matches!(&eval_x, Expr::Real(_)) || rounded.fract() != 0.0
      {
        return Ok(Expr::Real(rounded));
      }
      if rounded.fract() == 0.0 && rounded.abs() < i128::MAX as f64 {
        return Ok(Expr::Integer(rounded as i128));
      }
      return Ok(Expr::Real(rounded));
    }
    return Ok(Expr::FunctionCall {
      name: "Round".to_string(),
      args: args.to_vec().into(),
    });
  }
  // Complex[re, im]: round real and imaginary parts separately
  if let Expr::FunctionCall { name, args: cargs } = &args[0]
    && name == "Complex"
    && cargs.len() == 2
  {
    let re = round_ast(&[cargs[0].clone()])?;
    let im = round_ast(&[cargs[1].clone()])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Plus",
      &[
        re,
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![im, Expr::Identifier("I".to_string())].into(),
        },
      ],
    );
  }
  // Exact complex in Plus/Times form: extract and round parts separately
  if let Some(((rn, rd), (in_, id))) =
    crate::functions::math_ast::try_extract_complex_exact(&args[0])
    && in_ != 0
  {
    let re_rat = make_rational_pub(rn, rd);
    let im_rat = make_rational_pub(in_, id);
    let re_rounded = round_ast(&[re_rat])?;
    let im_rounded = round_ast(&[im_rat])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Plus",
      &[
        re_rounded,
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![im_rounded, Expr::Identifier("I".to_string())].into(),
        },
      ],
    );
  }
  if let Some(n) = try_eval_to_f64(&args[0]) {
    Ok(f64_to_int_expr(bankers_round(n)))
  } else if let Some(r) = try_symbolic_complex_rounding(&args[0], round_ast) {
    Ok(r)
  } else {
    Ok(Expr::FunctionCall {
      name: "Round".to_string(),
      args: args.to_vec().into(),
    })
  }
}

/// Returns true if `expr` is a complex number with a non-zero imaginary part.
/// Handles Complex[re, im], Plus[re, Times[im, I]], I, etc.
fn is_complex_number(expr: &Expr) -> bool {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Complex"
    && args.len() == 2
  {
    return !matches!(&args[1], Expr::Integer(0));
  }
  if let Some(((_, _), (in_, _))) =
    crate::functions::math_ast::try_extract_complex_exact(expr)
  {
    return in_ != 0;
  }
  false
}

/// Mod[m, n] - Modulus, or Mod[m, n, d] with offset
pub fn mod_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(format!(
      "Mod expects 2 or 3 arguments; {} given",
      args.len()
    )));
  }

  // 3-argument form: Mod[m, n, d] = m - n * Floor[(m - d) / n]
  if args.len() == 3 {
    return mod3_ast(&args[0], &args[1], &args[2]);
  }

  // 2-argument form: Mod[m, n]
  mod2_ast(&args[0], &args[1])
}

/// True if `expr` is a literal zero: `Integer(0)`, `BigInteger(0)`,
/// `Rational(0, q)`, or `Real(0.0)`. Used for division-by-zero detection in
/// Mod / Quotient / QuotientRemainder.
pub fn is_literal_zero(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(0) => true,
    Expr::Real(f) => *f == 0.0,
    Expr::BigInteger(n) => {
      use num_traits::Zero;
      n.is_zero()
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!(&args[0], Expr::Integer(0))
    }
    _ => false,
  }
}

/// True if `a` and `b` are exactly-equal rational literals (so `a - b == 0`).
/// Symbolic operands return `false`.
fn exact_rational_equal(a: &Expr, b: &Expr) -> bool {
  match (try_as_rational(a), try_as_rational(b)) {
    (Some((an, ad)), Some((bn, bd))) => an * bd == bn * ad,
    _ => false,
  }
}

/// Helper to extract (numerator, denominator) from Integer or Rational
pub fn try_as_rational(expr: &Expr) -> Option<(i128, i128)> {
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n, *d))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Mod[m, n] - 2-argument form
pub fn mod2_ast(m: &Expr, n: &Expr) -> Result<Expr, InterpreterError> {
  // Mod[m, 0] => Indeterminate for any m (including symbolic), matching
  // wolframscript. The numeric branches below also guard against zero, but
  // this catches the case where m is symbolic and never reaches them.
  if is_literal_zero(n) {
    crate::emit_message(&format!(
      "Mod::indet: Indeterminate expression Mod[{}, 0] encountered.",
      crate::syntax::expr_to_string(m)
    ));
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // BigInteger fast-path so `Mod[2^200, 10^100]` doesn't fall through
  // to the float branch and lose precision. Mirrors the
  // `((m % n) + n) % n` rule the small-integer path uses.
  let m_big = match m {
    Expr::Integer(v) => Some(num_bigint::BigInt::from(*v)),
    Expr::BigInteger(v) => Some(v.clone()),
    _ => None,
  };
  let n_big = match n {
    Expr::Integer(v) => Some(num_bigint::BigInt::from(*v)),
    Expr::BigInteger(v) => Some(v.clone()),
    _ => None,
  };
  if let (Some(mb), Some(nb)) = (m_big, n_big) {
    use num_bigint::BigInt;
    use num_traits::Zero;
    if nb.is_zero() {
      crate::emit_message(&format!(
        "Mod::indet: Indeterminate expression Mod[{}, 0] encountered.",
        crate::syntax::expr_to_string(m)
      ));
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    let rem: BigInt = ((&mb % &nb) + &nb) % &nb;
    // Demote to native i128 when the result fits, since downstream
    // code special-cases `Expr::Integer`.
    if let Ok(small) = i128::try_from(&rem) {
      return Ok(Expr::Integer(small));
    }
    return Ok(Expr::BigInteger(rem));
  }
  // Try exact rational arithmetic first
  if let (Some((mn, md)), Some((nn, nd))) =
    (try_as_rational(m), try_as_rational(n))
  {
    if nn == 0 && nd != 0 {
      // Mod[m, 0] => Indeterminate
      crate::emit_message(&format!(
        "Mod::indet: Indeterminate expression Mod[{}, 0] encountered.",
        crate::syntax::expr_to_string(m)
      ));
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // Convert to common denominator: m = mn/md, n = nn/nd
    // Mod[mn/md, nn/nd] = Mod[mn*nd, nn*md] / (md*nd)
    let num_m = mn * nd;
    let num_n = nn * md;
    let common_d = md * nd;
    // Wolfram Mod: result = ((num_m % num_n) + num_n) % num_n, then divide by common_d
    let rem = ((num_m % num_n) + num_n) % num_n;
    return Ok(make_rational(rem, common_d));
  }

  // Exact symbolic path: when both operands are exact (no machine reals)
  // but at least one is an exact real constant such as Pi, E, GoldenRatio,
  // or Sqrt[2] (so the rational fast-path above did not fire), keep the
  // result exact via the defining identity Mod[m, n] = m - n*Floor[m/n].
  // wolframscript: Mod[Pi, 1] = -3 + Pi, Mod[5, Pi] = 5 - Pi.
  if !mod_contains_inexact(m)
    && !mod_contains_inexact(n)
    && let (Some(_), Some(b)) = (try_eval_to_f64(m), try_eval_to_f64(n))
  {
    if b == 0.0 {
      crate::emit_message(&format!(
        "Mod::indet: Indeterminate expression Mod[{}, 0] encountered.",
        crate::syntax::expr_to_string(m)
      ));
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // floor_quot = Floor[m/n]; evaluating the quotient first lets exact
    // cancellations (e.g. 2*Pi/Pi -> 2) happen so the floor is exact.
    let quotient = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(m.clone()),
      right: Box::new(n.clone()),
    };
    let floor_quot = Expr::FunctionCall {
      name: "Floor".to_string(),
      args: vec![quotient].into(),
    };
    // result = m - n*Floor[m/n]
    let result = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(m.clone()),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(n.clone()),
        right: Box::new(floor_quot),
      }),
    };
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  // Gaussian Mod: when at least one operand is genuinely complex, wolframscript
  // uses Mod[m, n] = m - n*Round[m/n], rounding the complex quotient
  // component-wise (round-half-to-even). Real operands are handled above, so
  // this only fires for Gaussian integers/rationals, e.g.
  // Mod[7 + 3*I, 2] = -1 - I.
  if let (Some((_, (m_im, _))), Some((_, (n_im, _)))) =
    (try_extract_complex_exact(m), try_extract_complex_exact(n))
    && (m_im != 0 || n_im != 0)
  {
    let quotient = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(m.clone()),
      right: Box::new(n.clone()),
    };
    let round_quot = Expr::FunctionCall {
      name: "Round".to_string(),
      args: vec![quotient].into(),
    };
    let result = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(m.clone()),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(n.clone()),
        right: Box::new(round_quot),
      }),
    };
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  // Float fallback
  if let (Some(a), Some(b)) = (try_eval_to_f64(m), try_eval_to_f64(n)) {
    if b == 0.0 {
      crate::emit_message(&format!(
        "Mod::indet: Indeterminate expression Mod[{}, 0] encountered.",
        crate::syntax::expr_to_string(m)
      ));
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    let result = ((a % b) + b) % b;
    // This path is only reached when an operand is an inexact machine real, so
    // the result stays Real even when it is a whole number (Mod[10., 5.] = 0.,
    // not 0) — num_to_expr would otherwise collapse it to an Integer.
    return Ok(Expr::Real(result));
  }

  // Symbolic
  Ok(Expr::FunctionCall {
    name: "Mod".to_string(),
    args: vec![m.clone(), n.clone()].into(),
  })
}

/// True if `expr` contains any machine real (`Real`/`BigFloat`) atom, i.e.
/// it is not an exact expression.
fn mod_contains_inexact(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::UnaryOp { operand, .. } => mod_contains_inexact(operand),
    Expr::BinaryOp { left, right, .. } => {
      mod_contains_inexact(left) || mod_contains_inexact(right)
    }
    Expr::FunctionCall { args, .. } => args.iter().any(mod_contains_inexact),
    _ => false,
  }
}

/// Mod[m, n, d] - 3-argument form: m - n * Floor[(m - d) / n]
pub fn mod3_ast(
  m: &Expr,
  n: &Expr,
  d: &Expr,
) -> Result<Expr, InterpreterError> {
  // Mod[m, 0, d] => Indeterminate for any m (including symbolic).
  if is_literal_zero(n) {
    crate::emit_message(&format!(
      "Mod::indet: Indeterminate expression Mod[{}, 0, {}] encountered.",
      crate::syntax::expr_to_string(m),
      crate::syntax::expr_to_string(d)
    ));
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // Try exact rational arithmetic
  if let (Some((mn, md)), Some((nn, nd)), Some((dn, dd))) =
    (try_as_rational(m), try_as_rational(n), try_as_rational(d))
  {
    if nn == 0 && nd != 0 {
      crate::emit_message(&format!(
        "Mod::indet: Indeterminate expression Mod[{}, 0, {}] encountered.",
        crate::syntax::expr_to_string(m),
        crate::syntax::expr_to_string(d)
      ));
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // m - d = (mn*dd - dn*md) / (md*dd)
    let diff_n = mn * dd - dn * md;
    let diff_d = md * dd;
    // (m - d) / n = (diff_n * nd) / (diff_d * nn)
    let quot_n = diff_n * nd;
    let quot_d = diff_d * nn;
    // Floor of quot_n / quot_d
    let fl = floor_div(quot_n, quot_d);
    // result = m - n * fl = mn/md - nn/nd * fl = (mn*nd - nn*md*fl) / (md*nd)
    let res_n = mn * nd - nn * md * fl;
    let res_d = md * nd;
    return Ok(make_rational(res_n, res_d));
  }

  // Exact symbolic path: keep the result exact via the defining identity
  // Mod[m, n, d] = m - n*Floor[(m - d)/n] when all operands are exact and
  // numerically real. wolframscript: Mod[Pi, 2, -1] = -4 + Pi.
  if !mod_contains_inexact(m)
    && !mod_contains_inexact(n)
    && !mod_contains_inexact(d)
    && let (Some(_), Some(b), Some(_)) =
      (try_eval_to_f64(m), try_eval_to_f64(n), try_eval_to_f64(d))
  {
    if b == 0.0 {
      crate::emit_message(&format!(
        "Mod::indet: Indeterminate expression Mod[{}, 0, {}] encountered.",
        crate::syntax::expr_to_string(m),
        crate::syntax::expr_to_string(d)
      ));
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // floor_quot = Floor[(m - d)/n]
    let diff = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(m.clone()),
      right: Box::new(d.clone()),
    };
    let quotient = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(diff),
      right: Box::new(n.clone()),
    };
    let floor_quot = Expr::FunctionCall {
      name: "Floor".to_string(),
      args: vec![quotient].into(),
    };
    // result = m - n*Floor[(m - d)/n]
    let result = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(m.clone()),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(n.clone()),
        right: Box::new(floor_quot),
      }),
    };
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  // Float fallback
  if let (Some(a), Some(b), Some(c)) =
    (try_eval_to_f64(m), try_eval_to_f64(n), try_eval_to_f64(d))
  {
    if b == 0.0 {
      crate::emit_message(&format!(
        "Mod::indet: Indeterminate expression Mod[{}, 0, {}] encountered.",
        crate::syntax::expr_to_string(m),
        crate::syntax::expr_to_string(d)
      ));
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    let result = a - b * ((a - c) / b).floor();
    // Reached only when an operand is an inexact machine real, so the result
    // stays Real even when it is a whole number (Mod[6., 3, 0] = 0., not 0).
    return Ok(Expr::Real(result));
  }

  // Symbolic
  Ok(Expr::FunctionCall {
    name: "Mod".to_string(),
    args: vec![m.clone(), n.clone(), d.clone()].into(),
  })
}

/// Integer floor division: floor(a / b)
pub fn floor_div(a: i128, b: i128) -> i128 {
  if b == 0 {
    return 0;
  }
  let d = a / b;
  let r = a % b;
  // Adjust if remainder has opposite sign to divisor
  if r != 0 && (r ^ b) < 0 { d - 1 } else { d }
}

/// Quotient[a, b] - Integer quotient
pub fn quotient_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(format!(
      "Quotient expects 2 or 3 arguments; {} given",
      args.len()
    )));
  }

  // Zero divisor (args[1] == 0). Matches wolframscript:
  //   Quotient[n, 0]    -> ComplexInfinity   (Quotient[0, 0] -> Indeterminate)
  //   Quotient[n, 0, d] -> ComplexInfinity   ((n - d) == 0 -> Indeterminate)
  // A symbolic numerator is treated as non-zero, hence ComplexInfinity.
  if is_literal_zero(&args[1]) {
    let numerator_is_zero = if args.len() == 3 {
      exact_rational_equal(&args[0], &args[2])
    } else {
      is_literal_zero(&args[0])
    };
    let call = crate::syntax::expr_to_string(&Expr::FunctionCall {
      name: "Quotient".to_string(),
      args: args.to_vec().into(),
    });
    if numerator_is_zero {
      crate::emit_message(&format!(
        "Quotient::indet: Indeterminate expression {call} encountered."
      ));
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    crate::emit_message(&format!(
      "Quotient::infy: Infinite expression {call} encountered."
    ));
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }

  // 3-argument form: Quotient[n, m, d] = Floor[(n - d) / m]
  if args.len() == 3 {
    if let (Some((nn, nd)), Some((mn, md)), Some((dn, dd))) = (
      try_as_rational(&args[0]),
      try_as_rational(&args[1]),
      try_as_rational(&args[2]),
    ) {
      if mn == 0 {
        return Err(InterpreterError::EvaluationError(
          "Quotient: division by zero".into(),
        ));
      }
      // (n - d) = (nn*dd - dn*nd) / (nd*dd)
      let diff_n = nn * dd - dn * nd;
      let diff_d = nd * dd;
      // (n - d) / m = (diff_n * md) / (diff_d * mn)
      let q_n = diff_n * md;
      let q_d = diff_d * mn;
      return Ok(Expr::Integer(floor_div(q_n, q_d)));
    }
    if let (Some(a), Some(b), Some(c)) = (
      try_eval_to_f64(&args[0]),
      try_eval_to_f64(&args[1]),
      try_eval_to_f64(&args[2]),
    ) {
      if b == 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Quotient: division by zero".into(),
        ));
      }
      return Ok(Expr::Integer(((a - c) / b).floor() as i128));
    }
    return Ok(Expr::FunctionCall {
      name: "Quotient".to_string(),
      args: args.to_vec().into(),
    });
  }

  // 2-argument form: Quotient[n, m] = Floor[n / m]
  match (&args[0], &args[1]) {
    (Expr::Integer(a), Expr::Integer(b)) => {
      if *b == 0 {
        Err(InterpreterError::EvaluationError(
          "Quotient: division by zero".into(),
        ))
      } else {
        Ok(Expr::Integer(floor_div(*a, *b)))
      }
    }
    _ => {
      // Exact arbitrary-precision path: if both args are Integer or
      // BigInteger, compute floor division without falling back to f64
      // (which loses precision for numbers that don't fit in 53 bits).
      use num_bigint::BigInt;
      use num_traits::Zero;
      let as_bigint = |e: &Expr| -> Option<BigInt> {
        match e {
          Expr::Integer(n) => Some(BigInt::from(*n)),
          Expr::BigInteger(n) => Some(n.clone()),
          _ => None,
        }
      };
      if let (Some(a), Some(b)) = (as_bigint(&args[0]), as_bigint(&args[1])) {
        if b.is_zero() {
          return Err(InterpreterError::EvaluationError(
            "Quotient: division by zero".into(),
          ));
        }
        // Floor division: when the sign of the dividend differs from the
        // divisor and there is a non-zero remainder, round the truncated
        // quotient toward -infinity.
        let (mut q, r) = (&a / &b, &a % &b);
        if !r.is_zero()
          && ((a.sign() != num_bigint::Sign::NoSign
            && b.sign() != num_bigint::Sign::NoSign)
            && (a.sign() != b.sign()))
        {
          q -= 1;
        }
        return Ok(crate::functions::math_ast::bigint_to_expr(q));
      }
      if let (Some(a), Some(b)) =
        (try_eval_to_f64(&args[0]), try_eval_to_f64(&args[1]))
      {
        if b == 0.0 {
          Err(InterpreterError::EvaluationError(
            "Quotient: division by zero".into(),
          ))
        } else {
          Ok(Expr::Integer((a / b).floor() as i128))
        }
      } else if let (Some((_, a_im)), Some((c_re, c_im))) = (
        crate::functions::math_ast::try_extract_complex_float(&args[0]),
        crate::functions::math_ast::try_extract_complex_float(&args[1]),
      ) && (a_im != 0.0 || c_im != 0.0)
      {
        if c_re == 0.0 && c_im == 0.0 {
          return Err(InterpreterError::EvaluationError(
            "Quotient: division by zero".into(),
          ));
        }
        // Gaussian quotient = Round[z/w], rounding the complex quotient
        // component-wise (round-half-to-even). This matches wolframscript and
        // the identity Mod[m, n] = m - n*Quotient[m, n]. Building the symbolic
        // Round and evaluating it also normalises the display to `a + b*I`.
        let quotient = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(args[0].clone()),
          right: Box::new(args[1].clone()),
        };
        let round_quot = Expr::FunctionCall {
          name: "Round".to_string(),
          args: vec![quotient].into(),
        };
        crate::evaluator::evaluate_expr_to_expr(&round_quot)
      } else {
        Ok(Expr::FunctionCall {
          name: "Quotient".to_string(),
          args: args.to_vec().into(),
        })
      }
    }
  }
}

/// Clip[x
pub fn clip_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Clip expects 1 to 3 arguments".into(),
    ));
  }

  // Handle list first argument: thread Clip over each element
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|item| {
        let mut new_args = args.to_vec();
        new_args[0] = item.clone();
        clip_ast(&new_args)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "Clip".to_string(),
      args: args.to_vec().into(),
    })
  };

  // Special values that don't compare like an ordinary real.
  match &args[0] {
    // Clip[Indeterminate] -> Indeterminate.
    Expr::Identifier(s) | Expr::Constant(s) if s == "Indeterminate" => {
      return Ok(args[0].clone());
    }
    // ComplexInfinity has no definite ordering: emit nord, stay unevaluated.
    Expr::Identifier(s) | Expr::Constant(s) if s == "ComplexInfinity" => {
      crate::emit_message(
        "Clip::nord: Invalid comparison with ComplexInfinity attempted.",
      );
      return unevaluated();
    }
    _ => {}
  }
  // A genuine complex number can't be clipped: emit ncompl, stay unevaluated.
  if crate::functions::predicate_ast::is_complex_number(&args[0]) {
    crate::emit_message(
      "Clip::ncompl: Symbolic or noncomplex numerical arguments are expected.",
    );
    return unevaluated();
  }

  // Numeric value of x, used only to decide the branch. The exact input is
  // preserved in the result, so Clip[1/2] stays 1/2 and Clip[Pi, {0, 10}] stays
  // Pi rather than being floatified. Infinity / -Infinity resolve to ±inf so
  // they clamp to the upper / lower bound respectively.
  let x =
    match crate::functions::math_ast::try_eval_to_f64_with_infinity(&args[0]) {
      Some(v) => v,
      None => return unevaluated(),
    };

  // Bounds: exact expressions plus their numeric values. The default range is
  // {-1, 1}.
  let (min_expr, max_expr, min_val, max_val) = if args.len() >= 2 {
    match &args[1] {
      Expr::List(bounds) if bounds.len() == 2 => {
        let min = match crate::functions::math_ast::try_eval_to_f64(&bounds[0])
        {
          Some(v) => v,
          None => return unevaluated(),
        };
        let max = match crate::functions::math_ast::try_eval_to_f64(&bounds[1])
        {
          Some(v) => v,
          None => return unevaluated(),
        };
        (bounds[0].clone(), bounds[1].clone(), min, max)
      }
      _ => return unevaluated(),
    }
  } else {
    (Expr::Integer(-1), Expr::Integer(1), -1.0, 1.0)
  };

  // 3rd arg: replacement values {vBelow, vAbove} for out-of-range inputs;
  // otherwise the clamped bound itself is used.
  let (below_expr, above_expr) = if args.len() == 3 {
    match &args[2] {
      Expr::List(r) if r.len() == 2 => (r[0].clone(), r[1].clone()),
      _ => return unevaluated(),
    }
  } else {
    (min_expr, max_expr)
  };

  if x < min_val {
    Ok(below_expr)
  } else if x > max_val {
    Ok(above_expr)
  } else {
    // Within range: return the exact input unchanged.
    Ok(args[0].clone())
  }
}

/// Emit the IntegerExponent::ibase message for an invalid base `b`.
fn emit_integer_exponent_ibase(b: &Expr) {
  crate::emit_message(&format!(
    "IntegerExponent::ibase: Base {} is not an integer greater than 1.",
    crate::syntax::format_expr(b, crate::syntax::ExprForm::Output)
  ));
}

/// IntegerExponent[n, b] - largest power of b that divides n
/// IntegerExponent[n] - largest power of 2 that divides n
pub fn integer_exponent_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use num_bigint::BigInt;
  use num_traits::Zero;

  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "IntegerExponent expects 1 or 2 arguments".into(),
    ));
  }
  // Accept both Integer and BigInteger for the first argument so that
  // IntegerExponent[100!, 10] works even when the factorial overflows i128.
  let n: BigInt = match &args[0] {
    Expr::Integer(k) => BigInt::from(*k),
    Expr::BigInteger(k) => k.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "IntegerExponent".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let base: BigInt = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(b) => BigInt::from(*b),
      Expr::BigInteger(b) => b.clone(),
      // A non-integer base is invalid: emit IntegerExponent::ibase.
      _ => {
        emit_integer_exponent_ibase(&args[1]);
        return Ok(Expr::FunctionCall {
          name: "IntegerExponent".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    BigInt::from(10) // default base is 10 in Wolfram
  };

  // The base must be an integer greater than 1. This is validated before the
  // n == 0 short-circuit, so IntegerExponent[0, 1] emits ibase rather than
  // returning Infinity (matching wolframscript). Only an explicit base can be
  // invalid; the default base 10 is always valid.
  if args.len() == 2 && base <= BigInt::from(1) {
    emit_integer_exponent_ibase(&args[1]);
    return Ok(Expr::FunctionCall {
      name: "IntegerExponent".to_string(),
      args: args.to_vec().into(),
    });
  }

  if n.is_zero() {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }

  let mut count: i128 = 0;
  let mut val = n.clone();
  if val.sign() == num_bigint::Sign::Minus {
    val = -val;
  }
  while !val.is_zero() && (&val % &base).is_zero() {
    count += 1;
    val /= &base;
  }
  Ok(Expr::Integer(count))
}

// ─── IntegerPart / FractionalPart ──────────────────────────────────

/// True when `expr` carries a machine-precision (inexact) literal anywhere.
fn contains_inexact_literal(e: &Expr) -> bool {
  match e {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::BinaryOp { left, right, .. } => {
      contains_inexact_literal(left) || contains_inexact_literal(right)
    }
    Expr::UnaryOp { operand, .. } => contains_inexact_literal(operand),
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(contains_inexact_literal)
    }
    _ => false,
  }
}

/// Build `re + im*I`, dropping a zero imaginary part. Used by the complex
/// branches of IntegerPart/FractionalPart.
fn build_complex_result(re: Expr, im: Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_function_call_ast(
    "Plus",
    &[
      re,
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![im, Expr::Identifier("I".to_string())].into(),
      },
    ],
  )
}

/// Split a complex number into its real and imaginary parts as expressions, for
/// the two-argument Floor/Ceiling/Round forms. Prefers the symbolic split (to
/// keep exact rational/symbolic components) and falls back to the machine-float
/// split. Returns `None` for a purely real argument.
fn complex_parts_for_rounding(expr: &Expr) -> Option<(Expr, Expr)> {
  if let Some((re, im)) =
    crate::evaluator::dispatch::complex_and_special::split_real_imag_symbolic(
      expr,
    )
    && !is_literal_zero(&im)
  {
    return Some((re, im));
  }
  if let Some((re, im)) =
    crate::functions::math_ast::try_extract_complex_float(expr)
    && im != 0.0
  {
    return Some((Expr::Real(re), Expr::Real(im)));
  }
  None
}

/// Apply an integer-valued rounding function (Floor/Ceiling/Round/IntegerPart)
/// componentwise to a complex `a + b*I` whose parts are symbolic-real, e.g.
/// `Pi + E I` or `Sqrt[2] + Sqrt[3] I`. The scalar path already resolves such
/// real components (`Floor[Pi] == 3`), so we split, round each part and
/// recombine. Returns `None` when `arg` is not of real+imag*I form, has an
/// identically-zero imaginary part, or either rounded component fails to
/// resolve to an integer (so the caller leaves the call unevaluated, like
/// wolframscript does for `Floor[x + y I]`).
fn try_symbolic_complex_rounding(
  arg: &Expr,
  scalar: fn(&[Expr]) -> Result<Expr, InterpreterError>,
) -> Option<Expr> {
  let (re, im) =
    crate::evaluator::dispatch::complex_and_special::split_real_imag_symbolic(
      arg,
    )?;
  if matches!(im, Expr::Integer(0)) || matches!(im, Expr::Real(z) if z == 0.0) {
    return None;
  }
  let fr = scalar(std::slice::from_ref(&re)).ok()?;
  let fi = scalar(std::slice::from_ref(&im)).ok()?;
  let is_int = |e: &Expr| matches!(e, Expr::Integer(_) | Expr::BigInteger(_));
  if !is_int(&fr) || !is_int(&fi) {
    return None;
  }
  build_complex_result(fr, fi).ok()
}

/// IntegerPart[x] - Integer part (truncation towards zero)
pub fn integer_part_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "IntegerPart expects exactly 1 argument".into(),
    ));
  }
  if is_integer_valued_rounding(&args[0]) {
    return Ok(args[0].clone());
  }
  if let Some(inf) = infinity_passthrough(&args[0]) {
    return Ok(inf);
  }
  // Complex number: IntegerPart[a + b I] = IntegerPart[a] + IntegerPart[b] I,
  // truncating each component toward zero. Covers numeric/Real components;
  // symbolic-constant components (Pi + E I) extract no float and fall through.
  if let Some((re, im)) =
    crate::functions::math_ast::try_extract_complex_float(&args[0])
    && im != 0.0
  {
    return build_complex_result(
      Expr::Integer(re.trunc() as i128),
      Expr::Integer(im.trunc() as i128),
    );
  }
  match &args[0] {
    Expr::Integer(n) => Ok(Expr::Integer(*n)),
    Expr::Real(f) => Ok(f64_to_int_expr(f.trunc())),
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1])
        && *d != 0
      {
        // Truncate towards zero
        return Ok(Expr::Integer(n / d));
      }
      Ok(Expr::FunctionCall {
        name: "IntegerPart".to_string(),
        args: args.to_vec().into(),
      })
    }
    _ => {
      if let Some(f) = try_eval_to_f64(&args[0]) {
        Ok(f64_to_int_expr(f.trunc()))
      } else if let Some(r) =
        try_symbolic_complex_rounding(&args[0], integer_part_ast)
      {
        Ok(r)
      } else {
        Ok(Expr::FunctionCall {
          name: "IntegerPart".to_string(),
          args: args.to_vec().into(),
        })
      }
    }
  }
}

/// Compute FractionalPart of a rational `n/d` (assumes positive `d`),
/// truncating toward zero — matches Wolfram's `FractionalPart`.
/// Returns (num, den) with |num| < den.
fn rational_fractional_part(n: i128, d: i128) -> (i128, i128) {
  let d = d.abs().max(1);
  let n = if d == 0 { n } else { n };
  // Truncate n/d toward zero.
  let trunc = n / d;
  let frac = n - trunc * d;
  (frac, d)
}

/// FractionalPart[x] - Fractional part: x - IntegerPart[x]
pub fn fractional_part_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FractionalPart expects exactly 1 argument".into(),
    ));
  }
  // FractionalPart of an infinite quantity is the full unit interval in the
  // corresponding direction (wolframscript): Infinity -> Interval[{0, 1}],
  // -Infinity -> Interval[{-1, 0}], ComplexInfinity -> Interval[{0, 1}].
  {
    let unit_interval = |lo: i128, hi: i128| Expr::FunctionCall {
      name: "Interval".to_string(),
      args: vec![Expr::List(
        vec![Expr::Integer(lo), Expr::Integer(hi)].into(),
      )]
      .into(),
    };
    if matches!(&args[0], Expr::Identifier(s) if s == "Infinity" || s == "ComplexInfinity")
    {
      return Ok(unit_interval(0, 1));
    }
    if crate::functions::math_ast::is_neg_infinity(&args[0]) {
      return Ok(unit_interval(-1, 0));
    }
  }
  // FractionalPart[Indeterminate] -> Indeterminate.
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // Exact complex rational: apply FractionalPart to real and imag parts
  // separately (FractionalPart truncates toward zero).
  if let Some(((rn, rd), (in_, id))) =
    crate::functions::math_ast::try_extract_complex_exact(&args[0])
    && in_ != 0
  {
    let re_frac = rational_fractional_part(rn, rd);
    let im_frac = rational_fractional_part(in_, id);
    let build = |n: i128, d: i128| -> Expr {
      if n == 0 {
        Expr::Integer(0)
      } else {
        make_rational(n, d)
      }
    };
    return build_complex_result(
      build(re_frac.0, re_frac.1),
      build(im_frac.0, im_frac.1),
    );
  }
  // Inexact complex: FractionalPart[a + b I] = FractionalPart[a] +
  // FractionalPart[b] I with Real components. Forming a complex from a Real
  // promotes both parts to Real (Complex[2., 2.5]), so even an integer-valued
  // component yields `0.` here — matching wolframscript.
  if contains_inexact_literal(&args[0])
    && let Some((re, im)) =
      crate::functions::math_ast::try_extract_complex_float(&args[0])
    && im != 0.0
  {
    return build_complex_result(
      Expr::Real(re - re.trunc()),
      Expr::Real(im - im.trunc()),
    );
  }
  match &args[0] {
    Expr::Integer(_) => Ok(Expr::Integer(0)),
    // Preserve Real type: FractionalPart[3.0] = 0. (not 0)
    Expr::Real(f) => Ok(Expr::Real(*f - f.trunc())),
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1])
        && *d != 0
      {
        let int_part = n / d;
        let frac_n = n - int_part * d;
        if frac_n == 0 {
          return Ok(Expr::Integer(0));
        }
        return Ok(make_rational(frac_n, *d));
      }
      Ok(Expr::FunctionCall {
        name: "FractionalPart".to_string(),
        args: args.to_vec().into(),
      })
    }
    _ => {
      // For symbolic expressions that contain no Real/BigFloat literal but
      // can still be evaluated to a real f64 (e.g. Pi, E, Pi^20, Pi+E),
      // return the exact symbolic form `x - Floor[x]` to match wolframscript
      // (FractionalPart[Pi^20] -> -8769956796 + Pi^20).
      fn has_inexact(e: &Expr) -> bool {
        match e {
          Expr::Real(_) | Expr::BigFloat(_, _) => true,
          Expr::BinaryOp { left, right, .. } => {
            has_inexact(left) || has_inexact(right)
          }
          Expr::UnaryOp { operand, .. } => has_inexact(operand),
          Expr::FunctionCall { args, .. } | Expr::List(args) => {
            args.iter().any(has_inexact)
          }
          _ => false,
        }
      }
      if !has_inexact(&args[0])
        && let Ok(floor_val) =
          crate::functions::math_ast::floor_ast(&[args[0].clone()])
      {
        if let Expr::Integer(n) = &floor_val {
          if *n == 0 {
            return Ok(args[0].clone());
          }
          return crate::evaluator::evaluate_function_call_ast(
            "Plus",
            &[args[0].clone(), Expr::Integer(-*n)],
          );
        }
        if let Expr::BigInteger(_) = &floor_val {
          return crate::evaluator::evaluate_function_call_ast(
            "Plus",
            &[
              args[0].clone(),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), floor_val].into(),
              },
            ],
          );
        }
      }
      if let Some(f) = try_eval_to_f64(&args[0]) {
        let frac = f - f.trunc();
        if frac == 0.0 {
          Ok(Expr::Integer(0))
        } else {
          Ok(Expr::Real(frac))
        }
      } else {
        Ok(Expr::FunctionCall {
          name: "FractionalPart".to_string(),
          args: args.to_vec().into(),
        })
      }
    }
  }
}

// ─── MixedFractionParts ────────────────────────────────────────────

pub fn mixed_fraction_parts_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "MixedFractionParts expects exactly 1 argument".into(),
    ));
  }
  let int_part = integer_part_ast(args)?;
  let frac_part = fractional_part_ast(args)?;
  Ok(Expr::List(vec![int_part, frac_part].into()))
}

// ─── Chop ──────────────────────────────────────────────────────────

/// Chop[x] or Chop[x, delta] - Replaces approximate real numbers close to zero by exact 0
pub fn chop_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Chop expects 1 or 2 arguments".into(),
    ));
  }
  let tolerance = if args.len() == 2 {
    match try_eval_to_f64(&args[1]) {
      Some(t) => t,
      None => {
        return Ok(Expr::FunctionCall {
          name: "Chop".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    1e-10 // Default tolerance
  };

  chop_expr(&args[0], tolerance)
}

pub fn chop_expr(
  expr: &Expr,
  tolerance: f64,
) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Real(f) => {
      if f.abs() < tolerance {
        Ok(Expr::Integer(0))
      } else {
        Ok(expr.clone())
      }
    }
    Expr::List(items) => {
      let chopped: Result<Vec<Expr>, _> =
        items.iter().map(|e| chop_expr(e, tolerance)).collect();
      Ok(Expr::List(chopped?.into()))
    }
    // Recursively chop into function calls (Plus, Times, Complex, …) and
    // re-evaluate so that chopped subterms like `Complex[0, 0]` collapse
    // to 0 and additive identities are simplified.
    Expr::FunctionCall { name, args } => {
      let chopped_args: Vec<Expr> = args
        .iter()
        .map(|a| chop_expr(a, tolerance))
        .collect::<Result<_, _>>()?;
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: name.clone(),
        args: chopped_args.into(),
      })
    }
    Expr::BinaryOp { op, left, right } => {
      let new_left = chop_expr(left, tolerance)?;
      let new_right = chop_expr(right, tolerance)?;
      crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: *op,
        left: Box::new(new_left),
        right: Box::new(new_right),
      })
    }
    Expr::UnaryOp { op, operand } => {
      let new_operand = chop_expr(operand, tolerance)?;
      crate::evaluator::evaluate_expr_to_expr(&Expr::UnaryOp {
        op: *op,
        operand: Box::new(new_operand),
      })
    }
    _ => Ok(expr.clone()),
  }
}

// ─── CubeRoot ──────────────────────────────────────────────────────

/// CubeRoot[x] - Real-valued cube root
/// Extract the largest perfect cube factor from n.
/// Returns (cube_root_of_cube_part, remainder) such that n = cube_part^3 * remainder.
fn extract_cube_factor(mut n: u128) -> (u128, u128) {
  let mut cube_root = 1u128;
  // Trial division by small primes
  let mut p = 2u128;
  while p * p * p <= n {
    let mut count = 0u32;
    while n.is_multiple_of(p) {
      n /= p;
      count += 1;
    }
    let cube_groups = count / 3;
    let leftover = count % 3;
    for _ in 0..cube_groups {
      cube_root *= p;
    }
    for _ in 0..leftover {
      n *= p; // put non-cube parts back into remainder
    }
    p += if p == 2 { 1 } else { 2 };
  }
  (cube_root, n)
}

pub fn cube_root_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CubeRoot expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => {
      if *n == 0 {
        return Ok(Expr::Integer(0));
      }
      let sign = n.signum();
      let abs_n = n.unsigned_abs();
      // Check for perfect cube
      let root = (abs_n as f64).cbrt().round() as u128;
      if root * root * root == abs_n {
        return Ok(Expr::Integer(sign * root as i128));
      }
      // Factor out the largest perfect cube
      // Find prime factorization and extract cube parts
      let (cube_part, remainder) = extract_cube_factor(abs_n);
      if cube_part > 1 {
        // CubeRoot[n] = cube_part * CubeRoot[remainder]
        let cube_root_remainder = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(Expr::Integer(remainder as i128)),
          right: Box::new(Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(3)].into(),
          }),
        };
        let result = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(sign * cube_part as i128)),
          right: Box::new(cube_root_remainder),
        };
        crate::evaluator::evaluate_expr_to_expr(&result)
      } else {
        // No cube factor — CubeRoot is the real cube root, so return
        // Sign[n] * abs[n]^(1/3) (matching wolframscript: CubeRoot[-5]
        // -> -5^(1/3), i.e. -(5^(1/3)), not the complex (-5)^(1/3)).
        let pow = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(Expr::Integer(abs_n as i128)),
          right: Box::new(Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(3)].into(),
          }),
        };
        if sign < 0 {
          Ok(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(pow),
          })
        } else {
          Ok(pow)
        }
      }
    }
    Expr::Real(f) => Ok(Expr::Real(f.signum() * f.abs().cbrt())),
    _ => {
      if let Some(f) = try_eval_to_f64(&args[0]) {
        Ok(Expr::Real(f.signum() * f.abs().cbrt()))
      } else {
        // Canonicalize CubeRoot[x] → Surd[x, 3]
        Ok(Expr::FunctionCall {
          name: "Surd".to_string(),
          args: vec![args[0].clone(), Expr::Integer(3)].into(),
        })
      }
    }
  }
}

// ─── Subdivide ─────────────────────────────────────────────────────

/// Subdivide[n] - subdivide [0,1] into n equal parts
/// Subdivide[xmax, n] - subdivide [0, xmax] into n equal parts
/// Subdivide[xmin, xmax, n] - subdivide [xmin, xmax] into n equal parts
pub fn subdivide_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Subdivide expects 1 to 3 arguments".into(),
    ));
  }

  // Extract (xmin, xmax, n) based on arity
  let (xmin, xmax, n_expr) = match args.len() {
    1 => (Expr::Integer(0), Expr::Integer(1), args[0].clone()),
    2 => (Expr::Integer(0), args[0].clone(), args[1].clone()),
    3 => (args[0].clone(), args[1].clone(), args[2].clone()),
    _ => unreachable!(),
  };

  // n must be a positive integer
  let n_val = match &n_expr {
    Expr::Integer(n) if *n > 0 => *n,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Subdivide".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Fast path: both endpoints are integers
  if let (Expr::Integer(xmin_i), Expr::Integer(xmax_i)) = (&xmin, &xmax) {
    let mut items = Vec::with_capacity(n_val as usize + 1);
    let range = xmax_i - xmin_i;
    for i in 0..=n_val {
      let numer = xmin_i * n_val + i * range;
      items.push(make_rational(numer, n_val));
    }
    return Ok(Expr::List(items.into()));
  }

  // General path: build xmin + i*(xmax - xmin)/n symbolically and evaluate.
  // For vector endpoints, thread element-wise.
  let is_vector =
    matches!(&xmin, Expr::List(_)) || matches!(&xmax, Expr::List(_));
  if is_vector {
    // Both must be lists of the same length
    if let (Expr::List(xmin_items), Expr::List(xmax_items)) = (&xmin, &xmax) {
      if xmin_items.len() != xmax_items.len() {
        return Ok(Expr::FunctionCall {
          name: "Subdivide".to_string(),
          args: args.to_vec().into(),
        });
      }
      let dim = xmin_items.len();
      let mut result_items = Vec::with_capacity(n_val as usize + 1);
      for i in 0..=n_val {
        let mut point = Vec::with_capacity(dim);
        for d in 0..dim {
          let val =
            subdivide_scalar_at(&xmin_items[d], &xmax_items[d], i, n_val)?;
          point.push(val);
        }
        result_items.push(Expr::List(point.into()));
      }
      return Ok(Expr::List(result_items.into()));
    }
    return Ok(Expr::FunctionCall {
      name: "Subdivide".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Scalar, general (symbolic or float) endpoints
  let mut items = Vec::with_capacity(n_val as usize + 1);
  for i in 0..=n_val {
    items.push(subdivide_scalar_at(&xmin, &xmax, i, n_val)?);
  }
  Ok(Expr::List(items.into()))
}

/// Compute xmin*(n-i)/n + xmax*i/n for a single scalar pair of endpoints.
fn subdivide_scalar_at(
  xmin: &Expr,
  xmax: &Expr,
  i: i128,
  n: i128,
) -> Result<Expr, InterpreterError> {
  use crate::evaluator::evaluate_expr_to_expr;
  use crate::syntax::BinaryOperator;

  if i == 0 {
    return Ok(xmin.clone());
  }
  if i == n {
    return Ok(xmax.clone());
  }

  // Build: xmin*(n-i)/n + xmax*i/n  and evaluate
  let coeff_min = make_rational(n - i, n);
  let coeff_max = make_rational(i, n);

  let term_min = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(coeff_min),
    right: Box::new(xmin.clone()),
  };
  let term_max = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(coeff_max),
    right: Box::new(xmax.clone()),
  };
  let result = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(term_min),
    right: Box::new(term_max),
  };
  evaluate_expr_to_expr(&result)
}

/// Ramp[x] - returns max(0, x)
/// Ramp[list] - maps over lists
pub fn ramp_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Ramp expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => Ok(Expr::Integer((*n).max(0))),
    Expr::Real(f) => Ok(if *f > 0.0 {
      Expr::Real(*f)
    } else {
      Expr::Real(0.0)
    }),
    Expr::List(items) => {
      let results: Result<Vec<Expr>, InterpreterError> =
        items.iter().map(|x| ramp_ast(&[x.clone()])).collect();
      Ok(Expr::List(results?.into()))
    }
    // Exact rationals and real-valued symbolic numerics (Pi, Sqrt[2], E - 3,
    // …): Ramp[x] = x for x >= 0, else 0. The exact input is preserved; a
    // negative value yields the integer 0 (the Real case is handled above).
    other => match crate::functions::math_ast::try_eval_to_f64(other) {
      Some(v) if v >= 0.0 => Ok(other.clone()),
      Some(_) => Ok(Expr::Integer(0)),
      None => Ok(Expr::FunctionCall {
        name: "Ramp".to_string(),
        args: args.to_vec().into(),
      }),
    },
  }
}

/// KroneckerDelta[args...] - returns 1 if all arguments are equal, 0 otherwise
/// KroneckerDelta[n] - returns 1 if n==0, 0 if n!=0 (equivalent to KroneckerDelta[n, 0])
pub fn kronecker_delta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }

  // Single argument: KroneckerDelta[n] is 1 if n==0, 0 otherwise
  if args.len() == 1 {
    return match &args[0] {
      Expr::Integer(n) => Ok(Expr::Integer(if *n == 0 { 1 } else { 0 })),
      Expr::Real(f) => Ok(Expr::Integer(if *f == 0.0 { 1 } else { 0 })),
      _ => Ok(Expr::FunctionCall {
        name: "KroneckerDelta".to_string(),
        args: args.to_vec().into(),
      }),
    };
  }

  // Multi argument: check if all are equal
  // First check if any are symbolic (non-numeric)
  let mut has_symbolic = false;
  let mut all_equal = true;
  let first_str = crate::syntax::expr_to_string(&args[0]);
  for arg in &args[1..] {
    let s = crate::syntax::expr_to_string(arg);
    if s != first_str {
      all_equal = false;
      // Check if both are numeric and compare numerically
      if let (Some(a), Some(b)) =
        (try_eval_to_f64(&args[0]), try_eval_to_f64(arg))
      {
        if a != b {
          return Ok(Expr::Integer(0));
        } else {
          all_equal = true;
        }
      } else {
        has_symbolic = true;
      }
    }
  }
  if has_symbolic {
    Ok(Expr::FunctionCall {
      name: "KroneckerDelta".to_string(),
      args: args.to_vec().into(),
    })
  } else if all_equal {
    Ok(Expr::Integer(1))
  } else {
    Ok(Expr::Integer(0))
  }
}

/// DiscreteDelta[n1, n2, ...] - returns 1 if all ni are 0, 0 otherwise
/// DiscreteDelta[] - returns 1
pub fn discrete_delta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }
  let mut has_symbolic = false;
  for arg in args {
    match arg {
      Expr::Integer(n) => {
        if *n != 0 {
          return Ok(Expr::Integer(0));
        }
      }
      Expr::Real(f) => {
        if *f != 0.0 {
          return Ok(Expr::Integer(0));
        }
      }
      _ => {
        has_symbolic = true;
      }
    }
  }
  if has_symbolic {
    Ok(Expr::FunctionCall {
      name: "DiscreteDelta".to_string(),
      args: args.to_vec().into(),
    })
  } else {
    Ok(Expr::Integer(1))
  }
}

/// UnitStep[x] - returns 0 for x < 0, 1 for x >= 0
/// UnitStep[x1, x2, ...] - returns 1 if all xi >= 0, 0 if any xi < 0
/// UnitStep[list] - maps over lists (single arg only)
pub fn unit_step_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "UnitStep expects at least 1 argument".into(),
    ));
  }

  // Multi-arg: UnitStep[x1, x2, ...] = product of UnitStep[xi]; it is 0 if any
  // xi < 0 and 1 if all are >= 0. Arguments known to be >= 0 are dropped, and
  // the remaining symbolic arguments are deduplicated and sorted, matching
  // wolframscript (UnitStep[1/2, x] -> UnitStep[x], UnitStep[b, a] ->
  // UnitStep[a, b]).
  if args.len() > 1 {
    let mut remaining: Vec<Expr> = Vec::new();
    for arg in args {
      match crate::functions::math_ast::try_eval_to_f64(arg) {
        Some(v) if v < 0.0 => return Ok(Expr::Integer(0)),
        Some(_) => {} // >= 0: contributes 1, drop it
        None => {
          let s = crate::syntax::expr_to_string(arg);
          if !remaining
            .iter()
            .any(|e| crate::syntax::expr_to_string(e) == s)
          {
            remaining.push(arg.clone());
          }
        }
      }
    }
    if remaining.is_empty() {
      return Ok(Expr::Integer(1));
    }
    remaining.sort_by(crate::functions::list_helpers_ast::canonical_cmp);
    return Ok(Expr::FunctionCall {
      name: "UnitStep".to_string(),
      args: remaining.into(),
    });
  }

  // Single arg
  match &args[0] {
    Expr::Integer(n) => Ok(Expr::Integer(if *n >= 0 { 1 } else { 0 })),
    Expr::Real(f) => Ok(Expr::Integer(if *f >= 0.0 { 1 } else { 0 })),
    Expr::Constant(c) => match c.as_str() {
      "Pi" | "E" | "Degree" => Ok(Expr::Integer(1)),
      _ => Ok(Expr::FunctionCall {
        name: "UnitStep".to_string(),
        args: args.to_vec().into(),
      }),
    },
    Expr::Identifier(name) if name == "Infinity" => Ok(Expr::Integer(1)),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => match operand.as_ref() {
      Expr::Constant(c) if matches!(c.as_str(), "Pi" | "E" | "Degree") => {
        Ok(Expr::Integer(0))
      }
      Expr::Identifier(name) if name == "Infinity" => Ok(Expr::Integer(0)),
      _ => Ok(Expr::FunctionCall {
        name: "UnitStep".to_string(),
        args: args.to_vec().into(),
      }),
    },
    // Times[-1, x] pattern (e.g. -Pi parses as Times[-1, Pi])
    Expr::FunctionCall { name, args: fargs }
      if name == "Times"
        && fargs.len() == 2
        && matches!(fargs[0], Expr::Integer(-1)) =>
    {
      match &fargs[1] {
        Expr::Constant(c) if matches!(c.as_str(), "Pi" | "E" | "Degree") => {
          Ok(Expr::Integer(0))
        }
        Expr::Identifier(n) if n == "Infinity" => Ok(Expr::Integer(0)),
        _ => Ok(Expr::FunctionCall {
          name: "UnitStep".to_string(),
          args: args.to_vec().into(),
        }),
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(-1)) => match right.as_ref() {
      Expr::Constant(c) if matches!(c.as_str(), "Pi" | "E" | "Degree") => {
        Ok(Expr::Integer(0))
      }
      Expr::Identifier(n) if n == "Infinity" => Ok(Expr::Integer(0)),
      _ => Ok(Expr::FunctionCall {
        name: "UnitStep".to_string(),
        args: args.to_vec().into(),
      }),
    },
    Expr::List(items) => {
      let results: Result<Vec<Expr>, InterpreterError> =
        items.iter().map(|x| unit_step_ast(&[x.clone()])).collect();
      Ok(Expr::List(results?.into()))
    }
    // Exact rationals and real-valued symbolic numerics (Sqrt[2] - 2, …):
    // UnitStep[x] = 1 for x >= 0, else 0.
    other => match crate::functions::math_ast::try_eval_to_f64(other) {
      Some(v) if v >= 0.0 => Ok(Expr::Integer(1)),
      Some(_) => Ok(Expr::Integer(0)),
      None => Ok(Expr::FunctionCall {
        name: "UnitStep".to_string(),
        args: args.to_vec().into(),
      }),
    },
  }
}

pub fn heaviside_theta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "HeavisideTheta expects at least 1 argument".into(),
    ));
  }

  // Multi-arg: HeavisideTheta[x1, x2, ...] = product, 0 if any xi < 0
  if args.len() > 1 {
    let mut has_zero = false;
    let mut remaining = Vec::new();
    for arg in args {
      match arg {
        Expr::Integer(n) => {
          if *n < 0 {
            return Ok(Expr::Integer(0));
          }
          if *n == 0 {
            has_zero = true;
            remaining.push(arg.clone());
          }
          // n > 0: contributes 1, skip
        }
        Expr::Real(f) => {
          if *f < 0.0 {
            return Ok(Expr::Integer(0));
          }
          if *f == 0.0 {
            has_zero = true;
            remaining.push(arg.clone());
          }
        }
        _ => {
          remaining.push(arg.clone());
        }
      }
    }
    // If any arg is zero, HeavisideTheta[0] is undefined so the whole
    // expression stays unevaluated with ALL original arguments sorted.
    if has_zero {
      let mut sorted_args = args.to_vec();
      sorted_args.sort_by(crate::functions::list_helpers_ast::canonical_cmp);
      return Ok(Expr::FunctionCall {
        name: "HeavisideTheta".to_string(),
        args: sorted_args.into(),
      });
    }
    if remaining.is_empty() {
      return Ok(Expr::Integer(1));
    }
    // Sort remaining args for canonical form
    remaining.sort_by(crate::functions::list_helpers_ast::canonical_cmp);
    return Ok(Expr::FunctionCall {
      name: "HeavisideTheta".to_string(),
      args: remaining.into(),
    });
  }

  // Single arg
  match &args[0] {
    Expr::Integer(n) => {
      if *n > 0 {
        Ok(Expr::Integer(1))
      } else if *n < 0 {
        Ok(Expr::Integer(0))
      } else {
        // HeavisideTheta[0] stays symbolic
        Ok(Expr::FunctionCall {
          name: "HeavisideTheta".to_string(),
          args: vec![Expr::Integer(0)].into(),
        })
      }
    }
    Expr::Real(f) => {
      if *f > 0.0 {
        Ok(Expr::Integer(1))
      } else if *f < 0.0 {
        Ok(Expr::Integer(0))
      } else {
        Ok(Expr::FunctionCall {
          name: "HeavisideTheta".to_string(),
          args: vec![Expr::Real(0.0)].into(),
        })
      }
    }
    Expr::Constant(c) => match c.as_str() {
      "Pi" | "E" | "Degree" => Ok(Expr::Integer(1)),
      _ => Ok(Expr::FunctionCall {
        name: "HeavisideTheta".to_string(),
        args: args.to_vec().into(),
      }),
    },
    Expr::Identifier(name) if name == "Infinity" => Ok(Expr::Integer(1)),
    // Exact rationals and real-valued symbolic numerics (Sqrt[2] - 2, …):
    // HeavisideTheta[x] = 1 for x > 0, 0 for x < 0, and stays unevaluated at 0.
    other => match crate::functions::math_ast::try_eval_to_f64(other) {
      Some(v) if v > 0.0 => Ok(Expr::Integer(1)),
      Some(v) if v < 0.0 => Ok(Expr::Integer(0)),
      _ => Ok(Expr::FunctionCall {
        name: "HeavisideTheta".to_string(),
        args: args.to_vec().into(),
      }),
    },
  }
}

pub fn dirac_delta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "DiracDelta expects at least 1 argument".into(),
    ));
  }

  // For numeric arguments: 0 for any non-zero value, symbolic at 0
  for arg in args {
    match arg {
      Expr::Integer(n) if *n != 0 => return Ok(Expr::Integer(0)),
      Expr::Real(f) if *f != 0.0 => return Ok(Expr::Integer(0)),
      Expr::Constant(_) => return Ok(Expr::Integer(0)), // Pi, E, etc. are non-zero
      Expr::Identifier(name) if name == "Infinity" => {
        return Ok(Expr::Integer(0));
      }
      _ => {}
    }
  }

  // If all args are zero, or mixed with symbolic, stay symbolic
  Ok(Expr::FunctionCall {
    name: "DiracDelta".to_string(),
    args: args.to_vec().into(),
  })
}

/// UnitBox[x] = 1 for |x| <= 1/2, 0 otherwise
pub fn unit_box_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let x = &args[0];
  match x {
    Expr::Integer(n) => {
      let v = (*n).abs();
      // |n| <= 1/2 only if n == 0
      Ok(Expr::Integer(if v == 0 { 1 } else { 0 }))
    }
    Expr::Real(f) => {
      let v = f.abs();
      Ok(Expr::Integer(if v <= 0.5 { 1 } else { 0 }))
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Rational" && fargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&fargs[0], &fargs[1])
        && *d != 0
      {
        let abs_val = (*n as f64 / *d as f64).abs();
        return Ok(Expr::Integer(if abs_val <= 0.5 { 1 } else { 0 }));
      }
      Ok(Expr::FunctionCall {
        name: "UnitBox".to_string(),
        args: args.to_vec().into(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "UnitBox".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// HeavisidePi[x] = 1 for |x| < 1/2, 0 for |x| > 1/2, unevaluated at |x| = 1/2
pub fn heaviside_pi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let x = &args[0];
  match x {
    Expr::Integer(n) => {
      // |n| < 1/2 only if n == 0
      Ok(Expr::Integer(if *n == 0 { 1 } else { 0 }))
    }
    Expr::Real(f) => {
      let v = f.abs();
      if (v - 0.5).abs() < f64::EPSILON {
        // At boundary, return unevaluated
        Ok(Expr::FunctionCall {
          name: "HeavisidePi".to_string(),
          args: args.to_vec().into(),
        })
      } else if v < 0.5 {
        Ok(Expr::Integer(1))
      } else {
        Ok(Expr::Integer(0))
      }
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Rational" && fargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&fargs[0], &fargs[1])
        && *d != 0
      {
        // Compare |n/d| with 1/2: |2*n| vs |d|
        let two_n_abs = (2 * *n).unsigned_abs();
        let d_abs = d.unsigned_abs();
        return if two_n_abs == d_abs {
          // At boundary, return unevaluated
          Ok(Expr::FunctionCall {
            name: "HeavisidePi".to_string(),
            args: args.to_vec().into(),
          })
        } else if two_n_abs < d_abs {
          Ok(Expr::Integer(1))
        } else {
          Ok(Expr::Integer(0))
        };
      }
      Ok(Expr::FunctionCall {
        name: "HeavisidePi".to_string(),
        args: args.to_vec().into(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "HeavisidePi".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// UnitTriangle[x] = 1 - |x| for |x| <= 1, 0 otherwise
pub fn unit_triangle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let x = &args[0];
  match x {
    Expr::Integer(n) => {
      let v = (*n).abs();
      Ok(if v <= 1 {
        Expr::Integer(1 - v)
      } else {
        Expr::Integer(0)
      })
    }
    Expr::Real(f) => {
      let v = f.abs();
      Ok(if v <= 1.0 {
        Expr::Real(1.0 - v)
      } else {
        Expr::Integer(0)
      })
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Rational" && fargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&fargs[0], &fargs[1])
        && *d != 0
      {
        let abs_val = (*n as f64 / *d as f64).abs();
        if abs_val <= 1.0 {
          // 1 - |n/d| = (|d| - |n|) / |d|
          let abs_n = n.abs();
          let abs_d = d.abs();
          let num = abs_d - abs_n;
          if num == 0 {
            return Ok(Expr::Integer(0));
          }
          return Ok(Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(num), Expr::Integer(abs_d)].into(),
          });
        } else {
          return Ok(Expr::Integer(0));
        }
      }
      Ok(Expr::FunctionCall {
        name: "UnitTriangle".to_string(),
        args: args.to_vec().into(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "UnitTriangle".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// HeavisideLambda[x] = 1 - |x| for |x| < 1, 0 for |x| >= 1
pub fn heaviside_lambda_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let x = &args[0];
  match x {
    Expr::Integer(n) => {
      let v = (*n).abs();
      Ok(if v < 1 {
        Expr::Integer(1 - v)
      } else {
        Expr::Integer(0)
      })
    }
    Expr::Real(f) => {
      let v = f.abs();
      Ok(if v < 1.0 {
        Expr::Real(1.0 - v)
      } else {
        Expr::Real(0.0)
      })
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Rational" && fargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&fargs[0], &fargs[1])
        && *d != 0
      {
        let abs_n = n.abs();
        let abs_d = d.abs();
        // Compare |n/d| with 1: |n| vs |d|
        if abs_n >= abs_d {
          return Ok(Expr::Integer(0));
        }
        // 1 - |n/d| = (|d| - |n|) / |d|
        let num = abs_d - abs_n;
        return Ok(crate::functions::math_ast::make_rational_pub(num, abs_d));
      }
      Ok(Expr::FunctionCall {
        name: "HeavisideLambda".to_string(),
        args: args.to_vec().into(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "HeavisideLambda".to_string(),
      args: args.to_vec().into(),
    }),
  }
}
