#[allow(unused_imports)]
use super::*;

/// PolyLog[s, z] - Polylogarithm function Li_s(z)
pub fn polylog_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "PolyLog expects exactly 2 arguments".into(),
    ));
  }

  let s_expr = &args[0];
  let z_expr = &args[1];

  // PolyLog[s, 0] = 0 for any s
  if matches!(z_expr, Expr::Integer(0)) {
    return Ok(Expr::Integer(0));
  }

  // PolyLog[s, 1] for integer s >= 2: return Zeta[s].
  // For symbolic s, Wolfram keeps PolyLog[s, 1] unevaluated — we match that.
  if matches!(z_expr, Expr::Integer(1))
    && let Expr::Integer(n) = s_expr
    && *n >= 2
  {
    return zeta_ast(&[Expr::Integer(*n)]);
  }

  // PolyLog[s, 1] for an exact rational s > 1 is Zeta[s].
  if matches!(z_expr, Expr::Integer(1))
    && matches!(s_expr, Expr::FunctionCall { name, .. } if name == "Rational")
    && try_eval_to_f64(s_expr).is_some_and(|sf| sf > 1.0)
  {
    return zeta_ast(std::slice::from_ref(s_expr));
  }

  // An exact rational order with an inexact argument evaluates numerically.
  if matches!(s_expr, Expr::FunctionCall { name, .. } if name == "Rational")
    && matches!(z_expr, Expr::Real(_))
    && let (Some(sf), Some(zf)) =
      (try_eval_to_f64(s_expr), try_eval_to_f64(z_expr))
  {
    return Ok(Expr::Real(polylog_numeric(sf, zf)));
  }

  match s_expr {
    Expr::Integer(s) => {
      return polylog_integer_s(*s, z_expr, args);
    }
    Expr::Real(sf)
      if sf.fract() != 0.0
        && let Some(z) = try_extract_complex_float(z_expr)
        && (z.1 != 0.0 || z.0 < -0.5)
        && let Some((re, im)) = polylog_complex(*sf, z) =>
    {
      // A complex argument, or one near -1 where the power series stops
      // converging fast enough.
      if z.1 == 0.0 {
        return Ok(Expr::Real(re));
      }
      return Ok(build_complex_float_expr(re, im));
    }
    Expr::Real(sf) => {
      // A machine-real order numericizes the whole call, so the argument may be
      // any exact numeric (an integer, a Rational such as 1/2, or a constant
      // expression like Pi/4) — not only Integer/Real. `PolyLog[2., 1/2]` must
      // evaluate, matching wolframscript.
      if let Some(zf) = try_eval_to_f64(z_expr) {
        if sf.fract() == 0.0 && (2.0..=60.0).contains(sf) && zf > 1.0 {
          let (re, im) = polylog_integer_above_one(*sf as usize, zf);
          return Ok(build_complex_float_expr(re, im));
        }
        return Ok(Expr::Real(polylog_numeric(*sf, zf)));
      }
    }
    _ => {}
  }

  // PolyLog[s, -1] = -(1 - 2^(1 - s)) Zeta[s] for symbolic s (the alternating
  // zeta / Dirichlet-eta relation). Integer and real s are handled above.
  if matches!(z_expr, Expr::Integer(-1)) {
    return polylog_at_neg1_symbolic(s_expr);
  }

  Ok(unevaluated("PolyLog", args))
}

/// PolyLog[s, -1] = -(1 - 2^(1 - s)) Zeta[s] for a symbolic order `s`.
fn polylog_at_neg1_symbolic(s: &Expr) -> Result<Expr, InterpreterError> {
  // 2^(1 - s)
  let pow = pow(
    Expr::Integer(2),
    plus(vec![Expr::Integer(1), neg(s.clone())]),
  );
  // 1 - 2^(1 - s)
  let inner = plus(vec![Expr::Integer(1), neg(pow)]);
  // -(1 - 2^(1 - s)) Zeta[s]
  let result = times(vec![Expr::Integer(-1), inner, call1("Zeta", s.clone())]);
  crate::evaluator::evaluate_expr_to_expr(&result)
}

fn polylog_integer_s(
  s: i128,
  z_expr: &Expr,
  orig_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // s = 1: PolyLog[1, z] = -Log[1-z]
  if s == 1 {
    return polylog_s1(z_expr);
  }

  // s = 0: PolyLog[0, z] = z/(1-z)
  if s == 0 {
    return polylog_s0(z_expr);
  }

  // s < 0: rational function via Eulerian numbers
  if s < 0 {
    return polylog_negative_s((-s) as usize, z_expr);
  }

  // s >= 2: special values at z = 0, 1, -1, 1/2
  match z_expr {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Integer(1) => {
      // PolyLog[s, 1] = Zeta[s]
      return zeta_ast(&[Expr::Integer(s)]);
    }
    Expr::Integer(-1) => {
      // PolyLog[s, -1] = -(1 - 2^{1-s}) * Zeta[s]
      return Ok(polylog_at_neg1(s));
    }
    Expr::Real(f) if *f > 1.0 && s <= 60 => {
      let (re, im) = polylog_integer_above_one(s as usize, *f);
      return Ok(build_complex_float_expr(re, im));
    }
    Expr::Real(f) => {
      return Ok(Expr::Real(polylog_numeric(s as f64, *f)));
    }
    Expr::FunctionCall {
      name: rname,
      args: rargs,
    } if rname == "Rational"
      && rargs.len() == 2
      && matches!(
        (&rargs[0], &rargs[1]),
        (Expr::Integer(1), Expr::Integer(2))
      ) =>
    {
      // Known closed forms at z = 1/2.
      if let Some(expr) = polylog_at_half(s) {
        return Ok(expr);
      }
    }
    _ => {}
  }

  Ok(unevaluated("PolyLog", orig_args))
}

/// PolyLog[1, z] = -Log[1-z]
fn polylog_s1(z_expr: &Expr) -> Result<Expr, InterpreterError> {
  match z_expr {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(1) => Ok(id_expr("Infinity")),
    Expr::Integer(z) => {
      // 1-z is an integer, construct -Log[1-z]
      let one_minus_z = Expr::Integer(1 - z);
      Ok(times2(Expr::Integer(-1), call1("Log", one_minus_z)))
    }
    Expr::FunctionCall {
      name,
      args: rat_args,
    } if name == "Rational" && rat_args.len() == 2 => {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rat_args[0], &rat_args[1])
      {
        // 1 - p/q = (q-p)/q
        let one_minus_z = make_rational(q - p, *q);
        // Evaluate Log to get simplification (e.g., Log[1/2] → -Log[2])
        let log_val = log_ast(&[one_minus_z])?;
        // Negate: -Log[1 - p/q], simplifying double negation
        match &log_val {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left,
            right,
          } if matches!(left.as_ref(), Expr::Integer(-1)) => {
            // -(-expr) = expr
            Ok(*right.clone())
          }
          _ => Ok(times2(Expr::Integer(-1), log_val)),
        }
      } else {
        Ok(polylog_s1_symbolic(z_expr))
      }
    }
    Expr::Real(f) => {
      let result = -(1.0 - f).ln();
      Ok(Expr::Real(result))
    }
    _ => Ok(polylog_s1_symbolic(z_expr)),
  }
}

fn polylog_s1_symbolic(z_expr: &Expr) -> Expr {
  let one_minus_z = minus2(Expr::Integer(1), z_expr.clone());
  times2(Expr::Integer(-1), call1("Log", one_minus_z))
}

/// PolyLog[0, z] = z/(1-z)
fn polylog_s0(z_expr: &Expr) -> Result<Expr, InterpreterError> {
  match z_expr {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(1) => Ok(id_expr("ComplexInfinity")),
    Expr::Integer(z) => {
      // z/(1-z) as rational
      Ok(make_rational(*z, 1 - z))
    }
    Expr::Real(f) => Ok(Expr::Real(f / (1.0 - f))),
    _ => {
      // z/(1-z)
      let result =
        div2(z_expr.clone(), minus2(Expr::Integer(1), z_expr.clone()));
      // Fold an exact rational z (PolyLog[0, 1/2] = 1); keep symbolic z.
      if matches!(z_expr, Expr::FunctionCall { name, .. } if name == "Rational")
        || try_extract_complex_exact(z_expr).is_some()
      {
        crate::evaluator::evaluate_expr_to_expr(&result)
      } else {
        Ok(result)
      }
    }
  }
}

/// PolyLog[-n, z] for n >= 1 using Eulerian numbers
fn polylog_negative_s(
  n: usize,
  z_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  match z_expr {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Integer(1) => {
      return Ok(id_expr("ComplexInfinity"));
    }
    // Inexact input → machine number.
    Expr::Real(f) => return Ok(Expr::Real(polylog_numeric(-(n as f64), *f))),
    _ => {}
  }
  // Exact numeric z (integer or rational) folds via the rational closed form
  // below — the analytic continuation is valid for any z != 1, unlike the
  // numeric series which diverges for |z| >= 1.
  // An exact Gaussian rational (`I`, `2 + I`) folds the same way.
  let exact_numeric = matches!(z_expr, Expr::Integer(_))
    || matches!(z_expr, Expr::FunctionCall { name, .. } if name == "Rational")
    || try_extract_complex_exact(z_expr).is_some();

  // Compute Eulerian numbers A(n, k) for k = 0..n-1
  let eulerian = eulerian_numbers(n);

  // Build numerator: Σ A(n, k) * x^{k+1}
  let mut terms: Vec<Expr> = Vec::new();
  for (k, &a) in eulerian.iter().enumerate() {
    if a == 0 {
      continue;
    }
    let x_power = if k + 1 == 1 {
      z_expr.clone()
    } else {
      pow2(z_expr.clone(), Expr::Integer((k + 1) as i128))
    };
    let term = if a == 1 {
      x_power
    } else {
      times2(Expr::Integer(a), x_power)
    };
    terms.push(term);
  }

  let numerator = if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    plus(terms)
  };

  // Denominator: (1 - x)^{n+1}
  let one_minus_x = minus2(Expr::Integer(1), z_expr.clone());
  let denominator = pow2(one_minus_x, Expr::Integer((n + 1) as i128));

  let result = div2(numerator, denominator);
  // Exact numeric z: fold the rational expression to a number (e.g.
  // PolyLog[-1, 1/2] = 2). Symbolic z keeps the x/(1-x)^... display form.
  if exact_numeric {
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else {
    Ok(result)
  }
}

/// Compute Eulerian numbers A(n, k) for k = 0, ..., n-1
fn eulerian_numbers(n: usize) -> Vec<i128> {
  if n == 0 {
    return vec![];
  }
  let mut a = vec![0_i128; n];
  a[0] = 1;
  for m in 2..=n {
    let mut new_a = vec![0_i128; n];
    new_a[0] = 1;
    for k in 1..m {
      new_a[k] = (k as i128 + 1) * a[k] + (m as i128 - k as i128) * a[k - 1];
    }
    a = new_a;
  }
  a[..n].to_vec()
}

/// PolyLog[s, -1] = -(1 - 2^{1-s}) * Zeta[s] for s >= 2
fn polylog_at_neg1(s: i128) -> Expr {
  let s_usize = s as usize;

  if s % 2 == 0 {
    // Even s: Zeta[s] is exact
    if let Some((b_num, b_den)) = bernoulli_number(s_usize) {
      if b_num == 0 {
        return Expr::Integer(0);
      }

      // Compute Zeta coefficient: |B_{2n}| * 2^{2n-1} / (2n)!
      let mut znum = b_num.abs();
      let mut zden = b_den.abs();
      for _ in 0..(s_usize - 1) {
        znum = match znum.checked_mul(2) {
          Some(v) => v,
          None => return unevaluated_polylog(s, -1),
        };
        (znum, zden) = rat_reduce(znum, zden);
      }
      for k in 1..=s_usize {
        zden = match zden.checked_mul(k as i128) {
          Some(v) => v,
          None => return unevaluated_polylog(s, -1),
        };
        (znum, zden) = rat_reduce(znum, zden);
      }

      // Multiply by -(1 - 2^{1-s}) = (1 - 2^{s-1}) / 2^{s-1}
      let exp2 = 1_i128 << (s_usize - 1);
      let coeff_num = 1 - exp2; // negative
      let coeff_den = exp2;

      let Some(mut final_num) = coeff_num.checked_mul(znum) else {
        return unevaluated_polylog(s, -1);
      };
      let Some(mut final_den) = coeff_den.checked_mul(zden) else {
        return unevaluated_polylog(s, -1);
      };
      (final_num, final_den) = rat_reduce(final_num, final_den);

      // Build coefficient * Pi^s
      let pi_power = pow2(id_expr("Pi"), Expr::Integer(s));

      if final_num.abs() == 1 && final_den == 1 {
        if final_num == 1 {
          return pi_power;
        }
        return times2(Expr::Integer(-1), pi_power);
      } else if final_num.abs() == 1 {
        return times2(make_rational(final_num, final_den), pi_power);
      } else if final_den == 1 {
        return times2(Expr::Integer(final_num), pi_power);
      }
      return div2(
        times2(Expr::Integer(final_num), pi_power),
        Expr::Integer(final_den),
      );
    }
  } else {
    // Odd s >= 3: Zeta stays symbolic
    // PolyLog[s, -1] = cn/cd * Zeta[s] where cn < 0
    let exp2 = 1_i128 << (s_usize - 1);
    let coeff_num = 1 - exp2;
    let coeff_den = exp2;
    let (cn, cd) = rat_reduce(coeff_num, coeff_den);

    let zeta_expr = call1("Zeta", Expr::Integer(s));

    if cd == 1 {
      return times2(Expr::Integer(cn), zeta_expr);
    }
    // Use (cn*Zeta[s])/cd format
    return div2(times2(Expr::Integer(cn), zeta_expr), Expr::Integer(cd));
  }

  unevaluated_polylog(s, -1)
}

/// Known exact closed forms of `PolyLog[s, 1/2]`.
///
///   PolyLog[2, 1/2] = Pi^2/12 - Log[2]^2/2
///   PolyLog[3, 1/2] = (-2*Pi^2*Log[2] + 4*Log[2]^3 + 21*Zeta[3])/24
fn polylog_at_half(s: i128) -> Option<Expr> {
  let pi = id_expr("Pi");
  let log2 = call1("Log", Expr::Integer(2));

  if s == 2 {
    // Pi^2/12 - Log[2]^2/2
    let pi2 = pow2(pi, Expr::Integer(2));
    let log2_sq = pow2(log2, Expr::Integer(2));
    let lhs = div2(pi2, Expr::Integer(12));
    let rhs = div2(log2_sq, Expr::Integer(2));
    return Some(minus2(lhs, rhs));
  }

  if s == 3 {
    // (-2*Pi^2*Log[2] + 4*Log[2]^3 + 21*Zeta[3])/24
    let pi2 = pow2(pi, Expr::Integer(2));
    // -2*Pi^2*Log[2]
    let term1 = times(vec![Expr::Integer(-2), pi2, log2.clone()]);
    // 4*Log[2]^3
    let log2_cubed = pow2(log2, Expr::Integer(3));
    let term2 = times2(Expr::Integer(4), log2_cubed);
    // 21*Zeta[3]
    let zeta3 = call1("Zeta", Expr::Integer(3));
    let term3 = times2(Expr::Integer(21), zeta3);
    let numer = plus(vec![term1, term2, term3]);
    return Some(div2(numer, Expr::Integer(24)));
  }

  None
}

fn unevaluated_polylog(s: i128, z: i128) -> Expr {
  call("PolyLog", vec![Expr::Integer(s), Expr::Integer(z)])
}

/// Compute polylogarithm numerically using series summation
fn polylog_numeric(s: f64, z: f64) -> f64 {
  if z == 0.0 {
    return 0.0;
  }
  if z.abs() <= 1.0 {
    let mut sum = 0.0;
    let mut z_power = z;
    for k in 1..=10000 {
      let term = z_power / (k as f64).powf(s);
      sum += term;
      if term.abs() < 1e-15 * sum.abs().max(1e-300) {
        break;
      }
      z_power *= z;
    }
    sum
  } else if s <= 0.0 && s.fract() == 0.0 {
    polylog_nonpositive_integer(-s as usize, z)
  } else if z < -1.0 && s > 0.0 {
    -fermi_dirac_integral(s, (-z).ln())
  } else {
    f64::NAN
  }
}

/// Li_n(x) for an integer order n >= 2 and a real x > 1, on the branch cut
/// that starts at 1. The inversion formula
/// Li_n(z) = -(-1)^n Li_n(1/z) - (2 Pi I)^n/n! BernoulliB[n, 1/2 + Log[-z]/(2 Pi I)]
/// with the principal Log[-x] = Log[x] + I Pi gives the value wolframscript
/// returns, e.g. PolyLog[2, 2.] = Pi^2/4 - I Pi Log[2].
fn polylog_integer_above_one(n: usize, x: f64) -> (f64, f64) {
  use num_traits::ToPrimitive;
  type C = (f64, f64);
  let mul = |a: C, b: C| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
  let two_pi = 2.0 * std::f64::consts::PI;
  // t = 1/2 + (Log[x] + I Pi)/(2 Pi I) = 1 - I Log[x]/(2 Pi)
  let t: C = (1.0, -x.ln() / two_pi);
  // BernoulliB[n, t] = Σ_k Binomial[n, k] B_k t^(n-k), by Horner in t.
  let mut b: C = (0.0, 0.0);
  let mut binom = 1.0;
  let mut coeffs = Vec::with_capacity(n + 1);
  for k in 0..=n {
    let (num, den) = super::number_theory::bernoulli_big(k);
    let bk = num.to_f64().unwrap_or(0.0) / den.to_f64().unwrap_or(1.0);
    coeffs.push(binom * bk);
    binom = binom * (n - k) as f64 / (k + 1) as f64;
  }
  // coeffs[k] multiplies t^(n-k): Horner from k = 0 (highest power).
  for c in &coeffs {
    b = mul(b, t);
    b.0 += c;
  }
  // (2 Pi I)^n / n!
  let mut factor: C = (1.0, 0.0);
  for k in 1..=n {
    factor = mul(factor, (0.0, two_pi / k as f64));
  }
  let tail = mul(factor, b);
  let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
  let inverse = polylog_numeric(n as f64, 1.0 / x);
  (sign * inverse - tail.0, -tail.1)
}

/// Li_{-n}(z) = Σ A(n, k) z^(k+1) / (1 - z)^(n+1), valid for every z != 1.
fn polylog_nonpositive_integer(n: usize, z: f64) -> f64 {
  if n == 0 {
    return z / (1.0 - z);
  }
  let numerator: f64 = eulerian_numbers(n)
    .iter()
    .enumerate()
    .map(|(k, &a)| a as f64 * z.powi(k as i32 + 1))
    .sum();
  numerator / (1.0 - z).powi(n as i32 + 1)
}

/// The complete Fermi–Dirac integral
/// `1/Gamma(s) * Integral[t^(s-1) / (Exp[t - mu] + 1), {t, 0, Infinity}]`
/// for `s > 0`, which equals `-PolyLog[s, -Exp[mu]]`. The substitution
/// `t = u^(2/s)` removes the endpoint singularity of `t^(s-1)`, and the
/// remaining smooth integrand is summed with composite Gauss–Legendre panels.
fn fermi_dirac_integral(s: f64, mu: f64) -> f64 {
  const NODES: usize = 24;
  const PANELS: usize = 256;
  let (xs, ws) = gauss_legendre_nodes(NODES);
  let p = 2.0 / s;
  let t_max = mu.max(0.0) + 45.0;
  let u_max = t_max.powf(1.0 / p);
  let h = u_max / PANELS as f64;
  let mut sum = 0.0;
  for panel in 0..PANELS {
    let a = panel as f64 * h;
    for (x, w) in xs.iter().zip(&ws) {
      let u = a + 0.5 * h * (x + 1.0);
      let t = u.powf(p);
      let weight = 0.5 * h * w * p * u.powf(p * s - 1.0);
      sum += weight / ((t - mu).exp() + 1.0);
    }
  }
  sum / super::gamma::gamma_fn(s)
}

/// Nodes and weights of the `n`-point Gauss–Legendre rule on [-1, 1].
fn gauss_legendre_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
  let mut xs = vec![0.0; n];
  let mut ws = vec![0.0; n];
  for i in 0..n {
    let mut x =
      (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
    let mut dp = 1.0;
    for _ in 0..100 {
      let (mut p0, mut p1) = (1.0, x);
      for k in 2..=n {
        let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
        p0 = p1;
        p1 = p2;
      }
      dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
      let dx = p1 / dp;
      x -= dx;
      if dx.abs() < 1e-15 {
        break;
      }
    }
    xs[i] = x;
    ws[i] = 2.0 / ((1.0 - x * x) * dp * dp);
  }
  (xs, ws)
}

/// Li_s(z) for a real non-integer order `s` and complex `z` with |z| ≤ 1:
/// the power series near the origin, and otherwise the expansion in
/// μ = Log[z],  Li_s(e^μ) = Γ(1-s) (-μ)^(s-1) + Σ_k ζ(s-k) μ^k / k!,
/// which converges for |μ| < 2π and so covers the unit circle.
pub(crate) fn polylog_complex(s: f64, z: (f64, f64)) -> Option<(f64, f64)> {
  type C = (f64, f64);
  let mul = |a: C, b: C| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
  let r = z.0.hypot(z.1);
  if s.fract() == 0.0 || r > 1.0 + 1e-12 {
    return None;
  }
  if r <= 0.5 {
    let mut sum = (0.0, 0.0);
    let mut zk = z;
    for k in 1..=200 {
      let term = mul(zk, ((k as f64).powf(-s), 0.0));
      sum = (sum.0 + term.0, sum.1 + term.1);
      if term.0.hypot(term.1) <= 1e-17 * sum.0.hypot(sum.1).max(1e-300) {
        break;
      }
      zk = mul(zk, z);
    }
    return Some(sum);
  }
  let mu: C = (r.ln(), z.1.atan2(z.0));
  if mu.0.hypot(mu.1) > 5.0 {
    return None;
  }
  // Γ(1-s) (-μ)^(s-1) with the principal power.
  let neg_mu = (-mu.0, -mu.1);
  let (m, arg) = (neg_mu.0.hypot(neg_mu.1), neg_mu.1.atan2(neg_mu.0));
  let mag = super::gamma::gamma_fn(1.0 - s) * m.powf(s - 1.0);
  let mut sum = (mag * ((s - 1.0) * arg).cos(), mag * ((s - 1.0) * arg).sin());
  let mut mu_k: C = (1.0, 0.0);
  let mut factorial = 1.0;
  // ζ(s - k) is nearly zero close to the trivial zeros, so one small term
  // does not mean convergence: stop after several in a row.
  let mut small = 0;
  for k in 0..400 {
    if k > 0 {
      mu_k = mul(mu_k, mu);
      factorial *= k as f64;
    }
    let zeta = super::zeta_functions::zeta_real_f64(s - k as f64);
    let term = (zeta * mu_k.0 / factorial, zeta * mu_k.1 / factorial);
    sum = (sum.0 + term.0, sum.1 + term.1);
    if term.0.hypot(term.1) <= 1e-17 * sum.0.hypot(sum.1).max(1e-300) {
      small += 1;
      if small >= 4 {
        break;
      }
    } else {
      small = 0;
    }
  }
  Some(sum)
}
