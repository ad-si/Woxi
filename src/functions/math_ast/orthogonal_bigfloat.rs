//! Real-argument values of the orthogonal-polynomial family at non-integer
//! degree, at any precision: the single numeric core behind machine and
//! arbitrary-precision `LegendreP[ν, x]`, `LaguerreL[ν, (λ,) x]`,
//! `HermiteH[ν, x]`, `Fibonacci[ν, x]` and `QPochhammer[a, q, n]`.
//!
//! The polynomials are generalised through their hypergeometric forms
//! (DLMF 14.3.1, 18.5.12, 18.5.13 / 13.2.38), summed by [`pfq_bigfloat`] with
//! guard bits grown until the series' cancellation is absorbed. Machine values
//! are rounded from a 64-bit result.

use astro_float::{BigFloat, Consts, RoundingMode};

use crate::syntax::Expr;

use crate::functions::math_ast::{
  bigfloat_to_f64_approx, gamma_bigfloat, with_thread_consts,
};

/// Binary exponent of `b`, or a huge negative number for zero.
fn bf_exp(b: &BigFloat) -> i64 {
  if b.is_zero() {
    i64::MIN / 4
  } else {
    i64::from(b.exponent().unwrap_or(i32::MIN / 4))
  }
}

/// `b` is a non-positive integer (a pole of `Gamma`).
fn is_nonpositive_int(b: &BigFloat) -> bool {
  b.is_zero() || (b.is_int() && !b.is_positive())
}

/// `1/Γ(z)`, zero at the poles of `Γ`.
fn rgamma(
  z: &BigFloat,
  wp: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> BigFloat {
  if is_nonpositive_int(z) {
    return BigFloat::from_i32(0, wp);
  }
  BigFloat::from_i32(1, wp).div(&gamma_bigfloat(z, wp, rm, cc), wp, rm)
}

/// The generalised hypergeometric series `pFq(a; b; z)` summed at `wp` bits.
/// Returns the sum together with the number of bits its largest term exceeds
/// it by (the bits lost to cancellation), or `None` when a lower parameter is
/// a pole, or when the series does not converge within the term budget.
fn pfq_series(
  a: &[BigFloat],
  b: &[BigFloat],
  z: &BigFloat,
  wp: usize,
  rm: RoundingMode,
) -> Option<(BigFloat, i64)> {
  if b.iter().any(is_nonpositive_int) {
    return None;
  }
  // A non-positive integer upper parameter ends the series: a polynomial.
  let terminating = a.iter().any(is_nonpositive_int);
  if a.len() > b.len() + 1
    || (a.len() == b.len() + 1 && !terminating && bf_exp(z) > 0)
  {
    return None;
  }
  let mut term = BigFloat::from_i32(1, wp);
  let mut sum = BigFloat::from_i32(1, wp);
  let mut max_exp = 1i64;
  let mut small_run = 0;
  let max_terms = 64 * wp + 20_000;
  for k in 0..max_terms {
    let kf = BigFloat::from_i64(k as i64, wp);
    let mut num = z.clone();
    for ai in a {
      num = num.mul(&ai.add(&kf, wp, rm), wp, rm);
    }
    let mut den = BigFloat::from_i64(k as i64 + 1, wp);
    for bi in b {
      den = den.mul(&bi.add(&kf, wp, rm), wp, rm);
    }
    term = term.mul(&num, wp, rm).div(&den, wp, rm);
    if term.is_zero() {
      let lost = max_exp - bf_exp(&sum);
      return Some((sum, lost));
    }
    sum = sum.add(&term, wp, rm);
    max_exp = max_exp.max(bf_exp(&term));
    // Past the largest term (the ratio has settled below 1), stop once two
    // successive terms fall below the working precision of the sum.
    if bf_exp(&term) < bf_exp(&sum) - wp as i64 - 2 {
      small_run += 1;
      if small_run >= 2 {
        let lost = max_exp - bf_exp(&sum);
        return Some((sum, lost));
      }
    } else {
      small_run = 0;
    }
  }
  None
}

/// Runs `eval` at `bits` plus a guard, growing the guard until the bits the
/// evaluation reports losing to cancellation leave `bits` + 16 correct ones.
fn with_guard(
  bits: usize,
  mut eval: impl FnMut(usize) -> Option<(BigFloat, i64)>,
) -> Option<BigFloat> {
  let cap = 16 * bits + 8192;
  let mut guard = 48usize;
  loop {
    let (value, lost) = eval(bits + guard)?;
    let lost = lost.max(0) as usize;
    if lost + 16 <= guard || guard > cap {
      return Some(value);
    }
    guard = (lost + 64).min(cap + 1);
  }
}

/// `pFq(a; b; z)` correct to `bits` bits.
fn pfq_bigfloat(
  a: &[BigFloat],
  b: &[BigFloat],
  z: &BigFloat,
  bits: usize,
  rm: RoundingMode,
) -> Option<BigFloat> {
  with_guard(bits, |wp| pfq_series(a, b, z, wp, rm))
}

/// `LegendreP[ν, x]` (type 1) for real `ν` and real `x > -1`
/// (DLMF 14.3.1): `₂F₁(−ν, ν+1; 1; (1−x)/2)`, and for `x > 1` its Pfaff
/// transform `((1+x)/2)^ν ₂F₁(−ν, −ν; 1; (x−1)/(x+1))`, which converges
/// there.
pub(crate) fn legendre_p_bigfloat(
  nu: &BigFloat,
  x: &BigFloat,
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  let wp = bits + 64;
  let one = BigFloat::from_i32(1, wp);
  let minus_one = BigFloat::from_i32(-1, wp);
  if x.cmp(&minus_one)? <= 0 {
    return None;
  }
  let neg_nu = nu.neg();
  if x.cmp(&one)? > 0 {
    let z = x.sub(&one, wp, rm).div(&x.add(&one, wp, rm), wp, rm);
    let f = pfq_bigfloat(
      &[neg_nu.clone(), neg_nu],
      std::slice::from_ref(&one),
      &z,
      wp,
      rm,
    )?;
    let half_sum = x.add(&one, wp, rm).div(&BigFloat::from_i32(2, wp), wp, rm);
    return Some(half_sum.pow(nu, wp, rm, cc).mul(&f, bits, rm));
  }
  let z = one.sub(x, wp, rm).div(&BigFloat::from_i32(2, wp), wp, rm);
  let nu1 = nu.add(&one, wp, rm);
  pfq_bigfloat(&[neg_nu, nu1], &[one], &z, bits, rm)
}

/// `LaguerreL[ν, λ, x]` for real arguments (DLMF 18.5.12):
/// `Γ(ν+λ+1)/(Γ(ν+1) Γ(λ+1)) ₁F₁(−ν; λ+1; x)`.
pub(crate) fn laguerre_l_bigfloat(
  nu: &BigFloat,
  lambda: &BigFloat,
  x: &BigFloat,
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  let wp = bits + 64;
  let one = BigFloat::from_i32(1, wp);
  let nu1 = nu.add(&one, wp, rm);
  let lam1 = lambda.add(&one, wp, rm);
  let top = nu.add(&lam1, wp, rm);
  // The binomial is 0/0 or ∞ at the poles; those are exact limits the
  // symbolic layer owns.
  if is_nonpositive_int(&nu1)
    || is_nonpositive_int(&lam1)
    || is_nonpositive_int(&top)
  {
    return None;
  }
  let binomial = gamma_bigfloat(&top, wp, rm, cc)
    .mul(&rgamma(&nu1, wp, rm, cc), wp, rm)
    .mul(&rgamma(&lam1, wp, rm, cc), wp, rm);
  let f = pfq_bigfloat(&[nu.neg()], &[lam1], x, wp, rm)?;
  Some(binomial.mul(&f, bits, rm))
}

/// `HermiteH[ν, x]` for real arguments (DLMF 18.5.13 continued to real `ν`,
/// as `2^ν √π (₁F₁(−ν/2; 1/2; x²)/Γ((1−ν)/2) − 2x ₁F₁((1−ν)/2; 3/2; x²)/Γ(−ν/2))`).
pub(crate) fn hermite_h_bigfloat(
  nu: &BigFloat,
  x: &BigFloat,
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  // The two terms cancel for large positive x (the function grows only like
  // x^ν there while each term grows like e^{x²}); the guard loop measures
  // that loss across the whole combination.
  with_guard(bits, |wp| {
    let two = BigFloat::from_i32(2, wp);
    let one = BigFloat::from_i32(1, wp);
    let half = one.div(&two, wp, rm);
    let x2 = x.mul(x, wp, rm);
    let neg_half_nu = nu.neg().div(&two, wp, rm);
    let a2 = one.sub(nu, wp, rm).div(&two, wp, rm);
    let (f1, l1) = pfq_series(
      std::slice::from_ref(&neg_half_nu),
      std::slice::from_ref(&half),
      &x2,
      wp,
      rm,
    )?;
    let three_half = half.add(&one, wp, rm);
    let (f2, l2) =
      pfq_series(std::slice::from_ref(&a2), &[three_half], &x2, wp, rm)?;
    let t1 = f1.mul(&rgamma(&a2, wp, rm, cc), wp, rm);
    let t2 = f2
      .mul(&rgamma(&neg_half_nu, wp, rm, cc), wp, rm)
      .mul(x, wp, rm)
      .mul(&two, wp, rm);
    let diff = t1.sub(&t2, wp, rm);
    let lost =
      (bf_exp(&t1).max(bf_exp(&t2)) - bf_exp(&diff)).max(0) + l1.max(l2);
    let pi = cc.pi(wp, rm);
    let value = two
      .pow(nu, wp, rm, cc)
      .mul(&pi.sqrt(wp, rm), wp, rm)
      .mul(&diff, wp, rm);
    Some((value, lost))
  })
}

/// `Fibonacci[ν, x]` for real arguments: with `φ = (x + √(x²+4))/2`,
/// `(φ^ν − cos(πν) φ^−ν)/√(x²+4)`, which at `x = 1` is Binet's
/// `(φ^ν − cos(πν) φ^−ν)/√5` behind `Fibonacci[ν]`.
pub(crate) fn fibonacci_bigfloat(
  nu: &BigFloat,
  x: &BigFloat,
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  with_guard(bits, |wp| {
    let root = x
      .mul(x, wp, rm)
      .add(&BigFloat::from_i32(4, wp), wp, rm)
      .sqrt(wp, rm);
    let phi = x.add(&root, wp, rm).div(&BigFloat::from_i32(2, wp), wp, rm);
    let up = phi.pow(nu, wp, rm, cc);
    let down = BigFloat::from_i32(1, wp).div(&up, wp, rm);
    let cos = cc.pi(wp, rm).mul(nu, wp, rm).cos(wp, rm, cc);
    let t2 = cos.mul(&down, wp, rm);
    let diff = up.sub(&t2, wp, rm);
    let lost = (bf_exp(&up).max(bf_exp(&t2)) - bf_exp(&diff)).max(0);
    Some((diff.div(&root, wp, rm), lost))
  })
}

/// `0F1(; b; z)/Γ(b)`; at a pole `b = −n` of `Γ` the regularized limit
/// `z^(n+1) 0F1(; n+2; z)/(n+1)!`.
fn hypergeometric_0f1_regularized_bigfloat(
  b: &BigFloat,
  z: &BigFloat,
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  let wp = bits + 64;
  if is_nonpositive_int(b) {
    let one = BigFloat::from_i32(1, wp);
    let shift = one.sub(b, wp, rm); // n + 1
    let b2 = shift.add(&one, wp, rm); // n + 2
    let f = pfq_bigfloat(&[], std::slice::from_ref(&b2), z, wp, rm)?;
    let n1 = i64::try_from(crate::functions::math_ast::bigfloat_to_f64_approx(
      &shift,
    )? as i64)
    .ok()?;
    let zp =
      crate::functions::math_ast::bigfloat_powi(z, i128::from(n1), wp, rm);
    return Some(zp.mul(&f, wp, rm).mul(&rgamma(&b2, wp, rm, cc), bits, rm));
  }
  let f = pfq_bigfloat(&[], std::slice::from_ref(b), z, wp, rm)?;
  Some(f.mul(&rgamma(b, wp, rm, cc), bits, rm))
}

/// `FresnelS[z] = π z³/6 ₁F₂(3/4; 3/2, 7/4; −π² z⁴/16)` and
/// `FresnelC[z] = z ₁F₂(1/4; 1/2, 5/4; −π² z⁴/16)` (DLMF 7.2.7, 7.12).
fn fresnel_bigfloat(
  sine: bool,
  z: &BigFloat,
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  let wp = bits + 64;
  let q = |n: i32, d: i32| {
    BigFloat::from_i32(n, wp).div(&BigFloat::from_i32(d, wp), wp, rm)
  };
  let pi = cc.pi(wp, rm);
  let z2 = z.mul(z, wp, rm);
  let w = pi.mul(&pi, wp, rm).mul(&z2, wp, rm).mul(&z2, wp, rm).div(
    &BigFloat::from_i32(-16, wp),
    wp,
    rm,
  );
  if sine {
    let f = pfq_bigfloat(&[q(3, 4)], &[q(3, 2), q(7, 4)], &w, wp, rm)?;
    Some(
      pi.mul(&z2, wp, rm)
        .mul(z, wp, rm)
        .div(&BigFloat::from_i32(6, wp), wp, rm)
        .mul(&f, bits, rm),
    )
  } else {
    let f = pfq_bigfloat(&[q(1, 4)], &[q(1, 2), q(5, 4)], &w, wp, rm)?;
    Some(z.mul(&f, bits, rm))
  }
}

/// Machine `Fibonacci[ν, x]`, the same continuation as
/// [`fibonacci_bigfloat`] worked in machine arithmetic, as wolframscript
/// does: its `Fibonacci[-0.5]` is `0.35157758425414287`, one ulp from the
/// correctly rounded value.
fn fibonacci_f64(nu: f64, x: f64) -> f64 {
  let root = (x * x + 4.0).sqrt();
  let phi = f64::midpoint(x, root);
  (phi.powf(nu) - (std::f64::consts::PI * nu).cos() * phi.powf(-nu)) / root
}

/// `(a; q)_∞ = Π_{k≥0} (1 − a q^k)` for real `|q| < 1`.
fn qpochhammer_infinite(
  a: &BigFloat,
  q: &BigFloat,
  wp: usize,
  rm: RoundingMode,
) -> BigFloat {
  let one = BigFloat::from_i32(1, wp);
  let mut prod = one.clone();
  let mut aqk = a.clone();
  for _ in 0..(64 * wp + 1000) {
    if aqk.is_zero() || bf_exp(&aqk) < -(wp as i64) - 4 {
      break;
    }
    prod = prod.mul(&one.sub(&aqk, wp, rm), wp, rm);
    aqk = aqk.mul(q, wp, rm);
  }
  prod
}

/// `QPochhammer[a, q, n]` for real arguments with `0 < q < 1` and real `n`:
/// `(a; q)_∞ / (a q^n; q)_∞`.
pub(crate) fn qpochhammer_bigfloat(
  a: &BigFloat,
  q: &BigFloat,
  n: &BigFloat,
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  let wp = bits + 64;
  let one = BigFloat::from_i32(1, wp);
  if !q.is_positive() || q.cmp(&one)? >= 0 {
    return None;
  }
  let num = qpochhammer_infinite(a, q, wp, rm);
  let aqn = a.mul(&q.pow(n, wp, rm, cc), wp, rm);
  let den = qpochhammer_infinite(&aqn, q, wp, rm);
  if den.is_zero() {
    return None;
  }
  Some(num.div(&den, bits, rm))
}

/// The value of `name[args…]` from the cores above, for real `args`.
pub(crate) fn orthogonal_bigfloat(
  name: &str,
  v: &[BigFloat],
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  let one = || BigFloat::from_i32(1, bits);
  let zero = || BigFloat::from_i32(0, bits);
  match (name, v) {
    ("LegendreP", [nu, x]) => legendre_p_bigfloat(nu, x, bits, rm, cc),
    ("LaguerreL", [nu, x]) => laguerre_l_bigfloat(nu, &zero(), x, bits, rm, cc),
    ("LaguerreL", [nu, l, x]) => laguerre_l_bigfloat(nu, l, x, bits, rm, cc),
    ("HermiteH", [nu, x]) => hermite_h_bigfloat(nu, x, bits, rm, cc),
    ("Fibonacci", [nu]) => fibonacci_bigfloat(nu, &one(), bits, rm, cc),
    ("Fibonacci", [nu, x]) => fibonacci_bigfloat(nu, x, bits, rm, cc),
    ("QPochhammer", [a, q, n]) => qpochhammer_bigfloat(a, q, n, bits, rm, cc),
    ("Hypergeometric1F1", [a, b, z]) => pfq_bigfloat(
      std::slice::from_ref(a),
      std::slice::from_ref(b),
      z,
      bits,
      rm,
    ),
    ("Hypergeometric0F1", [b, z]) => {
      pfq_bigfloat(&[], std::slice::from_ref(b), z, bits, rm)
    }
    ("Hypergeometric0F1Regularized", [b, z]) => {
      hypergeometric_0f1_regularized_bigfloat(b, z, bits, rm, cc)
    }
    ("FresnelS" | "FresnelC", [z]) => {
      fresnel_bigfloat(name == "FresnelS", z, bits, rm, cc)
    }
    ("LogisticSigmoid", [z]) => {
      let wp = bits + 64;
      let one = BigFloat::from_i32(1, wp);
      Some(one.div(&one.add(&z.neg().exp(wp, rm, cc), wp, rm), bits, rm))
    }
    _ => None,
  }
}

/// `name[args…]` evaluated numerically when the arguments are real numbers
/// and at least one is inexact: a machine `Real` when any argument is one,
/// otherwise an arbitrary-precision number at the lowest argument
/// precision. `None` leaves the call to the caller (exact or symbolic
/// arguments, complex values, or a point the cores do not cover).
pub(crate) fn inexact_orthogonal_value(
  name: &str,
  args: &[Expr],
) -> Option<Expr> {
  let machine = args.iter().any(|a| matches!(a, Expr::Real(_)));
  let precision = args
    .iter()
    .filter_map(|a| match a {
      Expr::BigFloat(_, p) => Some(*p),
      _ => None,
    })
    .reduce(f64::min);
  if !machine && precision.is_none() {
    return None;
  }
  let reals: Vec<f64> = args
    .iter()
    .map(|a| {
      crate::functions::math_ast::is_real_numeric(a)
        .then(|| crate::functions::math_ast::try_eval_to_f64(a))
        .flatten()
    })
    .collect::<Option<_>>()?;
  if machine && name == "Fibonacci" {
    return Some(Expr::Real(fibonacci_f64(
      reals[0],
      reals.get(1).copied().unwrap_or(1.0),
    )));
  }
  if machine {
    let v = machine_from_bigfloat(&reals, |v, bits, rm, cc| {
      orthogonal_bigfloat(name, v, bits, rm, cc)
    })?;
    return Some(Expr::Real(v));
  }
  let call = Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  };
  crate::functions::math_ast::bigfloat_function_value(&call, precision?)
}

/// Machine value of one of the cores above: computed at 64 bits plus guard
/// and rounded once to `f64`.
pub(crate) fn machine_from_bigfloat(
  args: &[f64],
  f: impl FnOnce(&[BigFloat], usize, RoundingMode, &mut Consts) -> Option<BigFloat>,
) -> Option<f64> {
  if args.iter().any(|v| !v.is_finite()) {
    return None;
  }
  let bits = 64 + 64;
  let rm = RoundingMode::ToEven;
  let bfs: Vec<BigFloat> =
    args.iter().map(|&v| BigFloat::from_f64(v, bits)).collect();
  let r = with_thread_consts(|cc| f(&bfs, bits, rm, cc))?;
  let v = bigfloat_to_f64_approx(&r)?;
  v.is_finite().then_some(v)
}
