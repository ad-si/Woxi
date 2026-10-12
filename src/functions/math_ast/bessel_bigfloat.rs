//! Bessel functions of real order and positive real argument at any
//! precision: the single numeric core behind machine and arbitrary-precision
//! `BesselJ`, `BesselY`, `BesselI` and `BesselK`.
//!
//! Small and moderate arguments use the power series (DLMF 10.2.2, 10.8.1,
//! 10.25.2, 10.31.1), summed with enough guard bits to absorb their
//! cancellation — the series for `J_0(30)` adds terms near `10^11` to a
//! result near `0.09`. Large arguments use Hankel's asymptotic expansions
//! (DLMF 10.17.3–4, 10.40.1–2), which share the coefficients
//! `a_k(ν) = a_{k−1}(ν) (4ν² − (2k−1)²)/(8k)`. Machine values are rounded from
//! a 64-bit result.

#[allow(unused_imports)]
use super::*;
use astro_float::{BigFloat, Consts, RoundingMode};

/// Which Bessel function a computation is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BesselKind {
  J,
  Y,
  I,
  K,
}

/// Binary exponent of `b`, or a huge negative number for zero.
fn bf_exp(b: &BigFloat) -> i32 {
  if b.is_zero() {
    i32::MIN / 4
  } else {
    b.exponent().unwrap_or(i32::MIN / 4)
  }
}

/// Runs `eval` at `bits` plus a guard, growing the guard until the bits the
/// evaluation reports losing to cancellation leave `bits` + 16 correct ones.
/// `expected_loss` is the cancellation the caller anticipates, so the usual
/// case needs a single pass.
fn with_guard<T>(
  bits: usize,
  expected_loss: f64,
  mut eval: impl FnMut(usize) -> (T, i32),
) -> T {
  let cap = 16 * bits + 8192;
  let mut guard = 32 + expected_loss.max(0.0).ceil() as usize;
  loop {
    let (value, lost) = eval(bits + guard);
    let lost = lost.max(0) as usize;
    if lost + 16 <= guard || guard > cap {
      return value;
    }
    guard = (lost + 64).min(cap + 1);
  }
}

/// `ν` as a small integer, if it is one.
fn small_integer(nu: &BigFloat) -> Option<i64> {
  if !nu.is_int() {
    return None;
  }
  let f = crate::functions::math_ast::bigfloat_to_f64_approx(nu)?;
  (f.abs() < 1.0e6).then_some(f as i64)
}

/// The Bessel function `kind` of real order `nu` at `x > 0`. `None` for
/// `x ≤ 0` (the callers handle zero and the branch cut).
pub(crate) fn bessel_bigfloat(
  kind: BesselKind,
  nu: &BigFloat,
  x: &BigFloat,
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  if !x.is_positive() {
    return None;
  }
  let zero = BigFloat::from_i32(0, bits);
  // Integer orders reduce to non-negative ones:
  // J_{−n} = (−1)^n J_n, Y_{−n} = (−1)^n Y_n, I_{−n} = I_n, K_{−ν} = K_ν.
  let n_int = small_integer(nu);
  if kind == BesselKind::K && nu.is_negative() {
    return bessel_bigfloat(kind, &nu.neg(), x, bits, rm, cc);
  }
  if let Some(n) = n_int
    && n < 0
  {
    let v = bessel_bigfloat(kind, &nu.neg(), x, bits, rm, cc)?;
    let odd = n % 2 != 0;
    return Some(match kind {
      BesselKind::J | BesselKind::Y if odd => v.neg(),
      _ => v,
    });
  }

  let xf = crate::functions::math_ast::bigfloat_to_f64_approx(x)?;
  let nuf = crate::functions::math_ast::bigfloat_to_f64_approx(nu)?;
  // Hankel's expansion is usable once its smallest term, about e^(−2x),
  // is below the precision and the order is small next to the argument.
  let asymptotic = xf > 0.35 * (bits as f64 + 20.0) && xf > 1.5 * nuf * nuf;
  if asymptotic {
    let w = bits + 64 + (xf.log2().max(0.0) as usize);
    return Some(
      hankel_asymptotic(kind, nu, x, w, rm, cc).add(&zero, bits, rm),
    );
  }

  // The J and Y series sum terms near e^x to a result near 1/√x; K's
  // formulas cancel growing I's (~e^x) down to K (~e^(−x)).
  let log2_e = std::f64::consts::LOG2_E;
  let expected_loss = match kind {
    BesselKind::J | BesselKind::Y => log2_e * xf,
    BesselKind::I => 0.0,
    BesselKind::K => 2.0 * log2_e * xf,
  };
  let value = match kind {
    BesselKind::J | BesselKind::I => with_guard(bits, expected_loss, |w| {
      power_series(kind, nu, x, w, rm, cc)
    }),
    BesselKind::Y | BesselKind::K => match n_int {
      Some(n) => with_guard(bits, expected_loss, |w| {
        integer_order_second_kind(kind, n, x, w, rm, cc)
      }),
      None => with_guard(bits, expected_loss, |w| {
        // Y_ν = (J_ν cos νπ − J_{−ν})/sin νπ, K_ν = π/2 (I_{−ν} − I_ν)/sin νπ.
        let first = if kind == BesselKind::Y {
          BesselKind::J
        } else {
          BesselKind::I
        };
        let (pos, lost_p) = power_series(first, nu, x, w, rm, cc);
        let (neg, lost_n) = power_series(first, &nu.neg(), x, w, rm, cc);
        let pi = cc.pi(w, rm);
        let nu_pi = nu.mul(&pi, w, rm);
        let sin = nu_pi.sin(w, rm, cc);
        let (a, b) = if kind == BesselKind::Y {
          (pos.mul(&nu_pi.cos(w, rm, cc), w, rm), neg)
        } else {
          (neg, pos)
        };
        let diff = a.sub(&b, w, rm);
        let lost =
          bf_exp(&a).max(bf_exp(&b)) - bf_exp(&diff) + lost_p.max(lost_n);
        let mut value = diff.div(&sin, w, rm);
        if kind == BesselKind::K {
          value = value.mul(&pi, w, rm).div(&BigFloat::from_i32(2, w), w, rm);
        }
        (value, lost)
      }),
    },
  };
  Some(value.add(&zero, bits, rm))
}

/// `J_ν(x) = Σ (−1)^k (x/2)^(2k+ν)/(k! Γ(k+ν+1))` or the same series without
/// the sign for `I_ν`, at working precision `w`. Returns the sum and the bits
/// it lost to cancellation.
fn power_series(
  kind: BesselKind,
  nu: &BigFloat,
  x: &BigFloat,
  w: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> (BigFloat, i32) {
  let one = BigFloat::from_i32(1, w);
  let half_x = x.div(&BigFloat::from_i32(2, w), w, rm);
  let mut ratio = half_x.mul(&half_x, w, rm);
  if kind == BesselKind::J {
    ratio = ratio.neg();
  }
  // Integer orders were reduced to non-negative ones before this, so
  // Γ(ν+1) has no pole here.
  // The k = 0 term (x/2)^ν/Γ(ν+1): for an integer order (x/2)^n/n!, which
  // needs neither a logarithm nor Gamma.
  let first = match small_integer(nu) {
    Some(n) if n >= 0 => {
      let mut fact = one.clone();
      for j in 2..=n {
        fact = fact.mul(&BigFloat::from_i64(j, w), w, rm);
      }
      bigfloat_powi(&half_x, n as i128, w, rm).div(&fact, w, rm)
    }
    _ => {
      let gamma = cached_gamma(&nu.add(&one, w, rm), w, rm, cc);
      crate::functions::math_ast::ln_above_integer(&half_x, w, rm, cc)
        .mul(nu, w, rm)
        .exp(w, rm, cc)
        .div(&gamma, w, rm)
    }
  };
  // Σ_k ratio^k / Π_{j≤k} j(j+ν), kept as one fraction num/den so the loop
  // multiplies instead of dividing: num_k = num_{k−1} d_k + ratio^k and
  // den_k = den_{k−1} d_k with d_k = k(k+ν).
  let mut num = one.clone();
  let mut den = one.clone();
  let mut power = one.clone(); // ratio^k
  let mut max_exp = 0; // log₂ of the largest |term/first| so far
  let xf = crate::functions::math_ast::bigfloat_to_f64_approx(x).unwrap_or(0.0);
  for k in 1u64.. {
    let k_bf = BigFloat::from_u64(k, w);
    let d_k = k_bf.mul(&k_bf.add(nu, w, rm), w, rm);
    power = power.mul(&ratio, w, rm);
    num = num.mul(&d_k, w, rm).add(&power, w, rm);
    den = den.mul(&d_k, w, rm);
    let term_exp = bf_exp(&power) - bf_exp(&den);
    max_exp = max_exp.max(term_exp);
    if (k as f64) > xf && term_exp < max_exp - w as i32 - 8 {
      break;
    }
  }
  let lost = max_exp - (bf_exp(&num) - bf_exp(&den));
  (first.mul(&num, w, rm).div(&den, w, rm), lost)
}

/// `Γ(v)` at `w` bits, remembered for the last few `(v, w)`: a machine
/// evaluation of a non-integer order needs `Γ(ν+1)` and `Γ(1−ν)` on every
/// call, and a plot or table asks for the same order thousands of times.
fn cached_gamma(
  v: &BigFloat,
  w: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> BigFloat {
  thread_local! {
    static CACHE: std::cell::RefCell<Vec<(BigFloat, usize, BigFloat)>> =
      const { std::cell::RefCell::new(Vec::new()) };
  }
  let hit = CACHE.with(|c| {
    c.borrow()
      .iter()
      .find(|(key, kw, _)| *kw == w && key.cmp(v) == Some(0))
      .map(|(_, _, value)| value.clone())
  });
  if let Some(value) = hit {
    return value;
  }
  let value = crate::functions::math_ast::gamma_bigfloat(v, w, rm, cc);
  CACHE.with(|c| {
    let mut c = c.borrow_mut();
    if c.len() >= 16 {
      c.remove(0);
    }
    c.push((v.clone(), w, value.clone()));
  });
  value
}

/// `Y_n` (DLMF 10.8.1) and `K_n` (DLMF 10.31.1) for an integer `n ≥ 0`:
///
/// `Y_n = −(x/2)^(−n)/π Σ_{k<n} (n−k−1)!/k! (x²/4)^k + 2/π ln(x/2) J_n
///        − (x/2)^n/π Σ_k (ψ(k+1) + ψ(n+k+1)) (−x²/4)^k/(k!(n+k)!)`,
///
/// `K_n = ½(x/2)^(−n) Σ_{k<n} (n−k−1)!/k! (−x²/4)^k + (−1)^(n+1) ln(x/2) I_n
///        + (−1)^n ½(x/2)^n Σ_k (ψ(k+1) + ψ(n+k+1)) (x²/4)^k/(k!(n+k)!)`.
fn integer_order_second_kind(
  kind: BesselKind,
  n: i64,
  x: &BigFloat,
  w: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> (BigFloat, i32) {
  let one = BigFloat::from_i32(1, w);
  let two = BigFloat::from_i32(2, w);
  let pi = cc.pi(w, rm);
  let half_x = x.div(&two, w, rm);
  let quarter_x2 = half_x.mul(&half_x, w, rm);
  let n_bf = BigFloat::from_i64(n, w);
  let half_x_pow_n = bigfloat_powi(&half_x, n as i128, w, rm);
  let is_y = kind == BesselKind::Y;

  // The finite sum Σ_{k<n} (n−k−1)!/k! (∓x²/4)^k.
  let finite_ratio = if is_y {
    quarter_x2.clone()
  } else {
    quarter_x2.neg()
  };
  let mut finite = BigFloat::from_i32(0, w);
  if n > 0 {
    // k = 0 term: (n−1)!
    let mut fact = one.clone();
    for j in 2..n {
      fact = fact.mul(&BigFloat::from_i64(j, w), w, rm);
    }
    let mut term = fact;
    for k in 0..n {
      finite = finite.add(&term, w, rm);
      if k + 1 < n {
        // (n−k−2)!/(k+1)! = (n−k−1)!/k! · 1/((n−k−1)(k+1))
        term = term.mul(&finite_ratio, w, rm).div(
          &BigFloat::from_i64((n - k - 1) * (k + 1), w),
          w,
          rm,
        );
      }
    }
  }
  let finite = finite.div(&half_x_pow_n, w, rm);

  // The ψ sum: ψ(1) = −γ, ψ(m+1) = ψ(m) + 1/m.
  let gamma = crate::functions::math_ast::compute_euler_gamma(w, rm, cc);
  let mut psi_k1 = gamma.neg(); // ψ(k+1)
  let mut psi_nk1 = gamma.neg(); // ψ(n+k+1)
  for j in 1..=n {
    psi_nk1 = psi_nk1.add(&one.div(&BigFloat::from_i64(j, w), w, rm), w, rm);
  }
  let series_ratio = if is_y {
    quarter_x2.neg()
  } else {
    quarter_x2.clone()
  };
  let mut n_fact = one.clone();
  for j in 2..=n {
    n_fact = n_fact.mul(&BigFloat::from_i64(j, w), w, rm);
  }
  let mut base = one.div(&n_fact, w, rm); // (±x²/4)^k/(k!(n+k)!)
  let mut psi_sum = BigFloat::from_i32(0, w);
  let mut max_exp = i32::MIN / 4;
  let xf = crate::functions::math_ast::bigfloat_to_f64_approx(x).unwrap_or(0.0);
  for k in 0u64.. {
    if k > 0 {
      let k_bf = BigFloat::from_u64(k, w);
      base = base.mul(&series_ratio, w, rm).div(
        &k_bf.mul(&k_bf.add(&n_bf, w, rm), w, rm),
        w,
        rm,
      );
      psi_k1 = psi_k1.add(&one.div(&k_bf, w, rm), w, rm);
      psi_nk1 = psi_nk1.add(&one.div(&k_bf.add(&n_bf, w, rm), w, rm), w, rm);
    }
    let term = base.mul(&psi_k1.add(&psi_nk1, w, rm), w, rm);
    psi_sum = psi_sum.add(&term, w, rm);
    max_exp = max_exp.max(bf_exp(&term));
    if (k as f64) > xf && bf_exp(&term) < max_exp - w as i32 - 8 {
      break;
    }
  }
  let psi_part = half_x_pow_n.mul(&psi_sum, w, rm);
  let first_kind = if is_y { BesselKind::J } else { BesselKind::I };
  let (first, first_lost) = power_series(first_kind, &n_bf, x, w, rm, cc);
  let log_part =
    crate::functions::math_ast::ln_above_integer(&half_x, w, rm, cc)
      .mul(&first, w, rm);

  let parts = if is_y {
    // −finite/π + 2/π log_part − psi_part/π
    [
      finite.neg().div(&pi, w, rm),
      log_part.mul(&two, w, rm).div(&pi, w, rm),
      psi_part.neg().div(&pi, w, rm),
    ]
  } else {
    let sign = |v: BigFloat, odd: bool| if odd { v.neg() } else { v };
    let n_odd = n % 2 != 0;
    [
      finite.div(&two, w, rm),
      sign(log_part, !n_odd),
      sign(psi_part.div(&two, w, rm), n_odd),
    ]
  };
  let max_part = parts.iter().map(bf_exp).max().unwrap_or(0).max(max_exp);
  let value = parts[0].add(&parts[1], w, rm).add(&parts[2], w, rm);
  let lost = max_part - bf_exp(&value) + first_lost;
  (value, lost)
}

/// Hankel's asymptotic expansions for large `x`:
/// `J, Y = √(2/(πx)) (P cos ω ∓ Q sin ω …)` with `ω = x − νπ/2 − π/4`,
/// `I = e^x/√(2πx) Σ (−1)^k a_k/x^k`, `K = √(π/(2x)) e^(−x) Σ a_k/x^k`.
fn hankel_asymptotic(
  kind: BesselKind,
  nu: &BigFloat,
  x: &BigFloat,
  w: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> BigFloat {
  let one = BigFloat::from_i32(1, w);
  let two = BigFloat::from_i32(2, w);
  let pi = cc.pi(w, rm);
  let four_nu2 = nu.mul(nu, w, rm).mul(&BigFloat::from_i32(4, w), w, rm);
  // The terms a_k/x^k = num_k/den_k with num_k = Π (4ν² − (2j−1)²) and
  // den_k = Π 8j·x. Each sum is kept over the common denominator, so the
  // loop multiplies instead of dividing: sum_k = sum_{k−1} d_k + s_k num_k
  // with d_k = 8k·x. J/Y need the even (P) and odd (Q) parts with signs
  // (−1)^⌊k/2⌋, I the alternating sum, K the plain one.
  let jy = matches!(kind, BesselKind::J | BesselKind::Y);
  let mut num = one.clone();
  let mut den = one.clone();
  let mut even = one.clone();
  let mut odd = BigFloat::from_i32(0, w);
  let mut single = one.clone(); // Σ (−1)^k a_k/x^k for I, Σ a_k/x^k for K
  let mut prev_exp = 0;
  for k in 1u64..100_000 {
    let odd_sq = BigFloat::from_u64((2 * k - 1) * (2 * k - 1), w);
    let factor = four_nu2.sub(&odd_sq, w, rm);
    if factor.is_zero() {
      break; // a half-integer order: the expansion terminates exactly
    }
    num = num.mul(&factor, w, rm);
    let d_k = x.mul(&BigFloat::from_u64(8 * k, w), w, rm);
    den = den.mul(&d_k, w, rm);
    if jy {
      even = even.mul(&d_k, w, rm);
      odd = odd.mul(&d_k, w, rm);
      let signed = if (k / 2) % 2 == 1 {
        num.neg()
      } else {
        num.clone()
      };
      if k % 2 == 0 {
        even = even.add(&signed, w, rm);
      } else {
        odd = odd.add(&signed, w, rm);
      }
    } else {
      single = single.mul(&d_k, w, rm);
      let signed = if kind == BesselKind::I && k % 2 == 1 {
        num.neg()
      } else {
        num.clone()
      };
      single = single.add(&signed, w, rm);
    }
    let e = bf_exp(&num) - bf_exp(&den);
    if e < -(w as i32) - 8 || (e > prev_exp && k > 4) {
      break;
    }
    prev_exp = e;
  }
  let even = even.div(&den, w, rm);
  let odd = odd.div(&den, w, rm);
  let single = single.div(&den, w, rm);
  match kind {
    BesselKind::J | BesselKind::Y => {
      let quarter_pi = pi.div(&BigFloat::from_i32(4, w), w, rm);
      let omega = x.sub(&nu.mul(&pi, w, rm).div(&two, w, rm), w, rm).sub(
        &quarter_pi,
        w,
        rm,
      );
      let (c, s) = (omega.cos(w, rm, cc), omega.sin(w, rm, cc));
      let amp = two.div(&pi.mul(x, w, rm), w, rm).sqrt(w, rm);
      let core = if kind == BesselKind::J {
        even.mul(&c, w, rm).sub(&odd.mul(&s, w, rm), w, rm)
      } else {
        even.mul(&s, w, rm).add(&odd.mul(&c, w, rm), w, rm)
      };
      amp.mul(&core, w, rm)
    }
    BesselKind::I => {
      let denom = two.mul(&pi, w, rm).mul(x, w, rm).sqrt(w, rm);
      x.exp(w, rm, cc).div(&denom, w, rm).mul(&single, w, rm)
    }
    BesselKind::K => {
      let amp = pi.div(&two.mul(x, w, rm), w, rm).sqrt(w, rm);
      amp.mul(&x.neg().exp(w, rm, cc), w, rm).mul(&single, w, rm)
    }
  }
}

/// Machine Bessel function `kind` of real order `nu` at `x > 0`, rounded
/// from a 64-bit result. `None` for `x ≤ 0` or a non-finite input.
pub fn bessel_f64(kind: BesselKind, nu: f64, x: f64) -> Option<f64> {
  if x <= 0.0 || !nu.is_finite() || !x.is_finite() {
    return None;
  }
  crate::functions::math_ast::with_thread_consts(|cc| {
    let rm = RoundingMode::ToEven;
    let v = bessel_bigfloat(
      kind,
      &BigFloat::from_f64(nu, 64),
      &BigFloat::from_f64(x, 64),
      64,
      rm,
      cc,
    )?;
    crate::functions::math_ast::bigfloat_to_f64_approx(&v)
  })
}

/// Struve `H_ν(x)` (`modified == false`) or `L_ν(x)` (`modified == true`)
/// for real `ν` and `x > 0`: the power series
/// `Σ (∓1)^m (x/2)^(2m+ν+1)/(Γ(m+3/2) Γ(m+ν+3/2))` with guard bits, or for
/// large `x` the asymptotic forms (DLMF 11.6.1–2)
/// `H_ν − Y_ν ~ 1/π Σ Γ(k+½) (x/2)^(ν−2k−1)/Γ(ν+½−k)` and
/// `L_ν − I_{−ν} ~ 1/π Σ (−1)^(k+1) Γ(k+½) (x/2)^(ν−2k−1)/Γ(ν+½−k)`.
pub(crate) fn struve_bigfloat(
  modified: bool,
  nu: &BigFloat,
  x: &BigFloat,
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  if !x.is_positive() {
    return None;
  }
  let zero = BigFloat::from_i32(0, bits);
  let xf = crate::functions::math_ast::bigfloat_to_f64_approx(x)?;
  let nuf = crate::functions::math_ast::bigfloat_to_f64_approx(nu)?;
  // The asymptotic sum's smallest term is about e^(−x).
  if xf > 0.7 * (bits as f64 + 20.0) && xf > 1.5 * nuf * nuf {
    let w = bits + 64 + (xf.log2().max(0.0) as usize);
    let half = BigFloat::from_f64(0.5, w);
    let pi = cc.pi(w, rm);
    let half_x = x.div(&BigFloat::from_i32(2, w), w, rm);
    let a = nu.add(&half, w, rm); // ν + ½
    // Γ(k+½)/Γ(a−k) (x/2)^(ν−2k−1), from k = 0 by
    // ratio (k+½)(a−k−1)/(x/2)².
    let mut term = cc
      .pi(w, rm)
      .sqrt(w, rm)
      .div(&cached_gamma(&a, w, rm, cc), w, rm)
      .mul(
        &crate::functions::math_ast::ln_above_integer(&half_x, w, rm, cc)
          .mul(&nu.sub(&BigFloat::from_i32(1, w), w, rm), w, rm)
          .exp(w, rm, cc),
        w,
        rm,
      );
    let half_x2 = half_x.mul(&half_x, w, rm);
    let mut sum = BigFloat::from_i32(0, w);
    let mut prev_exp = i32::MAX;
    for k in 0u64..100_000 {
      let signed = if modified && k % 2 == 0 {
        term.neg()
      } else {
        term.clone()
      };
      sum = sum.add(&signed, w, rm);
      let k_bf = BigFloat::from_u64(k, w);
      term = term
        .mul(&k_bf.add(&half, w, rm), w, rm)
        .mul(
          &a.sub(&k_bf, w, rm).sub(&BigFloat::from_i32(1, w), w, rm),
          w,
          rm,
        )
        .div(&half_x2, w, rm);
      let e = bf_exp(&term);
      if term.is_zero()
        || e < bf_exp(&sum) - w as i32 - 8
        || (e > prev_exp && k > 4)
      {
        break;
      }
      prev_exp = e;
    }
    let correction = sum.div(&pi, w, rm);
    let base = if modified {
      bessel_bigfloat(BesselKind::I, &nu.neg(), x, w, rm, cc)?
    } else {
      bessel_bigfloat(BesselKind::Y, nu, x, w, rm, cc)?
    };
    return Some(base.add(&correction, w, rm).add(&zero, bits, rm));
  }
  let expected_loss = if modified {
    0.0
  } else {
    std::f64::consts::LOG2_E * xf
  };
  let value = with_guard(bits, expected_loss, |w| {
    let one = BigFloat::from_i32(1, w);
    let three_halves = BigFloat::from_f64(1.5, w);
    let half = BigFloat::from_f64(0.5, w);
    let half_x = x.div(&BigFloat::from_i32(2, w), w, rm);
    let mut ratio = half_x.mul(&half_x, w, rm);
    if !modified {
      ratio = ratio.neg();
    }
    // 1/Γ(m+ν+3/2) vanishes while m + ν + 3/2 is a non-positive integer:
    // start the series at the first m past those poles.
    let b = nu.add(&three_halves, w, rm);
    let m0 = match small_integer(&b) {
      Some(j) if j <= 0 => (1 - j) as u64,
      _ => 0,
    };
    let m0_bf = BigFloat::from_u64(m0, w);
    let gamma = |v: &BigFloat, cc: &mut Consts| cached_gamma(v, w, rm, cc);
    let power =
      crate::functions::math_ast::ln_above_integer(&half_x, w, rm, cc)
        .mul(
          &m0_bf.add(&m0_bf, w, rm).add(nu, w, rm).add(&one, w, rm),
          w,
          rm,
        )
        .exp(w, rm, cc);
    let mut term = power
      .div(&gamma(&m0_bf.add(&three_halves, w, rm), cc), w, rm)
      .div(&gamma(&m0_bf.add(&b, w, rm), cc), w, rm);
    if !modified && m0 % 2 == 1 {
      term = term.neg();
    }
    let mut sum = term.clone();
    let mut max_exp = bf_exp(&term);
    for m in (m0 + 1).. {
      let m_bf = BigFloat::from_u64(m, w);
      term = term.mul(&ratio, w, rm).div(
        &m_bf.add(&half, w, rm).mul(
          &m_bf.add(nu, w, rm).add(&half, w, rm),
          w,
          rm,
        ),
        w,
        rm,
      );
      sum = sum.add(&term, w, rm);
      max_exp = max_exp.max(bf_exp(&term));
      if (m as f64) > xf && bf_exp(&term) < max_exp - w as i32 - 8 {
        break;
      }
    }
    let lost = max_exp - bf_exp(&sum);
    (sum, lost)
  });
  Some(value.add(&zero, bits, rm))
}

/// Machine Struve `H_ν(x)`/`L_ν(x)` for `x > 0`, rounded from 64 bits.
pub fn struve_f64(modified: bool, nu: f64, x: f64) -> Option<f64> {
  if x <= 0.0 || !nu.is_finite() || !x.is_finite() {
    return None;
  }
  crate::functions::math_ast::with_thread_consts(|cc| {
    let rm = RoundingMode::ToEven;
    let v = struve_bigfloat(
      modified,
      &BigFloat::from_f64(nu, 64),
      &BigFloat::from_f64(x, 64),
      64,
      rm,
      cc,
    )?;
    crate::functions::math_ast::bigfloat_to_f64_approx(&v)
  })
}

/// Which Airy function an evaluation is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AiryKind {
  Ai,
  Bi,
  AiPrime,
  BiPrime,
}

/// Airy functions of a real `x ≠ 0` through Bessel functions of order
/// ±1/3, ±2/3 at `ζ = (2/3)|x|^(3/2)` (DLMF 9.6.1–9.6.9):
/// for `x > 0`, `Ai = √(x/3)/π K_{1/3}`, `Ai' = −x/(π√3) K_{2/3}`,
/// `Bi = √(x/3) (I_{−1/3} + I_{1/3})`, `Bi' = x/√3 (I_{−2/3} + I_{2/3})`;
/// for `x = −t < 0`, `Ai = √t/3 (J_{1/3} + J_{−1/3})`,
/// `Ai' = t/3 (J_{2/3} − J_{−2/3})`, `Bi = √(t/3) (J_{−1/3} − J_{1/3})`,
/// `Bi' = t/√3 (J_{−2/3} + J_{2/3})`.
pub(crate) fn airy_bigfloat(
  kind: AiryKind,
  x: &BigFloat,
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  if x.is_zero() {
    return None;
  }
  // Ai and Ai' decay like e^(−ζ) while the I's grow; the J combinations
  // cancel near their zeros. 64 extra bits cover both for machine use, and
  // the Bessel core guards its own sums.
  let w = bits + 64;
  let t = x.abs();
  let three = BigFloat::from_i32(3, w);
  let zeta = t
    .mul(&t.sqrt(w, rm), w, rm)
    .mul(&BigFloat::from_i32(2, w), w, rm)
    .div(&three, w, rm);
  let third = BigFloat::from_i32(1, w).div(&three, w, rm);
  let two_thirds = third.add(&third, w, rm);
  let pi = cc.pi(w, rm);
  let sqrt3 = three.sqrt(w, rm);
  let b = |k: BesselKind, nu: &BigFloat, cc: &mut Consts| {
    bessel_bigfloat(k, nu, &zeta, w, rm, cc)
  };
  use BesselKind::{I, J, K};
  let value = if x.is_positive() {
    match kind {
      AiryKind::Ai => t.div(&three, w, rm).sqrt(w, rm).div(&pi, w, rm).mul(
        &b(K, &third, cc)?,
        w,
        rm,
      ),
      AiryKind::AiPrime => t
        .div(&pi.mul(&sqrt3, w, rm), w, rm)
        .mul(&b(K, &two_thirds, cc)?, w, rm)
        .neg(),
      AiryKind::Bi => t.div(&three, w, rm).sqrt(w, rm).mul(
        &b(I, &third.neg(), cc)?.add(&b(I, &third, cc)?, w, rm),
        w,
        rm,
      ),
      AiryKind::BiPrime => t.div(&sqrt3, w, rm).mul(
        &b(I, &two_thirds.neg(), cc)?.add(&b(I, &two_thirds, cc)?, w, rm),
        w,
        rm,
      ),
    }
  } else {
    match kind {
      AiryKind::Ai => t.sqrt(w, rm).div(&three, w, rm).mul(
        &b(J, &third, cc)?.add(&b(J, &third.neg(), cc)?, w, rm),
        w,
        rm,
      ),
      AiryKind::AiPrime => t.div(&three, w, rm).mul(
        &b(J, &two_thirds, cc)?.sub(&b(J, &two_thirds.neg(), cc)?, w, rm),
        w,
        rm,
      ),
      AiryKind::Bi => t.div(&three, w, rm).sqrt(w, rm).mul(
        &b(J, &third.neg(), cc)?.sub(&b(J, &third, cc)?, w, rm),
        w,
        rm,
      ),
      AiryKind::BiPrime => t.div(&sqrt3, w, rm).mul(
        &b(J, &two_thirds.neg(), cc)?.add(&b(J, &two_thirds, cc)?, w, rm),
        w,
        rm,
      ),
    }
  };
  Some(value.add(&BigFloat::from_i32(0, bits), bits, rm))
}

/// Machine Airy function `kind` of a real `x ≠ 0`, rounded from 64 bits.
pub fn airy_f64(kind: AiryKind, x: f64) -> Option<f64> {
  if x == 0.0 || !x.is_finite() {
    return None;
  }
  crate::functions::math_ast::with_thread_consts(|cc| {
    let v = airy_bigfloat(
      kind,
      &BigFloat::from_f64(x, 64),
      64,
      RoundingMode::ToEven,
      cc,
    )?;
    crate::functions::math_ast::bigfloat_to_f64_approx(&v)
  })
}
