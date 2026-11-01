//! Arbitrary-precision elliptic integrals of the first kind and the real-axis
//! inverse Jacobi functions built on them: the single core behind
//! `N[EllipticK[m], p]`, `N[EllipticF[phi, m], p]`, `N[InverseJacobiPQ[x, m], p]`
//! and the machine values of those functions at real arguments (computed at
//! 128 bits and rounded once).
//!
//! `R_F` is summed with Carlson's duplication theorem until the deviations
//! from the mean drop below `2^(-wp/6)`, where the fifth-order Taylor
//! correction leaves an error below `2^-wp`; `K(m)` comes from the AGM.

use astro_float::{BigFloat, Consts, RoundingMode};

/// Binary exponent of `b`, or a huge negative number for zero.
fn bf_exp(b: &BigFloat) -> i64 {
  if b.is_zero() {
    i64::MIN / 4
  } else {
    i64::from(b.exponent().unwrap_or(i32::MIN / 4))
  }
}

fn int(v: i32, wp: usize) -> BigFloat {
  BigFloat::from_i32(v, wp)
}

fn is_neg(b: &BigFloat) -> bool {
  b.is_negative() && !b.is_zero()
}

/// `R_F(x, y, z)` for non-negative `x, y, z`, at most one of them zero.
pub(crate) fn carlson_rf_bigfloat(
  x: &BigFloat,
  y: &BigFloat,
  z: &BigFloat,
  wp: usize,
  rm: RoundingMode,
) -> Option<BigFloat> {
  if is_neg(x) || is_neg(y) || is_neg(z) {
    return None;
  }
  let zeros = [x, y, z].iter().filter(|v| v.is_zero()).count();
  if zeros > 1 {
    return None;
  }
  let quarter = BigFloat::from_f64(0.25, wp);
  let three = int(3, wp);
  let (mut x, mut y, mut z) = (x.clone(), y.clone(), z.clone());
  let target = -(wp as i64) / 6 - 2;
  for _ in 0..(wp + 200) {
    let (sx, sy, sz) = (x.sqrt(wp, rm), y.sqrt(wp, rm), z.sqrt(wp, rm));
    let lam =
      sx.mul(&sy.add(&sz, wp, rm), wp, rm)
        .add(&sy.mul(&sz, wp, rm), wp, rm);
    x = x.add(&lam, wp, rm).mul(&quarter, wp, rm);
    y = y.add(&lam, wp, rm).mul(&quarter, wp, rm);
    z = z.add(&lam, wp, rm).mul(&quarter, wp, rm);
    let a = x.add(&y, wp, rm).add(&z, wp, rm).div(&three, wp, rm);
    let dx = a.sub(&x, wp, rm).div(&a, wp, rm);
    let dy = a.sub(&y, wp, rm).div(&a, wp, rm);
    let dz = a.sub(&z, wp, rm).div(&a, wp, rm);
    if bf_exp(&dx).max(bf_exp(&dy)).max(bf_exp(&dz)) < target {
      let e2 = dx.mul(&dy, wp, rm).sub(&dz.mul(&dz, wp, rm), wp, rm);
      let e3 = dx.mul(&dy, wp, rm).mul(&dz, wp, rm);
      // 1 - E2/10 + E3/14 + E2^2/24 - 3 E2 E3/44
      let poly = int(1, wp)
        .sub(&e2.div(&int(10, wp), wp, rm), wp, rm)
        .add(&e3.div(&int(14, wp), wp, rm), wp, rm)
        .add(&e2.mul(&e2, wp, rm).div(&int(24, wp), wp, rm), wp, rm)
        .sub(
          &e2.mul(&e3, wp, rm).mul(&int(3, wp), wp, rm).div(
            &int(44, wp),
            wp,
            rm,
          ),
          wp,
          rm,
        );
      return Some(poly.div(&a.sqrt(wp, rm), wp, rm));
    }
  }
  None
}

/// `K(m) = Pi / (2 AGM(1, Sqrt[1 - m]))` for real `m < 1`.
pub(crate) fn elliptic_k_bigfloat(
  m: &BigFloat,
  wp: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  let one = int(1, wp);
  let mc = one.sub(m, wp, rm);
  if !mc.is_positive() || mc.is_zero() {
    return None;
  }
  let mut a = one;
  let mut b = mc.sqrt(wp, rm);
  let two = int(2, wp);
  for _ in 0..200 {
    let diff = a.sub(&b, wp, rm);
    if diff.is_zero() || bf_exp(&diff) < bf_exp(&a) - wp as i64 + 2 {
      break;
    }
    let a1 = a.add(&b, wp, rm).div(&two, wp, rm);
    b = a.mul(&b, wp, rm).sqrt(wp, rm);
    a = a1;
  }
  Some(cc.pi(wp, rm).div(&a.mul(&two, wp, rm), wp, rm))
}

/// `F(phi, m)` for real `phi` and real `m` where the value is real:
/// `Sin[r] R_F(Cos[r]^2, 1 - m Sin[r]^2, 1) + 2 k K(m)` with
/// `phi = r + k Pi`, `|r| <= Pi/2`.
pub(crate) fn elliptic_f_bigfloat(
  phi: &BigFloat,
  m: &BigFloat,
  wp: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  if phi.is_zero() {
    return Some(int(0, wp));
  }
  let pi = cc.pi(wp, rm);
  let ratio =
    crate::functions::math_ast::bigfloat_to_f64_approx(&phi.div(&pi, 64, rm))?;
  let k = ratio.round();
  if !k.is_finite() || k.abs() > 1e15 {
    return None;
  }
  let r = phi.sub(&pi.mul(&BigFloat::from_f64(k, wp), wp, rm), wp, rm);
  let s = r.sin(wp, rm, cc);
  let c = r.cos(wp, rm, cc);
  let one = int(1, wp);
  let delta = one.sub(&m.mul(&s.mul(&s, wp, rm), wp, rm), wp, rm);
  if is_neg(&delta) {
    return None;
  }
  let f = s.mul(
    &carlson_rf_bigfloat(&c.mul(&c, wp, rm), &delta, &one, wp, rm)?,
    wp,
    rm,
  );
  if k == 0.0 {
    return Some(f);
  }
  let kk = elliptic_k_bigfloat(m, wp, rm, cc)?;
  Some(f.add(&kk.mul(&BigFloat::from_f64(2.0 * k, wp), wp, rm), wp, rm))
}

/// The amplitude whose `F(·, m)` is the inverse function value.
enum Amplitude {
  ArcSin(BigFloat),
  ArcCos(BigFloat),
  ArcTan(BigFloat),
  ArcCot(BigFloat),
}

fn amplitude_value(
  a: Amplitude,
  wp: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  let one = int(1, wp);
  let in_unit = |v: &BigFloat| v.abs().cmp(&one).is_some_and(|c| c <= 0);
  match a {
    Amplitude::ArcSin(v) => in_unit(&v).then(|| v.asin(wp, rm, cc)),
    Amplitude::ArcCos(v) => in_unit(&v).then(|| v.acos(wp, rm, cc)),
    Amplitude::ArcTan(v) => Some(v.atan(wp, rm, cc)),
    Amplitude::ArcCot(v) => Some(if v.is_zero() {
      cc.pi(wp, rm).div(&int(2, wp), wp, rm)
    } else {
      one.div(&v, wp, rm).atan(wp, rm, cc)
    }),
  }
}

/// `InverseJacobi<kind>[x, m]` (kind such as "SN", "DC") for real `x` and
/// real `m`, as `(re, im)`. Covers the values that are real (with `m < 1`)
/// and, for `0 < m < 1`, the complex values of `dn^-1` and `nd^-1` on the
/// real axis, which lie on the lines `K + i y` and `i y` (+ `2 i K'`).
/// `None` elsewhere (the caller then uses the machine complex path).
pub(crate) fn inverse_jacobi_bigfloat(
  kind: &str,
  x: &BigFloat,
  m: &BigFloat,
  wp: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<(BigFloat, BigFloat)> {
  let zero = int(0, wp);
  let one = int(1, wp);
  let recip = |v: &BigFloat| (!v.is_zero()).then(|| one.div(v, wp, rm));
  match kind {
    "NS" => return inverse_jacobi_bigfloat("SN", &recip(x)?, m, wp, rm, cc),
    "NC" => return inverse_jacobi_bigfloat("CN", &recip(x)?, m, wp, rm, cc),
    "ND" => return inverse_jacobi_bigfloat("DN", &recip(x)?, m, wp, rm, cc),
    "DS" => return inverse_jacobi_bigfloat("SD", &recip(x)?, m, wp, rm, cc),
    "DC" => return inverse_jacobi_bigfloat("CD", &recip(x)?, m, wp, rm, cc),
    _ => {}
  }
  if !m.cmp(&one).is_some_and(|c| c < 0) {
    return None;
  }
  let x2 = x.mul(x, wp, rm);
  let real = |a: Amplitude, cc: &mut Consts| -> Option<(BigFloat, BigFloat)> {
    let phi = amplitude_value(a, wp, rm, cc)?;
    Some((elliptic_f_bigfloat(&phi, m, wp, rm, cc)?, int(0, wp)))
  };
  match kind {
    "SN" => real(Amplitude::ArcSin(x.clone()), cc),
    "CN" => real(Amplitude::ArcCos(x.clone()), cc),
    "SC" => real(Amplitude::ArcTan(x.clone()), cc),
    "CS" => real(Amplitude::ArcCot(x.clone()), cc),
    "SD" => {
      let d = one.add(&m.mul(&x2, wp, rm), wp, rm);
      if !d.is_positive() || d.is_zero() {
        return None;
      }
      real(Amplitude::ArcSin(x.div(&d.sqrt(wp, rm), wp, rm)), cc)
    }
    "CD" => {
      // cn = cd Sqrt[(1 - m)/(1 - m cd^2)]
      let d = one.sub(&m.mul(&x2, wp, rm), wp, rm);
      if !d.is_positive() || d.is_zero() {
        return None;
      }
      let q = one.sub(m, wp, rm).div(&d, wp, rm).sqrt(wp, rm);
      real(Amplitude::ArcCos(x.mul(&q, wp, rm)), cc)
    }
    "DN" => {
      if !m.is_positive() || m.is_zero() {
        return None;
      }
      let mc = one.sub(m, wp, rm);
      let kc = mc.sqrt(wp, rm);
      let abs_x = x.abs();
      let below = |a: &BigFloat, b: &BigFloat| a.cmp(b).is_some_and(|c| c < 0);
      // 2 i K' shifts dn to -dn.
      let two_kp = |cc: &mut Consts| -> Option<BigFloat> {
        Some(elliptic_k_bigfloat(&mc, wp, rm, cc)?.mul(&int(2, wp), wp, rm))
      };
      if below(&abs_x, &kc) {
        // dn(K + i y | m) = k' cd(y | 1 - m)
        let (y, _) =
          inverse_jacobi_bigfloat("CD", &x.div(&kc, wp, rm), &mc, wp, rm, cc)?;
        return Some((elliptic_k_bigfloat(m, wp, rm, cc)?, y));
      }
      if below(&one, &abs_x) {
        // dn(i y | m) = dc(y | 1 - m)
        let (y, _) = inverse_jacobi_bigfloat("DC", &abs_x, &mc, wp, rm, cc)?;
        if is_neg(x) {
          let tk = two_kp(cc)?;
          return Some((zero, tk.sub(&y, wp, rm)));
        }
        return Some((zero, y));
      }
      // k' <= |x| <= 1: real, shifted by 2 i K' for negative x.
      let s2 = one.sub(&x2, wp, rm).div(m, wp, rm);
      let (u, _) = real(Amplitude::ArcSin(s2.sqrt(wp, rm)), cc)?;
      if is_neg(x) {
        return Some((u, two_kp(cc)?));
      }
      Some((u, zero))
    }
    _ => None,
  }
}

/// `α = p/q` with small `p`, `q` when `alpha` is (to rounding) such a
/// fraction, so `1/Γ(α k + β)` follows a recurrence in steps of `q`.
fn small_fraction(alpha: f64) -> Option<(i64, i64)> {
  (1..=64i64).find_map(|q| {
    let p = (alpha * q as f64).round();
    ((alpha * q as f64 - p).abs() <= 1e-12 * p.abs().max(1.0)
      && (1.0..=64.0).contains(&p))
    .then_some((p as i64, q))
  })
}

/// `E_{α,β}(z) = Σ_k z^k / Γ(α k + β)` for real `α`, `β`, `z` at `bits` bits.
/// The series is summed with guard bits grown past its cancellation (for a
/// negative `z` its terms reach `e^(|z|^(1/α))` before the sum settles);
/// a negative `α` goes through `E_{α,β}(z) = 1/Γ(β) - E_{-α,β}(1/z)`.
pub(crate) fn mittag_leffler_bigfloat(
  alpha: &BigFloat,
  beta: &BigFloat,
  z: &BigFloat,
  bits: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> Option<BigFloat> {
  use crate::functions::math_ast::{bigfloat_to_f64_approx, gamma_bigfloat};
  let rgamma = |x: &BigFloat, wp: usize, cc: &mut Consts| {
    if x.is_zero() || (x.is_int() && x.is_negative()) {
      int(0, wp)
    } else {
      int(1, wp).div(&gamma_bigfloat(x, wp, rm, cc), wp, rm)
    }
  };
  if alpha.is_zero() {
    return None;
  }
  if alpha.is_negative() {
    if z.is_zero() {
      return None;
    }
    let wp = bits + 32;
    let inv = int(1, wp).div(z, wp, rm);
    let e = mittag_leffler_bigfloat(&alpha.neg(), beta, &inv, wp, rm, cc)?;
    return Some(rgamma(beta, wp, cc).sub(&e, wp, rm));
  }
  let (af, bf, zf) = (
    bigfloat_to_f64_approx(alpha)?,
    bigfloat_to_f64_approx(beta)?,
    bigfloat_to_f64_approx(z)?,
  );
  if z.is_zero() {
    return Some(rgamma(beta, bits, cc));
  }
  // log2 of the largest term, to size the first guard.
  let lz = zf.abs().log2();
  let mut peak = 0.0f64;
  let mut k = 0usize;
  loop {
    let x = af * k as f64 + bf;
    let t = k as f64 * lz
      - if x > 0.0 {
        crate::functions::math_ast::lgamma(x) / std::f64::consts::LN_2
      } else {
        0.0
      };
    peak = peak.max(t);
    if (x > 2.0 && t < peak - 64.0) || k > 1_000_000 {
      break;
    }
    k += 1;
  }
  let fraction = small_fraction(af);
  let mut guard = 64 + peak.max(0.0) as usize;
  loop {
    // Rounded up so repeated calls (a Plot) share the cached gammas below.
    let wp = (bits + guard).div_ceil(128) * 128;
    let mut sum = int(0, wp);
    let mut zpow = int(1, wp);
    let mut max_exp = i64::MIN / 4;
    let mut small = 0;
    // The last q (α k + β, 1/Γ(α k + β)) pairs, for the α = p/q recurrence.
    let mut ring: Vec<(BigFloat, BigFloat)> = Vec::new();
    let limit = 64 * wp + 2_000_000;
    let mut converged = false;
    let mut arg = beta.clone();
    for k in 0..limit {
      if k > 0 {
        arg = arg.add(alpha, wp, rm);
      }
      let rg = match fraction {
        Some((p, q)) if k >= q as usize => {
          // 1/Γ(x + p) = 1/Γ(x) / (x (x+1) … (x+p-1)), x = α(k-q) + β
          let (x, prev) = &ring[k % q as usize];
          let mut den = x.clone();
          for j in 1..p {
            den = den.mul(&x.add(&BigFloat::from_i64(j, wp), wp, rm), wp, rm);
          }
          if den.is_zero() {
            rgamma(&arg, wp, cc)
          } else {
            prev.div(&den, wp, rm)
          }
        }
        Some(_) => cached_rgamma(&arg, wp, rm, cc),
        None => rgamma(&arg, wp, cc),
      };
      if let Some((_, q)) = fraction {
        let q = q as usize;
        if ring.len() < q {
          ring.push((arg.clone(), rg.clone()));
        } else {
          ring[k % q] = (arg.clone(), rg.clone());
        }
      }
      let term = zpow.mul(&rg, wp, rm);
      sum = sum.add(&term, wp, rm);
      let te = bf_exp(&term);
      max_exp = max_exp.max(te);
      let past_peak = (af * k as f64 + bf) > 2.0
        && (k as f64) * lz
          - crate::functions::math_ast::lgamma(af * k as f64 + bf)
            / std::f64::consts::LN_2
          < peak - 8.0;
      // The cancelled sum keeps `bits` significant bits: terms below its
      // last one no longer matter.
      if past_peak && te < bf_exp(&sum) - bits as i64 - 8 {
        small += 1;
        if small >= 2 {
          converged = true;
          break;
        }
      } else {
        small = 0;
      }
      zpow = zpow.mul(z, wp, rm);
    }
    if !converged {
      return None;
    }
    let lost = (max_exp - bf_exp(&sum)).max(0) as usize;
    if lost + 16 <= guard || guard > 16 * bits + 8192 {
      return Some(sum);
    }
    guard = lost + 64;
  }
}

/// `1/Γ(x)` (0 at the poles), memoised per thread: the Mittag-Leffler
/// recurrence needs the same few starting values at every point of a plot.
fn cached_rgamma(
  x: &BigFloat,
  wp: usize,
  rm: RoundingMode,
  cc: &mut Consts,
) -> BigFloat {
  use std::cell::RefCell;
  use std::collections::HashMap;
  thread_local! {
    static CACHE: RefCell<HashMap<(String, usize), BigFloat>> =
      RefCell::new(HashMap::new());
  }
  let key = (format!("{x:?}"), wp);
  if let Some(v) = CACHE.with(|c| c.borrow().get(&key).cloned()) {
    return v;
  }
  let v = if x.is_zero() || (x.is_int() && x.is_negative()) {
    int(0, wp)
  } else {
    int(1, wp).div(
      &crate::functions::math_ast::gamma_bigfloat(x, wp, rm, cc),
      wp,
      rm,
    )
  };
  CACHE.with(|c| {
    let mut c = c.borrow_mut();
    if c.len() > 4096 {
      c.clear();
    }
    c.insert(key, v.clone());
  });
  v
}
