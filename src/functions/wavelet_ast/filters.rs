//! Filter-coefficient construction for all wavelet families.
//!
//! Filters follow the Wolfram Language normalization: lowpass coefficients
//! sum to 1 and highpass coefficients sum to 0. The transforms multiply by
//! Sqrt[2] per level, so `Sqrt[2] h` is the orthonormal filter. For
//! biorthogonal families the primal lowpass is the synthesis filter and the
//! dual lowpass the analysis filter; for orthogonal families they coincide.

use crate::syntax::Expr;

/// One filter as ascending (index, coefficient) pairs.
pub type Filter = Vec<(i64, f64)>;
/// Exact symbolic filter, when the family has a closed form.
pub type ExactFilter = Vec<(i64, Expr)>;

pub struct WaveletFilters {
  pub primal_lo: Filter,
  pub dual_lo: Filter,
  pub primal_lo_exact: Option<ExactFilter>,
  pub dual_lo_exact: Option<ExactFilter>,
}

impl WaveletFilters {
  fn orthogonal(lo: Filter, exact: Option<ExactFilter>) -> Self {
    WaveletFilters {
      primal_lo: lo.clone(),
      dual_lo: lo,
      primal_lo_exact: exact.clone(),
      dual_lo_exact: exact,
    }
  }
}

/// Highpass from a lowpass via the quadrature-mirror relation
/// g[k] = (-1)^k h[1-k]. The primal highpass mirrors the dual lowpass and
/// vice versa, which yields a perfect-reconstruction filter bank.
pub fn highpass_from(lo: &Filter) -> Filter {
  let mut g: Filter = lo
    .iter()
    .map(|&(i, c)| {
      let k = 1 - i;
      let sign = if k.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
      (k, sign * c)
    })
    .collect();
  g.sort_by_key(|p| p.0);
  g
}

pub fn highpass_from_exact(lo: &ExactFilter) -> ExactFilter {
  let mut g: ExactFilter = lo
    .iter()
    .map(|(i, c)| {
      let k = 1 - i;
      let e = if k.rem_euclid(2) == 0 {
        c.clone()
      } else {
        negate_expr(c)
      };
      (k, e)
    })
    .collect();
  g.sort_by_key(|a| a.0);
  g
}

fn negate_expr(e: &Expr) -> Expr {
  let call = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(-1), e.clone()].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&call).unwrap_or(call)
}

/// Flip the sign of every coefficient (indices preserved).
pub fn negate_filter(f: &Filter) -> Filter {
  f.iter().map(|&(i, c)| (i, -c)).collect()
}

pub fn negate_filter_exact(f: &ExactFilter) -> ExactFilter {
  f.iter().map(|(i, c)| (*i, negate_expr(c))).collect()
}

/// Parse and evaluate a Wolfram Language expression used to build exact
/// filter values (input is always internally generated, never user data).
fn wl(code: &str) -> Expr {
  crate::syntax::string_to_expr(code)
    .ok()
    .and_then(|e| crate::evaluator::evaluate_expr_to_expr(&e).ok())
    .unwrap_or_else(|| panic!("internal wavelet expression failed: {code}"))
}

fn rational(num: i128, den: i128) -> Expr {
  crate::functions::math_ast::make_rational(num, den)
}

// ---------------------------------------------------------------------------
// Rational arithmetic used for the exact spline-family filters
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Rat {
  num: i128,
  den: i128,
}

impl Rat {
  fn new(num: i128, den: i128) -> Self {
    debug_assert!(den != 0);
    let g = gcd(num.unsigned_abs(), den.unsigned_abs()).max(1) as i128;
    let sign = if den < 0 { -1 } else { 1 };
    Rat {
      num: sign * num / g,
      den: sign * den / g,
    }
  }
  const ZERO: Rat = Rat { num: 0, den: 1 };
  fn add(self, other: Rat) -> Rat {
    Rat::new(
      self.num * other.den + other.num * self.den,
      self.den * other.den,
    )
  }
  fn mul(self, other: Rat) -> Rat {
    Rat::new(self.num * other.num, self.den * other.den)
  }
  fn to_f64(self) -> f64 {
    self.num as f64 / self.den as f64
  }
}

use crate::functions::math_ast::gcd_u128 as gcd;

fn binomial(n: u64, k: u64) -> i128 {
  if k > n {
    return 0;
  }
  let k = k.min(n - k);
  let mut result: i128 = 1;
  for i in 0..k {
    result = result * (n - i) as i128 / (i + 1) as i128;
  }
  result
}

/// Laurent polynomial in u = e^{-i w/2} with rational coefficients,
/// represented as a map exponent -> coefficient.
#[derive(Clone)]
struct LaurentRat(std::collections::BTreeMap<i64, Rat>);

impl LaurentRat {
  fn one() -> Self {
    let mut m = std::collections::BTreeMap::new();
    m.insert(0, Rat::new(1, 1));
    LaurentRat(m)
  }
  fn term(exp: i64, coef: Rat) -> Self {
    let mut m = std::collections::BTreeMap::new();
    if coef.num != 0 {
      m.insert(exp, coef);
    }
    LaurentRat(m)
  }
  fn add(&self, other: &Self) -> Self {
    let mut m = self.0.clone();
    for (&e, &c) in &other.0 {
      let v = m.get(&e).copied().unwrap_or(Rat::ZERO).add(c);
      if v.num == 0 {
        m.remove(&e);
      } else {
        m.insert(e, v);
      }
    }
    LaurentRat(m)
  }
  fn mul(&self, other: &Self) -> Self {
    let mut m = std::collections::BTreeMap::new();
    for (&e1, &c1) in &self.0 {
      for (&e2, &c2) in &other.0 {
        let e = e1 + e2;
        let v: Rat = m.get(&e).copied().unwrap_or(Rat::ZERO).add(c1.mul(c2));
        m.insert(e, v);
      }
    }
    m.retain(|_, c: &mut Rat| c.num != 0);
    LaurentRat(m)
  }
  fn pow(&self, n: u32) -> Self {
    let mut result = LaurentRat::one();
    for _ in 0..n {
      result = result.mul(self);
    }
    result
  }
  fn scale(&self, s: Rat) -> Self {
    let mut m = self.0.clone();
    m.retain(|_, c| {
      *c = c.mul(s);
      c.num != 0
    });
    LaurentRat(m)
  }
}

/// cos(w/2) = (u + u^{-1}) / 2 in u = e^{-i w/2}.
fn cos_half() -> LaurentRat {
  LaurentRat::term(1, Rat::new(1, 2)).add(&LaurentRat::term(-1, Rat::new(1, 2)))
}

/// sin^2(w/2) = -(u - u^{-1})^2 / 4.
fn sin2_half() -> LaurentRat {
  let diff = LaurentRat::term(1, Rat::new(1, 1))
    .add(&LaurentRat::term(-1, Rat::new(-1, 1)));
  diff.mul(&diff).scale(Rat::new(-1, 4))
}

/// Extract the integer-indexed filter h_k from a Laurent polynomial
/// Sum h_k u^{2k}: all exponents must be even.
fn laurent_to_filter(p: &LaurentRat) -> Vec<(i64, Rat)> {
  p.0
    .iter()
    .map(|(&e, &c)| {
      debug_assert!(e % 2 == 0, "odd exponent in wavelet filter expansion");
      (e / 2, c)
    })
    .collect()
}

// ---------------------------------------------------------------------------
// Complex arithmetic + polynomial root finding for the Daubechies family
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct C64 {
  re: f64,
  im: f64,
}

impl C64 {
  fn new(re: f64, im: f64) -> Self {
    C64 { re, im }
  }
  fn add(self, o: C64) -> C64 {
    C64::new(self.re + o.re, self.im + o.im)
  }
  fn sub(self, o: C64) -> C64 {
    C64::new(self.re - o.re, self.im - o.im)
  }
  fn mul(self, o: C64) -> C64 {
    C64::new(
      self.re * o.re - self.im * o.im,
      self.re * o.im + self.im * o.re,
    )
  }
  fn div(self, o: C64) -> C64 {
    let d = o.re * o.re + o.im * o.im;
    C64::new(
      (self.re * o.re + self.im * o.im) / d,
      (self.im * o.re - self.re * o.im) / d,
    )
  }
  fn abs(self) -> f64 {
    self.re.hypot(self.im)
  }
  fn sqrt(self) -> C64 {
    let r = self.abs();
    let re = ((r + self.re) / 2.0).max(0.0).sqrt();
    let im_mag = ((r - self.re) / 2.0).max(0.0).sqrt();
    C64::new(re, if self.im < 0.0 { -im_mag } else { im_mag })
  }
}

/// All roots of a polynomial with real coefficients (ascending order) via
/// the Durand-Kerner iteration.
fn poly_roots(coeffs: &[f64]) -> Vec<C64> {
  let n = coeffs.len() - 1;
  if n == 0 {
    return vec![];
  }
  let lead = coeffs[n];
  let monic: Vec<f64> = coeffs.iter().map(|c| c / lead).collect();
  let eval = |z: C64| -> C64 {
    let mut acc = C64::new(0.0, 0.0);
    for &c in monic.iter().rev() {
      acc = acc.mul(z).add(C64::new(c, 0.0));
    }
    acc
  };
  // Initial guesses on a spiral to break symmetry.
  let mut roots: Vec<C64> = (0..n)
    .map(|k| {
      let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64 + 0.4;
      let r = 0.8 + 0.1 * k as f64 / n as f64;
      C64::new(r * angle.cos(), r * angle.sin())
    })
    .collect();
  for _ in 0..500 {
    let mut max_delta: f64 = 0.0;
    for i in 0..n {
      let mut denom = C64::new(1.0, 0.0);
      for j in 0..n {
        if i != j {
          denom = denom.mul(roots[i].sub(roots[j]));
        }
      }
      let delta = eval(roots[i]).div(denom);
      roots[i] = roots[i].sub(delta);
      max_delta = max_delta.max(delta.abs());
    }
    if max_delta < 1e-15 {
      break;
    }
  }
  roots
}

/// Daubechies extremal-phase lowpass filter of order n (2n coefficients,
/// indices 0..2n-1, summing to 1) via spectral factorization of the
/// half-band polynomial P(y) = Sum C(n-1+k, k) y^k.
fn daubechies_lowpass(n: usize) -> Filter {
  if n == 1 {
    return vec![(0, 0.5), (1, 0.5)];
  }
  // Roots of P(y), degree n-1.
  let coeffs: Vec<f64> = (0..n)
    .map(|k| binomial((n - 1 + k) as u64, k as u64) as f64)
    .collect();
  let y_roots = poly_roots(&coeffs);
  // Each y-root maps to a z-root pair z + 1/z = 2 - 4 y; keep |z| < 1.
  let mut z_roots: Vec<C64> = Vec::new();
  for y in y_roots {
    let b = C64::new(2.0, 0.0).sub(y.mul(C64::new(4.0, 0.0)));
    let disc = b.mul(b).sub(C64::new(4.0, 0.0)).sqrt();
    let z1 = b.add(disc).div(C64::new(2.0, 0.0));
    let z2 = b.sub(disc).div(C64::new(2.0, 0.0));
    z_roots.push(if z1.abs() < z2.abs() { z1 } else { z2 });
  }
  // h(z) = (1+z)^n * prod (z - z_j), then normalize to sum 1 and reverse so
  // that h_0 carries the largest leading coefficient (standard ordering,
  // e.g. db2 = {(1+Sqrt[3])/8, (3+Sqrt[3])/8, (3-Sqrt[3])/8, (1-Sqrt[3])/8}).
  let mut poly: Vec<C64> = vec![C64::new(1.0, 0.0)];
  for _ in 0..n {
    poly = poly_mul(&poly, &[C64::new(1.0, 0.0), C64::new(1.0, 0.0)]);
  }
  for zj in &z_roots {
    poly = poly_mul(&poly, &[C64::new(-zj.re, -zj.im), C64::new(1.0, 0.0)]);
  }
  let sum: f64 = poly.iter().map(|c| c.re).sum();
  let mut h: Vec<f64> = poly.iter().map(|c| c.re / sum).collect();
  h.reverse();
  h.into_iter()
    .enumerate()
    .map(|(i, c)| (i as i64, c))
    .collect()
}

fn poly_mul(a: &[C64], b: &[C64]) -> Vec<C64> {
  let mut out = vec![C64::new(0.0, 0.0); a.len() + b.len() - 1];
  for (i, &ai) in a.iter().enumerate() {
    for (j, &bj) in b.iter().enumerate() {
      out[i + j] = out[i + j].add(ai.mul(bj));
    }
  }
  out
}

// ---------------------------------------------------------------------------
// Family constructors
// ---------------------------------------------------------------------------

pub fn haar_filters() -> WaveletFilters {
  let exact = vec![(0, rational(1, 2)), (1, rational(1, 2))];
  WaveletFilters::orthogonal(vec![(0, 0.5), (1, 0.5)], Some(exact))
}

pub fn daubechies_filters(n: usize) -> WaveletFilters {
  if n == 1 {
    return haar_filters();
  }
  let exact = if n == 2 {
    Some(vec![
      (0, wl("(1 + Sqrt[3])/8")),
      (1, wl("(3 + Sqrt[3])/8")),
      (2, wl("(3 - Sqrt[3])/8")),
      (3, wl("(1 - Sqrt[3])/8")),
    ])
  } else {
    None
  };
  WaveletFilters::orthogonal(daubechies_lowpass(n), exact)
}

/// Lookup into the static tables (values sum to Sqrt[2], so divide).
fn table_filter(rec_lo: &[f64]) -> Filter {
  let s = std::f64::consts::SQRT_2;
  rec_lo
    .iter()
    .enumerate()
    .map(|(i, &c)| (i as i64, c / s))
    .collect()
}

pub fn symlet_filters(n: usize) -> Option<WaveletFilters> {
  if n == 1 {
    return Some(haar_filters());
  }
  let table = super::tables::SYMLET_REC_LO.get(n - 2)?;
  Some(WaveletFilters::orthogonal(table_filter(table), None))
}

pub fn coiflet_filters(n: usize) -> Option<WaveletFilters> {
  let table = super::tables::COIFLET_REC_LO.get(n - 1)?;
  Some(WaveletFilters::orthogonal(table_filter(table), None))
}

/// Biorthogonal spline wavelet of order (n, m): the primal lowpass is the
/// B-spline binomial filter of order n; the dual lowpass is the shortest
/// CDF complement with m factors of cos(w/2). Both are exact dyadic
/// rationals. Requires n + m even.
pub fn biorthogonal_spline_filters(n: u32, m: u32) -> WaveletFilters {
  let eps = (n % 2) as i64;
  // Primal: u^eps cos^n(w/2)
  let primal_poly =
    LaurentRat::term(eps, Rat::new(1, 1)).mul(&cos_half().pow(n));
  // Dual: u^eps cos^m(w/2) Sum_{j<q} C(q-1+j, j) sin^{2j}(w/2), q = (n+m)/2
  let q = (n + m) / 2;
  let s2 = sin2_half();
  let mut series = LaurentRat::term(0, Rat::ZERO);
  let mut s_pow = LaurentRat::one();
  for j in 0..q {
    let c = binomial((q - 1 + j) as u64, j as u64);
    series = series.add(&s_pow.scale(Rat::new(c, 1)));
    s_pow = s_pow.mul(&s2);
  }
  let dual_poly = LaurentRat::term(eps, Rat::new(1, 1))
    .mul(&cos_half().pow(m))
    .mul(&series);

  let primal_rat = laurent_to_filter(&primal_poly);
  let dual_rat = laurent_to_filter(&dual_poly);
  let to_f64 = |f: &Vec<(i64, Rat)>| -> Filter {
    f.iter().map(|&(i, c)| (i, c.to_f64())).collect()
  };
  let to_exact = |f: &Vec<(i64, Rat)>| -> ExactFilter {
    f.iter()
      .map(|&(i, c)| (i, rational(c.num, c.den)))
      .collect()
  };
  WaveletFilters {
    primal_lo: to_f64(&primal_rat),
    dual_lo: to_f64(&dual_rat),
    primal_lo_exact: Some(to_exact(&primal_rat)),
    dual_lo_exact: Some(to_exact(&dual_rat)),
  }
}

pub fn reverse_biorthogonal_spline_filters(n: u32, m: u32) -> WaveletFilters {
  let f = biorthogonal_spline_filters(n, m);
  WaveletFilters {
    primal_lo: f.dual_lo,
    dual_lo: f.primal_lo,
    primal_lo_exact: f.dual_lo_exact,
    dual_lo_exact: f.primal_lo_exact,
  }
}

/// CDF 9/7 (lossy JPEG2000): spectral factorization of the order-4
/// half-band polynomial into its real linear factor (primal, 7 taps) and
/// the remaining quadratic (dual, 9 taps). Computed at full f64 precision.
pub fn cdf_97_filters() -> WaveletFilters {
  // P(y) = 1 + 4 y + 10 y^2 + 20 y^3; real root by Newton iteration.
  let p = |y: f64| 1.0 + 4.0 * y + 10.0 * y * y + 20.0 * y * y * y;
  let dp = |y: f64| 4.0 + 20.0 * y + 60.0 * y * y;
  let mut r = -0.342;
  for _ in 0..60 {
    r -= p(r) / dp(r);
  }
  // Primal factor: (1 - y/r); dual factor: P(y) / (1 - y/r) =
  // 20 r (y^2 + b y + c) with the quotient computed by synthetic division.
  // P(y) = (1 - y/r) * (a0 + a1 y + a2 y^2) requires
  // a0 = 1, a1 = 4 + a0/r, a2 = 10 + a1/r  (and 20 + a2/r ~ 0).
  let a0 = 1.0;
  let a1 = 4.0 + a0 / r;
  let a2 = 10.0 + a1 / r;

  // Laurent expansion with f64 coefficients: reuse the rational machinery
  // shapes but inline with f64 maps keyed by exponent.
  use std::collections::BTreeMap;
  type LF = BTreeMap<i64, f64>;
  let mul = |a: &LF, b: &LF| -> LF {
    let mut out = LF::new();
    for (&e1, &c1) in a {
      for (&e2, &c2) in b {
        *out.entry(e1 + e2).or_insert(0.0) += c1 * c2;
      }
    }
    out
  };
  let cos_half: LF = [(1i64, 0.5f64), (-1i64, 0.5f64)].into_iter().collect();
  let sin2: LF = {
    // -(u - 1/u)^2 / 4 = -u^2/4 + 1/2 - u^-2/4
    [(2i64, -0.25f64), (0i64, 0.5f64), (-2i64, -0.25f64)]
      .into_iter()
      .collect()
  };
  let mut cos4 = LF::from([(0i64, 1.0f64)]);
  for _ in 0..4 {
    cos4 = mul(&cos4, &cos_half);
  }
  // primal: cos^4 * (1 - y/r)
  let lin: LF = {
    let mut l = LF::from([(0i64, 1.0f64)]);
    for (e, c) in &sin2 {
      *l.entry(*e).or_insert(0.0) += -c / r;
    }
    l
  };
  let primal_poly = mul(&cos4, &lin);
  // dual: cos^4 * (a0 + a1 y + a2 y^2)
  let quad: LF = {
    let mut q = LF::from([(0i64, a0)]);
    for (e, c) in &sin2 {
      *q.entry(*e).or_insert(0.0) += a1 * c;
    }
    let y2 = mul(&sin2, &sin2);
    for (e, c) in &y2 {
      *q.entry(*e).or_insert(0.0) += a2 * c;
    }
    q
  };
  let dual_poly = mul(&cos4, &quad);
  let collect = |p: &LF| -> Filter {
    p.iter()
      .filter(|(_, c)| c.abs() > 1e-14)
      .map(|(&e, &c)| (e / 2, c))
      .collect()
  };
  WaveletFilters {
    primal_lo: collect(&primal_poly),
    dual_lo: collect(&dual_poly),
    primal_lo_exact: None,
    dual_lo_exact: None,
  }
}

/// CDF 5/3 (lossless JPEG2000, LeGall): identical to the biorthogonal
/// spline pair of order (2, 2) — exact dyadic rationals.
pub fn cdf_53_filters() -> WaveletFilters {
  biorthogonal_spline_filters(2, 2)
}

/// Shannon lowpass: h_k = Sinc(k/2)/2 truncated to |k| <= lim.
pub fn shannon_filters(lim: f64) -> WaveletFilters {
  let l = lim.floor() as i64;
  let mut lo: Filter = Vec::new();
  let mut exact: ExactFilter = Vec::new();
  for k in -l..=l {
    let (v, e) = if k == 0 {
      (0.5, rational(1, 2))
    } else if k % 2 == 0 {
      (0.0, Expr::Integer(0))
    } else {
      // h_k = sin(pi k/2)/(pi k) is even in k, so use |k| throughout.
      let sign = if (k.unsigned_abs() as i64 - 1) / 2 % 2 == 0 {
        1i128
      } else {
        -1i128
      };
      let v = (std::f64::consts::PI * k as f64 / 2.0).sin()
        / (std::f64::consts::PI * k as f64);
      (v, wl(&format!("{}/({}*Pi)", sign, k.unsigned_abs())))
    };
    lo.push((k, v));
    exact.push((k, e));
  }
  WaveletFilters::orthogonal(lo, Some(exact))
}

/// Meyer auxiliary polynomial nu_n(x) = x^{n+1} Sum_{k<=n} C(n+k,k)(1-x)^k,
/// clamped to [0, 1] outside the transition band.
pub fn meyer_nu(n: u32, x: f64) -> f64 {
  if x <= 0.0 {
    return 0.0;
  }
  if x >= 1.0 {
    return 1.0;
  }
  let mut sum = 0.0;
  let mut one_minus_pow = 1.0;
  for k in 0..=n {
    sum += binomial((n + k) as u64, k as u64) as f64 * one_minus_pow;
    one_minus_pow *= 1.0 - x;
  }
  x.powi(n as i32 + 1) * sum
}

/// Meyer lowpass filter: inverse Fourier coefficients of
/// m0(w) = PhiHat(2 w) on [-pi, pi], truncated to |k| <= lim.
pub fn meyer_filters(n: u32, lim: f64) -> WaveletFilters {
  let l = lim.floor() as i64;
  let pi = std::f64::consts::PI;
  // PhiHat(2 w): 1 for |w| <= pi/3, cos(pi/2 nu(3 |w|/pi - 1)) in the
  // transition band pi/3..2 pi/3, 0 beyond.
  let m0 = |w: f64| -> f64 {
    let a = w.abs();
    if a <= pi / 3.0 {
      1.0
    } else if a <= 2.0 * pi / 3.0 {
      (pi / 2.0 * meyer_nu(n, 3.0 * a / pi - 1.0)).cos()
    } else {
      0.0
    }
  };
  let lo = (-l..=l)
    .map(|k| {
      // h_k = (1/pi) [ Integral_0^{2pi/3} m0(w) cos(k w) dw ] by symmetry
      let steps = 4000;
      let upper = 2.0 * pi / 3.0;
      let dw = upper / steps as f64;
      let mut acc = 0.0;
      for i in 0..=steps {
        let w = i as f64 * dw;
        let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
        acc += weight * m0(w) * (k as f64 * w).cos();
      }
      (k, acc * dw / pi)
    })
    .collect();
  WaveletFilters::orthogonal(lo, None)
}

/// Battle-Lemarie lowpass of order n (orthonormalized B-spline of degree n):
/// m0(w) = u^eps cos^{n+1}(w/2) sqrt(S(w)/S(2w)) with
/// S(w) = Sum_k |sinc((w + 2 pi k)/2)|^{2(n+1)}, truncated to |k| <= lim.
pub fn battle_lemarie_filters(n: u32, lim: f64) -> WaveletFilters {
  let l = lim.floor() as i64;
  let pi = std::f64::consts::PI;
  let p = 2 * (n + 1) as i32;
  let s = |w: f64| -> f64 {
    let mut acc = 0.0;
    for k in -60i64..=60 {
      let x = (w + 2.0 * pi * k as f64) / 2.0;
      let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
      acc += sinc.powi(p);
    }
    acc
  };
  let eps = ((n + 1) % 2) as f64;
  // Complex integrand: m0(w) e^{i k w}; the result is real.
  let lo = (-l..=l)
    .map(|k| {
      let steps = 4096;
      let dw = 2.0 * pi / steps as f64;
      let mut acc = 0.0;
      for i in 0..steps {
        let w = -pi + (i as f64 + 0.5) * dw;
        let mag = (w / 2.0).cos().abs().powi((n + 1) as i32)
          * (s(w) / s(2.0 * w)).sqrt();
        // cos^{n+1} sign handling: cos(w/2) >= 0 on [-pi, pi], so the
        // magnitude form is exact here.
        let phase = -eps * w / 2.0 + k as f64 * w;
        acc += mag * phase.cos();
      }
      (k, acc * dw / (2.0 * pi))
    })
    .collect();
  WaveletFilters::orthogonal(lo, None)
}
