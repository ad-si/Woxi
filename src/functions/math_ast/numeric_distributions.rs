//! Continuous distributions whose PDF and CDF Woxi evaluates numerically:
//! `BeckmannDistribution[μ1, μ2, σ1, σ2]`, `LandauDistribution[μ, σ]`,
//! `NoncentralBetaDistribution[α, β, δ]` and the Pearson family
//! `PearsonDistribution[a1, a0, b2, b1, b0]` (optionally preceded by the
//! type).
//!
//! Like wolframscript, a property only becomes a number when the point or a
//! parameter is an inexact number; with exact arguments it stays unevaluated
//! (the noncentral beta PDF, which has a closed form, is the exception).
//! Quantiles bisect the numeric CDF and sit at the ends of the support for
//! the probabilities 0 and 1.

use crate::InterpreterError;
use crate::helpers::{call, unevaluated};
use crate::syntax::Expr;

use super::numeric_utils::try_eval_to_f64;

/// The heads handled here.
pub(crate) fn is_numeric_distribution(name: &str, n: usize) -> bool {
  matches!(
    (name, n),
    ("BeckmannDistribution", 4)
      | ("LandauDistribution", 2)
      | ("NoncentralBetaDistribution", 3)
      | ("PearsonDistribution", 5 | 6)
  )
}

/// `property[name[dargs], x]` for `property` in PDF, CDF and Quantile, or
/// `None` when `name[dargs]` is not handled here.
pub(crate) fn numeric_distribution_property(
  property: &str,
  name: &str,
  dargs: &[Expr],
  x: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  if !is_numeric_distribution(name, dargs.len()) {
    return None;
  }
  let stay = || Ok(call(property, vec![unevaluated(name, dargs), x.clone()]));
  if property == "PDF"
    && name == "NoncentralBetaDistribution"
    && !is_inexact(x)
    && !dargs.iter().any(is_inexact)
  {
    return Some(noncentral_beta_pdf_closed_form(dargs, x));
  }
  let Some(dist) = Dist::new(name, dargs) else {
    return Some(stay());
  };
  let inexact = is_inexact(x) || dargs.iter().any(is_inexact);
  Some(match property {
    "PDF" | "CDF" => {
      let Some(xv) = try_eval_to_f64(x).filter(|_| inexact) else {
        return Some(stay());
      };
      let value = if property == "PDF" {
        dist.pdf(xv)
      } else {
        dist.cdf(xv)
      };
      match value {
        v if v.is_finite() => Ok(Expr::Real(v)),
        _ => stay(),
      }
    }
    "Quantile" => {
      let (lo, hi) = dist.support();
      match x {
        Expr::Integer(0) => Ok(bound_expr(lo)),
        Expr::Integer(1) => Ok(bound_expr(hi)),
        Expr::Real(q) if *q == 0.0 => Ok(bound_expr(lo)),
        Expr::Real(q) if *q == 1.0 => Ok(bound_expr(hi)),
        Expr::Real(q) if *q > 0.0 && *q < 1.0 => {
          match dist.quantile(*q, lo, hi) {
            Some(v) => Ok(Expr::Real(v)),
            None => stay(),
          }
        }
        _ => stay(),
      }
    }
    _ => stay(),
  })
}

fn is_inexact(e: &Expr) -> bool {
  matches!(e, Expr::Real(_) | Expr::BigFloat(..))
}

/// An end of the support as Wolfram writes it: an integer when it is one,
/// `±Infinity` when unbounded.
fn bound_expr(b: f64) -> Expr {
  if b == f64::INFINITY {
    Expr::Identifier("Infinity".into())
  } else if b == f64::NEG_INFINITY {
    crate::evaluator::evaluate_expr_to_expr(&call(
      "Times",
      vec![Expr::Integer(-1), Expr::Identifier("Infinity".into())],
    ))
    .unwrap_or_else(|_| Expr::Identifier("Indeterminate".into()))
  } else if b.fract() == 0.0 && b.abs() < 1e15 {
    Expr::Integer(b as i128)
  } else {
    Expr::Real(b)
  }
}

fn noncentral_beta_pdf_closed_form(
  dargs: &[Expr],
  x: &Expr,
) -> Result<Expr, InterpreterError> {
  let src = "Piecewise[{{((1 - x)^(-1 + p2)*x^(-1 + p1)*Hypergeometric1F1[p1 + p2, p1, (p3*x)/2])/(E^(p3/2)*Beta[p1, p2]), 0 < x < 1}}, 0]";
  let body = crate::syntax::string_to_expr(src)?;
  let bindings = [
    ("p1", &dargs[0]),
    ("p2", &dargs[1]),
    ("p3", &dargs[2]),
    ("x", x),
  ];
  crate::evaluator::evaluate_expr_to_expr(&crate::syntax::substitute_variables(
    &body, &bindings,
  ))
}

/// A Pearson density `exp(-G(x))` with `G' = (a1 x + a0)/Q(x)`,
/// `Q = b2 x^2 + b1 x + b0`, on the interval of its support.
#[derive(Clone, Copy)]
enum PearsonKernel {
  /// `Q` constant: a normal density.
  Constant,
  /// `Q` linear with root `r`.
  Linear { r: f64 },
  /// `Q` quadratic without real roots.
  NoRealRoots { disc: f64 },
  /// `Q` quadratic with the double root `r`.
  DoubleRoot { r: f64 },
  /// `Q` quadratic with the roots `r1 < r2`; `G = A log|x-r1| + B log|x-r2|`.
  TwoRoots { r1: f64, r2: f64, a: f64, b: f64 },
}

enum Dist {
  Beckmann { m1: f64, m2: f64, s1: f64, s2: f64 },
  Landau { m: f64, s: f64 },
  NoncentralBeta { a: f64, b: f64, d: f64 },
  Pearson(Box<Pearson>),
}

struct Pearson {
  a1: f64,
  a0: f64,
  b2: f64,
  b1: f64,
  b0: f64,
  kernel: PearsonKernel,
  lo: f64,
  hi: f64,
  /// Where the density peaks (clamped into the support), the scale of the
  /// peak, and the log-density there, which every evaluation subtracts.
  mode: f64,
  scale: f64,
  g_mode: f64,
  /// The normalization: the integral of `exp(g_mode - G(x))`.
  norm: f64,
}

impl Dist {
  fn new(name: &str, dargs: &[Expr]) -> Option<Dist> {
    let p: Vec<f64> = dargs
      .iter()
      .map(try_eval_to_f64)
      .collect::<Option<Vec<_>>>()?;
    if p.iter().any(|v| !v.is_finite()) {
      return None;
    }
    match name {
      "BeckmannDistribution" if p[2] > 0.0 && p[3] > 0.0 => {
        Some(Dist::Beckmann {
          m1: p[0],
          m2: p[1],
          s1: p[2],
          s2: p[3],
        })
      }
      "LandauDistribution" if p[1] > 0.0 => {
        Some(Dist::Landau { m: p[0], s: p[1] })
      }
      "NoncentralBetaDistribution"
        if p[0] > 0.0 && p[1] > 0.0 && p[2] >= 0.0 =>
      {
        Some(Dist::NoncentralBeta {
          a: p[0],
          b: p[1],
          d: p[2],
        })
      }
      "PearsonDistribution" => {
        let c = &p[p.len() - 5..];
        Pearson::new(c[0], c[1], c[2], c[3], c[4])
          .map(|p| Dist::Pearson(Box::new(p)))
      }
      _ => None,
    }
  }

  fn support(&self) -> (f64, f64) {
    match self {
      Dist::Beckmann { .. } => (0.0, f64::INFINITY),
      Dist::Landau { .. } => (f64::NEG_INFINITY, f64::INFINITY),
      Dist::NoncentralBeta { .. } => (0.0, 1.0),
      Dist::Pearson(p) => (p.lo, p.hi),
    }
  }

  fn pdf(&self, x: f64) -> f64 {
    match self {
      Dist::Beckmann { m1, m2, s1, s2 } => beckmann_pdf(*m1, *m2, *s1, *s2, x),
      Dist::Landau { m, s } => {
        landau_standard_pdf(landau_standardize(*m, *s, x)) / s
      }
      Dist::NoncentralBeta { a, b, d } => {
        if x <= 0.0 || x >= 1.0 {
          return 0.0;
        }
        let ln = (b - 1.0) * (1.0 - x).ln() + (a - 1.0) * x.ln()
          - d / 2.0
          - ln_beta(*a, *b);
        ln.exp()
          * super::hypergeometric::hypergeometric_1f1(a + b, *a, d * x / 2.0)
      }
      Dist::Pearson(p) => p.pdf(x),
    }
  }

  fn cdf(&self, x: f64) -> f64 {
    match self {
      Dist::Beckmann { m1, m2, s1, s2 } => beckmann_cdf(*m1, *m2, *s1, *s2, x),
      Dist::Landau { m, s } => {
        landau_standard_cdf(landau_standardize(*m, *s, x))
      }
      Dist::NoncentralBeta { a, b, d } => noncentral_beta_cdf(*a, *b, *d, x),
      Dist::Pearson(p) => p.cdf(x),
    }
  }

  /// Bisect the CDF for the `q` quantile inside the support `[lo, hi]`.
  fn quantile(&self, q: f64, lo: f64, hi: f64) -> Option<f64> {
    let cdf = |x: f64| Some(self.cdf(x)).filter(|c| c.is_finite());
    // Bracket by stepping outwards from a finite point of the support.
    let start = if lo.is_finite() && hi.is_finite() {
      f64::midpoint(lo, hi)
    } else if lo.is_finite() {
      lo + 1.0
    } else if hi.is_finite() {
      hi - 1.0
    } else {
      0.0
    };
    let (mut a, mut b) = if cdf(start)? < q {
      let mut step = 1.0;
      let mut b = start;
      let mut a = start;
      while cdf(b)? < q {
        a = b;
        b = if hi.is_finite() {
          f64::midpoint(b, hi)
        } else {
          b + step
        };
        step *= 2.0;
        if !b.is_finite() || step > 1e300 || (hi.is_finite() && hi - b < 1e-300)
        {
          return None;
        }
      }
      (a, b)
    } else {
      let mut step = 1.0;
      let mut a = start;
      let mut b = start;
      while cdf(a)? >= q {
        b = a;
        a = if lo.is_finite() {
          f64::midpoint(lo, a)
        } else {
          a - step
        };
        step *= 2.0;
        if !a.is_finite() || step > 1e300 || (lo.is_finite() && a - lo < 1e-300)
        {
          return None;
        }
      }
      (a, b)
    };
    for _ in 0..200 {
      let mid = f64::midpoint(a, b);
      if mid <= a || mid >= b {
        break;
      }
      if cdf(mid)? < q {
        a = mid;
      } else {
        b = mid;
      }
    }
    Some(b)
  }
}

fn ln_beta(a: f64, b: f64) -> f64 {
  libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b)
}

// ---------------------------------------------------------------------------
// Quadrature

const GK_X: [f64; 8] = [
  0.991_455_371_120_812_6,
  0.949_107_912_342_758_5,
  0.864_864_423_359_769_1,
  0.741_531_185_599_394_4,
  0.586_087_235_467_691_1,
  0.405_845_151_377_397_2,
  0.207_784_955_007_898_5,
  0.0,
];
const GK_WK: [f64; 8] = [
  0.022_935_322_010_529_22,
  0.063_092_092_629_978_55,
  0.104_790_010_322_250_18,
  0.140_653_259_715_525_92,
  0.169_004_726_639_267_9,
  0.190_350_578_064_785_4,
  0.204_432_940_075_298_9,
  0.209_482_141_084_727_83,
];
const GK_WG: [f64; 4] = [
  0.129_484_966_168_869_7,
  0.279_705_391_489_276_7,
  0.381_830_050_505_118_9,
  0.417_959_183_673_469_4,
];

/// 15-point Kronrod estimate and its difference from the embedded 7-point
/// Gauss rule, together with the Kronrod estimate of the integral of `|f|`.
fn gk15(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> (f64, f64, f64) {
  let c = f64::midpoint(a, b);
  let h = 0.5 * (b - a);
  let val = |x: f64| {
    let v = f(x);
    if v.is_finite() { v } else { 0.0 }
  };
  let fc = val(c);
  let mut k = GK_WK[7] * fc;
  let mut g = GK_WG[3] * fc;
  let mut abs = GK_WK[7] * fc.abs();
  for i in 0..7 {
    let d = h * GK_X[i];
    let (f1, f2) = (val(c - d), val(c + d));
    k += GK_WK[i] * (f1 + f2);
    abs += GK_WK[i] * (f1.abs() + f2.abs());
    if i % 2 == 1 {
      g += GK_WG[i / 2] * (f1 + f2);
    }
  }
  (k * h, ((k - g) * h).abs(), abs * h.abs())
}

/// Globally adaptive Gauss–Kronrod quadrature of `f` over the finite `[a, b]`,
/// to a relative tolerance on the integral of `|f|` (so an integral that
/// cancels to almost nothing is resolved to the absolute accuracy its parts
/// allow).
fn adaptive(f: &dyn Fn(f64) -> f64, a: f64, b: f64, pieces: usize) -> f64 {
  if a == b {
    return 0.0;
  }
  let mut intervals: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
  let n = pieces.max(1);
  for i in 0..n {
    let lo = a + (b - a) * i as f64 / n as f64;
    let hi = if i + 1 == n {
      b
    } else {
      a + (b - a) * (i + 1) as f64 / n as f64
    };
    let (v, e, s) = gk15(f, lo, hi);
    intervals.push((lo, hi, v, e, s));
  }
  for _ in 0..4000 {
    let total: f64 = intervals.iter().map(|i| i.2).sum();
    let abs: f64 = intervals.iter().map(|i| i.4).sum();
    let err: f64 = intervals.iter().map(|i| i.3).sum();
    if err <= 1e-14 * total.abs().max(1e-3 * abs) || abs == 0.0 {
      break;
    }
    let (idx, _) = intervals
      .iter()
      .enumerate()
      .max_by(|x, y| x.1.3.total_cmp(&y.1.3))
      .unwrap();
    let (lo, hi, ..) = intervals.swap_remove(idx);
    let mid = f64::midpoint(lo, hi);
    if mid <= lo || mid >= hi {
      break;
    }
    let (v1, e1, s1) = gk15(f, lo, mid);
    let (v2, e2, s2) = gk15(f, mid, hi);
    intervals.push((lo, mid, v1, e1, s1));
    intervals.push((mid, hi, v2, e2, s2));
  }
  let mut sum = 0.0;
  let mut comp = 0.0;
  for i in &intervals {
    let t = sum + i.2;
    comp += if sum.abs() >= i.2.abs() {
      (sum - t) + i.2
    } else {
      (i.2 - t) + sum
    };
    sum = t;
  }
  sum + comp
}

/// `∫ exp(logf(x)) dx` over `[a, b]`, where `a` or `b` may be infinite and
/// the density may be singular or peak at an end, by a double-exponential
/// change of variable followed by adaptive Gauss–Kronrod. `scale` is the
/// width of the peak at a finite end of a half-infinite interval.
fn de_integral(logf: &dyn Fn(f64) -> f64, a: f64, b: f64, scale: f64) -> f64 {
  use std::f64::consts::FRAC_PI_2;
  if a >= b {
    return 0.0;
  }
  match (a.is_finite(), b.is_finite()) {
    (true, true) => {
      // tanh-sinh, each node placed by its distance from the nearer end.
      let half = 0.5 * (b - a);
      let g = |t: f64| {
        let s = FRAC_PI_2 * t.sinh();
        let e = (-2.0 * s.abs()).exp();
        let dist = 2.0 * e / (1.0 + e);
        let x = if t >= 0.0 {
          b - half * dist
        } else {
          a + half * dist
        };
        let ln_w = (half * FRAC_PI_2 * t.cosh() * 4.0 * e
          / ((1.0 + e) * (1.0 + e)))
          .ln();
        (logf(x) + ln_w).exp()
      };
      adaptive(&g, -4.5, 4.5, 16)
    }
    (true, false) => {
      let g = |t: f64| {
        let s = FRAC_PI_2 * t.sinh();
        let x = a + scale * s.exp();
        (logf(x) + scale.ln() + s + (FRAC_PI_2 * t.cosh()).ln()).exp()
      };
      adaptive(&g, -6.5, 6.5, 24)
    }
    (false, true) => {
      let g = |t: f64| {
        let s = FRAC_PI_2 * t.sinh();
        let x = b - scale * s.exp();
        (logf(x) + scale.ln() + s + (FRAC_PI_2 * t.cosh()).ln()).exp()
      };
      adaptive(&g, -6.5, 6.5, 24)
    }
    (false, false) => {
      de_integral(logf, f64::NEG_INFINITY, 0.0, scale)
        + de_integral(logf, 0.0, f64::INFINITY, scale)
    }
  }
}

// ---------------------------------------------------------------------------
// Beckmann: the length of a bivariate normal vector with independent
// components N(μ1, σ1) and N(μ2, σ2).

fn beckmann_pdf(m1: f64, m2: f64, s1: f64, s2: f64, r: f64) -> f64 {
  if r <= 0.0 {
    return 0.0;
  }
  let expo = |th: f64| {
    let u = (r * th.cos() - m1) / s1;
    let v = (r * th.sin() - m2) / s2;
    -0.5 * (u * u + v * v)
  };
  // Factor out the largest exponent so a far tail keeps its relative accuracy.
  let peak = (0..256)
    .map(|k| expo(std::f64::consts::TAU * k as f64 / 256.0))
    .fold(f64::NEG_INFINITY, f64::max);
  let integral =
    adaptive(&|th| (expo(th) - peak).exp(), 0.0, std::f64::consts::TAU, 8);
  r / (std::f64::consts::TAU * s1 * s2) * integral * peak.exp()
}

/// `Φ(hi) - Φ(lo)` for the standard normal, from whichever tails avoid
/// cancellation.
fn normal_mass(lo: f64, hi: f64) -> f64 {
  let c = std::f64::consts::FRAC_1_SQRT_2;
  if hi <= 0.0 {
    0.5 * (libm::erfc(-hi * c) - libm::erfc(-lo * c))
  } else if lo >= 0.0 {
    0.5 * (libm::erfc(lo * c) - libm::erfc(hi * c))
  } else {
    1.0 - f64::midpoint(libm::erfc(hi * c), libm::erfc(-lo * c))
  }
}

fn beckmann_cdf(m1: f64, m2: f64, s1: f64, s2: f64, x: f64) -> f64 {
  if x <= 0.0 {
    return 0.0;
  }
  // P(X^2 + Y^2 <= x^2) = ∫ φ_X(u) P(|Y| <= sqrt(x^2 - u^2)) du.
  let log_integrand = |u: f64| {
    let w = ((x - u) * (x + u)).max(0.0).sqrt();
    let mass = normal_mass((-w - m2) / s2, (w - m2) / s2);
    let z = (u - m1) / s1;
    -0.5 * z * z - (s1 * (std::f64::consts::TAU).sqrt()).ln() + mass.ln()
  };
  de_integral(&log_integrand, -x, x, 1.0).min(1.0)
}

// ---------------------------------------------------------------------------
// Landau. LandauDistribution[μ, σ] has the characteristic function
// exp(i t (μ + 2 σ Log[σ]/π) - |σ t| (1 + 2 i Sign[t] Log|σ t|/π)), so it is
// μ + 2 σ Log[σ]/π + σ Y for the standard variable Y with
// φ(t) = exp(-|t| - 2 i t Log|t|/π).

fn landau_standardize(m: f64, s: f64, x: f64) -> f64 {
  (x - m - 2.0 * s * s.ln() / std::f64::consts::PI) / s
}

/// `z` for which `Y = (2/π)(Z - Log[π/2])`, `Z` being the Landau variable of
/// the form `(1/π) ∫ exp(-t Log t - z t) Sin[π t] dt`.
fn landau_z(y: f64) -> f64 {
  use std::f64::consts::{FRAC_PI_2, PI};
  PI * y / 2.0 + FRAC_PI_2.ln()
}

fn landau_standard_pdf(y: f64) -> f64 {
  use std::f64::consts::{FRAC_2_PI, FRAC_PI_2, PI};
  let z = landau_z(y);
  if z >= -1.0 {
    let f = |t: f64| {
      if t == 0.0 {
        return 0.0;
      }
      (-t * t.ln() - z * t).exp() * (PI * t).sin()
    };
    let upper = landau_upper_limit(z);
    FRAC_PI_2 * adaptive(&f, 0.0, upper, 32) / PI
  } else {
    let f = |t: f64| {
      if t == 0.0 {
        return 1.0;
      }
      (-t).exp() * (t * y + FRAC_2_PI * t * t.ln()).cos()
    };
    adaptive(&f, 0.0, 50.0, 64) / PI
  }
}

fn landau_standard_cdf(y: f64) -> f64 {
  use std::f64::consts::{FRAC_2_PI, PI};
  let z = landau_z(y);
  if z >= -1.0 {
    // The upper tail, (1/π) ∫ exp(-t Log t - z t) Sin[π t]/t dt.
    let f = |t: f64| {
      if t == 0.0 {
        return PI;
      }
      (-t * t.ln() - z * t).exp() * (PI * t).sin() / t
    };
    let upper = landau_upper_limit(z);
    1.0 - adaptive(&f, 0.0, upper, 32) / PI
  } else {
    // Gil-Pelaez: 1/2 + (1/π) ∫ exp(-t) Sin[t y + 2 t Log t/π]/t dt.
    let f = |t: f64| {
      if t == 0.0 {
        return y;
      }
      (-t).exp() * (t * y + FRAC_2_PI * t * t.ln()).sin() / t
    };
    (0.5 + adaptive(&f, 0.0, 50.0, 64) / PI).max(0.0)
  }
}

/// Where `exp(-t Log t - z t)` has fallen below 1e-300.
fn landau_upper_limit(z: f64) -> f64 {
  let mut t: f64 = 1.0;
  while t * t.ln() + z * t < 700.0 && t < 1e6 {
    t *= 1.5;
  }
  t
}

// ---------------------------------------------------------------------------
// Noncentral beta: a Poisson(δ/2) mixture of Beta(α + j, β).

fn noncentral_beta_cdf(a: f64, b: f64, d: f64, x: f64) -> f64 {
  if x <= 0.0 {
    return 0.0;
  }
  if x >= 1.0 {
    return 1.0;
  }
  let lambda = d / 2.0;
  let mut sum = 0.0;
  let mut weight_total = 0.0;
  let mut j = 0u32;
  loop {
    let jf = f64::from(j);
    let ln_w = if lambda > 0.0 {
      -lambda + jf * lambda.ln() - libm::lgamma(jf + 1.0)
    } else if j == 0 {
      0.0
    } else {
      f64::NEG_INFINITY
    };
    let w = ln_w.exp();
    sum += w * super::gamma::beta_regularized_numeric(x, a + jf, b);
    weight_total += w;
    if (jf > lambda && w < 1e-17 * sum.max(1e-300)) || j > 100_000 {
      break;
    }
    if lambda == 0.0 || 1.0 - weight_total < 1e-17 && jf > lambda {
      break;
    }
    j += 1;
  }
  sum.min(1.0)
}

// ---------------------------------------------------------------------------
// Pearson family.

impl Pearson {
  fn new(a1: f64, a0: f64, b2: f64, b1: f64, b0: f64) -> Option<Pearson> {
    let inf = f64::INFINITY;
    let (kernel, lo, hi) = if b2 == 0.0 && b1 == 0.0 {
      if b0 == 0.0 || a1 / b0 <= 0.0 {
        return None;
      }
      (PearsonKernel::Constant, -inf, inf)
    } else if b2 == 0.0 {
      let r = -b0 / b1;
      // Density ∝ |x - r|^(-(a1 r + a0)/b1) exp(-a1 x/b1): it decays on the
      // side a1/b1 points to and must be integrable at r.
      if a1 == 0.0 || (a1 * r + a0) / b1 >= 1.0 {
        return None;
      }
      if a1 / b1 > 0.0 {
        (PearsonKernel::Linear { r }, r, inf)
      } else {
        (PearsonKernel::Linear { r }, -inf, r)
      }
    } else {
      let disc = b1 * b1 - 4.0 * b2 * b0;
      if disc < 0.0 {
        if a1 / b2 <= 1.0 {
          return None;
        }
        (PearsonKernel::NoRealRoots { disc: -disc }, -inf, inf)
      } else if disc == 0.0 {
        let r = -b1 / (2.0 * b2);
        if a1 / b2 <= 1.0 {
          return None;
        }
        let c = (a1 * r + a0) / b2;
        if c < 0.0 {
          (PearsonKernel::DoubleRoot { r }, r, inf)
        } else if c > 0.0 {
          (PearsonKernel::DoubleRoot { r }, -inf, r)
        } else {
          return None;
        }
      } else {
        let sq = disc.sqrt();
        let (mut r1, mut r2) =
          ((-b1 - sq) / (2.0 * b2), (-b1 + sq) / (2.0 * b2));
        if r1 > r2 {
          std::mem::swap(&mut r1, &mut r2);
        }
        let a = (a1 * r1 + a0) / (b2 * (r1 - r2));
        let b = (a1 * r2 + a0) / (b2 * (r2 - r1));
        let kernel = PearsonKernel::TwoRoots { r1, r2, a, b };
        if a < 1.0 && b < 1.0 {
          (kernel, r1, r2)
        } else if b < 1.0 && a + b > 1.0 {
          (kernel, r2, inf)
        } else if a < 1.0 && a + b > 1.0 {
          (kernel, -inf, r1)
        } else {
          return None;
        }
      }
    };
    let mut p = Pearson {
      a1,
      a0,
      b2,
      b1,
      b0,
      kernel,
      lo,
      hi,
      mode: 0.0,
      scale: 1.0,
      g_mode: 0.0,
      norm: 1.0,
    };
    // The density peaks where a1 x + a0 = 0; clamp that into the support.
    let mut mode = -a0 / a1;
    if !(mode > lo && mode < hi) {
      mode = if lo.is_finite() && hi.is_finite() {
        f64::midpoint(lo, hi)
      } else if lo.is_finite() {
        lo + 1.0
      } else {
        hi - 1.0
      };
    }
    let q = b2 * mode * mode + b1 * mode + b0;
    let scale = (q.abs() / a1.abs()).sqrt();
    p.mode = mode;
    p.scale = if scale.is_finite() && scale > 0.0 {
      scale
    } else {
      1.0
    };
    p.g_mode = p.g(mode);
    let norm = p.integral(lo, mode) + p.integral(mode, hi);
    if !(norm.is_finite() && norm > 0.0) {
      return None;
    }
    p.norm = norm;
    Some(p)
  }

  /// `G(x)`, an antiderivative of `(a1 x + a0)/Q(x)`.
  fn g(&self, x: f64) -> f64 {
    let (a1, a0, b2, b1) = (self.a1, self.a0, self.b2, self.b1);
    match self.kernel {
      PearsonKernel::Constant => (a1 * x * x / 2.0 + a0 * x) / self.b0,
      PearsonKernel::Linear { r } => {
        (a1 / b1) * x + ((a1 * r + a0) / b1) * (x - r).abs().ln()
      }
      PearsonKernel::NoRealRoots { disc } => {
        let ln_q = if x.abs() > 1e100 {
          b2.abs().ln()
            + 2.0 * x.abs().ln()
            + (1.0 + b1 / (b2 * x) + self.b0 / (b2 * x * x)).abs().ln()
        } else {
          (b2 * x * x + b1 * x + self.b0).abs().ln()
        };
        let sq = disc.sqrt();
        (a1 / (2.0 * b2)) * ln_q
          + (a0 - a1 * b1 / (2.0 * b2))
            * (2.0 / sq)
            * ((2.0 * b2 * x + b1) / sq).atan()
      }
      PearsonKernel::DoubleRoot { r } => {
        (a1 / b2) * (x - r).abs().ln() - (a1 * r + a0) / (b2 * (x - r))
      }
      PearsonKernel::TwoRoots { r1, r2, a, b } => {
        a * (x - r1).abs().ln() + b * (x - r2).abs().ln()
      }
    }
  }

  fn log_density(&self, x: f64) -> f64 {
    if x <= self.lo || x >= self.hi {
      return f64::NEG_INFINITY;
    }
    self.g_mode - self.g(x)
  }

  /// The unnormalized mass on `[a, b]`.
  fn integral(&self, a: f64, b: f64) -> f64 {
    de_integral(&|x| self.log_density(x), a, b, self.scale)
  }

  fn pdf(&self, x: f64) -> f64 {
    self.log_density(x).exp() / self.norm
  }

  fn cdf(&self, x: f64) -> f64 {
    if x <= self.lo {
      return 0.0;
    }
    if x >= self.hi {
      return 1.0;
    }
    if x <= self.mode {
      (self.integral(self.lo, x) / self.norm).min(1.0)
    } else {
      (1.0 - self.integral(x, self.hi) / self.norm).max(0.0)
    }
  }
}
