//! `DistributionFitTest` and the `HypothesisTestData` object it can produce.
//!
//! Six goodness-of-fit tests are supported for a *fully specified* continuous
//! univariate distribution (no parameters estimated inside the test itself —
//! `dist`'s parameters are already concrete numbers, e.g. because the caller
//! ran `FindDistributionParameters` beforehand). This is the classical
//! "Case 0" of EDF (empirical distribution function) goodness-of-fit theory,
//! and is what lets the asymptotic p-value formulas below be exact/verified
//! algorithms rather than guesses:
//!
//! - **KolmogorovSmirnov**, **Kuiper**: Stephens' (1970) modified-statistic
//!   asymptotic series, reproduced in D'Agostino & Stephens (1986),
//!   "Goodness-of-Fit Techniques", and independently verified against the
//!   `circular` R package's `kuiper.test`/`ks.test` statistic formulas.
//! - **AndersonDarling**: the exact finite-sample algorithm from Marsaglia &
//!   Marsaglia (2004), "Evaluating the Anderson-Darling Distribution",
//!   Journal of Statistical Software 9(2) — ported from the paper's own
//!   published `ADinf.c`/`AnDarl.c`, cross-checked against the CRAN
//!   `ADGofTest` package's `ad.test.pvalue`/`ad.test.statistic`.
//! - **CramerVonMises**: the *n → ∞* term of Csörgő & Faraway's (1996) exact
//!   asymptotic distribution ("Vinf"), as implemented in the CRAN `goftest`
//!   package's `pCvM`. The paper's additional O(1/n) finite-sample
//!   correction term (`psi1`, which needs Bessel-K orders 3/4 and 5/4) is
//!   not applied, so results for very small samples (n < ~15) are a little
//!   less accurate than Mathematica's; for the sample sizes this function is
//!   normally used with, the omitted term is negligible.
//! - **PearsonChiSquare**: classical equal-probability binning
//!   (`k = ceil(2 n^0.4)` cells, the Mann-Wald rule) against the reference
//!   chi-square distribution with `k - 1` degrees of freedom.
//! - **WatsonUSquare**: the statistic is Watson's (1961) exact formula; the
//!   p-value is a monotone log-linear interpolation through the standard
//!   critical-value table (Stephens 1970, as reproduced in the `circular`
//!   package's `watson.test`), since no simple closed-form asymptotic CDF
//!   for it is as readily available as for the other five tests.
//!
//! `ShapiroWilk` and `JarqueBeraALM` (normality-specific tests) are not
//! implemented, so `"AllTests"` reports the six tests above — minus
//! `CramerVonMises` for fewer than 7 data points, where wolframscript deems it
//! invalid (it still computes it on request, with `DistributionFitTest::htdrng`).

use super::*;

const TEST_NAMES: [&str; 6] = [
  "AndersonDarling",
  "CramerVonMises",
  "KolmogorovSmirnov",
  "Kuiper",
  "PearsonChiSquare",
  "WatsonUSquare",
];

/// Every property name wolframscript's `HypothesisTestData` accepts. Asking
/// for one Woxi doesn't compute leaves the call unevaluated; any other name
/// is rejected with `DistributionFitTest::invprp`.
const PROPERTIES: [&str; 30] = [
  "AllTests",
  "AndersonDarling",
  "AutomaticTest",
  "BaringhausHenze",
  "CramerVonMises",
  "DegreesOfFreedom",
  "DistanceToBoundary",
  "FittedDistribution",
  "FittedDistributionParameters",
  "HypothesisTestData",
  "JarqueBeraALM",
  "KolmogorovSmirnov",
  "Kuiper",
  "MardiaCombined",
  "MardiaKurtosis",
  "MardiaSkewness",
  "PearsonChiSquare",
  "Properties",
  "PValue",
  "PValueTable",
  "ShapiroWilk",
  "ShortTestConclusion",
  "SzekelyEnergy",
  "TestConclusion",
  "TestData",
  "TestDataTable",
  "TestEntries",
  "TestStatistic",
  "TestStatisticTable",
  "WatsonUSquare",
];

/// Smallest sample the CramerVonMises test is valid for.
const CVM_MIN_SAMPLE: usize = 7;

/// Emit `DistributionFitTest::invprp` for an unknown property `spec`.
fn emit_invalid_property(spec: &Expr) {
  crate::emit_message(&format!(
    "DistributionFitTest::invprp: The argument {} is not a valid property. Specify \"Properties\" to obtain a list of valid properties.",
    crate::syntax::expr_to_output(spec)
  ));
}

/// Requesting a test that `"AllTests"` leaves out for this sample (only
/// CramerVonMises, on fewer than 7 points) still runs it, with a warning.
fn warn_if_out_of_range(pairs: &[(Expr, Expr)], test_name: &str) {
  let listed = matches!(association_get(pairs, "AllTests"),
    Some(Expr::List(names))
      if names.iter().any(|n| matches!(n, Expr::String(s) if s == test_name)));
  if test_name == "CramerVonMises" && !listed {
    crate::emit_message(&format!(
      "DistributionFitTest::htdrng: The CramerVonMises test is only valid for sample sizes between {CVM_MIN_SAMPLE} and Infinity."
    ));
  }
}

struct TestResult {
  statistic: f64,
  p_value: f64,
}

// ─── generic setup ──────────────────────────────────────────────────────

/// `u_(1) <= … <= u_(n)`, the probability-integral-transformed, sorted data:
/// `u_i = CDF[dist, x_i]`. Under a true null hypothesis these are a sorted
/// sample from `UniformDistribution[{0, 1}]`, which is what every test below
/// is built on. Returns `None` when `data` isn't a list or a value doesn't
/// evaluate numerically.
fn sorted_probability_transform(data: &Expr, dist: &Expr) -> Option<Vec<f64>> {
  let Expr::List(items) = data else {
    return None;
  };
  let mut u = Vec::with_capacity(items.len());
  for item in items {
    let x = try_eval_to_f64(item)?;
    let f = cdf_ast(&[dist.clone(), num_to_expr(x)]).ok()?;
    u.push(try_eval_to_f64(&f)?);
  }
  u.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  Some(u)
}

/// The one-sided Kolmogorov-Smirnov deviations `D+ = max_i(i/n - u_(i))` and
/// `D- = max_i(u_(i) - (i-1)/n)`, shared by the KS and Kuiper tests.
fn edf_deviations(u: &[f64]) -> (f64, f64) {
  let n = u.len() as f64;
  let mut d_plus = f64::NEG_INFINITY;
  let mut d_minus = f64::NEG_INFINITY;
  for (i0, ui) in u.iter().enumerate() {
    let i = (i0 + 1) as f64;
    d_plus = d_plus.max(i / n - ui);
    d_minus = d_minus.max(ui - (i - 1.0) / n);
  }
  (d_plus, d_minus)
}

// ─── Kolmogorov-Smirnov ─────────────────────────────────────────────────

fn ks_test(u: &[f64]) -> TestResult {
  let n = u.len();
  let (d_plus, d_minus) = edf_deviations(u);
  let d = d_plus.max(d_minus);
  let p_value = if let Some(cdf) = kolmogorov_cdf_exact(n, d) {
    (1.0 - cdf).clamp(0.0, 1.0)
  } else {
    let nf = n as f64;
    kolmogorov_tail((nf.sqrt() + 0.12 + 0.11 / nf.sqrt()) * d)
  };
  TestResult {
    statistic: d,
    p_value,
  }
}

/// `P(D_n < d)` for the one-sample Kolmogorov statistic, exactly, by
/// Marsaglia, Tsang & Wang (2003), "Evaluating Kolmogorov's distribution"
/// (Journal of Statistical Software 8(18)), as wolframscript computes it.
/// When the matrix would be too large to raise to the n-th power, the
/// paper's tail approximation, or `None` away from the tail.
fn kolmogorov_cdf_exact(n: usize, d: f64) -> Option<f64> {
  let nf = n as f64;
  let k = (nf * d) as usize + 1;
  let m = 2 * k - 1;
  if m > 300 {
    let s = d * d * nf;
    return (s > 3.76).then(|| {
      1.0 - 2.0 * (-(2.000071 + 0.331 / nf.sqrt() + 1.409 / nf) * s).exp()
    });
  }
  let h = k as f64 - nf * d;
  let mut hm = vec![vec![0.0f64; m]; m];
  for (i, row) in hm.iter_mut().enumerate() {
    for (j, cell) in row.iter_mut().enumerate() {
      if i + 1 >= j {
        *cell = 1.0;
      }
    }
  }
  for i in 0..m {
    hm[i][0] -= h.powi(i as i32 + 1);
    hm[m - 1][i] -= h.powi((m - i) as i32);
  }
  if 2.0 * h - 1.0 > 0.0 {
    hm[m - 1][0] += (2.0 * h - 1.0).powi(m as i32);
  }
  for (i, row) in hm.iter_mut().enumerate() {
    for (j, cell) in row.iter_mut().enumerate() {
      if i + 1 > j {
        for g in 1..=(i + 1 - j) {
          *cell /= g as f64;
        }
      }
    }
  }
  // H^n with a running power-of-ten exponent against overflow.
  fn multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let mut c = vec![vec![0.0; m]; m];
    for i in 0..m {
      for l in 0..m {
        let ail = a[i][l];
        if ail == 0.0 {
          continue;
        }
        for j in 0..m {
          c[i][j] += ail * b[l][j];
        }
      }
    }
    c
  }
  fn power(a: &[Vec<f64>], ea: i32, n: usize) -> (Vec<Vec<f64>>, i32) {
    if n == 1 {
      return (a.to_vec(), ea);
    }
    let (half, eh) = power(a, ea, n / 2);
    let mut v = multiply(&half, &half);
    let mut ev = 2 * eh;
    if n % 2 == 1 {
      v = multiply(a, &v);
      ev += ea;
    }
    let m = v.len();
    if v[m / 2][m / 2] > 1e140 {
      for row in &mut v {
        for x in row.iter_mut() {
          *x *= 1e-140;
        }
      }
      ev += 140;
    }
    (v, ev)
  }
  let (q, mut eq) = power(&hm, 0, n);
  let mut result = q[k - 1][k - 1];
  for i in 1..=n {
    result = result * i as f64 / nf;
    if result < 1e-140 {
      result *= 1e140;
      eq -= 140;
    }
  }
  Some(result * 10f64.powi(eq))
}

/// `Q_KS(λ) = 2 Σ_{k=1}^∞ (-1)^{k-1} exp(-2k²λ²)`, the asymptotic tail
/// probability of the Kolmogorov distribution.
fn kolmogorov_tail(lambda: f64) -> f64 {
  if lambda < 1e-10 {
    return 1.0;
  }
  let mut sum = 0.0;
  for k in 1..=200i64 {
    let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
    let term = sign * (-2.0 * (k as f64).powi(2) * lambda * lambda).exp();
    sum += term;
    if term.abs() < 1e-17 {
      break;
    }
  }
  (2.0 * sum).clamp(0.0, 1.0)
}

// ─── Kuiper ─────────────────────────────────────────────────────────────

fn kuiper_test(u: &[f64]) -> TestResult {
  let n = u.len() as f64;
  let (d_plus, d_minus) = edf_deviations(u);
  let v = d_plus + d_minus;
  let vstar = v * (n.sqrt() + 0.155 + 0.24 / n.sqrt());
  let mut sum = 0.0;
  for k in 1..=200i64 {
    let kf = k as f64;
    let term = (4.0 * kf * kf * vstar * vstar - 1.0)
      * (-2.0 * kf * kf * vstar * vstar).exp();
    sum += term;
    if term.abs() < 1e-17 {
      break;
    }
  }
  TestResult {
    statistic: v,
    p_value: (2.0 * sum).clamp(0.0, 1.0),
  }
}

// ─── Anderson-Darling ───────────────────────────────────────────────────
//
// Ported from Marsaglia & Marsaglia (2004)'s own `ADinf.c`/`AnDarl.c`
// (Journal of Statistical Software 9(2), CC-BY licensed). `adinf` is the
// asymptotic (n = ∞) CDF of the AD statistic; `errfix` is the finite-sample
// correction so that `adinf(z) + errfix(n, adinf(z))` is the exact
// finite-sample CDF to about 4-5 significant digits.

fn ad_adinf(z: f64) -> f64 {
  if z < 2.0 {
    (-1.2337141 / z).exp() / z.sqrt()
      * (2.00012
        + (0.247105
          - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z)
            * z)
          * z)
  } else {
    let poly = 1.0776
      - (2.30695
        - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z)
        * z;
    (-poly.exp()).exp()
  }
}

fn ad_errfix(n: f64, x: f64) -> f64 {
  if x > 0.8 {
    return (-130.2137
      + (745.2337
        - (1705.091 - (1950.646 - (1116.360 - 255.7844 * x) * x) * x) * x)
        * x)
      / n;
  }
  let c = 0.01265 + 0.1757 / n;
  if x < c {
    let t = x / c;
    let t = t.sqrt() * (1.0 - t) * (49.0 * t - 102.0);
    return t * (0.0037 / (n * n) + 0.00078 / n + 0.00006) / n;
  }
  let t = (x - c) / (0.8 - c);
  let t = -0.00022633
    + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t) * t;
  t * (0.04213 + 0.01365 / n) / n
}

/// `Prob(A_n < z)` for finite sample size `n`.
fn ad_cdf(n: f64, z: f64) -> f64 {
  if z < 0.01 {
    return 0.0;
  }
  let x = ad_adinf(z);
  (x + ad_errfix(n, x)).clamp(0.0, 1.0)
}

fn anderson_darling_test(u: &[f64]) -> TestResult {
  let n = u.len();
  let nf = n as f64;
  let mut s = 0.0;
  for (i0, ui) in u.iter().enumerate() {
    let t = ui * (1.0 - u[n - 1 - i0]);
    s -= ((2 * i0 + 1) as f64) * t.ln();
  }
  let statistic = -nf + s / nf;
  TestResult {
    statistic,
    p_value: (1.0 - ad_cdf(nf, statistic)).clamp(0.0, 1.0),
  }
}

// ─── Cramér-von Mises ───────────────────────────────────────────────────
//
// `cvm_vinf` is the n = ∞ term ("Vinf") of Csörgő & Faraway (1996)'s exact
// asymptotic distribution of the CvM statistic, ported from the CRAN
// `goftest` package's `pCvM` (GPL-2, Adrian Baddeley / Julian Faraway).

fn cvm_test(u: &[f64]) -> TestResult {
  let n = u.len();
  let nf = n as f64;
  let mut w2 = 1.0 / (12.0 * nf);
  for (i0, ui) in u.iter().enumerate() {
    let d = ui - (2.0 * (i0 + 1) as f64 - 1.0) / (2.0 * nf);
    w2 += d * d;
  }
  TestResult {
    statistic: w2,
    p_value: (1.0 - cvm_vinf(w2)).clamp(0.0, 1.0),
  }
}

/// `C(2k, k) / 4^k`, computed via the stable product `Π_{j=1}^k (2j-1)/(2j)`
/// (no factorial overflow, decays like `1/sqrt(πk)`).
fn central_binomial_over_4k(k: i64) -> f64 {
  let mut result = 1.0;
  for j in 1..=k {
    result *= (2 * j - 1) as f64 / (2 * j) as f64;
  }
  result
}

fn cvm_vinf(x: f64) -> f64 {
  if x <= 0.0 {
    return 0.0;
  }
  let mut tot = 0.0;
  for k in 0..=200i64 {
    let kf = k as f64;
    let q = (4.0 * kf + 1.0).powi(2) / (16.0 * x);
    let term = central_binomial_over_4k(k)
      * (4.0 * kf + 1.0).sqrt()
      * exp_neg_z_bessel_k(0.25, q)
      / x.sqrt();
    tot += term;
    if k > 5 && term.abs() < 1e-13 {
      break;
    }
  }
  (tot / std::f64::consts::PI).clamp(0.0, 1.0)
}

/// `exp(-z) * K_ν(z)`, computed so the result stays accurate even though
/// `K_ν(z)` decays like `e^{-z}` while the reflection formula used for small
/// arguments (`K_ν(z) = (π/2)(I_{-ν}(z) - I_ν(z))/sin(νπ)`) is built from
/// terms that grow like `e^{z}`. For `z` past the point where that
/// cancellation would cost more than a few digits, the standard asymptotic
/// series for `K_ν` is summed directly against the `exp(-z)` factor instead.
fn exp_neg_z_bessel_k(nu: f64, z: f64) -> f64 {
  if z <= 3.0 {
    (-z).exp() * bessel_k(nu, z)
  } else {
    let mu = 4.0 * nu * nu;
    let mut sum = 1.0;
    let mut term = 1.0;
    for m in 1..40 {
      let mf = m as f64;
      let factor = -(mu - (2.0 * mf - 1.0).powi(2)) / (8.0 * z * mf);
      term *= factor;
      if term.abs() < 1e-17 {
        break;
      }
      sum += term;
    }
    (std::f64::consts::PI / (2.0 * z)).sqrt() * (-2.0 * z).exp() * sum
  }
}

// ─── Pearson chi-square ─────────────────────────────────────────────────

fn pearson_chi_square_test(u: &[f64]) -> TestResult {
  pearson_chi_square_test_estimated(u, 0)
}

/// Pearson's test of `data` against `dist` whose `estimated` parameters
/// were fitted to the data, each of which takes away a degree of freedom:
/// `(statistic, p-value)`.
pub fn pearson_chi_square_fitted(
  data: &Expr,
  dist: &Expr,
  estimated: usize,
) -> Option<(f64, f64)> {
  let u = sorted_probability_transform(data, dist)?;
  let r = pearson_chi_square_test_estimated(&u, estimated);
  Some((r.statistic, r.p_value))
}

fn pearson_chi_square_test_estimated(
  u: &[f64],
  estimated: usize,
) -> TestResult {
  let n = u.len();
  let nf = n as f64;
  let k = (2.0 * nf.powf(0.4)).ceil().max(2.0) as usize;
  let mut counts = vec![0usize; k];
  for &ui in u {
    let mut bin = (ui * k as f64).floor() as isize;
    if bin < 0 {
      bin = 0;
    }
    if bin as usize >= k {
      bin = k as isize - 1;
    }
    counts[bin as usize] += 1;
  }
  let expected = nf / k as f64;
  let chi2: f64 = counts
    .iter()
    .map(|&o| {
      let d = o as f64 - expected;
      d * d / expected
    })
    .sum();
  let df = (k - 1).saturating_sub(estimated).max(1) as i128;
  let p = cdf_ast(&[
    call1("ChiSquareDistribution", Expr::Integer(df)),
    num_to_expr(chi2),
  ])
  .ok()
  .and_then(|e| try_eval_to_f64(&e))
  .map_or(f64::NAN, |f0| (1.0 - f0).clamp(0.0, 1.0));
  TestResult {
    statistic: chi2,
    p_value: p,
  }
}

// ─── Watson U² ──────────────────────────────────────────────────────────

fn watson_u_square_test(u: &[f64]) -> TestResult {
  let n = u.len();
  let nf = n as f64;
  let ubar: f64 = u.iter().sum::<f64>() / nf;
  let mut w2 = 1.0 / (12.0 * nf);
  for (i0, ui) in u.iter().enumerate() {
    let d = ui - (2.0 * (i0 + 1) as f64 - 1.0) / (2.0 * nf);
    w2 += d * d;
  }
  let u2 = w2 - nf * (ubar - 0.5).powi(2);
  let statistic = (u2 - 0.1 / nf + 0.1 / (nf * nf)) * (1.0 + 0.8 / nf);
  TestResult {
    statistic,
    p_value: watson_u_square_p_value(statistic),
  }
}

/// Monotone log-linear interpolation through the standard critical-value
/// table for Watson's U² (Stephens 1970, as reproduced by the `circular`
/// package's `watson.test`): `(statistic, p)` pairs `(0.152, 0.10)`,
/// `(0.187, 0.05)`, `(0.221, 0.025)`, `(0.267, 0.01)`. The four segments'
/// slopes in log-p space are all close to -20, so a single continuous curve
/// through them (extrapolated linearly in log-p beyond either end) is a
/// reasonable stand-in for the exact asymptotic CDF, which — unlike the
/// other five tests here — has no simple closed form in the literature.
fn watson_u_square_p_value(u2: f64) -> f64 {
  const POINTS: [(f64, f64); 4] =
    [(0.152, 0.10), (0.187, 0.05), (0.221, 0.025), (0.267, 0.01)];
  let log_p = |p: f64| p.ln();

  if u2 <= POINTS[0].0 {
    let (x0, p0) = POINTS[0];
    let (x1, p1) = POINTS[1];
    let slope = (log_p(p1) - log_p(p0)) / (x1 - x0);
    return (log_p(p0) + slope * (u2 - x0)).exp().min(1.0);
  }
  if u2 >= POINTS[3].0 {
    let (x0, p0) = POINTS[2];
    let (x1, p1) = POINTS[3];
    let slope = (log_p(p1) - log_p(p0)) / (x1 - x0);
    return (log_p(p1) + slope * (u2 - x1)).exp().max(0.0);
  }
  for w in POINTS.windows(2) {
    let (x0, p0) = w[0];
    let (x1, p1) = w[1];
    if u2 >= x0 && u2 <= x1 {
      let t = (u2 - x0) / (x1 - x0);
      return (log_p(p0) * (1.0 - t) + log_p(p1) * t).exp();
    }
  }
  0.0
}

// ─── dispatch ───────────────────────────────────────────────────────────

fn run_test(name: &str, u: &[f64]) -> Option<TestResult> {
  match name {
    "AndersonDarling" => Some(anderson_darling_test(u)),
    "CramerVonMises" => Some(cvm_test(u)),
    "KolmogorovSmirnov" => Some(ks_test(u)),
    "Kuiper" => Some(kuiper_test(u)),
    "PearsonChiSquare" => Some(pearson_chi_square_test(u)),
    "WatsonUSquare" => Some(watson_u_square_test(u)),
    _ => None,
  }
}

/// A test's name as wolframscript labels it in a table, with the inline
/// box syntax (`\!\(\*SuperscriptBox[…]\)`) its strings carry.
fn test_label(name: &str) -> String {
  const BOX_START: &str = "\u{F7C1}";
  const OPEN: &str = "\u{F7C9}";
  const CLOSE: &str = "\u{F7C0}";
  const STAR: &str = "\u{F7C8}";
  let superscript = |base: &str| {
    format!(
      "{BOX_START}{OPEN}{STAR}SuperscriptBox[{OPEN}{base}{CLOSE},{OPEN}2{CLOSE}]{CLOSE}"
    )
  };
  match name {
    "AndersonDarling" => "Anderson\u{2010}Darling".to_string(),
    "CramerVonMises" => "Cram\u{E9}r\u{2010}von Mises".to_string(),
    "KolmogorovSmirnov" => "Kolmogorov\u{2010}Smirnov".to_string(),
    "PearsonChiSquare" => format!("Pearson {}", superscript("\u{3C7}")),
    "WatsonUSquare" => format!("Watson {}", superscript("U")),
    other => other.to_string(),
  }
}

fn test_data_table(name: &str, stat: f64, p: f64) -> Expr {
  let header = Expr::List(
    vec![
      Expr::String(String::new()),
      str_expr("Statistic"),
      str_expr("P\u{2010}Value"),
    ]
    .into(),
  );
  let row = Expr::List(
    vec![
      Expr::String(test_label(name)),
      Expr::Real(stat),
      Expr::Real(p),
    ]
    .into(),
  );
  let at_second = || {
    Expr::List(
      vec![call(
        "Rule",
        vec![Expr::Integer(2), call1("GrayLevel", Expr::Real(0.7))],
      )]
      .into(),
    )
  };
  let grid = call(
    "Grid",
    vec![
      Expr::List(vec![header, row].into()),
      call(
        "Rule",
        vec![
          id_expr("Alignment"),
          Expr::List(vec![id_expr("Left"), id_expr("Automatic")].into()),
        ],
      ),
      call(
        "Rule",
        vec![
          id_expr("Dividers"),
          Expr::List(vec![at_second(), at_second()].into()),
        ],
      ),
      call("Rule", vec![id_expr("Spacings"), id_expr("Automatic")]),
    ],
  );
  call("Style", vec![grid, str_expr("DialogStyle")])
}

/// Build the `HypothesisTestData[<|…|>]` association: `"FittedDistribution"`,
/// `"AllTests"`, and a `"Tests"` sub-association of
/// `name -> <|"TestStatistic" -> …, "PValue" -> …|>` for each of the six
/// tests in [`TEST_NAMES`].
fn build_hypothesis_test_data(u: &[f64], dist: &Expr) -> Expr {
  let mut tests = Vec::with_capacity(TEST_NAMES.len());
  for &name in &TEST_NAMES {
    if let Some(r) = run_test(name, u) {
      tests.push((
        str_expr(name),
        Expr::Association(vec![
          (str_expr("TestStatistic"), Expr::Real(r.statistic)),
          (str_expr("PValue"), Expr::Real(r.p_value)),
        ]),
      ));
    }
  }
  Expr::Association(vec![
    (str_expr("FittedDistribution"), dist.clone()),
    (
      str_expr("AllTests"),
      Expr::List(
        TEST_NAMES
          .iter()
          .filter(|n| **n != "CramerVonMises" || u.len() >= CVM_MIN_SAMPLE)
          .map(|n| str_expr(n))
          .collect(),
      ),
    ),
    (str_expr("Tests"), Expr::Association(tests)),
  ])
}

fn association_get<'a>(
  pairs: &'a [(Expr, Expr)],
  key: &str,
) -> Option<&'a Expr> {
  pairs
    .iter()
    .find(|(k, _)| matches!(k, Expr::String(s) if s == key))
    .map(|(_, v)| v)
}

/// The result of a single named test from within a `HypothesisTestData`
/// association's `"Tests"` entry, defaulting to `"AndersonDarling"` — the
/// generally most powerful of the six — when no test name is given.
fn selected_test<'a>(
  assoc_pairs: &'a [(Expr, Expr)],
  test_name: Option<&str>,
) -> Option<&'a Expr> {
  let Some(Expr::Association(tests)) = association_get(assoc_pairs, "Tests")
  else {
    return None;
  };
  let name = test_name.unwrap_or("AndersonDarling");
  association_get(tests, name)
}

/// The p-value recorded for a test.
fn association_get_p_value(test_assoc: &Expr) -> Option<f64> {
  property_from_test(test_assoc, "PValue").and_then(|p| try_eval_to_f64(&p))
}

fn property_from_test(test_assoc: &Expr, prop: &str) -> Option<Expr> {
  let Expr::Association(pairs) = test_assoc else {
    return None;
  };
  association_get(pairs, prop).cloned()
}

/// `HypothesisTestData[<|…|>][prop]` / `[prop, testName]` property access.
pub fn apply_hypothesis_test_data(
  func_args: &[Expr],
  index_args: &[Expr],
) -> Option<Expr> {
  let Some(Expr::Association(pairs)) = func_args.first() else {
    return None;
  };
  match index_args {
    [Expr::String(prop)] => match prop.as_str() {
      "FittedDistribution" | "AllTests" | "Properties" => {
        association_get(pairs, prop).cloned()
      }
      "TestStatistic" | "PValue" => {
        let test = selected_test(pairs, None)?;
        property_from_test(test, prop)
      }
      "TestDataTable" => {
        let name = "AndersonDarling";
        let test = selected_test(pairs, Some(name))?;
        let Expr::Association(tp) = test else {
          return None;
        };
        let stat = try_eval_to_f64(association_get(tp, "TestStatistic")?)?;
        let p = try_eval_to_f64(association_get(tp, "PValue")?)?;
        Some(test_data_table(name, stat, p))
      }
      _ => {
        if TEST_NAMES.contains(&prop.as_str()) {
          warn_if_out_of_range(pairs, prop);
          let test = selected_test(pairs, Some(prop))?;
          property_from_test(test, "PValue")
        } else {
          if !PROPERTIES.contains(&prop.as_str()) {
            emit_invalid_property(&index_args[0]);
          }
          None
        }
      }
    },
    [Expr::String(prop), Expr::String(test_name)]
      if TEST_NAMES.contains(&test_name.as_str()) =>
    {
      if !PROPERTIES.contains(&prop.as_str()) {
        emit_invalid_property(&Expr::List(index_args.to_vec().into()));
        return None;
      }
      warn_if_out_of_range(pairs, test_name);
      let test = selected_test(pairs, Some(test_name))?;
      match prop.as_str() {
        "TestDataTable" => {
          let Expr::Association(tp) = test else {
            return None;
          };
          let stat = try_eval_to_f64(association_get(tp, "TestStatistic")?)?;
          let p = try_eval_to_f64(association_get(tp, "PValue")?)?;
          Some(test_data_table(test_name, stat, p))
        }
        "TestStatistic" | "PValue" => property_from_test(test, prop),
        _ => None,
      }
    }
    _ => None,
  }
}

/// `DistributionFitTest[data, dist]`, `DistributionFitTest[data, dist,
/// property]` — `property` may be `"HypothesisTestData"`, `"PValue"`,
/// `"TestStatistic"`, one of [`TEST_NAMES`] (interpreted as that test's
/// p-value), or omitted (defaults to `"PValue"`).
pub fn distribution_fit_test_ast(
  all_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (args, opts) =
    crate::evaluator::dispatch::split_trailing_options(all_args);
  // `SignificanceLevel` is the cutoff of "ShortTestConclusion".
  let settings = match opts.as_slice() {
    [] => super::statistics::HypothesisTestSettings::parse(&[]),
    [(name, _)] if name == "SignificanceLevel" => {
      super::statistics::HypothesisTestSettings::parse(&opts)
    }
    _ => None,
  };
  let Some(settings) = settings else {
    return Ok(unevaluated("DistributionFitTest", all_args));
  };
  if args.len() < 2 || args.len() > 3 {
    return Ok(unevaluated("DistributionFitTest", all_args));
  }
  let data = &args[0];
  let dist = &args[1];
  // `{property, test}` asks for one test's property.
  let (property, chosen_test): (&str, Option<&str>) = match args.get(2) {
    Some(Expr::String(s)) => (s.as_str(), None),
    Some(Expr::List(spec)) => match spec.as_slice() {
      [Expr::String(prop), Expr::String(test)]
        if TEST_NAMES.contains(&test.as_str()) =>
      {
        (prop.as_str(), Some(test.as_str()))
      }
      _ => return Ok(unevaluated("DistributionFitTest", all_args)),
    },
    None => ("PValue", None),
    _ => return Ok(unevaluated("DistributionFitTest", all_args)),
  };

  let Some(u) = sorted_probability_transform(data, dist) else {
    return Ok(unevaluated("DistributionFitTest", all_args));
  };
  if u.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "DistributionFitTest: data must have at least 2 elements".into(),
    ));
  }

  let assoc = build_hypothesis_test_data(&u, dist);
  let Expr::Association(pairs) = &assoc else {
    unreachable!()
  };

  if let Some(name) = chosen_test {
    let test = selected_test(pairs, Some(name)).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "DistributionFitTest: unknown test".into(),
      )
    })?;
    return match property {
      "PValue" | "TestStatistic" => {
        if property == "PValue" {
          warn_if_out_of_range(pairs, name);
        }
        Ok(property_from_test(test, property).unwrap_or(assoc))
      }
      "ShortTestConclusion" => Ok(
        association_get_p_value(test)
          .map(|p| settings.short_conclusion(p))
          .unwrap_or(assoc),
      ),
      "TestDataTable" => {
        let Expr::Association(tp) = test else {
          return Ok(unevaluated("DistributionFitTest", all_args));
        };
        let (Some(stat), Some(p)) = (
          association_get(tp, "TestStatistic").and_then(try_eval_to_f64),
          association_get(tp, "PValue").and_then(try_eval_to_f64),
        ) else {
          return Ok(unevaluated("DistributionFitTest", all_args));
        };
        Ok(test_data_table(name, stat, p))
      }
      "HypothesisTestData" => Ok(call1("HypothesisTestData", assoc)),
      _ => Ok(unevaluated("DistributionFitTest", all_args)),
    };
  }

  match property {
    "HypothesisTestData" => Ok(call1("HypothesisTestData", assoc)),
    "FittedDistribution" | "AllTests" => {
      Ok(association_get(pairs, property).cloned().unwrap_or(assoc))
    }
    "PValue" | "TestStatistic" => {
      let test = selected_test(pairs, None).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "DistributionFitTest: no test available".into(),
        )
      })?;
      Ok(property_from_test(test, property).unwrap_or(assoc))
    }
    "ShortTestConclusion" => {
      let test = selected_test(pairs, None).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "DistributionFitTest: no test available".into(),
        )
      })?;
      Ok(
        association_get_p_value(test)
          .map(|p| settings.short_conclusion(p))
          .unwrap_or(assoc),
      )
    }
    name if TEST_NAMES.contains(&name) => {
      warn_if_out_of_range(pairs, name);
      let test = selected_test(pairs, Some(name)).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "DistributionFitTest: unknown test".into(),
        )
      })?;
      Ok(property_from_test(test, "PValue").unwrap_or(assoc))
    }
    other => {
      if !PROPERTIES.contains(&other) {
        emit_invalid_property(&args[2]);
      }
      Ok(unevaluated("DistributionFitTest", all_args))
    }
  }
}

/// `KolmogorovSmirnovTest[data, dist, property]` and the other named
/// goodness-of-fit tests: `DistributionFitTest` with that one test chosen.
pub fn named_fit_test_ast(
  head: &str,
  all_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let test = head.strip_suffix("Test").unwrap_or(head);
  let (args, _) = crate::evaluator::dispatch::split_trailing_options(all_args);
  let options = &all_args[args.len()..];
  if !(2..=3).contains(&args.len()) || !TEST_NAMES.contains(&test) {
    return Ok(unevaluated(head, all_args));
  }
  let property = match args.get(2) {
    None => str_expr("PValue"),
    Some(Expr::String(p)) => str_expr(p),
    Some(_) => return Ok(unevaluated(head, args)),
  };
  let spec = Expr::List(vec![property, str_expr(test)].into());
  let mut call_args = vec![args[0].clone(), args[1].clone(), spec];
  call_args.extend(options.iter().cloned());
  let result = distribution_fit_test_ast(&call_args)?;
  Ok(match &result {
    Expr::FunctionCall { name, .. } if name == "DistributionFitTest" => {
      unevaluated(head, all_args)
    }
    _ => result,
  })
}
