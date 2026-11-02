//! `VarianceEquivalenceTest[{data1, data2, …}]`: whether samples share a
//! variance, by the Bartlett, Brown–Forsythe, Conover, Fisher ratio (two
//! samples only) and Levene tests.
//!
//! The automatic test is the Fisher ratio test for two samples (Levene's
//! for more) when the pooled within-sample residuals look normal, and the
//! rank-based Conover test otherwise. Normality is judged by an
//! Anderson–Darling test with estimated mean and deviation (Stephens'
//! p-value approximation) at the 5% level.

use crate::InterpreterError;
use crate::functions::math_ast::try_eval_to_f64;
use crate::helpers::{call, unevaluated};
use crate::syntax::Expr;

/// The tests in the order of the table, with their display names.
const TESTS: [(&str, &str); 5] = [
  ("Bartlett", "Bartlett"),
  ("BrownForsythe", "Brown\u{2010}Forsythe"),
  ("Conover", "Conover"),
  ("FisherRatio", "Fisher Ratio"),
  ("Levene", "Levene"),
];

/// Marker key of the `HypothesisTestData` association this test builds.
const MARKER: &str = "VarianceEquivalenceTest";

fn chi_square_sf(x: f64, k: f64) -> f64 {
  crate::functions::math_ast::gamma::gamma_regularized_numeric(k / 2.0, x / 2.0)
}

fn f_sf(f: f64, d1: f64, d2: f64) -> f64 {
  if f <= 0.0 {
    return 1.0;
  }
  crate::functions::math_ast::gamma::beta_regularized_numeric(
    d2 / (d2 + d1 * f),
    d2 / 2.0,
    d1 / 2.0,
  )
}

fn normal_sf(z: f64) -> f64 {
  0.5 * libm::erfc(z / std::f64::consts::SQRT_2)
}

fn mean(v: &[f64]) -> f64 {
  v.iter().sum::<f64>() / v.len() as f64
}

fn variance(v: &[f64]) -> f64 {
  let m = mean(v);
  v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)
}

fn median(v: &[f64]) -> f64 {
  let mut s = v.to_vec();
  s.sort_by(f64::total_cmp);
  let n = s.len();
  if n % 2 == 1 {
    s[n / 2]
  } else {
    f64::midpoint(s[n / 2 - 1], s[n / 2])
  }
}

/// Ranks with ties averaged.
fn ranks(v: &[f64]) -> Vec<f64> {
  let mut idx: Vec<usize> = (0..v.len()).collect();
  idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
  let mut r = vec![0.0; v.len()];
  let mut i = 0;
  while i < idx.len() {
    let mut j = i;
    while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
      j += 1;
    }
    let avg = (i + j) as f64 / 2.0 + 1.0;
    for &k in &idx[i..=j] {
      r[k] = avg;
    }
    i = j + 1;
  }
  r
}

/// One-way ANOVA F of `groups` (Levene and Brown–Forsythe use it on the
/// absolute deviations): statistic and p-value.
fn anova(groups: &[Vec<f64>]) -> (f64, f64) {
  let k = groups.len() as f64;
  let n: f64 = groups.iter().map(|g| g.len() as f64).sum();
  let all: Vec<f64> = groups.iter().flatten().copied().collect();
  let grand = mean(&all);
  let between: f64 = groups
    .iter()
    .map(|g| g.len() as f64 * (mean(g) - grand).powi(2))
    .sum();
  let within: f64 = groups
    .iter()
    .map(|g| {
      let m = mean(g);
      g.iter().map(|x| (x - m).powi(2)).sum::<f64>()
    })
    .sum();
  let w = (n - k) / (k - 1.0) * between / within;
  (w, f_sf(w, k - 1.0, n - k))
}

fn deviations(data: &[Vec<f64>], center: fn(&[f64]) -> f64) -> Vec<Vec<f64>> {
  data
    .iter()
    .map(|g| {
      let c = center(g);
      g.iter().map(|x| (x - c).abs()).collect()
    })
    .collect()
}

/// `(statistic, p-value)` of each applicable test, in table order.
fn run_tests(data: &[Vec<f64>]) -> Vec<(&'static str, f64, f64)> {
  let k = data.len() as f64;
  let n: f64 = data.iter().map(|g| g.len() as f64).sum();
  let mut out = Vec::new();
  // Bartlett.
  let vars: Vec<f64> = data.iter().map(|g| variance(g)).collect();
  let pooled = data
    .iter()
    .zip(&vars)
    .map(|(g, v)| (g.len() as f64 - 1.0) * v)
    .sum::<f64>()
    / (n - k);
  let numerator = (n - k) * pooled.ln()
    - data
      .iter()
      .zip(&vars)
      .map(|(g, v)| (g.len() as f64 - 1.0) * v.ln())
      .sum::<f64>();
  let correction = 1.0
    + (data
      .iter()
      .map(|g| 1.0 / (g.len() as f64 - 1.0))
      .sum::<f64>()
      - 1.0 / (n - k))
      / (3.0 * (k - 1.0));
  let bartlett = numerator / correction;
  out.push(("Bartlett", bartlett, chi_square_sf(bartlett, k - 1.0)));
  // Brown–Forsythe: Levene's test about the medians.
  let (bf, bf_p) = anova(&deviations(data, median));
  out.push(("BrownForsythe", bf, bf_p));
  // Conover's squared ranks of the deviations from the medians.
  let dev: Vec<Vec<f64>> = deviations(data, median);
  let flat: Vec<f64> = dev.iter().flatten().copied().collect();
  let squares: Vec<f64> = ranks(&flat).iter().map(|r| r * r).collect();
  let s_mean = mean(&squares);
  let s_sq: f64 = squares.iter().map(|s| s * s).sum();
  let mut start = 0;
  let sums: Vec<f64> = dev
    .iter()
    .map(|g| {
      let s: f64 = squares[start..start + g.len()].iter().sum();
      start += g.len();
      s
    })
    .collect();
  if data.len() == 2 {
    let (n1, n2) = (data[0].len() as f64, data[1].len() as f64);
    let var = n1 * n2 / (n * (n - 1.0)) * (s_sq - n * s_mean * s_mean);
    let z = (sums[0] - n1 * s_mean) / var.sqrt();
    out.push(("Conover", z, 2.0 * normal_sf(z.abs())));
  } else {
    let d2 = (s_sq - n * s_mean * s_mean) / (n - 1.0);
    let t = (dev
      .iter()
      .zip(&sums)
      .map(|(g, s)| s * s / g.len() as f64)
      .sum::<f64>()
      - n * s_mean * s_mean)
      / d2;
    out.push(("Conover", t, chi_square_sf(t, k - 1.0)));
  }
  // Fisher ratio, for two samples.
  if data.len() == 2 {
    let f = vars[0] / vars[1];
    let (d1, d2) = (data[0].len() as f64 - 1.0, data[1].len() as f64 - 1.0);
    let sf = f_sf(f, d1, d2);
    out.push(("FisherRatio", f, (2.0 * sf.min(1.0 - sf)).min(1.0)));
  }
  // Levene: about the means.
  let (lev, lev_p) = anova(&deviations(data, mean));
  out.push(("Levene", lev, lev_p));
  out
}

/// Whether the pooled residuals pass an Anderson–Darling normality test
/// with estimated parameters at the 5% level.
fn residuals_look_normal(data: &[Vec<f64>]) -> bool {
  let mut res: Vec<f64> = data
    .iter()
    .flat_map(|g| {
      let m = mean(g);
      g.iter().map(move |x| x - m)
    })
    .collect();
  let n = res.len() as f64;
  let sd = (res.iter().map(|x| x * x).sum::<f64>() / (n - 1.0)).sqrt();
  if sd == 0.0 {
    return true;
  }
  res.sort_by(f64::total_cmp);
  let cdf = |x: f64| (1.0 - normal_sf(x / sd)).clamp(1e-300, 1.0 - 1e-16);
  let a2 = -n
    - (0..res.len())
      .map(|i| {
        let fi = cdf(res[i]);
        let fj = cdf(res[res.len() - 1 - i]);
        (2.0 * i as f64 + 1.0) * (fi.ln() + (1.0 - fj).ln())
      })
      .sum::<f64>()
      / n;
  let a = a2 * (1.0 + 0.75 / n + 2.25 / (n * n));
  let p = if a >= 0.6 {
    (1.2937 - 5.709 * a + 0.0186 * a * a).exp()
  } else if a >= 0.34 {
    (0.9177 - 4.279 * a - 1.38 * a * a).exp()
  } else if a >= 0.2 {
    1.0 - (-8.318 + 42.796 * a - 59.938 * a * a).exp()
  } else {
    1.0 - (-13.436 + 101.14 * a - 223.73 * a * a).exp()
  };
  p >= 0.05
}

fn automatic_test(data: &[Vec<f64>]) -> &'static str {
  if !residuals_look_normal(data) {
    "Conover"
  } else if data.len() == 2 {
    "FisherRatio"
  } else {
    "Levene"
  }
}

fn grid(rows: Vec<Expr>) -> Expr {
  let header = Expr::List(
    vec![
      Expr::String(String::new()),
      Expr::String("Statistic".into()),
      Expr::String("P\u{2010}Value".into()),
    ]
    .into(),
  );
  let mut all = vec![header];
  all.extend(rows);
  let rule = |k: &str, v: Expr| Expr::Rule {
    pattern: Box::new(Expr::Identifier(k.into())),
    replacement: Box::new(v),
  };
  let at_second = || {
    Expr::List(
      vec![Expr::Rule {
        pattern: Box::new(Expr::Integer(2)),
        replacement: Box::new(call("GrayLevel", vec![Expr::Real(0.7)])),
      }]
      .into(),
    )
  };
  call(
    "Grid",
    vec![
      Expr::List(all.into()),
      rule(
        "Alignment",
        Expr::List(
          vec![
            Expr::Identifier("Left".into()),
            Expr::Identifier("Automatic".into()),
          ]
          .into(),
        ),
      ),
      rule(
        "Dividers",
        Expr::List(vec![at_second(), at_second()].into()),
      ),
      rule("Spacings", Expr::Identifier("Automatic".into())),
    ],
  )
}

fn assoc_get<'a>(pairs: &'a [(Expr, Expr)], key: &str) -> Option<&'a Expr> {
  pairs.iter().find_map(|(k, v)| match k {
    Expr::String(s) if s == key => Some(v),
    _ => None,
  })
}

/// The `{statistic, p}` of `test` in the object's results.
fn result_of(pairs: &[(Expr, Expr)], test: &str) -> Option<(f64, f64)> {
  let Expr::Association(results) = assoc_get(pairs, "Results")? else {
    return None;
  };
  let Expr::List(sp) = assoc_get(results, test)? else {
    return None;
  };
  Some((try_eval_to_f64(&sp[0])?, try_eval_to_f64(&sp[1])?))
}

fn row(pairs: &[(Expr, Expr)], test: &str) -> Option<Expr> {
  let (s, p) = result_of(pairs, test)?;
  let label = TESTS.iter().find(|(n, _)| *n == test)?.1;
  Some(Expr::List(
    vec![Expr::String(label.into()), Expr::Real(s), Expr::Real(p)].into(),
  ))
}

/// Properties of the `HypothesisTestData` object, or `None` when it is not
/// one of these.
pub fn property(pairs: &[(Expr, Expr)], index: &[Expr]) -> Option<Expr> {
  assoc_get(pairs, MARKER)?;
  let automatic = match assoc_get(pairs, "AutomaticTest")? {
    Expr::String(s) => s.clone(),
    _ => return None,
  };
  let alpha = try_eval_to_f64(assoc_get(pairs, "SignificanceLevel")?)?;
  let test_name = |e: &Expr| match e {
    Expr::String(s) if TESTS.iter().any(|(n, _)| n == s) => Some(s.clone()),
    Expr::Identifier(a) if a == "Automatic" => Some(automatic.clone()),
    _ => None,
  };
  match index {
    [Expr::String(p)] => match p.as_str() {
      "AutomaticTest" => Some(Expr::String(automatic.clone())),
      "PValue" => result_of(pairs, &automatic).map(|r| Expr::Real(r.1)),
      "TestStatistic" => result_of(pairs, &automatic).map(|r| Expr::Real(r.0)),
      "ShortTestConclusion" => {
        let p = result_of(pairs, &automatic)?.1;
        Some(Expr::String(
          if p < alpha { "Reject" } else { "Do not reject" }.into(),
        ))
      }
      "TestDataTable" => Some(grid(vec![row(pairs, &automatic)?])),
      name if TESTS.iter().any(|(n, _)| *n == name) => {
        result_of(pairs, name).map(|r| Expr::Real(r.1))
      }
      _ => None,
    },
    [Expr::String(p), which] => {
      let names: Vec<String> = match which {
        Expr::Identifier(a) if a == "All" => TESTS
          .iter()
          .map(|(n, _)| n.to_string())
          .filter(|n| result_of(pairs, n).is_some())
          .collect(),
        other => vec![test_name(other)?],
      };
      let pick = |f: fn((f64, f64)) -> f64| -> Option<Expr> {
        let values: Vec<Expr> = names
          .iter()
          .map(|n| result_of(pairs, n).map(|r| Expr::Real(f(r))))
          .collect::<Option<_>>()?;
        Some(if matches!(which, Expr::Identifier(a) if a == "All") {
          Expr::List(values.into())
        } else {
          values.into_iter().next()?
        })
      };
      match p.as_str() {
        "TestDataTable" => Some(grid(
          names
            .iter()
            .map(|n| row(pairs, n))
            .collect::<Option<Vec<_>>>()?,
        )),
        "PValue" => pick(|r| r.1),
        "TestStatistic" => pick(|r| r.0),
        _ => None,
      }
    }
    _ => None,
  }
}

/// `VarianceEquivalenceTest[{data…}]`, `[{data…}, test]` (a p-value, or
/// a list of them for `All`), `[{data…}, "HypothesisTestData" | property]`.
pub fn variance_equivalence_test_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original = || Ok(unevaluated("VarianceEquivalenceTest", args));
  let Some(Expr::List(samples)) = args.first() else {
    return original();
  };
  let data: Option<Vec<Vec<f64>>> = samples
    .iter()
    .map(|s| match s {
      Expr::List(v) if v.len() >= 2 => v.iter().map(try_eval_to_f64).collect(),
      _ => None,
    })
    .collect();
  let Some(data) = data.filter(|d| d.len() >= 2) else {
    return original();
  };
  let mut alpha = 0.05;
  let mut positional: Vec<&Expr> = Vec::new();
  for a in &args[1..] {
    match a {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "SignificanceLevel") =>
      {
        alpha = try_eval_to_f64(replacement).unwrap_or(0.05);
      }
      other => positional.push(other),
    }
  }
  let results = run_tests(&data);
  let automatic = automatic_test(&data);
  let object_pairs: Vec<(Expr, Expr)> = vec![
    (Expr::String(MARKER.into()), Expr::Identifier("True".into())),
    (
      Expr::String("Results".into()),
      Expr::Association(
        results
          .iter()
          .map(|(n, s, p)| {
            (
              Expr::String((*n).into()),
              Expr::List(vec![Expr::Real(*s), Expr::Real(*p)].into()),
            )
          })
          .collect(),
      ),
    ),
    (
      Expr::String("AutomaticTest".into()),
      Expr::String(automatic.into()),
    ),
    (Expr::String("SignificanceLevel".into()), Expr::Real(alpha)),
  ];
  let object = call(
    "HypothesisTestData",
    vec![Expr::Association(object_pairs.clone())],
  );
  match positional.as_slice() {
    [] => {
      Ok(property(&object_pairs, &[Expr::String("PValue".into())]).unwrap())
    }
    [Expr::String(h)] if h == "HypothesisTestData" => Ok(object),
    [Expr::String(prop)] if !TESTS.iter().any(|(n, _)| n == prop) => Ok(
      property(&object_pairs, &[Expr::String(prop.clone())])
        .unwrap_or_else(|| unevaluated("VarianceEquivalenceTest", args)),
    ),
    [which] => Ok(
      property(
        &object_pairs,
        &[Expr::String("PValue".into()), (*which).clone()],
      )
      .unwrap_or_else(|| unevaluated("VarianceEquivalenceTest", args)),
    ),
    _ => original(),
  }
}
