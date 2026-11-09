//! `MannWhitneyTest[{x, y}, μ0, property, opts]`: whether the median
//! difference of two samples is μ0, by the Mann–Whitney U test.
//!
//! The rules below were read off Wolfram's results (they are not the
//! textbook ones):
//!
//! - The statistic is U1, the number of pairs with `x - μ0 > y` (ties
//!   counting one half); the two-sided test reports `Min[U1, U2]`.
//! - With at most 40 values in all, the p-value is exact, from the
//!   distribution of U *without* ties, and counts strictly beyond the
//!   observed value: a two-sided p of `2 P[U <= Umin - 1]`, a "Less" p of
//!   `P[U <= U1 - 1]` and a "Greater" p of `P[U >= U1 + 1]`.
//! - Past 40 values it is the normal approximation with the tie-corrected
//!   variance, shifted by a half towards the mean when the first sample is
//!   the smaller one and away from it otherwise: `2 Φ((Umin - μ + c)/σ)`,
//!   capped at an exact 1, with `c = -1/2` for `n1 >= n2` and `+1/2`
//!   otherwise; "Less" is `Φ((U1 - μ + c)/σ)` and "Greater" its complement.

use crate::InterpreterError;
use crate::functions::math_ast::{
  Alternative, HypothesisTestSettings, erfc_f64, try_eval_to_f64,
};
use crate::helpers::{call, str_expr, unevaluated};
use crate::syntax::Expr;

const NAME: &str = "MannWhitneyTest";

/// The properties the test answers, in Wolfram's order.
const PROPERTIES: [&str; 13] = [
  "DegreesOfFreedom",
  "HypothesisTestData",
  "MannWhitney",
  "Properties",
  "PValue",
  "PValueTable",
  "ShortTestConclusion",
  "TestConclusion",
  "TestData",
  "TestDataTable",
  "TestEntries",
  "TestStatistic",
  "TestStatisticTable",
];

/// Up to this many values in both samples the p-value is exact.
const EXACT_LIMIT: usize = 40;

pub fn mann_whitney_test_ast(
  full_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (args, opts) =
    crate::evaluator::dispatch::split_trailing_options(full_args);
  if args.is_empty() || args.len() > 3 {
    return Ok(unevaluated(NAME, full_args));
  }
  let Some((x, y)) = two_samples(&args[0]) else {
    crate::emit_message(&format!(
      "{NAME}::rctndmln: The argument {} at position 1 should be a list \
       containing two rectangular arrays of real numbers where each array \
       has length greater than its dimension.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(unevaluated(NAME, full_args));
  };
  let mu0_expr = match args.get(1) {
    None => Expr::Integer(0),
    Some(Expr::Identifier(a)) if a == "Automatic" => Expr::Integer(0),
    Some(e) => e.clone(),
  };
  let Some(mu0) = try_eval_to_f64(&mu0_expr) else {
    return Ok(unevaluated(NAME, full_args));
  };
  let Some(settings) = HypothesisTestSettings::parse(&opts) else {
    return Ok(unevaluated(NAME, full_args));
  };
  let properties: Vec<String> = match args.get(2) {
    None => vec!["PValue".to_string()],
    Some(Expr::String(p)) => vec![p.clone()],
    Some(Expr::List(ps)) => {
      let mut out = Vec::with_capacity(ps.len());
      for p in ps {
        let Expr::String(p) = p else {
          return Ok(unevaluated(NAME, full_args));
        };
        out.push(p.clone());
      }
      out
    }
    Some(_) => return Ok(unevaluated(NAME, full_args)),
  };
  if let Some(bad) = properties
    .iter()
    .find(|p| !PROPERTIES.contains(&p.as_str()))
  {
    crate::emit_message(&format!(
      "{NAME}::invprp: The argument {bad} is not a valid property. Specify \
       \"Properties\" to obtain a list of valid properties."
    ));
    return Ok(unevaluated(NAME, full_args));
  }

  let shifted: Vec<f64> = x.iter().map(|v| v - mu0).collect();
  let test = MannWhitney::new(&shifted, &y, settings.alternative);
  let level = significance_level_text(&opts);
  let mut results = Vec::with_capacity(properties.len());
  for property in &properties {
    let Some(value) =
      test.property(property, &settings, &mu0_expr, level.as_str())
    else {
      return Ok(unevaluated(NAME, full_args));
    };
    results.push(value);
  }
  Ok(match args.get(2) {
    Some(Expr::List(_)) => Expr::List(results.into()),
    _ => results.remove(0),
  })
}

/// The two samples of `{x, y}`, each a list of at least two real numbers.
fn two_samples(data: &Expr) -> Option<(Vec<f64>, Vec<f64>)> {
  let Expr::List(samples) = data else {
    return None;
  };
  let [Expr::List(x), Expr::List(y)] = samples.as_slice() else {
    return None;
  };
  let reals = |s: &[Expr]| -> Option<Vec<f64>> {
    if s.len() < 2 {
      return None;
    }
    s.iter().map(try_eval_to_f64).collect()
  };
  Some((reals(x)?, reals(y)?))
}

/// The level in the conclusion text, in percent: `100 α` as Wolfram prints
/// it (`5` for the default 1/20, `1.` for `SignificanceLevel -> 0.01`).
fn significance_level_text(opts: &[(String, Expr)]) -> String {
  let given = opts
    .iter()
    .rev()
    .find(|(name, value)| {
      name == "SignificanceLevel"
        && !matches!(value, Expr::Identifier(a) if a == "Automatic")
    })
    .map(|(_, value)| value.clone());
  let alpha = given.unwrap_or_else(|| {
    call("Rational", vec![Expr::Integer(1), Expr::Integer(20)])
  });
  let percent = crate::evaluator::evaluate_expr_to_expr(&call(
    "Times",
    vec![Expr::Integer(100), alpha],
  ))
  .unwrap_or(Expr::Integer(5));
  crate::syntax::expr_to_string(&percent)
}

struct MannWhitney {
  n1: usize,
  n2: usize,
  /// Pairs with `x > y`, ties counting one half.
  u1: f64,
  /// The variance of U under the null hypothesis, corrected for ties.
  variance: f64,
  alternative: Alternative,
}

impl MannWhitney {
  fn new(x: &[f64], y: &[f64], alternative: Alternative) -> Self {
    let (n1, n2) = (x.len(), y.len());
    let mut all: Vec<(f64, bool)> = x
      .iter()
      .map(|&v| (v, true))
      .chain(y.iter().map(|&v| (v, false)))
      .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Midranks and the tie sizes.
    let mut rank_sum_x = 0.0;
    let mut tie_term = 0.0;
    let mut i = 0;
    while i < all.len() {
      let mut j = i;
      while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
        j += 1;
      }
      let midrank = (i + j) as f64 / 2.0 + 1.0;
      let in_x = all[i..=j].iter().filter(|(_, from_x)| *from_x).count();
      rank_sum_x += midrank * in_x as f64;
      let t = (j - i + 1) as f64;
      tie_term += t * t * t - t;
      i = j + 1;
    }
    let (f1, f2) = (n1 as f64, n2 as f64);
    let n = f1 + f2;
    let u1 = rank_sum_x - f1 * (f1 + 1.0) / 2.0;
    let variance = f1 * f2 / 12.0 * ((n + 1.0) - tie_term / (n * (n - 1.0)));
    MannWhitney {
      n1,
      n2,
      u1,
      variance,
      alternative,
    }
  }

  fn mean(&self) -> f64 {
    self.n1 as f64 * self.n2 as f64 / 2.0
  }

  fn statistic(&self) -> f64 {
    match self.alternative {
      Alternative::Unequal => {
        self.u1.min(self.n1 as f64 * self.n2 as f64 - self.u1)
      }
      _ => self.u1,
    }
  }

  fn p_value(&self) -> Expr {
    if self.n1 + self.n2 <= EXACT_LIMIT {
      return Expr::Real(self.exact_p_value());
    }
    let shift = if self.n1 >= self.n2 { -0.5 } else { 0.5 };
    let sd = self.variance.sqrt();
    let phi = |u: f64| normal_cdf((u - self.mean() + shift) / sd);
    match self.alternative {
      Alternative::Less => Expr::Real(phi(self.u1)),
      Alternative::Greater => Expr::Real(1.0 - phi(self.u1)),
      Alternative::Unequal => {
        let p = 2.0 * phi(self.statistic());
        if p >= 1.0 {
          Expr::Integer(1)
        } else {
          Expr::Real(p)
        }
      }
    }
  }

  /// The p-value from the distribution of U for samples without ties.
  fn exact_p_value(&self) -> f64 {
    let counts = u_distribution(self.n1, self.n2);
    let total: u128 = counts.iter().sum();
    let at_most = |bound: f64| -> u128 {
      counts
        .iter()
        .enumerate()
        .filter(|(u, _)| (*u as f64) <= bound)
        .map(|(_, c)| c)
        .sum()
    };
    let at_least = |bound: f64| -> u128 {
      counts
        .iter()
        .enumerate()
        .filter(|(u, _)| (*u as f64) >= bound)
        .map(|(_, c)| c)
        .sum()
    };
    let favorable = match self.alternative {
      Alternative::Less => at_most(self.u1 - 1.0),
      Alternative::Greater => at_least(self.u1 + 1.0),
      Alternative::Unequal => 2 * at_most(self.statistic() - 1.0),
    };
    (favorable as f64 / total as f64).min(1.0)
  }

  fn property(
    &self,
    property: &str,
    settings: &HypothesisTestSettings,
    mu0: &Expr,
    level: &str,
  ) -> Option<Expr> {
    let statistic = Expr::Real(self.statistic());
    let p = self.p_value();
    let p_f64 = match &p {
      Expr::Real(v) => *v,
      _ => 1.0,
    };
    Some(match property {
      "PValue" => p,
      "TestStatistic" => statistic,
      "TestEntries" => Expr::List(vec![statistic, p].into()),
      "DegreesOfFreedom" => call("Missing", vec![str_expr("NotApplicable")]),
      "Properties" => Expr::List(
        PROPERTIES
          .iter()
          .map(|p| str_expr(p))
          .collect::<Vec<_>>()
          .into(),
      ),
      "ShortTestConclusion" => settings.short_conclusion(p_f64),
      "TestConclusion" => {
        let hypothesis = match self.alternative {
          Alternative::Unequal => "",
          Alternative::Less => "greater than or equal to ",
          Alternative::Greater => "less than or equal to ",
        };
        let verdict = if p_f64 < settings.alpha {
          "is rejected"
        } else {
          "is not rejected"
        };
        str_expr(&format!(
          "The null hypothesis that the median difference is {hypothesis}{} \
           {verdict} at the {level} percent level based on the \
           Mann\u{2010}Whitney test.",
          crate::syntax::expr_to_string(mu0)
        ))
      }
      "TestDataTable" => {
        table(&["Statistic", "P\u{2010}Value"], vec![statistic, p])
      }
      "PValueTable" => table(&["P\u{2010}Value"], vec![p]),
      "TestStatisticTable" => table(&["Statistic"], vec![statistic]),
      _ => return None,
    })
  }
}

/// `Grid[{{"", headers…}, {"Mann‐Whitney", values…}}, …]`, the layout of
/// the hypothesis tests' tables.
fn table(headers: &[&str], values: Vec<Expr>) -> Expr {
  let mut header = vec![str_expr("")];
  header.extend(headers.iter().map(|h| str_expr(h)));
  let mut row = vec![str_expr("Mann\u{2010}Whitney")];
  row.extend(values);
  let rule = |name: &str, value: Expr| {
    call("Rule", vec![Expr::Identifier(name.to_string()), value])
  };
  let divider = || {
    call(
      "Rule",
      vec![Expr::Integer(2), call("GrayLevel", vec![Expr::Real(0.7)])],
    )
  };
  call(
    "Grid",
    vec![
      Expr::List(
        vec![Expr::List(header.into()), Expr::List(row.into())].into(),
      ),
      rule(
        "Alignment",
        Expr::List(
          vec![
            Expr::Identifier("Left".to_string()),
            Expr::Identifier("Automatic".to_string()),
          ]
          .into(),
        ),
      ),
      rule(
        "Dividers",
        Expr::List(
          vec![
            Expr::List(vec![divider()].into()),
            Expr::List(vec![divider()].into()),
          ]
          .into(),
        ),
      ),
      rule("Spacings", Expr::Identifier("Automatic".to_string())),
    ],
  )
}

/// How many of the `C(n1 + n2, n1)` orderings of two tie-free samples give
/// each value of U (index = U).
fn u_distribution(n1: usize, n2: usize) -> Vec<u128> {
  let size = n1 * n2 + 1;
  // counts[j][u] for the current number of x values and j y values.
  let mut counts = vec![vec![0u128; size]; n2 + 1];
  for row in &mut counts {
    row[0] = 1;
  }
  for _ in 1..=n1 {
    let previous = counts;
    counts = vec![vec![0u128; size]; n2 + 1];
    counts[0][0] = 1;
    for j in 1..=n2 {
      for u in 0..size {
        // The largest value is a y (adds nothing to U) or an x (beats all j
        // y values).
        let mut c = counts[j - 1][u];
        if u >= j {
          c += previous[j][u - j];
        }
        counts[j][u] = c;
      }
    }
  }
  counts.swap_remove(n2)
}

/// The standard normal distribution function.
fn normal_cdf(z: f64) -> f64 {
  let t = -z / std::f64::consts::SQRT_2;
  if t >= 0.0 {
    erfc_f64(t) / 2.0
  } else {
    1.0 - erfc_f64(-t) / 2.0
  }
}
