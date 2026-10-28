//! Linear control systems: `TransferFunctionModel`, its poles, zeros and
//! responses, and the frequency-response plots (`BodePlot`, `NyquistPlot`,
//! `NicholsPlot`).
//!
//! A model is kept in wolframscript's normal form,
//! `TransferFunctionModel[{nums, den}, s, opts…]`: `nums` is the matrix of
//! numerators and `den` either the one denominator every element shares or a
//! matrix of them. A time delay `E^(-τ s)` stays a factor of its numerator.

use crate::InterpreterError;
use crate::syntax::Expr;

fn ev(e: &Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(e)
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn unevaluated(name: &str, args: &[Expr]) -> Expr {
  crate::helpers::unevaluated(name, args)
}

fn rule_name(e: &Expr) -> Option<&str> {
  match e {
    Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
      match pattern.as_ref() {
        Expr::Identifier(n) => Some(n.as_str()),
        _ => None,
      }
    }
    _ => None,
  }
}

/// A transfer-function matrix read from its normal form.
pub(crate) struct TfModel {
  /// Numerators, one row per output and one column per input.
  pub(crate) nums: Vec<Vec<Expr>>,
  /// Denominators, element by element (a shared one repeated).
  pub(crate) dens: Vec<Vec<Expr>>,
  /// The complex variable (`s`, or `z` for a sampled system).
  pub(crate) var: String,
  /// `SamplingPeriod -> T` of a discrete-time model.
  pub(crate) sampling_period: Option<Expr>,
}

fn matrix(e: &Expr) -> Option<Vec<Vec<Expr>>> {
  let Expr::List(rows) = e else { return None };
  let rows: Option<Vec<Vec<Expr>>> = rows
    .iter()
    .map(|r| match r {
      Expr::List(items) => Some(items.to_vec()),
      _ => None,
    })
    .collect();
  let rows = rows?;
  let width = rows.first()?.len();
  (width > 0 && rows.iter().all(|r| r.len() == width)).then_some(rows)
}

/// Whether `e` is `{nums, den}`: a numerator matrix and a denominator that
/// is a matrix of the same shape or a single expression.
fn normal_form(e: &Expr) -> Option<(Vec<Vec<Expr>>, Vec<Vec<Expr>>)> {
  let Expr::List(parts) = e else { return None };
  let [nums, den] = parts.as_slice() else {
    return None;
  };
  let nums = matrix(nums)?;
  let dens = match den {
    Expr::List(_) => {
      let dens = matrix(den)?;
      (dens.len() == nums.len() && dens[0].len() == nums[0].len())
        .then_some(dens)?
    }
    shared => vec![vec![shared.clone(); nums[0].len()]; nums.len()],
  };
  Some((nums, dens))
}

impl TfModel {
  pub(crate) fn from_expr(e: &Expr) -> Option<TfModel> {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    if name != "TransferFunctionModel" || args.len() < 2 {
      return None;
    }
    let (nums, dens) = normal_form(&args[0])?;
    let Expr::Identifier(var) = &args[1] else {
      return None;
    };
    let sampling_period = args[2..].iter().find_map(|o| match o {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "SamplingPeriod") => {
        Some(replacement.as_ref().clone())
      }
      _ => None,
    });
    Some(TfModel {
      nums,
      dens,
      var: var.clone(),
      sampling_period,
    })
  }

  pub(crate) fn outputs(&self) -> usize {
    self.nums.len()
  }

  pub(crate) fn inputs(&self) -> usize {
    self.nums[0].len()
  }
}

/// The exponent of `E^x`, in either representation of a power.
fn exp_exponent(e: &Expr) -> Option<&Expr> {
  let (base, exponent) = match e {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => (left.as_ref(), right.as_ref()),
    _ => return None,
  };
  matches!(base, Expr::Constant(c) | Expr::Identifier(c) if c == "E")
    .then_some(exponent)
}

/// A factor `E^(-τ s)` of `e`, split off: `(τ, rest)`. `τ` is 0 without one.
fn split_delay(e: &Expr, var: &str) -> Result<(Expr, Expr), InterpreterError> {
  let factors = crate::functions::polynomial_ast::together::flatten_times_args(
    std::slice::from_ref(e),
  );
  let mut delay = Vec::new();
  let mut rest = Vec::new();
  for f in factors {
    match exp_exponent(&f) {
      Some(x) if !crate::functions::calculus_ast::is_constant_wrt(x, var) => {
        delay.push(call(
          "Times",
          vec![
            Expr::Integer(-1),
            call("D", vec![x.clone(), Expr::Identifier(var.into())]),
          ],
        ));
      }
      _ => rest.push(f),
    }
  }
  let tau = if delay.is_empty() {
    Expr::Integer(0)
  } else {
    ev(&call("Plus", delay))?
  };
  let rest = match rest.len() {
    0 => Expr::Integer(1),
    1 => rest.pop().unwrap(),
    _ => ev(&call("Times", rest))?,
  };
  Ok((tau, rest))
}

/// The numerator and denominator of one transfer-function element. A
/// ratio of polynomials keeps its own (no common factor is cancelled, as in
/// wolframscript); anything else is combined with `Together`. A delay
/// stays in the numerator.
fn element_parts(
  f: &Expr,
  var: &str,
) -> Result<(Expr, Expr), InterpreterError> {
  let (tau, g) = split_delay(f, var)?;
  let polynomial = |e: &Expr| -> Result<bool, InterpreterError> {
    Ok(matches!(
      ev(&call("PolynomialQ", vec![e.clone(), Expr::Identifier(var.into())]))?,
      Expr::Identifier(ref t) if t == "True"
    ))
  };
  let mut num = ev(&call("Numerator", vec![g.clone()]))?;
  let mut den = ev(&call("Denominator", vec![g.clone()]))?;
  if !polynomial(&num)? || !polynomial(&den)? {
    let combined = ev(&call("Together", vec![g]))?;
    num = ev(&call("Numerator", vec![combined.clone()]))?;
    den = ev(&call("Denominator", vec![combined]))?;
  }
  if !matches!(tau, Expr::Integer(0)) {
    // wolframscript writes an inexact delay exactly and moves the
    // inexactness onto the rest: `10/(E^(2.*s) (1 + 7.*s))` is
    // `{{{10./E^(2*s)}}, 1. + 7.*s}`.
    let tau =
      if crate::functions::math_ast::numeric_utils::contains_inexact_real(&tau)
      {
        num = ev(&call("N", vec![num]))?;
        den = ev(&call("N", vec![den]))?;
        ev(&call("Rationalize", vec![tau, Expr::Integer(0)]))?
      } else {
        tau
      };
    num = ev(&call(
      "Times",
      vec![
        num,
        call(
          "Power",
          vec![
            Expr::Constant("E".into()),
            call(
              "Times",
              vec![Expr::Integer(-1), tau, Expr::Identifier(var.into())],
            ),
          ],
        ),
      ],
    ))?;
  }
  Ok((num, den))
}

fn normal_form_expr(nums: Vec<Vec<Expr>>, dens: Vec<Vec<Expr>>) -> Expr {
  let shared = dens
    .iter()
    .flatten()
    .all(|d| crate::evaluator::pattern_matching::expr_equal(d, &dens[0][0]));
  let to_matrix = |m: Vec<Vec<Expr>>| {
    Expr::List(m.into_iter().map(|r| Expr::List(r.into())).collect())
  };
  let den = if shared {
    dens[0][0].clone()
  } else {
    to_matrix(dens)
  };
  Expr::List(vec![to_matrix(nums), den].into())
}

/// `TransferFunctionModel[expr, s]`, `TransferFunctionModel[{{…}}, s]` and
/// the normal form itself, with options such as `SamplingPeriod -> T`.
pub fn transfer_function_model_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Ok(unevaluated("TransferFunctionModel", args));
  }
  let Expr::Identifier(var) = &args[1] else {
    return Ok(unevaluated("TransferFunctionModel", args));
  };
  // Already in normal form.
  if normal_form(&args[0]).is_some() {
    return Ok(unevaluated("TransferFunctionModel", args));
  }
  let elements = match &args[0] {
    Expr::List(_) => match matrix(&args[0]) {
      Some(m) => m,
      None => return Ok(unevaluated("TransferFunctionModel", args)),
    },
    scalar => vec![vec![scalar.clone()]],
  };
  let mut nums = Vec::with_capacity(elements.len());
  let mut dens = Vec::with_capacity(elements.len());
  for row in &elements {
    let mut num_row = Vec::with_capacity(row.len());
    let mut den_row = Vec::with_capacity(row.len());
    for f in row {
      let (n, d) = element_parts(f, var)?;
      num_row.push(n);
      den_row.push(d);
    }
    nums.push(num_row);
    dens.push(den_row);
  }
  let mut out = vec![normal_form_expr(nums, dens), args[1].clone()];
  out.extend(args[2..].iter().cloned());
  Ok(call("TransferFunctionModel", out))
}

/// `model[x]`: the transfer-function matrix at `x`. A delay `E^(-τ s)`
/// shows as `SystemsModelDelay[τ]`, as in wolframscript.
pub fn apply_transfer_function_model(
  model: &Expr,
  call_args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let tf = TfModel::from_expr(model)?;
  let [x] = call_args else { return None };
  Some((|| {
    let var = Expr::Identifier(tf.var.clone());
    let mut rows = Vec::with_capacity(tf.outputs());
    for (nrow, drow) in tf.nums.iter().zip(&tf.dens) {
      let mut row = Vec::with_capacity(nrow.len());
      for (n, d) in nrow.iter().zip(drow) {
        let (tau, rest) = split_delay(n, &tf.var)?;
        let value = call(
          "Times",
          vec![
            crate::functions::polynomial_ast::solve::substitute_expr(
              &rest, &var, x,
            ),
            call(
              "Power",
              vec![
                crate::functions::polynomial_ast::solve::substitute_expr(
                  d, &var, x,
                ),
                Expr::Integer(-1),
              ],
            ),
          ],
        );
        let value = if matches!(tau, Expr::Integer(0)) {
          value
        } else {
          call("Times", vec![value, call("SystemsModelDelay", vec![tau])])
        };
        row.push(ev(&value)?);
      }
      rows.push(Expr::List(row.into()));
    }
    Ok(Expr::List(rows.into()))
  })())
}

/// The roots of a polynomial in `var`, in `Solve`'s order.
fn roots(p: &Expr, var: &str) -> Result<Vec<Expr>, InterpreterError> {
  let var_expr = Expr::Identifier(var.into());
  if crate::functions::calculus_ast::is_constant_wrt(p, var) {
    return Ok(Vec::new());
  }
  let solved = ev(&call(
    "ReplaceAll",
    vec![
      var_expr.clone(),
      call(
        "Solve",
        vec![
          Expr::Comparison {
            operands: vec![p.clone(), Expr::Integer(0)],
            operators: vec![crate::syntax::ComparisonOp::Equal],
          },
          var_expr,
        ],
      ),
    ],
  ))?;
  match &solved {
    Expr::List(items) => Ok(items.to_vec()),
    _ => Ok(Vec::new()),
  }
}

fn per_element(
  args: &[Expr],
  name: &str,
  pick: impl Fn(&Expr, &Expr) -> Expr,
) -> Result<Expr, InterpreterError> {
  let Some(tf) = args.first().and_then(TfModel::from_expr) else {
    return Ok(unevaluated(name, args));
  };
  let mut rows = Vec::with_capacity(tf.outputs());
  for (nrow, drow) in tf.nums.iter().zip(&tf.dens) {
    let mut row = Vec::with_capacity(nrow.len());
    for (n, d) in nrow.iter().zip(drow) {
      let (_, rest) = split_delay(n, &tf.var)?;
      row.push(Expr::List(roots(&pick(&rest, d), &tf.var)?.into()));
    }
    rows.push(Expr::List(row.into()));
  }
  Ok(Expr::List(rows.into()))
}

/// `TransferFunctionPoles[model]`: the roots of each denominator.
pub fn transfer_function_poles_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  per_element(args, "TransferFunctionPoles", |_, d| d.clone())
}

/// `TransferFunctionZeros[model]`: the roots of each numerator.
pub fn transfer_function_zeros_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  per_element(args, "TransferFunctionZeros", |n, _| n.clone())
}

// ── Numeric frequency and time responses ────────────────────────────────

/// One element as numbers: polynomial coefficients (ascending) and a delay.
struct NumericElement {
  num: Vec<f64>,
  den: Vec<f64>,
  delay: f64,
}

fn coefficients(p: &Expr, var: &str) -> Option<Vec<f64>> {
  let list = ev(&call(
    "CoefficientList",
    vec![p.clone(), Expr::Identifier(var.into())],
  ))
  .ok()?;
  let Expr::List(items) = &list else {
    return crate::functions::math_ast::try_eval_to_f64(&list).map(|v| vec![v]);
  };
  let mut out: Vec<f64> = items
    .iter()
    .map(crate::functions::math_ast::try_eval_to_f64)
    .collect::<Option<_>>()?;
  while out.len() > 1 && *out.last().unwrap() == 0.0 {
    out.pop();
  }
  if out.is_empty() {
    out.push(0.0);
  }
  Some(out)
}

impl NumericElement {
  fn new(num: &Expr, den: &Expr, var: &str) -> Option<NumericElement> {
    let (tau, rest) = split_delay(num, var).ok()?;
    Some(NumericElement {
      num: coefficients(&rest, var)?,
      den: coefficients(den, var)?,
      delay: crate::functions::math_ast::try_eval_to_f64(&tau)?,
    })
  }

  /// The value at `s = (re, im)`.
  fn at(&self, re: f64, im: f64) -> (f64, f64) {
    let horner = |c: &[f64]| {
      c.iter().rev().fold((0.0, 0.0), |(ar, ai), &k| {
        (ar * re - ai * im + k, ar * im + ai * re)
      })
    };
    let (nr, ni) = horner(&self.num);
    let (dr, di) = horner(&self.den);
    let scale = dr * dr + di * di;
    let (mut vr, mut vi) =
      ((nr * dr + ni * di) / scale, (ni * dr - nr * di) / scale);
    if self.delay != 0.0 {
      // E^(-τ s) at s = re + i im.
      let m = (-self.delay * re).exp();
      let (c, s) = ((-self.delay * im).cos(), (-self.delay * im).sin());
      (vr, vi) = (m * (vr * c - vi * s), m * (vr * s + vi * c));
    }
    (vr, vi)
  }

  /// The continuous phase at `s = i ω` (`ω > 0`): the angles from each zero
  /// less those from each pole, the sign of the leading coefficients and
  /// the delay's `-τ ω`. Unlike an `Arg` of the value it does not wrap.
  fn phase(&self, omega: f64) -> f64 {
    use std::f64::consts::PI;
    let angles = |c: &[f64]| -> f64 {
      let lead = *c.last().unwrap_or(&1.0);
      let sign = if lead < 0.0 { PI } else { 0.0 };
      let roots = if c.len() > 1 {
        crate::functions::polynomial_ast::solve::durand_kerner_roots(c)
      } else {
        Vec::new()
      };
      sign
        + roots
          .iter()
          .map(|&(zr, zi)| (omega - zi).atan2(-zr))
          .sum::<f64>()
    };
    angles(&self.num) - angles(&self.den) - self.delay * omega
  }

  /// The nonzero |poles| and |zeros|, and whether one lies at the origin.
  fn corner_frequencies(&self) -> (Vec<f64>, bool) {
    let mut corners = Vec::new();
    let mut origin = false;
    for c in [&self.num, &self.den] {
      if c.len() > 1 {
        for (r, i) in
          crate::functions::polynomial_ast::solve::durand_kerner_roots(c)
        {
          let m = r.hypot(i);
          if m < 1e-12 {
            origin = true;
          } else {
            corners.push(m);
          }
        }
      }
    }
    (corners, origin)
  }
}

/// One frequency-response series: an element of a model, and the model's
/// sampling period if it is discrete.
struct Series {
  element: NumericElement,
  period: Option<f64>,
}

impl Series {
  /// The response at the frequency `ω`: `G(i ω)`, or `G(E^(i ω T))`.
  fn at(&self, omega: f64) -> (f64, f64) {
    match self.period {
      None => self.element.at(0.0, omega),
      Some(t) => self.element.at((omega * t).cos(), (omega * t).sin()),
    }
  }

  /// The phases (radians) over increasing frequencies, without jumps: the
  /// analytic angle sum for a continuous model, the unwrapped argument for
  /// a sampled one.
  fn phases(&self, omegas: &[f64]) -> Vec<f64> {
    if self.period.is_none() {
      return omegas.iter().map(|&w| self.element.phase(w)).collect();
    }
    let mut out: Vec<f64> = Vec::with_capacity(omegas.len());
    for &w in omegas {
      let (r, i) = self.at(w);
      let mut p = i.atan2(r);
      if let Some(&prev) = out.last() {
        let tau = std::f64::consts::TAU;
        p += tau * ((prev - p) / tau).round();
      }
      out.push(p);
    }
    out
  }
}

/// The numeric elements of every model in `spec` (a model or a list of
/// them), one series per element.
fn numeric_series(spec: &Expr) -> Option<Vec<Series>> {
  let models: Vec<&Expr> = match spec {
    Expr::List(items) => items.iter().collect(),
    single => vec![single],
  };
  let mut out = Vec::new();
  for m in models {
    let tf = TfModel::from_expr(m)?;
    let period = match &tf.sampling_period {
      None => None,
      Some(Expr::Identifier(n)) if n == "None" => None,
      Some(p) => Some(
        crate::functions::math_ast::try_eval_to_f64(p).filter(|p| *p > 0.0)?,
      ),
    };
    for (nrow, drow) in tf.nums.iter().zip(&tf.dens) {
      for (n, d) in nrow.iter().zip(drow) {
        out.push(Series {
          element: NumericElement::new(n, d, &tf.var)?,
          period,
        });
      }
    }
  }
  (!out.is_empty()).then_some(out)
}

/// wolframscript's automatic frequency range: a decade and a third either
/// side of the corner frequencies (`ω/20` to `20 ω`), from `0.001` when a
/// pole or zero lies at the origin, and up to 10 without any.
fn automatic_frequency_range(series: &[Series]) -> (f64, f64) {
  // A sampled model's response repeats past the Nyquist frequency π/T.
  if let Some(t) = series.iter().filter_map(|s| s.period).reduce(f64::min) {
    let nyquist = std::f64::consts::PI / t;
    return (nyquist / 1000.0, nyquist);
  }
  let mut corners = Vec::new();
  let mut origin = false;
  for s in series {
    let (c, o) = s.element.corner_frequencies();
    corners.extend(c);
    origin |= o;
  }
  let max = corners.iter().copied().fold(f64::NAN, f64::max);
  let min = corners.iter().copied().fold(f64::NAN, f64::min);
  let high = if max.is_finite() { 20.0 * max } else { 10.0 };
  let low = if origin || !min.is_finite() {
    0.001
  } else {
    min / 20.0
  };
  (low, high)
}

/// `{ωmin, ωmax}` from the argument after the model, if one is given.
fn frequency_range(arg: Option<&Expr>) -> Option<(f64, f64)> {
  let Expr::List(items) = arg? else { return None };
  let [lo, hi] = items.as_slice() else {
    return None;
  };
  let lo = crate::functions::math_ast::try_eval_to_f64(lo)?;
  let hi = crate::functions::math_ast::try_eval_to_f64(hi)?;
  (lo > 0.0 && hi > lo).then_some((lo, hi))
}

fn log_grid(low: f64, high: f64, n: usize) -> Vec<f64> {
  let (a, b) = (low.log10(), high.log10());
  (0..n)
    .map(|k| 10f64.powf(a + (b - a) * k as f64 / (n - 1) as f64))
    .collect()
}

fn point(x: f64, y: f64) -> Expr {
  Expr::List(vec![Expr::Real(x), Expr::Real(y)].into())
}

/// The options a frequency-response plot hands to the plot it draws:
/// everything but its own.
fn passed_options(opts: &[Expr]) -> Vec<Expr> {
  const OWN: [&str; 6] = [
    "PlotLayout",
    "StabilityMargins",
    "StabilityMarginsStyle",
    "PhaseRange",
    "FeedbackType",
    "SamplingPeriod",
  ];
  opts
    .iter()
    .filter(|o| rule_name(o).is_some_and(|n| !OWN.contains(&n)))
    .cloned()
    .collect()
}

fn split_options(args: &[Expr]) -> (Vec<&Expr>, Vec<Expr>) {
  let mut positional = Vec::new();
  let mut options = Vec::new();
  for a in args {
    if rule_name(a).is_some() {
      options.push(a.clone());
    } else {
      positional.push(a);
    }
  }
  (positional, options)
}

const SAMPLES: usize = 400;

/// `BodePlot[model]`, `BodePlot[{model, …}]`, `BodePlot[model, {ωmin,
/// ωmax}]`: the magnitude in decibels over the phase in degrees, both
/// against a logarithmic frequency axis, as a two-row `Grid`.
pub fn bode_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (positional, options) = split_options(args);
  let Some(series) = positional.first().and_then(|m| numeric_series(m)) else {
    return Ok(unevaluated("BodePlot", args));
  };
  let (low, high) = frequency_range(positional.get(1).copied())
    .unwrap_or_else(|| automatic_frequency_range(&series));
  let omegas = log_grid(low, high, SAMPLES);
  let mut magnitude = Vec::with_capacity(series.len());
  let mut phase = Vec::with_capacity(series.len());
  for e in &series {
    magnitude.push(Expr::List(
      omegas
        .iter()
        .map(|&w| {
          let (r, i) = e.at(w);
          point(w, 20.0 * r.hypot(i).log10())
        })
        .collect(),
    ));
    phase.push(Expr::List(
      omegas
        .iter()
        .zip(e.phases(&omegas))
        .map(|(&w, p)| point(w, p.to_degrees()))
        .collect(),
    ));
  }
  let panel = |data: Vec<Expr>| -> Result<Expr, InterpreterError> {
    let mut plot_args = vec![
      Expr::List(data.into()),
      rule("Joined", id("True")),
      rule("Frame", id("True")),
      rule("Axes", id("True")),
      rule(
        "AspectRatio",
        call("Power", vec![id("GoldenRatio"), Expr::Integer(-1)]),
      ),
      rule("PlotRange", id("All")),
    ];
    plot_args.extend(passed_options(&options));
    ev(&call("ListLogLinearPlot", plot_args))
  };
  let grid = Expr::List(
    vec![
      Expr::List(vec![panel(magnitude)?].into()),
      Expr::List(vec![panel(phase)?].into()),
    ]
    .into(),
  );
  ev(&call("Grid", vec![grid]))
}

/// `NicholsPlot[model]`: the magnitude in decibels against the phase in
/// degrees as the frequency runs over its range.
pub fn nichols_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (positional, options) = split_options(args);
  let Some(series) = positional.first().and_then(|m| numeric_series(m)) else {
    return Ok(unevaluated("NicholsPlot", args));
  };
  let (low, high) = frequency_range(positional.get(1).copied())
    .unwrap_or_else(|| automatic_frequency_range(&series));
  let omegas = log_grid(low, high, SAMPLES);
  let curves: Vec<Expr> = series
    .iter()
    .map(|e| {
      Expr::List(
        omegas
          .iter()
          .zip(e.phases(&omegas))
          .map(|(&w, p)| {
            let (r, i) = e.at(w);
            point(p.to_degrees(), 20.0 * r.hypot(i).log10())
          })
          .collect(),
      )
    })
    .collect();
  let mut plot_args = vec![
    Expr::List(curves.into()),
    rule("Axes", id("True")),
    rule("AspectRatio", Expr::Integer(1)),
    rule("PlotRange", id("All")),
  ];
  plot_args.extend(passed_options(&options));
  ev(&call("ListLinePlot", plot_args))
}

/// `NyquistPlot[model]`: the frequency response in the complex plane, for
/// positive frequencies and their mirror image.
pub fn nyquist_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (positional, options) = split_options(args);
  let Some(series) = positional.first().and_then(|m| numeric_series(m)) else {
    return Ok(unevaluated("NyquistPlot", args));
  };
  let (low, high) = match positional.get(1) {
    // `{ωmin, ωmax}` may run through negative frequencies; the curve is
    // symmetric, so its positive part is drawn and mirrored.
    Some(Expr::List(items)) if items.len() == 2 => {
      let lo = crate::functions::math_ast::try_eval_to_f64(&items[0]);
      let hi = crate::functions::math_ast::try_eval_to_f64(&items[1]);
      match (lo, hi) {
        (Some(lo), Some(hi)) if hi > lo => {
          let top = lo.abs().max(hi.abs());
          let bottom = if lo < 0.0 && hi > 0.0 {
            top * 1e-4
          } else {
            lo.abs().min(hi.abs()).max(top * 1e-4)
          };
          (bottom, top)
        }
        _ => automatic_frequency_range(&series),
      }
    }
    _ => {
      let (low, high) = automatic_frequency_range(&series);
      (low / 10.0, high * 10.0)
    }
  };
  let omegas = log_grid(low, high, SAMPLES);
  let mut curves = Vec::with_capacity(2 * series.len());
  for e in &series {
    let positive: Vec<(f64, f64)> = omegas.iter().map(|&w| e.at(w)).collect();
    curves.push(Expr::List(
      positive.iter().map(|&(r, i)| point(r, i)).collect(),
    ));
    curves.push(Expr::List(
      positive.iter().rev().map(|&(r, i)| point(r, -i)).collect(),
    ));
  }
  let mut plot_args = vec![
    Expr::List(curves.into()),
    rule("Axes", id("True")),
    rule("AspectRatio", id("Automatic")),
  ];
  plot_args.extend(passed_options(&options));
  ev(&call("ListLinePlot", plot_args))
}

fn rule(name: &str, value: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(id(name)),
    replacement: Box::new(value),
  }
}

fn id(name: &str) -> Expr {
  Expr::Identifier(name.into())
}

// ── OutputResponse ──────────────────────────────────────────────────────

/// Simulate `num(s)/den(s)` (proper) with the classical Runge–Kutta method
/// on its controllable canonical realization. `u` holds three input values
/// per step: just after its start, at its midpoint and just before its end,
/// so that an input that jumps at a step boundary is integrated from the
/// jump on rather than a fraction of a step early. Returns the output at
/// the start of every step and at the end of the last.
fn simulate(
  e: &NumericElement,
  u: &[f64],
  h: f64,
  impulse: Option<(usize, f64)>,
) -> Option<Vec<f64>> {
  let lead = *e.den.last()?;
  if lead == 0.0 || e.num.len() > e.den.len() {
    return None;
  }
  let n = e.den.len() - 1;
  let a: Vec<f64> = e.den.iter().map(|c| c / lead).collect();
  let mut b: Vec<f64> = e.num.iter().map(|c| c / lead).collect();
  b.resize(n + 1, 0.0);
  // y = d u + c . x with d = b[n] and c[k] = b[k] - d a[k].
  let d = b[n];
  let c: Vec<f64> = (0..n).map(|k| b[k] - d * a[k]).collect();
  let deriv = |x: &[f64], uu: f64| -> Vec<f64> {
    let mut dx = vec![0.0; n];
    for k in 0..n.saturating_sub(1) {
      dx[k] = x[k + 1];
    }
    if n > 0 {
      dx[n - 1] = uu - (0..n).map(|k| a[k] * x[k]).sum::<f64>();
    }
    dx
  };
  let steps = u.len() / 3;
  let mut x = vec![0.0; n];
  let mut out = Vec::with_capacity(steps + 1);
  let output = |x: &[f64], uu: f64| {
    d * uu + c.iter().zip(x).map(|(c, x)| c * x).sum::<f64>()
  };
  out.push(output(&x, u[0]));
  for k in 0..steps {
    // An impulse of weight w kicks the state by w B (B = e_n here).
    if let Some((at, weight)) = impulse
      && at == k
      && n > 0
    {
      x[n - 1] += weight;
      if k == 0 {
        out[0] = output(&x, u[0]);
      }
    }
    let (u0, um, u1) = (u[3 * k], u[3 * k + 1], u[3 * k + 2]);
    let k1 = deriv(&x, u0);
    let x2: Vec<f64> =
      x.iter().zip(&k1).map(|(x, k)| x + h / 2.0 * k).collect();
    let k2 = deriv(&x2, um);
    let x3: Vec<f64> =
      x.iter().zip(&k2).map(|(x, k)| x + h / 2.0 * k).collect();
    let k3 = deriv(&x3, um);
    let x4: Vec<f64> = x.iter().zip(&k3).map(|(x, k)| x + h * k).collect();
    let k4 = deriv(&x4, u1);
    for i in 0..n {
      x[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    if x.iter().any(|v| !v.is_finite()) {
      return None;
    }
    out.push(output(&x, u.get(3 * k + 3).copied().unwrap_or(u1)));
  }
  Some(out)
}

/// `OutputResponse[model, u, t]` (symbolic) and `OutputResponse[model, u,
/// {t, tmin, tmax}]` (simulated, as `{InterpolatingFunction[…][t], …}`).
/// `u` is one input or a list with one per model input; the system starts
/// at rest.
pub fn output_response_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated_call = || Ok(unevaluated("OutputResponse", args));
  if !(2..=3).contains(&args.len()) {
    return unevaluated_call();
  }
  let Some(tf) = TfModel::from_expr(&args[0]) else {
    return unevaluated_call();
  };
  let inputs: Vec<Expr> = match &args[1] {
    // One list per input; a list of numbers is the samples of one input.
    Expr::List(items)
      if args.len() == 2
        && items.iter().all(|i| !matches!(i, Expr::List(_))) =>
    {
      vec![args[1].clone()]
    }
    Expr::List(items) => items.to_vec(),
    single => vec![single.clone()],
  };
  if inputs.len() != tf.inputs() {
    return unevaluated_call();
  }
  // `{t, tmax}` starts at 0.
  let span = |spec: &Expr| -> Option<(String, f64, f64)> {
    let Expr::List(items) = spec else { return None };
    let (var, lo, hi) = match items.as_slice() {
      [v, hi] => (v, Expr::Integer(0), hi.clone()),
      [v, lo, hi] => (v, lo.clone(), hi.clone()),
      _ => return None,
    };
    let Expr::Identifier(var) = var else {
      return None;
    };
    let lo = crate::functions::math_ast::try_eval_to_f64(&lo)?;
    let hi = crate::functions::math_ast::try_eval_to_f64(&hi)?;
    (hi > lo).then(|| (var.clone(), lo, hi))
  };
  if let Some(period) = &tf.sampling_period {
    let Some(period) =
      crate::functions::math_ast::try_eval_to_f64(period).filter(|p| *p > 0.0)
    else {
      return unevaluated_call();
    };
    let result = match args.get(2) {
      None => discrete_response(&tf, period, &inputs, None)?,
      Some(spec) => match span(spec) {
        Some((var, lo, hi)) => {
          discrete_response(&tf, period, &inputs, Some((&var, lo, hi)))?
        }
        None => None,
      },
    };
    return Ok(result.unwrap_or_else(|| unevaluated("OutputResponse", args)));
  }
  match args.get(2) {
    Some(Expr::Identifier(t)) => symbolic_response(&tf, &inputs, t, args),
    Some(spec) => match span(spec) {
      Some((t, t0, t1)) => simulated_response(&tf, &inputs, &t, t0, t1)
        .map(|r| r.unwrap_or_else(|| unevaluated("OutputResponse", args))),
      None => unevaluated_call(),
    },
    None => unevaluated_call(),
  }
}

fn contains_head(e: &Expr, head: &str) -> bool {
  match e {
    Expr::FunctionCall { name, args } => {
      name == head || args.iter().any(|a| contains_head(a, head))
    }
    Expr::List(items) => items.iter().any(|a| contains_head(a, head)),
    Expr::BinaryOp { left, right, .. } => {
      contains_head(left, head) || contains_head(right, head)
    }
    Expr::UnaryOp { operand, .. } => contains_head(operand, head),
    _ => false,
  }
}

fn symbolic_response(
  tf: &TfModel,
  inputs: &[Expr],
  t: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let s = Expr::Identifier(tf.var.clone());
  let time = Expr::Identifier(t.into());
  let mut outputs = Vec::with_capacity(tf.outputs());
  for (nrow, drow) in tf.nums.iter().zip(&tf.dens) {
    let mut terms = Vec::with_capacity(nrow.len());
    for ((n, d), u) in nrow.iter().zip(drow).zip(inputs) {
      let (tau, _) = split_delay(n, &tf.var)?;
      if !matches!(tau, Expr::Integer(0)) {
        crate::emit_message(
          "OutputResponse::delaysm: The system contains time delays. An exact \
           simulation is not possible; try simulating over a time span.",
        );
        return Ok(unevaluated("OutputResponse", args));
      }
      let laplace =
        call("LaplaceTransform", vec![u.clone(), time.clone(), s.clone()]);
      terms.push(call(
        "Times",
        vec![
          n.clone(),
          call("Power", vec![d.clone(), Expr::Integer(-1)]),
          laplace,
        ],
      ));
    }
    let y = ev(&call(
      "InverseLaplaceTransform",
      vec![call("Plus", terms), s.clone(), time.clone()],
    ))?;
    if matches!(&y, Expr::FunctionCall { name, .. } if name == "InverseLaplaceTransform")
    {
      return Ok(unevaluated("OutputResponse", args));
    }
    // The response to an impulse starts at it: wolframscript multiplies it
    // by `HeavisideTheta[t]`.
    let y = if inputs.iter().any(|u| contains_head(u, "DiracDelta")) {
      call("Times", vec![y, call("HeavisideTheta", vec![time.clone()])])
    } else {
      y
    };
    outputs.push(ev(&call("Together", vec![y]))?);
  }
  Ok(Expr::List(outputs.into()))
}

/// The weight `c` of an input `c DiracDelta[t]`.
fn impulse_weight(u: &Expr, t: &str) -> Option<f64> {
  let is_delta = |e: &Expr| {
    matches!(e, Expr::FunctionCall { name, args }
      if name == "DiracDelta"
        && matches!(args.as_slice(), [Expr::Identifier(v)] if v == t))
  };
  let factors = crate::functions::polynomial_ast::together::flatten_times_args(
    std::slice::from_ref(u),
  );
  let deltas = factors.iter().filter(|f| is_delta(f)).count();
  if deltas != 1 {
    return None;
  }
  factors
    .iter()
    .filter(|f| !is_delta(f))
    .map(crate::functions::math_ast::try_eval_to_f64)
    .try_fold(1.0, |acc, v| v.map(|v| acc * v))
}

fn simulated_response(
  tf: &TfModel,
  inputs: &[Expr],
  t: &str,
  t0: f64,
  t1: f64,
) -> Result<Option<Expr>, InterpreterError> {
  let mut elements = Vec::with_capacity(tf.outputs());
  for (nrow, drow) in tf.nums.iter().zip(&tf.dens) {
    let mut row = Vec::with_capacity(nrow.len());
    for (n, d) in nrow.iter().zip(drow) {
      match NumericElement::new(n, d, &tf.var) {
        Some(e) => row.push(e),
        None => return Ok(None),
      }
    }
    elements.push(row);
  }
  // A step small against the fastest pole, and fine enough to plot.
  let mut fastest: f64 = 0.0;
  for e in elements.iter().flatten() {
    let (c, _) = e.corner_frequencies();
    fastest = c.into_iter().fold(fastest, f64::max);
  }
  // The output is kept at `OUTPUT_POINTS` times and simulated `REFINE`
  // times as finely, which bounds the error an input step between two
  // simulation steps can cause.
  const OUTPUT_POINTS: usize = 2000;
  const REFINE: usize = 10;
  let span = t1 - t0;
  let mut steps = OUTPUT_POINTS;
  if fastest > 0.0 {
    steps = steps.max((span * fastest * 20.0).ceil() as usize);
  }
  let steps = steps.min(100_000) * REFINE;
  let h = span / steps as f64;
  // The inputs over every step from the start of time (the system is at
  // rest before 0, and a delay looks that far back): see `simulate`.
  let start = 0f64.min(t0);
  let total_steps = ((t1 - start) / h).ceil() as usize;
  let epsilon = h * 1e-7;
  let sample_input =
    |u: &Expr, delay: f64| -> Result<Option<Vec<f64>>, InterpreterError> {
      let times: Vec<Expr> = (0..total_steps)
        .flat_map(|k| {
          let tk = start + k as f64 * h - delay;
          [tk + epsilon, tk + h / 2.0, tk + h - epsilon]
        })
        .map(Expr::Real)
        .collect();
      let values = ev(&call(
        "Map",
        vec![
          Expr::Function {
            body: Box::new(crate::syntax::substitute_variable(
              u,
              t,
              &Expr::Slot(1),
            )),
          },
          Expr::List(times.clone().into()),
        ],
      ))?;
      let Expr::List(values) = &values else {
        return Ok(None);
      };
      let mut out = Vec::with_capacity(values.len());
      for (v, time) in values.iter().zip(&times) {
        let Expr::Real(tv) = time else { unreachable!() };
        if *tv < 0.0 {
          out.push(0.0);
          continue;
        }
        match crate::functions::math_ast::try_eval_to_f64(v) {
          Some(x) => out.push(x),
          None => return Ok(None),
        }
      }
      Ok(Some(out))
    };
  let mut results = Vec::with_capacity(elements.len());
  for row in &elements {
    let mut total: Vec<f64> = vec![0.0; total_steps + 1];
    for (e, u) in row.iter().zip(inputs) {
      let rational = NumericElement {
        num: e.num.clone(),
        den: e.den.clone(),
        delay: 0.0,
      };
      // `c DiracDelta[t]` acts through the state rather than as samples.
      let y = if let Some(weight) = impulse_weight(u, t) {
        let at = ((e.delay - start) / h).round();
        if at < 0.0 {
          return Ok(None);
        }
        let zeros = vec![0.0; 3 * total_steps];
        simulate(&rational, &zeros, h, Some((at as usize, weight)))
      } else {
        let Some(samples) = sample_input(u, e.delay)? else {
          return Ok(None);
        };
        simulate(&rational, &samples, h, None)
      };
      let Some(y) = y else {
        return Ok(None);
      };
      for (acc, v) in total.iter_mut().zip(y) {
        *acc += v;
      }
    }
    // Keep the part from t0 on, at every `REFINE`th step. Linear
    // interpolation follows the kink where a delayed or stepped input
    // starts without overshooting it.
    let first = ((t0 - start) / h).round() as usize;
    let data: Vec<Expr> = total[first..]
      .iter()
      .enumerate()
      .step_by(REFINE)
      .map(|(k, v)| point(t0 + k as f64 * h, *v))
      .collect();
    let interpolation = ev(&call(
      "Interpolation",
      vec![
        Expr::List(data.into()),
        rule("InterpolationOrder", Expr::Integer(1)),
      ],
    ))?;
    results.push(Expr::CurriedCall {
      func: Box::new(interpolation),
      args: vec![Expr::Identifier(t.into())],
    });
  }
  Ok(Some(Expr::List(results.into())))
}

// ── Discrete-time models ────────────────────────────────────────────────

/// The algebraic maps between `s` and `z` that a conversion method stands
/// for (`T` the sampling period).
#[derive(Clone, Copy)]
enum Discretization {
  /// `s = (2/T) (z - 1)/(z + 1)`.
  Bilinear,
  /// `s = (z - 1)/T`.
  ForwardRectangular,
  /// `s = (z - 1)/(z T)`.
  BackwardRectangular,
}

fn discretization(opts: &[Expr]) -> Option<Discretization> {
  let mut method = Discretization::Bilinear;
  for o in opts {
    let Expr::Rule {
      pattern,
      replacement,
    } = o
    else {
      continue;
    };
    if !matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Method") {
      continue;
    }
    method = match replacement.as_ref() {
      Expr::String(m) => match m.as_str() {
        "BilinearTransform" => Discretization::Bilinear,
        "ForwardRectangularRule" => Discretization::ForwardRectangular,
        "BackwardRectangularRule" => Discretization::BackwardRectangular,
        _ => return None,
      },
      _ => return None,
    };
  }
  Some(method)
}

/// `Σ c_k a^k b^(n - k)` for the coefficients `c` (ascending) of `p` in
/// `var`: the polynomial `p(a/b)` cleared of its denominator `b^n`.
fn homogenize(
  p: &Expr,
  var: &str,
  n: usize,
  a: &Expr,
  b: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  let coefficients = ev(&call(
    "CoefficientList",
    vec![p.clone(), Expr::Identifier(var.into())],
  ))?;
  let coefficients: Vec<Expr> = match &coefficients {
    Expr::List(items) => items.to_vec(),
    single => vec![single.clone()],
  };
  if coefficients.len() > n + 1 {
    return Ok(None);
  }
  let terms: Vec<Expr> = coefficients
    .into_iter()
    .enumerate()
    .map(|(k, c)| {
      call(
        "Times",
        vec![
          c,
          call("Power", vec![a.clone(), Expr::Integer(k as i128)]),
          call("Power", vec![b.clone(), Expr::Integer((n - k) as i128)]),
        ],
      )
    })
    .collect();
  Ok(Some(ev(&call("Plus", terms))?))
}

fn degree(p: &Expr, var: &str) -> Result<usize, InterpreterError> {
  let d = ev(&call(
    "Exponent",
    vec![p.clone(), Expr::Identifier(var.into())],
  ))?;
  Ok(match d {
    Expr::Integer(k) if k > 0 => k as usize,
    _ => 0,
  })
}

/// Convert every element with the substitution `old = a/b` (`a`, `b`
/// polynomials in the new variable), clearing `b^n` from numerator and
/// denominator alike. `expand_numerator` writes the numerator out.
fn convert_model(
  tf: &TfModel,
  new_var: &Expr,
  a: &Expr,
  b: &Expr,
  extra_factor: &Expr,
  expand_numerator: bool,
  opts: Vec<Expr>,
) -> Result<Option<Expr>, InterpreterError> {
  let mut nums = Vec::with_capacity(tf.outputs());
  let mut dens = Vec::with_capacity(tf.outputs());
  for (nrow, drow) in tf.nums.iter().zip(&tf.dens) {
    let mut num_row = Vec::with_capacity(nrow.len());
    let mut den_row = Vec::with_capacity(nrow.len());
    for (num, den) in nrow.iter().zip(drow) {
      let (tau, _) = split_delay(num, &tf.var)?;
      if !matches!(tau, Expr::Integer(0)) {
        return Ok(None);
      }
      let n = degree(den, &tf.var)?.max(degree(num, &tf.var)?);
      let factor = call(
        "Power",
        vec![extra_factor.clone(), Expr::Integer(n as i128)],
      );
      let (Some(new_num), Some(new_den)) = (
        homogenize(num, &tf.var, n, a, b)?,
        homogenize(den, &tf.var, n, a, b)?,
      ) else {
        return Ok(None);
      };
      let new_num = ev(&call("Times", vec![factor.clone(), new_num]))?;
      let new_den = ev(&call("Times", vec![factor, new_den]))?;
      num_row.push(if expand_numerator {
        ev(&call("Expand", vec![new_num]))?
      } else {
        new_num
      });
      den_row.push(ev(&call("Expand", vec![new_den]))?);
    }
    nums.push(Expr::List(num_row.into()));
    dens.push(Expr::List(den_row.into()));
  }
  let mut args = vec![
    Expr::List(vec![Expr::List(nums.into()), Expr::List(dens.into())].into()),
    new_var.clone(),
  ];
  args.extend(opts);
  Ok(Some(call("TransferFunctionModel", args)))
}

fn sampling_rule(value: Expr) -> Expr {
  rule("SamplingPeriod", value)
}

/// `ToDiscreteTimeModel[model, T, z, Method -> m]`: the sampled model by
/// the bilinear transform (the default) or a rectangular rule.
pub fn to_discrete_time_model_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated_call = || Ok(unevaluated("ToDiscreteTimeModel", args));
  let (positional, options) = split_options(args);
  let (Some(model), Some(period)) = (positional.first(), positional.get(1))
  else {
    return unevaluated_call();
  };
  let Some(tf) = TfModel::from_expr(model) else {
    return unevaluated_call();
  };
  if tf.sampling_period.is_some() {
    return unevaluated_call();
  }
  let Some(method) = discretization(&options) else {
    return unevaluated_call();
  };
  let z = match positional.get(2) {
    Some(v @ Expr::Identifier(_)) => (*v).clone(),
    Some(_) => return unevaluated_call(),
    None => Expr::Identifier("z".into()),
  };
  let period = (*period).clone();
  let z_minus_1 = call("Plus", vec![z.clone(), Expr::Integer(-1)]);
  let z_plus_1 = call("Plus", vec![z.clone(), Expr::Integer(1)]);
  // s = a/b, with every element multiplied through by T^n.
  let (a, b) = match method {
    Discretization::Bilinear => (
      call("Times", vec![Expr::Integer(2), z_minus_1]),
      call("Times", vec![period.clone(), z_plus_1]),
    ),
    Discretization::ForwardRectangular => (z_minus_1, period.clone()),
    Discretization::BackwardRectangular => {
      (z_minus_1, call("Times", vec![period.clone(), z.clone()]))
    }
  };
  let converted = convert_model(
    &tf,
    &z,
    &a,
    &b,
    &Expr::Integer(1),
    false,
    vec![sampling_rule(period)],
  )?;
  Ok(converted.unwrap_or_else(|| unevaluated("ToDiscreteTimeModel", args)))
}

/// `ToContinuousTimeModel[model, s, Method -> m]`: the inverse of
/// `ToDiscreteTimeModel`'s maps.
pub fn to_continuous_time_model_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated_call = || Ok(unevaluated("ToContinuousTimeModel", args));
  let (positional, options) = split_options(args);
  let Some(tf) = positional.first().and_then(|m| TfModel::from_expr(m)) else {
    return unevaluated_call();
  };
  let Some(period) = tf.sampling_period.clone() else {
    return unevaluated_call();
  };
  if matches!(&period, Expr::Identifier(n) if n == "None") {
    return unevaluated_call();
  }
  let Some(method) = discretization(&options) else {
    return unevaluated_call();
  };
  let s = match positional.get(1) {
    Some(v @ Expr::Identifier(_)) => (*v).clone(),
    Some(_) => return unevaluated_call(),
    None => Expr::Identifier("s".into()),
  };
  let st = call("Times", vec![s.clone(), period.clone()]);
  let half_st = call(
    "Times",
    vec![
      call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
      st.clone(),
    ],
  );
  // z = a/b.
  let (a, b) = match method {
    Discretization::Bilinear => (
      call("Plus", vec![Expr::Integer(1), half_st.clone()]),
      call(
        "Plus",
        vec![
          Expr::Integer(1),
          call("Times", vec![Expr::Integer(-1), half_st]),
        ],
      ),
    ),
    Discretization::ForwardRectangular => {
      (call("Plus", vec![Expr::Integer(1), st]), Expr::Integer(1))
    }
    Discretization::BackwardRectangular => (
      Expr::Integer(1),
      call(
        "Plus",
        vec![Expr::Integer(1), call("Times", vec![Expr::Integer(-1), st])],
      ),
    ),
  };
  let converted = convert_model(
    &tf,
    &s,
    &a,
    &b,
    &Expr::Integer(1),
    true,
    vec![sampling_rule(Expr::Identifier("None".into()))],
  )?;
  Ok(converted.unwrap_or_else(|| unevaluated("ToContinuousTimeModel", args)))
}

/// Run `Σ a_k y[j-n+k] = Σ b_k u[j-n+k]` from rest over the samples `u`.
fn difference_equation(e: &NumericElement, u: &[f64]) -> Option<Vec<f64>> {
  let lead = *e.den.last()?;
  if lead == 0.0 || e.num.len() > e.den.len() {
    return None;
  }
  let n = e.den.len() - 1;
  let at = |v: &[f64], i: isize| if i < 0 { 0.0 } else { v[i as usize] };
  let mut y: Vec<f64> = Vec::with_capacity(u.len());
  for j in 0..u.len() {
    let base = j as isize - n as isize;
    let forced: f64 = e
      .num
      .iter()
      .enumerate()
      .map(|(k, b)| b * at(u, base + k as isize))
      .sum();
    let past: f64 = (0..n).map(|k| e.den[k] * at(&y, base + k as isize)).sum();
    y.push((forced - past) / lead);
  }
  Some(y)
}

/// Discrete `OutputResponse`: over `{n, nmin, nmax}` (in time units, one
/// value per sampling period) or for a list of input samples.
fn discrete_response(
  tf: &TfModel,
  period: f64,
  inputs: &[Expr],
  span: Option<(&str, f64, f64)>,
) -> Result<Option<Expr>, InterpreterError> {
  let mut elements = Vec::with_capacity(tf.outputs());
  for (nrow, drow) in tf.nums.iter().zip(&tf.dens) {
    let mut row = Vec::with_capacity(nrow.len());
    for (n, d) in nrow.iter().zip(drow) {
      match NumericElement::new(n, d, &tf.var) {
        Some(e) if e.delay == 0.0 => row.push(e),
        _ => return Ok(None),
      }
    }
    elements.push(row);
  }
  // The input samples from time 0, and the first one reported.
  let (samples, first): (Vec<Vec<f64>>, usize) =
    if let Some((var, t0, t1)) = span {
      let count = ((t1 / period) + 1e-9).floor() as usize + 1;
      let first = ((t0.max(0.0) / period) - 1e-9).ceil().max(0.0) as usize;
      let mut all = Vec::with_capacity(inputs.len());
      for u in inputs {
        let times: Vec<Expr> =
          (0..count).map(|k| Expr::Real(k as f64 * period)).collect();
        let values = ev(&call(
          "Map",
          vec![
            Expr::Function {
              body: Box::new(crate::syntax::substitute_variable(
                u,
                var,
                &Expr::Slot(1),
              )),
            },
            Expr::List(times.into()),
          ],
        ))?;
        let Expr::List(values) = &values else {
          return Ok(None);
        };
        let numbers: Option<Vec<f64>> = values
          .iter()
          .map(crate::functions::math_ast::try_eval_to_f64)
          .collect();
        let Some(numbers) = numbers else {
          return Ok(None);
        };
        all.push(numbers);
      }
      (all, first)
    } else {
      let mut all = Vec::with_capacity(inputs.len());
      for u in inputs {
        let Expr::List(items) = u else {
          return Ok(None);
        };
        let numbers: Option<Vec<f64>> = items
          .iter()
          .map(crate::functions::math_ast::try_eval_to_f64)
          .collect();
        let Some(numbers) = numbers else {
          return Ok(None);
        };
        all.push(numbers);
      }
      (all, 0)
    };
  let length = samples.iter().map(Vec::len).min().unwrap_or(0);
  let mut outputs = Vec::with_capacity(elements.len());
  for row in &elements {
    let mut total = vec![0.0; length];
    for (e, u) in row.iter().zip(&samples) {
      let Some(y) = difference_equation(e, &u[..length]) else {
        return Ok(None);
      };
      for (acc, v) in total.iter_mut().zip(y) {
        *acc += v;
      }
    }
    outputs.push(Expr::List(
      total[first.min(length)..]
        .iter()
        .map(|v| Expr::Real(*v))
        .collect(),
    ));
  }
  Ok(Some(Expr::List(outputs.into())))
}
