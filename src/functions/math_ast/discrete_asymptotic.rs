//! `DiscreteAsymptotic[expr, n -> Infinity]`: the leading term of a sequence
//! as the integer `n` tends to infinity.
//!
//! The leading term of a compound expression is assembled from the leading
//! terms of its parts. Every leading term carries a numeric growth key
//! (`n^(a n) * E^(b n) * n^p * Log[n]^q`) so the terms of a sum can be
//! compared: the dominant one survives and terms of equal growth are added.
//! Elementary functions are rewritten into exponentials and logarithms, and
//! the special sequences (Gamma, Fibonacci, Binomial, ...) use their classical
//! asymptotic formulas. `SeriesCoefficient[f, {z, 0, n}]` is handled by
//! singularity analysis of the nearest simple pole of a meromorphic `f`.

use super::*;
use crate::functions::calculus_ast::is_constant_wrt;
use std::cmp::Ordering;

/// Growth exponents of `n^(nlogn n) E^(exp n) n^pow Log[n]^log`, compared
/// lexicographically. NaN marks a component that is not known numerically.
#[derive(Clone, Copy, Debug)]
struct Growth {
  nlogn: f64,
  exp: f64,
  pow: f64,
  log: f64,
}

impl Growth {
  const ONE: Growth = Growth {
    nlogn: 0.0,
    exp: 0.0,
    pow: 0.0,
    log: 0.0,
  };
  const UNKNOWN: Growth = Growth {
    nlogn: f64::NAN,
    exp: f64::NAN,
    pow: f64::NAN,
    log: f64::NAN,
  };

  fn mul(self, o: Growth) -> Growth {
    Growth {
      nlogn: self.nlogn + o.nlogn,
      exp: self.exp + o.exp,
      pow: self.pow + o.pow,
      log: self.log + o.log,
    }
  }

  fn scale(self, k: f64) -> Growth {
    Growth {
      nlogn: self.nlogn * k,
      exp: self.exp * k,
      pow: self.pow * k,
      log: self.log * k,
    }
  }

  fn components(&self) -> [f64; 4] {
    [self.nlogn, self.exp, self.pow, self.log]
  }

  /// Lexicographic comparison; None when a deciding component is unknown.
  fn compare(&self, o: &Growth) -> Option<Ordering> {
    for (a, b) in self.components().iter().zip(o.components().iter()) {
      if a.is_nan() || b.is_nan() {
        return None;
      }
      if (a - b).abs() > 1e-9 {
        return a.partial_cmp(b);
      }
    }
    Some(Ordering::Equal)
  }

  /// Whether the sequence grows without bound.
  fn tends_to_infinity(&self) -> Option<bool> {
    for c in self.components() {
      if c.is_nan() {
        return None;
      }
      if c.abs() > 1e-9 {
        return Some(c > 0.0);
      }
    }
    Some(false)
  }

  fn is_constant(&self) -> bool {
    self.components().iter().all(|c| c.abs() <= 1e-9)
  }
}

/// A leading term: `expr` with growth `growth`, or the zero sequence.
#[derive(Clone, Debug)]
struct Lead {
  expr: Expr,
  growth: Growth,
  zero: bool,
}

impl Lead {
  fn new(expr: Expr, growth: Growth) -> Lead {
    Lead {
      expr,
      growth,
      zero: false,
    }
  }
}

fn ev(e: &Expr) -> Option<Expr> {
  crate::evaluator::evaluate_expr_to_expr(e).ok()
}

fn ev_call(name: &str, args: &[Expr]) -> Option<Expr> {
  crate::evaluator::evaluate_function_call_ast(name, args).ok()
}

/// Parse a formula template and substitute its placeholder symbols.
fn template(src: &str, bindings: &[(&str, &Expr)]) -> Option<Expr> {
  let parsed = crate::syntax::string_to_expr(src).ok()?;
  Some(crate::syntax::substitute_variables(&parsed, bindings))
}

fn is_zero(e: &Expr) -> bool {
  match e {
    Expr::Integer(0) => true,
    Expr::Real(f) => *f == 0.0,
    _ => false,
  }
}

fn numeric(e: &Expr) -> Option<f64> {
  try_eval_to_f64(e)
    .or_else(|| ev(&call1("N", e.clone())).and_then(|v| try_eval_to_f64(&v)))
}

/// The head and arguments of `e` in full form, translating the parser's
/// binary/unary operator nodes.
fn full_form(e: &Expr) -> Option<(String, Vec<Expr>)> {
  match e {
    Expr::FunctionCall { name, args } => {
      Some((name.clone(), args.iter().cloned().collect()))
    }
    Expr::BinaryOp { op, left, right } => {
      let (l, r) = (*left.clone(), *right.clone());
      Some(match op {
        BinaryOperator::Plus => ("Plus".into(), vec![l, r]),
        BinaryOperator::Minus => ("Plus".into(), vec![l, neg(r)]),
        BinaryOperator::Times => ("Times".into(), vec![l, r]),
        BinaryOperator::Divide => {
          ("Times".into(), vec![l, pow(r, Expr::Integer(-1))])
        }
        BinaryOperator::Power => ("Power".into(), vec![l, r]),
        _ => return None,
      })
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(("Times".into(), vec![Expr::Integer(-1), *operand.clone()])),
    _ => None,
  }
}

/// `e == c var + d` with `c`, `d` free of `var`.
fn linear_parts(e: &Expr, var: &str) -> Option<(Expr, Expr)> {
  let c = ev_call("D", &[e.clone(), id_expr(var)])?;
  if !is_constant_wrt(&c, var) {
    return None;
  }
  let d = ev(&crate::syntax::substitute_variable(
    e,
    var,
    &Expr::Integer(0),
  ))?;
  Some((c, d))
}

/// `e / 2 == var`, the argument shape of `BernoulliB[2 n]` and `EulerE[2 n]`.
fn is_twice_var(e: &Expr, var: &str) -> bool {
  matches!(linear_parts(e, var), Some((Expr::Integer(2), d)) if is_zero(&d))
}

/// Leading term of `expr` as `var -> Infinity`. A sum whose leading terms
/// cancel, or a logarithm of a sequence tending to 1, falls back to the
/// expansion `Series` finds at Infinity.
fn lead(expr: &Expr, var: &str) -> Option<Lead> {
  lead_rules(expr, var).or_else(|| match full_form(expr) {
    Some((h, _)) if h == "Plus" || h == "Log" => lead_by_series(expr, var),
    _ => None,
  })
}

/// The first nonzero term `c n^-k` of `Series[expr, {n, Infinity, order}]`.
fn lead_by_series(expr: &Expr, var: &str) -> Option<Lead> {
  let inf = id_expr("Infinity");
  for order in [4i128, 12] {
    let spec =
      Expr::List(vec![id_expr(var), inf.clone(), Expr::Integer(order)].into());
    let series = ev_call("Series", &[expr.clone(), spec])?;
    if let Some(term) =
      crate::functions::calculus_ast::leading_series_term(&series, var, &inf)
    {
      let term = ev(&term)?;
      let growth = lead_rules(&term, var)?.growth;
      return Some(Lead::new(term, growth));
    }
  }
  None
}

fn lead_rules(expr: &Expr, var: &str) -> Option<Lead> {
  if is_constant_wrt(expr, var) {
    if is_zero(expr) {
      return Some(Lead {
        expr: Expr::Integer(0),
        growth: Growth::ONE,
        zero: true,
      });
    }
    return Some(Lead::new(expr.clone(), Growth::ONE));
  }
  if matches!(expr, Expr::Identifier(n) if n == var) {
    return Some(Lead::new(
      expr.clone(),
      Growth {
        pow: 1.0,
        ..Growth::ONE
      },
    ));
  }
  if let Expr::CurriedCall { func, args } = expr {
    return lead_inactive(func, &args[..], var);
  }
  let (head, args) = full_form(expr)?;
  match (head.as_str(), args.as_slice()) {
    ("Plus", terms) => lead_sum(terms, var),
    ("Times", factors) => {
      let mut out = Lead::new(Expr::Integer(1), Growth::ONE);
      let mut exprs = Vec::new();
      for f in factors {
        let l = lead(f, var)?;
        if l.zero {
          return Some(l);
        }
        out.growth = out.growth.mul(l.growth);
        exprs.push(l.expr);
      }
      out.expr = ev(&times(exprs))?;
      Some(out)
    }
    ("Power", [base, exponent]) => lead_power(base, exponent, var),
    ("Sqrt", [x]) => lead_power(x, &make_rational(1, 2), var),
    ("Exp", [x]) => lead_power(&const_expr("E"), x, var),
    ("Log", [x]) => lead_log(x, var),
    // Hyperbolic and inverse hyperbolic functions in exponential and
    // logarithmic form.
    ("Sinh" | "Cosh" | "Tanh" | "Coth" | "Sech" | "Csch", [x]) => {
      let src = match head.as_str() {
        "Sinh" => "(E^daX$ - E^(-daX$))/2",
        "Cosh" => "(E^daX$ + E^(-daX$))/2",
        "Tanh" => "(E^daX$ - E^(-daX$))/(E^daX$ + E^(-daX$))",
        "Coth" => "(E^daX$ + E^(-daX$))/(E^daX$ - E^(-daX$))",
        "Sech" => "2/(E^daX$ + E^(-daX$))",
        _ => "2/(E^daX$ - E^(-daX$))",
      };
      lead(&template(src, &[("daX$", x)])?, var)
    }
    ("ArcSinh", [x]) => {
      lead_log(&template("daX$ + Sqrt[daX$^2 + 1]", &[("daX$", x)])?, var)
    }
    ("ArcCosh", [x]) => {
      lead_log(&template("daX$ + Sqrt[daX$^2 - 1]", &[("daX$", x)])?, var)
    }
    ("Factorial", [x]) => {
      lead_gamma(&plus(vec![x.clone(), Expr::Integer(1)]), var, 1)
    }
    ("Gamma", [x]) => lead_gamma(x, var, 1),
    ("Pochhammer", [a, x]) if is_constant_wrt(a, var) => lead(
      &times(vec![
        call1("Gamma", plus(vec![a.clone(), x.clone()])),
        pow(call1("Gamma", a.clone()), Expr::Integer(-1)),
      ]),
      var,
    ),
    ("FactorialPower", [a, x]) => lead(
      &template(
        "Gamma[1 + daA$]/Gamma[1 + daA$ - daX$]",
        &[("daA$", a), ("daX$", x)],
      )?,
      var,
    ),
    ("Binomial", [x, k]) => lead_binomial(x, k, var),
    ("HarmonicNumber" | "PolyGamma", [x]) => lead_log(x, var),
    ("PolyGamma", [Expr::Integer(0), x]) => lead_log(x, var),
    ("Zeta", [x]) => {
      let l = lead(x, var)?;
      (l.growth.tends_to_infinity()?
        && numeric(&l.expr).is_none_or(|v| v > 0.0))
      .then(|| Lead::new(Expr::Integer(1), Growth::ONE))
    }
    ("Prime", [x]) => {
      lead(&times(vec![x.clone(), call1("Log", x.clone())]), var)
    }
    ("Fibonacci" | "LucasL", [x]) => {
      linear_parts(x, var)?;
      let src = if head == "Fibonacci" {
        "GoldenRatio^daX$/Sqrt[5]"
      } else {
        "GoldenRatio^daX$"
      };
      lead(&template(src, &[("daX$", x)])?, var)
    }
    // Bell numbers: E^(n/W - n - 1) (n/W)^(n + 1/2) / Sqrt[n], W = ProductLog[n].
    ("BellB", [x]) if matches!(x, Expr::Identifier(n) if n == var) => {
      let e = template(
        "E^(-1 - daN$ + daN$/ProductLog[daN$]) (daN$/ProductLog[daN$])^(1/2 + daN$)/Sqrt[daN$]",
        &[("daN$", x)],
      )?;
      Some(Lead::new(
        ev(&e)?,
        Growth {
          nlogn: 1.0,
          ..Growth::UNKNOWN
        },
      ))
    }
    // B[2n] ~ (-1)^(n+1) 4 Sqrt[Pi n] (n/(Pi E))^(2n).
    ("BernoulliB", [x]) if is_twice_var(x, var) => {
      let e = template(
        "4 (-1)^(-1 + daN$) E^(-2 daN$) daN$^(1/2 + 2 daN$) Pi^(1/2 - 2 daN$)",
        &[("daN$", &id_expr(var))],
      )?;
      Some(Lead::new(
        ev(&e)?,
        Growth {
          nlogn: 2.0,
          exp: -2.0 - 2.0 * std::f64::consts::PI.ln(),
          pow: 0.5,
          log: 0.0,
        },
      ))
    }
    // E[2n] ~ (-1)^n 8 Sqrt[n/Pi] (4 n/(Pi E))^(2n).
    ("EulerE", [x]) if is_twice_var(x, var) => {
      let e = template(
        "(-1)^daN$ 2^(3 + 4 daN$) E^(-2 daN$) daN$^(1/2 + 2 daN$) Pi^(-1/2 - 2 daN$)",
        &[("daN$", &id_expr(var))],
      )?;
      Some(Lead::new(
        ev(&e)?,
        Growth {
          nlogn: 2.0,
          exp: 4.0 * 2f64.ln() - 2.0 - 2.0 * std::f64::consts::PI.ln(),
          pow: 0.5,
          log: 0.0,
        },
      ))
    }
    ("SeriesCoefficient", [f, Expr::List(spec)]) => {
      lead_series_coefficient(f, spec, var)
    }
    _ => None,
  }
}

/// `Inactive[h][args]`: activate it, and use the result when `h` evaluated.
fn lead_inactive(func: &Expr, args: &[Expr], var: &str) -> Option<Lead> {
  let Expr::FunctionCall { name, args: inner } = func else {
    return None;
  };
  if name != "Inactive" || inner.len() != 1 {
    return None;
  }
  let Expr::Identifier(h) = &inner[0] else {
    return None;
  };
  if h == "SeriesCoefficient"
    && let [f, Expr::List(spec)] = args
  {
    return lead_series_coefficient(f, spec, var);
  }
  let active = ev(&call(h, args.to_vec()))?;
  if matches!(&active, Expr::FunctionCall { name, .. } if name == h) {
    return None;
  }
  lead(&active, var)
}

/// The dominant term of a sum; terms of equal growth are added.
fn lead_sum(terms: &[Expr], var: &str) -> Option<Lead> {
  let mut best: Option<Lead> = None;
  let mut tied: Vec<Expr> = Vec::new();
  for t in terms {
    let l = lead(t, var)?;
    if l.zero {
      continue;
    }
    match &best {
      None => {
        tied = vec![l.expr.clone()];
        best = Some(l);
      }
      Some(b) => match l.growth.compare(&b.growth)? {
        Ordering::Greater => {
          tied = vec![l.expr.clone()];
          best = Some(l);
        }
        Ordering::Equal => tied.push(l.expr),
        Ordering::Less => {}
      },
    }
  }
  let Some(mut best) = best else {
    return Some(Lead {
      expr: Expr::Integer(0),
      growth: Growth::ONE,
      zero: true,
    });
  };
  if tied.len() > 1 {
    best.expr = ev(&plus(tied))?;
    // The leading terms cancel: the next order would be needed.
    if is_zero(&best.expr) {
      return None;
    }
  }
  Some(best)
}

fn lead_power(base: &Expr, exponent: &Expr, var: &str) -> Option<Lead> {
  let base_const = is_constant_wrt(base, var);
  if is_constant_wrt(exponent, var) {
    let l = lead(base, var)?;
    if l.zero {
      return None;
    }
    let k = numeric(exponent).unwrap_or(f64::NAN);
    let raised = ev(&pow(l.expr, exponent.clone()))?;
    let expanded = ev_call("PowerExpand", &[raised])?;
    return Some(Lead::new(expanded, l.growth.scale(k)));
  }
  let (c, d) = linear_parts(exponent, var)?;
  let c_f = numeric(&c).unwrap_or(f64::NAN);
  if base_const {
    // b^(c n + d): exponential growth Log[|b|] c.
    let log_abs =
      numeric(&call1("Log", call1("Abs", base.clone()))).unwrap_or(f64::NAN);
    let expr = ev(&pow(base.clone(), exponent.clone()))?;
    return Some(Lead::new(
      expr,
      Growth {
        exp: c_f * log_abs,
        ..Growth::ONE
      },
    ));
  }
  // n^(c n + d)
  if matches!(base, Expr::Identifier(n) if n == var) {
    let d_f = numeric(&d).unwrap_or(f64::NAN);
    return Some(Lead::new(
      ev(&pow(base.clone(), exponent.clone()))?,
      Growth {
        nlogn: c_f,
        pow: d_f,
        ..Growth::ONE
      },
    ));
  }
  None
}

fn lead_log(x: &Expr, var: &str) -> Option<Lead> {
  if matches!(x, Expr::Identifier(n) if n == var) {
    return Some(Lead::new(
      call1("Log", x.clone()),
      Growth {
        log: 1.0,
        ..Growth::ONE
      },
    ));
  }
  let l = lead(x, var)?;
  if l.zero {
    return None;
  }
  if l.growth.is_constant() {
    // Log of a sequence tending to a constant c: Log[c], unless c == 1.
    if !is_constant_wrt(&l.expr, var) || matches!(l.expr, Expr::Integer(1)) {
      return None;
    }
    return Some(Lead::new(ev(&call1("Log", l.expr))?, Growth::ONE));
  }
  if !l.growth.tends_to_infinity()? {
    return None;
  }
  // Log of the leading term, expanded into a sum of simpler pieces.
  let expanded = ev_call("PowerExpand", &[ev(&call1("Log", l.expr.clone()))?])?;
  if let Some((h, a)) = full_form(&expanded)
    && h == "Log"
    && a.len() == 1
    && !matches!(&a[0], Expr::Identifier(n) if n == var)
  {
    return None;
  }
  lead(&expanded, var)
}

/// Stirling: Gamma[c1 n + c0] ~ Sqrt[2 Pi] (c1 n)^(c1 n + c0 - 1/2) E^(-c1 n),
/// times the correction series to `goal` terms when `c1 == 1`; for
/// `c1 == -1`, the reflection formula reduces to the former.
fn lead_gamma(x: &Expr, var: &str, goal: usize) -> Option<Lead> {
  let (c1, c0) = linear_parts(x, var)?;
  let c1_f = numeric(&c1)?;
  let c0_f = numeric(&c0).unwrap_or(f64::NAN);
  let n = id_expr(var);
  if c1_f > 0.0 {
    let mut e = template(
      "Sqrt[2 Pi] (daC1$ daN$)^(daC1$ daN$ + daC0$ - 1/2) E^(-daC1$ daN$)",
      &[("daC1$", &c1), ("daC0$", &c0), ("daN$", &n)],
    )?;
    if goal > 1 {
      if !matches!(c1, Expr::Integer(1)) {
        return None;
      }
      e = times(vec![e, stirling_correction(&c0, var, goal)?]);
    }
    let e = ev_call("PowerExpand", &[ev(&e)?])?;
    let ln_c1 = c1_f.ln();
    return Some(Lead::new(
      e,
      Growth {
        nlogn: c1_f,
        exp: c1_f * ln_c1 - c1_f,
        pow: c0_f - 0.5,
        log: 0.0,
      },
    ));
  }
  if c1_f < 0.0 && goal == 1 {
    // Gamma[x] = Pi/(Sin[Pi x] Gamma[1 - x]).
    let reflected =
      lead_gamma(&plus(vec![Expr::Integer(1), neg(x.clone())]), var, 1)?;
    // Pi/Sqrt[2 Pi] = Sqrt[Pi/2], kept together as one radical.
    let rest = ev(&times(vec![
      reflected.expr,
      pow(
        make_sqrt(times(vec![Expr::Integer(2), const_expr("Pi")])),
        Expr::Integer(-1),
      ),
    ]))?;
    let e = template(
      "Sqrt[Pi/2] / (Sin[Pi daX$] daR$)",
      &[("daX$", x), ("daR$", &rest)],
    )?;
    return Some(Lead::new(ev(&e)?, Growth::UNKNOWN));
  }
  None
}

/// The Stirling correction factor of Gamma[n + c] to `goal` terms:
/// Exp[Sum_k (-1)^(k+1) BernoulliB[k+1, c]/(k (k+1) n^k)] in powers of 1/n.
fn stirling_correction(c: &Expr, var: &str, goal: usize) -> Option<Expr> {
  let t = id_expr("daT$");
  let order = (goal - 1) as i128;
  let mut terms = Vec::new();
  for k in 1..=order {
    terms.push(template(
      "(-1)^(daK$ + 1) BernoulliB[daK$ + 1, daC$] daT$^daK$/(daK$ (daK$ + 1))",
      &[("daK$", &Expr::Integer(k)), ("daC$", c), ("daT$", &t)],
    )?);
  }
  let series = ev_call(
    "Series",
    &[
      call1("Exp", plus(terms)),
      Expr::List(
        vec![t.clone(), Expr::Integer(0), Expr::Integer(order)].into(),
      ),
    ],
  )?;
  let normal = ev_call("Normal", &[series])?;
  ev(&crate::syntax::substitute_variable(
    &normal,
    "daT$",
    &pow(id_expr(var), Expr::Integer(-1)),
  ))
}

fn lead_binomial(x: &Expr, k: &Expr, var: &str) -> Option<Lead> {
  // Binomial[x, k] ~ x^k/k! for fixed k.
  if is_constant_wrt(k, var) {
    let l = lead(x, var)?;
    if !l.growth.tends_to_infinity()? {
      return None;
    }
    let kf = numeric(k).unwrap_or(f64::NAN);
    let e = ev(&times(vec![
      ev_call("PowerExpand", &[pow(l.expr, k.clone())])?,
      pow(call1("Factorial", k.clone()), Expr::Integer(-1)),
    ]))?;
    return Some(Lead::new(e, l.growth.scale(kf)));
  }
  // Binomial[a n, b n] ~ (a^a/(b^b (a-b)^(a-b)))^n Sqrt[a/(2 b (a-b))]/Sqrt[Pi n].
  let (a, a0) = linear_parts(x, var)?;
  let (b, b0) = linear_parts(k, var)?;
  if !is_zero(&a0) || !is_zero(&b0) {
    return None;
  }
  let (af, bf) = (numeric(&a)?, numeric(&b)?);
  if !(af > bf && bf > 0.0) {
    return None;
  }
  let base = ev(&template(
    "daA$^daA$/(daB$^daB$ (daA$ - daB$)^(daA$ - daB$))",
    &[("daA$", &a), ("daB$", &b)],
  )?)?;
  let coef = ev(&template(
    "Sqrt[daA$/(2 daB$ (daA$ - daB$))]",
    &[("daA$", &a), ("daB$", &b)],
  )?)?;
  let n = id_expr(var);
  let e = ev(&times(vec![
    pow(base.clone(), n.clone()),
    coef,
    pow(make_sqrt(n), Expr::Integer(-1)),
    pow(make_sqrt(const_expr("Pi")), Expr::Integer(-1)),
  ]))?;
  Some(Lead::new(
    e,
    Growth {
      exp: numeric(&base)?.ln(),
      pow: -0.5,
      ..Growth::ONE
    },
  ))
}

/// Whether `e` is entire in `z`: built from constants and `z` by sums,
/// products, nonnegative integer powers and exponentials.
fn is_entire(e: &Expr, z: &str) -> bool {
  if is_constant_wrt(e, z) {
    return true;
  }
  if matches!(e, Expr::Identifier(n) if n == z) {
    return true;
  }
  let Some((h, a)) = full_form(e) else {
    return false;
  };
  match (h.as_str(), a.as_slice()) {
    ("Plus" | "Times", xs) => xs.iter().all(|x| is_entire(x, z)),
    ("Power", [b, Expr::Integer(k)]) if *k >= 0 => is_entire(b, z),
    ("Power", [b, x]) if is_constant_wrt(b, z) => is_entire(x, z),
    ("Exp" | "Sin" | "Cos" | "Sinh" | "Cosh", [x]) => is_entire(x, z),
    _ => false,
  }
}

/// `SeriesCoefficient[g/h, {z, 0, n}]` for entire `g`, `h`: the nearest
/// zero `z0` of `h`, if it is unique and a simple pole, contributes
/// `-g(z0)/(h'(z0) z0) (1/z0)^n`.
fn lead_series_coefficient(f: &Expr, spec: &[Expr], var: &str) -> Option<Lead> {
  let [Expr::Identifier(z), z0, nn] = spec else {
    return None;
  };
  if !is_zero(z0) || !matches!(nn, Expr::Identifier(n) if n == var) {
    return None;
  }
  if !is_constant_wrt(f, var) {
    return None;
  }
  let together = ev_call("Together", std::slice::from_ref(f))?;
  let g = ev_call("Numerator", std::slice::from_ref(&together))?;
  let h = ev_call("Denominator", &[together])?;
  if is_constant_wrt(&h, z) || !is_entire(&g, z) || !is_entire(&h, z) {
    return None;
  }
  let eq = Expr::Comparison {
    operands: vec![h.clone(), Expr::Integer(0)],
    operators: vec![ComparisonOp::Equal],
  };
  let sols = ev_call("Solve", &[eq, id_expr(z)])?;
  let Expr::List(sols) = &sols else {
    return None;
  };
  // Candidate roots with their moduli; a ConditionalExpression family is
  // sampled at its principal member and its neighbours.
  let mut roots: Vec<(Expr, f64)> = Vec::new();
  for s in sols {
    let Expr::List(rules) = s else {
      return None;
    };
    let [Expr::Rule { replacement, .. }] = rules.as_slice() else {
      return None;
    };
    let r = replacement.as_ref();
    let members: Vec<Expr> = match full_form(r) {
      Some((hd, a)) if hd == "ConditionalExpression" && a.len() == 2 => {
        let c1 = call1("C", Expr::Integer(1));
        [-1i128, 0, 1]
          .iter()
          .map(|k| {
            ev_call(
              "ReplaceAll",
              &[a[0].clone(), rule_expr(c1.clone(), Expr::Integer(*k))],
            )
          })
          .collect::<Option<Vec<_>>>()?
      }
      _ => vec![r.clone()],
    };
    for m in members {
      let m = ev(&m)?;
      let modulus = numeric(&call1("Abs", m.clone()))?;
      roots.push((m, modulus));
    }
  }
  roots.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
  let (root, modulus) = roots.first()?.clone();
  if modulus < 1e-12 || roots.get(1).is_some_and(|r| r.1 - modulus < 1e-9) {
    return None;
  }
  let at = |e: &Expr| ev(&crate::syntax::substitute_variable(e, z, &root));
  let dh = ev_call("D", &[h.clone(), id_expr(z)])?;
  let (g0, dh0) = (at(&g)?, at(&dh)?);
  if numeric(&call1("Abs", g0.clone()))? < 1e-12
    || numeric(&call1("Abs", dh0.clone()))? < 1e-12
  {
    return None;
  }
  // -g/(z h') is cancelled as a function of z before z0 is substituted, so
  // a factor z0 shared with g drops out exactly.
  let ratio = ev_call(
    "Cancel",
    &[ev(&times(vec![
      Expr::Integer(-1),
      g.clone(),
      pow(dh.clone(), Expr::Integer(-1)),
      pow(id_expr(z), Expr::Integer(-1)),
    ]))?],
  )?;
  let coef = ev_call("Simplify", &[at(&ratio)?])?;
  let rho = ev(&pow(root, Expr::Integer(-1)))?;
  let e = ev(&times(vec![coef, positive_power(&rho, &id_expr(var))?]))?;
  Some(Lead::new(
    e,
    Growth {
      exp: -modulus.ln(),
      ..Growth::ONE
    },
  ))
}

/// `rho^n`, folding `(b^e)^n` into `b^(e n)` for a positive numeric base `b`
/// and a real exponent `e`.
fn positive_power(rho: &Expr, n: &Expr) -> Option<Expr> {
  if let Some((h, a)) = full_form(rho)
    && h == "Power"
    && let [b, e] = a.as_slice()
    && numeric(b).is_some_and(|v| v > 0.0)
    && numeric(e).is_some()
  {
    return ev(&pow(b.clone(), times(vec![e.clone(), n.clone()])));
  }
  ev(&pow(rho.clone(), n.clone()))
}

/// Combine several exponential factors `r^(c n + d)` with positive rational
/// bases `r` into one `R^n` times a constant: `4^n 2^-n` is `2^n`.
fn merge_rational_exponentials(e: Expr, var: &str) -> Option<Expr> {
  let Some((h, factors)) = full_form(&e) else {
    return Some(e);
  };
  if h != "Times" {
    return Some(e);
  }
  let is_pos_rational = |b: &Expr| match b {
    Expr::Integer(k) => *k > 0,
    Expr::FunctionCall { name, args } if name == "Rational" => {
      matches!(&args[0], Expr::Integer(k) if *k > 0)
    }
    _ => false,
  };
  let mut base_factors = Vec::new();
  let mut constant_factors = Vec::new();
  let mut rest = Vec::new();
  for f in factors {
    if let Some((fh, fa)) = full_form(&f)
      && fh == "Power"
      && let [b, x] = fa.as_slice()
      && is_pos_rational(b)
      && !is_constant_wrt(x, var)
      && let Some((c, d)) = linear_parts(x, var)
    {
      base_factors.push(pow(b.clone(), c));
      constant_factors.push(pow(b.clone(), d));
    } else {
      rest.push(f);
    }
  }
  if base_factors.len() < 2 {
    return Some(e);
  }
  let base = ev(&times(base_factors))?;
  rest.push(pow(base, id_expr(var)));
  rest.extend(constant_factors);
  ev(&times(rest))
}

/// Leading term of `expr`, with `goal` terms of its expansion for the
/// Gamma-type sequences.
fn discrete_asymptotic(expr: &Expr, var: &str, goal: usize) -> Option<Expr> {
  let result = if goal > 1 {
    // c Gamma[n + c0]-type products: the correction series multiplies the
    // single growing factor.
    let (h, factors) = full_form(expr).unwrap_or((String::new(), vec![]));
    let factors = if h == "Times" {
      factors
    } else {
      vec![expr.clone()]
    };
    let mut constant = Vec::new();
    let mut growing = None;
    for f in factors {
      if is_constant_wrt(&f, var) {
        constant.push(f);
      } else if growing.is_none() {
        growing = Some(f);
      } else {
        return None;
      }
    }
    let gamma_arg = match full_form(&growing?)? {
      (h, a) if h == "Gamma" && a.len() == 1 => a[0].clone(),
      (h, a) if h == "Factorial" && a.len() == 1 => {
        plus(vec![a[0].clone(), Expr::Integer(1)])
      }
      (h, a)
        if h == "Pochhammer" && a.len() == 2 && is_constant_wrt(&a[0], var) =>
      {
        constant.push(pow(call1("Gamma", a[0].clone()), Expr::Integer(-1)));
        plus(vec![a[0].clone(), a[1].clone()])
      }
      _ => return None,
    };
    let l = lead_gamma(&ev(&gamma_arg)?, var, goal)?;
    constant.push(l.expr);
    ev(&times(constant))?
  } else {
    let l = lead(expr, var)?;
    if l.zero { Expr::Integer(0) } else { l.expr }
  };
  merge_rational_exponentials(result, var)
}

/// `DiscreteAsymptotic[expr, n -> Infinity]`,
/// `DiscreteAsymptotic[expr, n -> Infinity, SeriesTermGoal -> m]` and
/// `DiscreteAsymptotic[expr, {n, Infinity, m}]`.
pub fn discrete_asymptotic_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DiscreteAsymptotic", args));
  if args.len() < 2 || args.len() > 3 {
    return unevaluated();
  }
  let is_infinity =
    |e: &Expr| matches!(e, Expr::Identifier(s) if s == "Infinity");
  let positive_goal = |e: &Expr| match e {
    Expr::Integer(m) if *m >= 1 => Some(*m as usize),
    _ => None,
  };
  let (var, mut goal) = match &args[1] {
    Expr::Rule {
      pattern,
      replacement,
    } => match (pattern.as_ref(), replacement.as_ref()) {
      (Expr::Identifier(s), inf) if is_infinity(inf) => (s.clone(), 1),
      _ => return unevaluated(),
    },
    Expr::List(items) if items.len() == 3 => match items.as_slice() {
      [Expr::Identifier(s), inf, m] if is_infinity(inf) => {
        match positive_goal(m) {
          Some(m) => (s.clone(), m),
          None => return unevaluated(),
        }
      }
      _ => return unevaluated(),
    },
    _ => return unevaluated(),
  };
  if let Some(opt) = args.get(2) {
    match opt {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "SeriesTermGoal") => {
        match positive_goal(replacement) {
          Some(m) => goal = m,
          None => return unevaluated(),
        }
      }
      _ => return unevaluated(),
    }
  }
  // Intermediate Solve/Series steps may warn; the answer does not depend
  // on them.
  let snapshot = crate::snapshot_warnings();
  let result = discrete_asymptotic(&args[0], &var, goal);
  crate::restore_warnings(snapshot);
  match result {
    Some(r) => Ok(r),
    None => unevaluated(),
  }
}
