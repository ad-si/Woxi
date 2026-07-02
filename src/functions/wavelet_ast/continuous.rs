//! ContinuousWaveletTransform, InverseContinuousWaveletTransform, and the
//! ContinuousWaveletData object.
//!
//! Canonical form:
//!   ContinuousWaveletData[{{oct,voc} -> coef, …}, wave, opts]
//! with opts a list of rules carrying "Voices", "WaveletScale", "DataMean",
//! "SampleRate", and "DataDimensions".
//!
//! Scales follow s(oct, voc) = a 2^(oct-1) 2^(voc/nvoc) with a the smallest
//! resolvable scale (WaveletScale). Coefficients are
//! w(u, s) = 1/Sqrt[s] Sum_k x_k Conjugate[psi]((k - u)/s), the same length
//! as the data.

use super::{ContinuousWaveletSpec, parse_continuous_wavelet, unevaluated};
use crate::InterpreterError;
use crate::syntax::{Expr, expr_to_string};

fn num(e: &Expr) -> Option<f64> {
  crate::functions::math_ast::expr_to_num(e)
}

/// Gamma(n + 1/2) = (2n)! Sqrt[Pi] / (4^n n!)
fn gamma_half(n: u32) -> f64 {
  let mut v = std::f64::consts::PI.sqrt();
  for k in 1..=n {
    v *= k as f64 - 0.5;
  }
  v
}

fn factorial(n: u32) -> f64 {
  (1..=n).map(|k| k as f64).product()
}

/// Probabilists' Hermite polynomial He_n(x).
fn hermite_he(n: u32, x: f64) -> f64 {
  let (mut h0, mut h1) = (1.0, x);
  if n == 0 {
    return h0;
  }
  for k in 1..n {
    let h2 = x * h1 - k as f64 * h0;
    h0 = h1;
    h1 = h2;
  }
  h1
}

/// Numeric wavelet function psi(t) -> (re, im).
pub fn psi_numeric(
  spec: &ContinuousWaveletSpec,
) -> Option<Box<dyn Fn(f64) -> (f64, f64)>> {
  let pi = std::f64::consts::PI;
  match spec {
    ContinuousWaveletSpec::MexicanHat(sigma) => {
      let s = num(sigma)?;
      if s <= 0.0 {
        return None;
      }
      let c = 2.0 / (3.0f64.sqrt() * s.sqrt() * pi.powf(0.25));
      Some(Box::new(move |t| {
        let r = t / s;
        (c * (1.0 - r * r) * (-r * r / 2.0).exp(), 0.0)
      }))
    }
    ContinuousWaveletSpec::Morlet => {
      let sigma = pi * (2.0 / 2.0f64.ln()).sqrt();
      let kappa = (-sigma * sigma / 2.0).exp();
      let c = pi.powf(-0.25);
      Some(Box::new(move |t| {
        (c * (-t * t / 2.0).exp() * ((sigma * t).cos() - kappa), 0.0)
      }))
    }
    ContinuousWaveletSpec::Gabor(omega) => {
      let w = num(omega)?;
      let c = pi.powf(-0.25);
      Some(Box::new(move |t| {
        let a = c * (-t * t / 2.0).exp();
        ((w * t).cos() * a, (w * t).sin() * a)
      }))
    }
    ContinuousWaveletSpec::DGaussian(order) => {
      let n = match order {
        Expr::Integer(n) if *n >= 1 => *n as u32,
        _ => return None,
      };
      let c = -1.0 / gamma_half(n).sqrt();
      Some(Box::new(move |t| {
        (c * hermite_he(n, t) * (-t * t / 2.0).exp(), 0.0)
      }))
    }
    ContinuousWaveletSpec::Paul(order) => {
      let m = match order {
        Expr::Integer(m) if *m >= 1 => *m as u32,
        _ => return None,
      };
      // (2^m i^m m!)/Sqrt[Pi (2m)!] * (1 - i t)^(-(m+1))
      let amp =
        2f64.powi(m as i32) * factorial(m) / (pi * factorial(2 * m)).sqrt();
      // i^m
      let (ire, iim) = match m % 4 {
        0 => (1.0, 0.0),
        1 => (0.0, 1.0),
        2 => (-1.0, 0.0),
        _ => (0.0, -1.0),
      };
      Some(Box::new(move |t| {
        // (1 - i t)^(-(m+1)) via polar form
        let r = (1.0 + t * t).sqrt();
        let theta = (-t).atan2(1.0);
        let p = r.powi(-(m as i32) - 1);
        let ang = -(m as f64 + 1.0) * theta;
        let (zre, zim) = (p * ang.cos(), p * ang.sin());
        (amp * (ire * zre - iim * zim), amp * (ire * zim + iim * zre))
      }))
    }
  }
}

/// In-memory view of a ContinuousWaveletData expression.
pub struct Cwd {
  pub rules: Vec<((i64, i64), Vec<(f64, f64)>)>,
  pub wavelet: Expr,
  pub voices: i64,
  pub wavelet_scale: f64,
  pub data_mean: f64,
  pub sample_rate: f64,
  pub dims: Vec<usize>,
}

fn ov_to_expr(ov: (i64, i64)) -> Expr {
  Expr::List(
    vec![Expr::Integer(ov.0 as i128), Expr::Integer(ov.1 as i128)].into(),
  )
}

fn complex_to_expr(c: (f64, f64)) -> Expr {
  if c.1 == 0.0 {
    Expr::Real(c.0)
  } else {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Complex".to_string(),
      args: vec![Expr::Real(c.0), Expr::Real(c.1)].into(),
    })
    .unwrap_or(Expr::Real(c.0))
  }
}

fn expr_to_complex(e: &Expr) -> Option<(f64, f64)> {
  if let Some(v) = num(e) {
    return Some((v, 0.0));
  }
  if let Expr::FunctionCall { name, args } = e
    && name == "Complex"
    && args.len() == 2
  {
    return Some((num(&args[0])?, num(&args[1])?));
  }
  // Fall back to Re/Im through the evaluator for canonical complex forms.
  let re = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Re".to_string(),
    args: vec![e.clone()].into(),
  })
  .ok()
  .and_then(|r| num(&r))?;
  let im = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Im".to_string(),
    args: vec![e.clone()].into(),
  })
  .ok()
  .and_then(|r| num(&r))?;
  Some((re, im))
}

impl Cwd {
  pub fn octaves(&self) -> i64 {
    self.rules.iter().map(|((o, _), _)| *o).max().unwrap_or(0)
  }

  pub fn scale(&self, oct: i64, voc: i64) -> f64 {
    self.wavelet_scale
      * 2f64.powf((oct - 1) as f64 + voc as f64 / self.voices as f64)
  }

  pub fn to_expr(&self) -> Expr {
    let rules: Vec<Expr> = self
      .rules
      .iter()
      .map(|(ov, coefs)| Expr::Rule {
        pattern: Box::new(ov_to_expr(*ov)),
        replacement: Box::new(Expr::List(
          coefs
            .iter()
            .map(|&c| complex_to_expr(c))
            .collect::<Vec<_>>()
            .into(),
        )),
      })
      .collect();
    let opt = |k: &str, v: Expr| Expr::Rule {
      pattern: Box::new(Expr::String(k.to_string())),
      replacement: Box::new(v),
    };
    let opts = Expr::List(
      vec![
        opt("Voices", Expr::Integer(self.voices as i128)),
        opt("WaveletScale", Expr::Real(self.wavelet_scale)),
        opt("DataMean", Expr::Real(self.data_mean)),
        opt("SampleRate", Expr::Real(self.sample_rate)),
        opt(
          "DataDimensions",
          Expr::List(
            self
              .dims
              .iter()
              .map(|&d| Expr::Integer(d as i128))
              .collect::<Vec<_>>()
              .into(),
          ),
        ),
      ]
      .into(),
    );
    Expr::FunctionCall {
      name: "ContinuousWaveletData".to_string(),
      args: vec![Expr::List(rules.into()), self.wavelet.clone(), opts].into(),
    }
  }

  pub fn from_expr(e: &Expr) -> Option<Cwd> {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    if name != "ContinuousWaveletData" || !(2..=3).contains(&args.len()) {
      return None;
    }
    let Expr::List(rule_items) = &args[0] else {
      return None;
    };
    let mut rules = Vec::new();
    for r in rule_items.iter() {
      let Expr::Rule {
        pattern,
        replacement,
      } = r
      else {
        return None;
      };
      let Expr::List(ov) = pattern.as_ref() else {
        return None;
      };
      if ov.len() != 2 {
        return None;
      }
      let (Expr::Integer(o), Expr::Integer(v)) = (&ov[0], &ov[1]) else {
        return None;
      };
      let Expr::List(coefs) = replacement.as_ref() else {
        return None;
      };
      let mut values = Vec::new();
      for c in coefs.iter() {
        values.push(expr_to_complex(c)?);
      }
      rules.push(((*o as i64, *v as i64), values));
    }
    let wavelet = args[1].clone();
    let mut voices = rules.iter().map(|((_, v), _)| *v).max().unwrap_or(4);
    let mut wavelet_scale = 2.0;
    let mut data_mean = 0.0;
    let mut sample_rate = 1.0;
    let mut dims = vec![rules.first().map_or(0, |(_, c)| c.len())];
    if let Some(Expr::List(opts)) = args.get(2) {
      for o in opts.iter() {
        if let Expr::Rule {
          pattern,
          replacement,
        } = o
          && let Expr::String(k) = pattern.as_ref()
        {
          match k.as_str() {
            "Voices" => {
              if let Some(v) = num(replacement) {
                voices = v as i64;
              }
            }
            "WaveletScale" => {
              if let Some(v) = num(replacement) {
                wavelet_scale = v;
              }
            }
            "DataMean" => {
              if let Some(v) = num(replacement) {
                data_mean = v;
              }
            }
            "SampleRate" => {
              if let Some(v) = num(replacement) {
                sample_rate = v;
              }
            }
            "DataDimensions" => {
              if let Expr::List(ds) = replacement.as_ref() {
                dims = ds
                  .iter()
                  .filter_map(|d| num(d).map(|v| v as usize))
                  .collect();
              }
            }
            _ => {}
          }
        }
      }
    }
    Some(Cwd {
      rules,
      wavelet,
      voices,
      wavelet_scale,
      data_mean,
      sample_rate,
      dims,
    })
  }
}

// ---------------------------------------------------------------------------
// ContinuousWaveletTransform
// ---------------------------------------------------------------------------

pub fn continuous_wavelet_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let fname = "ContinuousWaveletTransform";
  let positional: Vec<&Expr> = args
    .iter()
    .filter(|a| !matches!(a, Expr::Rule { .. }))
    .collect();
  if positional.is_empty() || positional.len() > 3 {
    crate::emit_message(&format!(
      "{fname}::argt: {fname} called with an invalid number of arguments."
    ));
    return Ok(unevaluated(fname, args));
  }
  // Data: 1D numeric list.
  let data: Vec<f64> = match positional[0] {
    Expr::List(items) => {
      let vals: Option<Vec<f64>> = items.iter().map(num).collect();
      match vals {
        Some(v) if v.len() >= 2 => v,
        _ => {
          crate::emit_message(&format!(
            "{fname}::invdata: The data should be a list of at least two real numbers."
          ));
          return Ok(unevaluated(fname, args));
        }
      }
    }
    _ => {
      crate::emit_message(&format!(
        "{fname}::invdata: The data should be a list of at least two real numbers."
      ));
      return Ok(unevaluated(fname, args));
    }
  };
  let default_wavelet = Expr::FunctionCall {
    name: "MexicanHatWavelet".to_string(),
    args: vec![].into(),
  };
  let wavelet_expr = match positional.get(1) {
    None => default_wavelet,
    Some(Expr::Identifier(a)) if a == "Automatic" => default_wavelet,
    Some(w) => (*w).clone(),
  };
  let Some(spec) = parse_continuous_wavelet(&wavelet_expr) else {
    crate::emit_message(&format!(
      "{}::invw: {} is not a valid continuous wavelet.",
      fname,
      expr_to_string(&wavelet_expr)
    ));
    return Ok(unevaluated(fname, args));
  };
  let Some(psi) = psi_numeric(&spec) else {
    crate::emit_message(&format!(
      "{}::invw: The parameters of {} are not numeric.",
      fname,
      expr_to_string(&wavelet_expr)
    ));
    return Ok(unevaluated(fname, args));
  };
  let n = data.len();
  let (noct, nvoc) = match positional.get(2) {
    None => ((((n as f64) / 2.0).log2().floor() as i64).max(1), 4),
    Some(Expr::List(ov)) if ov.len() == 2 => match (num(&ov[0]), num(&ov[1])) {
      (Some(o), Some(v)) if o >= 1.0 && v >= 1.0 => (o as i64, v as i64),
      _ => {
        crate::emit_message(&format!(
          "{}::invov: {} is not a valid octave/voice specification.",
          fname,
          expr_to_string(positional[2])
        ));
        return Ok(unevaluated(fname, args));
      }
    },
    Some(other) => {
      crate::emit_message(&format!(
        "{}::invov: {} is not a valid octave/voice specification.",
        fname,
        expr_to_string(other)
      ));
      return Ok(unevaluated(fname, args));
    }
  };

  let mean = data.iter().sum::<f64>() / n as f64;
  let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();
  let alpha = 2.0;
  let mut rules = Vec::new();
  for oct in 1..=noct {
    for voc in 1..=nvoc {
      let s = alpha * 2f64.powf((oct - 1) as f64 + voc as f64 / nvoc as f64);
      let coefs: Vec<(f64, f64)> = (0..n)
        .map(|u| {
          let mut re = 0.0;
          let mut im = 0.0;
          for (k, &x) in centered.iter().enumerate() {
            let (pr, pi_) = psi((k as f64 - u as f64) / s);
            // Conjugate[psi]
            re += x * pr;
            im -= x * pi_;
          }
          (re / s.sqrt(), im / s.sqrt())
        })
        .collect();
      rules.push(((oct, voc), coefs));
    }
  }
  let cwd = Cwd {
    rules,
    wavelet: wavelet_expr,
    voices: nvoc,
    wavelet_scale: alpha,
    data_mean: mean,
    sample_rate: 1.0,
    dims: vec![n],
  };
  Ok(cwd.to_expr())
}

// ---------------------------------------------------------------------------
// InverseContinuousWaveletTransform
// ---------------------------------------------------------------------------

pub fn inverse_continuous_wavelet_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let fname = "InverseContinuousWaveletTransform";
  let positional: Vec<&Expr> = args
    .iter()
    .filter(|a| !matches!(a, Expr::Rule { .. }))
    .collect();
  if positional.is_empty() || positional.len() > 3 {
    crate::emit_message(&format!(
      "{fname}::argt: {fname} called with an invalid number of arguments."
    ));
    return Ok(unevaluated(fname, args));
  }
  let Some(cwd) = Cwd::from_expr(positional[0]) else {
    crate::emit_message(&format!(
      "{}::invcwd: {} is not a valid ContinuousWaveletData object.",
      fname,
      expr_to_string(positional[0])
    ));
    return Ok(unevaluated(fname, args));
  };
  let wavelet_expr = match positional.get(1) {
    None => cwd.wavelet.clone(),
    Some(Expr::Identifier(a)) if a == "Automatic" => cwd.wavelet.clone(),
    Some(w) => (*w).clone(),
  };
  let Some(spec) = parse_continuous_wavelet(&wavelet_expr) else {
    crate::emit_message(&format!(
      "{}::invw: {} is not a valid continuous wavelet.",
      fname,
      expr_to_string(&wavelet_expr)
    ));
    return Ok(unevaluated(fname, args));
  };
  let Some(psi) = psi_numeric(&spec) else {
    return Ok(unevaluated(fname, args));
  };

  // Coefficient subset.
  let selected: Vec<&((i64, i64), Vec<(f64, f64)>)> = match positional.get(2) {
    None => cwd.rules.iter().collect(),
    Some(Expr::Identifier(a)) if a == "Automatic" || a == "All" => {
      cwd.rules.iter().collect()
    }
    Some(spec_expr) => {
      let winds = select_ovs(&cwd, spec_expr);
      match winds {
        Some(ovs) => cwd
          .rules
          .iter()
          .filter(|(ov, _)| ovs.contains(ov))
          .collect(),
        None => {
          crate::emit_message(&format!(
            "{}::invov: {} is not a valid octave/voice specification.",
            fname,
            expr_to_string(spec_expr)
          ));
          return Ok(unevaluated(fname, args));
        }
      }
    }
  };

  let n = cwd.dims.first().copied().unwrap_or(0);
  if n == 0 || selected.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Least-squares inverse: find x minimizing ||A x - w|| where
  // A[(s,u), k] = Conjugate[psi]((k-u)/s)/Sqrt[s]. Solved via the normal
  // equations with a tiny ridge for numerical stability.
  let mut g = vec![vec![0.0f64; n]; n];
  let mut b = vec![0.0f64; n];
  for ((oct, voc), coefs) in &selected {
    let s = cwd.scale(*oct, *voc);
    // Row block for this scale: rows indexed by u.
    let mut row = vec![(0.0, 0.0); 2 * n - 1]; // psi((k-u)/s) for k-u in -(n-1)..n-1
    for (d, r) in row.iter_mut().enumerate() {
      let t = (d as i64 - (n as i64 - 1)) as f64 / s;
      let (pr, pi_) = psi(t);
      *r = (pr / s.sqrt(), -pi_ / s.sqrt());
    }
    let at = |k: usize, u: usize| row[k + n - 1 - u];
    for u in 0..n {
      let w = coefs.get(u).copied().unwrap_or((0.0, 0.0));
      for k in 0..n {
        let a = at(k, u);
        b[k] += a.0 * w.0 + a.1 * w.1;
        for k2 in k..n {
          let a2 = at(k2, u);
          g[k][k2] += a.0 * a2.0 + a.1 * a2.1;
        }
      }
    }
  }
  for k in 0..n {
    for k2 in 0..k {
      g[k][k2] = g[k2][k];
    }
    g[k][k] += 1e-9 * (1.0 + g[k][k].abs());
  }
  let x = solve_linear(&mut g, &mut b);
  let result: Vec<Expr> =
    x.iter().map(|v| Expr::Real(v + cwd.data_mean)).collect();
  Ok(Expr::List(result.into()))
}

fn solve_linear(g: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
  let n = b.len();
  for col in 0..n {
    // Partial pivoting.
    let mut best = col;
    for r in col + 1..n {
      if g[r][col].abs() > g[best][col].abs() {
        best = r;
      }
    }
    g.swap(col, best);
    b.swap(col, best);
    let pivot = g[col][col];
    if pivot.abs() < 1e-300 {
      continue;
    }
    for r in col + 1..n {
      let f = g[r][col] / pivot;
      if f == 0.0 {
        continue;
      }
      for c in col..n {
        g[r][c] -= f * g[col][c];
      }
      b[r] -= f * b[col];
    }
  }
  let mut x = vec![0.0; n];
  for r in (0..n).rev() {
    let mut acc = b[r];
    for c in r + 1..n {
      acc -= g[r][c] * x[c];
    }
    x[r] = if g[r][r].abs() < 1e-300 {
      0.0
    } else {
      acc / g[r][r]
    };
  }
  x
}

// ---------------------------------------------------------------------------
// cwd[...] accessors
// ---------------------------------------------------------------------------

fn select_ovs(cwd: &Cwd, spec: &Expr) -> Option<Vec<(i64, i64)>> {
  let as_ov = |e: &Expr| -> Option<(i64, i64)> {
    let Expr::List(items) = e else { return None };
    if items.len() != 2 {
      return None;
    }
    match (&items[0], &items[1]) {
      (Expr::Integer(o), Expr::Integer(v)) => Some((*o as i64, *v as i64)),
      _ => None,
    }
  };
  if let Some(ov) = as_ov(spec) {
    return Some(vec![ov]);
  }
  if let Expr::List(items) = spec {
    let ovs: Option<Vec<(i64, i64)>> = items.iter().map(&as_ov).collect();
    if let Some(ovs) = ovs {
      return Some(ovs);
    }
  }
  // Pattern matching against {oct, voc} lists.
  let mut matched = Vec::new();
  for (ov, _) in &cwd.rules {
    if crate::evaluator::pattern_matching::match_pattern(&ov_to_expr(*ov), spec)
      .is_some()
    {
      matched.push(*ov);
    }
  }
  if matched.is_empty() {
    None
  } else {
    Some(matched)
  }
}

pub fn apply_cwd(func: &Expr, args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Some(cwd) = Cwd::from_expr(func) else {
    return Ok(Expr::CurriedCall {
      func: Box::new(func.clone()),
      args: args.to_vec(),
    });
  };
  if args.is_empty() || args.len() > 2 {
    return Ok(Expr::CurriedCall {
      func: Box::new(func.clone()),
      args: args.to_vec(),
    });
  }
  if let Expr::String(prop) = &args[0] {
    return cwd_property(&cwd, prop);
  }
  let form = match args.get(1) {
    Some(Expr::String(f)) => f.as_str(),
    _ => "Rules",
  };
  let ovs: Vec<(i64, i64)> = match &args[0] {
    Expr::Identifier(a) if a == "All" || a == "Automatic" => {
      cwd.rules.iter().map(|(ov, _)| *ov).collect()
    }
    spec => match select_ovs(&cwd, spec) {
      Some(ovs) => ovs,
      None => {
        crate::emit_message(&format!(
          "ContinuousWaveletData::ov: {} is not a valid octave/voice specification.",
          expr_to_string(&args[0])
        ));
        return Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        });
      }
    },
  };
  let single = select_single(&args[0]);
  let mut out = Vec::new();
  for ov in &ovs {
    let Some((_, coefs)) = cwd.rules.iter().find(|(o, _)| o == ov) else {
      continue;
    };
    let values = Expr::List(
      coefs
        .iter()
        .map(|&c| complex_to_expr(c))
        .collect::<Vec<_>>()
        .into(),
    );
    if form == "Rules" {
      out.push(Expr::Rule {
        pattern: Box::new(ov_to_expr(*ov)),
        replacement: Box::new(values),
      });
    } else {
      out.push(values);
    }
  }
  if single && out.len() == 1 && form != "Rules" {
    return Ok(out.pop().unwrap());
  }
  Ok(Expr::List(out.into()))
}

fn select_single(spec: &Expr) -> bool {
  if let Expr::List(items) = spec {
    return items.len() == 2
      && items.iter().all(|i| matches!(i, Expr::Integer(_)));
  }
  false
}

fn cwd_property(cwd: &Cwd, prop: &str) -> Result<Expr, InterpreterError> {
  match prop {
    "Properties" => Ok(Expr::List(
      [
        "DataDimensions",
        "DataMean",
        "Octaves",
        "SampleRate",
        "Scales",
        "Voices",
        "Wavelet",
        "WaveletIndex",
        "WaveletScale",
      ]
      .iter()
      .map(|s| Expr::String(s.to_string()))
      .collect::<Vec<_>>()
      .into(),
    )),
    "Octaves" => Ok(Expr::Integer(cwd.octaves() as i128)),
    "Voices" => Ok(Expr::Integer(cwd.voices as i128)),
    "Wavelet" => Ok(cwd.wavelet.clone()),
    "WaveletScale" => Ok(Expr::Real(cwd.wavelet_scale)),
    "DataMean" => Ok(Expr::Real(cwd.data_mean)),
    "SampleRate" => Ok(Expr::Real(cwd.sample_rate)),
    "DataDimensions" => Ok(Expr::List(
      cwd
        .dims
        .iter()
        .map(|&d| Expr::Integer(d as i128))
        .collect::<Vec<_>>()
        .into(),
    )),
    "WaveletIndex" => Ok(Expr::List(
      cwd
        .rules
        .iter()
        .map(|(ov, _)| ov_to_expr(*ov))
        .collect::<Vec<_>>()
        .into(),
    )),
    "Scales" => Ok(Expr::List(
      cwd
        .rules
        .iter()
        .map(|(ov, _)| Expr::Rule {
          pattern: Box::new(ov_to_expr(*ov)),
          replacement: Box::new(Expr::Real(cwd.scale(ov.0, ov.1))),
        })
        .collect::<Vec<_>>()
        .into(),
    )),
    _ => {
      crate::emit_message(&format!(
        "ContinuousWaveletData::prop: {prop} is not a valid property."
      ));
      Ok(Expr::FunctionCall {
        name: "Missing".to_string(),
        args: vec![Expr::String("NotAvailable".into())].into(),
      })
    }
  }
}

/// WaveletMapIndexed over a ContinuousWaveletData object; None if `e` is
/// not a cwd.
pub fn cwd_map_indexed(
  f: &Expr,
  e: &Expr,
  ovspec: Option<&Expr>,
) -> Result<Option<Expr>, InterpreterError> {
  let Some(cwd) = Cwd::from_expr(e) else {
    return Ok(None);
  };
  let ovs: Vec<(i64, i64)> = match ovspec {
    None => cwd.rules.iter().map(|(ov, _)| *ov).collect(),
    Some(Expr::Identifier(a)) if a == "All" || a == "Automatic" => {
      cwd.rules.iter().map(|(ov, _)| *ov).collect()
    }
    Some(spec) => match select_ovs(&cwd, spec) {
      Some(ovs) => ovs,
      None => return Ok(None),
    },
  };
  let mut new_rules = cwd.rules.clone();
  for (ov, coefs) in new_rules.iter_mut() {
    if !ovs.contains(ov) {
      continue;
    }
    let coef_expr = Expr::List(
      coefs
        .iter()
        .map(|&c| complex_to_expr(c))
        .collect::<Vec<_>>()
        .into(),
    );
    let mapped = crate::evaluator::function_application::apply_curried_call(
      f,
      &[coef_expr, ov_to_expr(*ov)],
    )?;
    if let Expr::List(items) = &mapped {
      let parsed: Option<Vec<(f64, f64)>> =
        items.iter().map(expr_to_complex).collect();
      if let Some(values) = parsed
        && values.len() == coefs.len()
      {
        *coefs = values;
        continue;
      }
    }
    crate::emit_message(
      "WaveletMapIndexed::invres: The function result is not an array of the same dimensions as the coefficients.",
    );
  }
  let new_cwd = Cwd {
    rules: new_rules,
    wavelet: cwd.wavelet.clone(),
    voices: cwd.voices,
    wavelet_scale: cwd.wavelet_scale,
    data_mean: cwd.data_mean,
    sample_rate: cwd.sample_rate,
    dims: cwd.dims.clone(),
  };
  Ok(Some(new_cwd.to_expr()))
}
