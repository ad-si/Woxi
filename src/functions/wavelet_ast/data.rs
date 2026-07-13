//! The DiscreteWaveletData object and the discrete-transform entry points:
//! DiscreteWaveletTransform, StationaryWaveletTransform,
//! DiscreteWaveletPacketTransform, StationaryWaveletPacketTransform,
//! LiftingWaveletTransform, InverseWaveletTransform, WaveletThreshold,
//! WaveletMapIndexed, and WaveletBestBasis.
//!
//! Canonical form:
//!   DiscreteWaveletData[{wind1 -> coef1, …}, wave, wtrans, dims]
//! where wtrans is the transform name string, or {name, opt -> val, …} when
//! extra state is carried (non-default Padding, a best-basis BasisIndex,
//! threshold values).

use std::collections::BTreeMap;

use super::transforms::{
  CoefArray, Padding, TransformKind, basis_index, default_refinement,
  forward_transform, inverse_transform, node_dims,
};
use super::{WaveletSpec, parse_discrete_wavelet, unevaluated};
use crate::InterpreterError;
use crate::syntax::{Expr, expr_to_string};

fn num(e: &Expr) -> Option<f64> {
  crate::functions::math_ast::expr_to_num(e)
}

pub fn wind_to_expr(wind: &[u8]) -> Expr {
  Expr::List(
    wind
      .iter()
      .map(|&d| Expr::Integer(d as i128))
      .collect::<Vec<_>>()
      .into(),
  )
}

fn expr_to_wind(e: &Expr) -> Option<Vec<u8>> {
  let Expr::List(items) = e else { return None };
  let mut wind = Vec::new();
  for i in items.iter() {
    match i {
      Expr::Integer(d) if (0..=7).contains(d) => wind.push(*d as u8),
      _ => return None,
    }
  }
  Some(wind)
}

fn dims_to_expr(dims: &[usize]) -> Expr {
  Expr::List(
    dims
      .iter()
      .map(|&d| Expr::Integer(d as i128))
      .collect::<Vec<_>>()
      .into(),
  )
}

/// In-memory view of a DiscreteWaveletData expression.
pub struct Dwd {
  /// wind -> coefficient array expression (kept as Expr so exact symbolic
  /// coefficients survive round trips)
  pub rules: Vec<(Vec<u8>, Expr)>,
  pub wavelet: Expr,
  pub kind: TransformKind,
  pub padding: Padding,
  pub basis_override: Option<Vec<Vec<u8>>>,
  pub threshold_values: Option<Expr>,
  pub dims: Vec<usize>,
}

impl Dwd {
  pub fn refinement(&self) -> usize {
    self.rules.iter().map(|(w, _)| w.len()).max().unwrap_or(0)
  }

  pub fn rank(&self) -> usize {
    self.dims.len()
  }

  pub fn basis(&self) -> Vec<Vec<u8>> {
    if let Some(b) = &self.basis_override {
      return b.clone();
    }
    let stored: std::collections::BTreeSet<&Vec<u8>> =
      self.rules.iter().map(|(w, _)| w).collect();
    basis_index(self.kind, self.refinement(), self.rank())
      .into_iter()
      .filter(|w| stored.contains(w))
      .collect()
  }

  pub fn coef(&self, wind: &[u8]) -> Option<&Expr> {
    self
      .rules
      .iter()
      .find(|(w, _)| w.as_slice() == wind)
      .map(|(_, c)| c)
  }

  pub fn to_expr(&self) -> Expr {
    let rules: Vec<Expr> = self
      .rules
      .iter()
      .map(|(w, c)| Expr::Rule {
        pattern: Box::new(wind_to_expr(w)),
        replacement: Box::new(c.clone()),
      })
      .collect();
    let mut extras: Vec<Expr> = Vec::new();
    if self.padding != Padding::Periodic {
      extras.push(Expr::Rule {
        pattern: Box::new(Expr::String("Padding".into())),
        replacement: Box::new(self.padding.to_expr()),
      });
    }
    if let Some(basis) = &self.basis_override {
      extras.push(Expr::Rule {
        pattern: Box::new(Expr::String("BasisIndex".into())),
        replacement: Box::new(Expr::List(
          basis
            .iter()
            .map(|w| wind_to_expr(w))
            .collect::<Vec<_>>()
            .into(),
        )),
      });
    }
    if let Some(tv) = &self.threshold_values {
      extras.push(Expr::Rule {
        pattern: Box::new(Expr::String("ThresholdValues".into())),
        replacement: Box::new(tv.clone()),
      });
    }
    let name = Expr::String(self.kind.name().to_string());
    let wtrans = if extras.is_empty() {
      name
    } else {
      let mut items = vec![name];
      items.extend(extras);
      Expr::List(items.into())
    };
    Expr::FunctionCall {
      name: "DiscreteWaveletData".to_string(),
      args: vec![
        Expr::List(rules.into()),
        self.wavelet.clone(),
        wtrans,
        dims_to_expr(&self.dims),
      ]
      .into(),
    }
  }

  /// Parse a DiscreteWaveletData[…] expression (3 or 4 args).
  pub fn from_expr(e: &Expr) -> Option<Dwd> {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    if name != "DiscreteWaveletData" || !(3..=4).contains(&args.len()) {
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
      rules.push((expr_to_wind(pattern)?, replacement.as_ref().clone()));
    }
    let wavelet = args[1].clone();
    let (kind, padding, basis_override, threshold_values) =
      parse_wtrans(&args[2])?;
    let dims: Vec<usize> = if args.len() == 4 {
      let Expr::List(ds) = &args[3] else {
        return None;
      };
      let mut dims = Vec::new();
      for d in ds.iter() {
        match d {
          Expr::Integer(n) if *n > 0 => dims.push(*n as usize),
          _ => return None,
        }
      }
      dims
    } else {
      // Infer the data dimensions from the coarsest stored coefficients:
      // each level roughly doubles the length.
      let (w, c) = rules.iter().min_by_key(|(w, _)| w.len())?;
      let arr = CoefArray::from_expr(c)?;
      arr.dims().iter().map(|&d| d << w.len().max(1)).collect()
    };
    Some(Dwd {
      rules,
      wavelet,
      kind,
      padding,
      basis_override,
      threshold_values,
      dims,
    })
  }
}

fn parse_wtrans(
  e: &Expr,
) -> Option<(TransformKind, Padding, Option<Vec<Vec<u8>>>, Option<Expr>)> {
  match e {
    Expr::String(s) => {
      TransformKind::from_name(s).map(|k| (k, Padding::Periodic, None, None))
    }
    Expr::List(items) if !items.is_empty() => {
      let Expr::String(s) = &items[0] else {
        return None;
      };
      let kind = TransformKind::from_name(s)?;
      let mut padding = Padding::Periodic;
      let mut basis = None;
      let mut thresh = None;
      for item in items.iter().skip(1) {
        let Expr::Rule {
          pattern,
          replacement,
        } = item
        else {
          continue;
        };
        if let Expr::String(key) = pattern.as_ref() {
          match key.as_str() {
            "Padding" => padding = Padding::parse(replacement)?,
            "BasisIndex" => {
              if let Expr::List(winds) = replacement.as_ref() {
                let mut b = Vec::new();
                for w in winds.iter() {
                  b.push(expr_to_wind(w)?);
                }
                basis = Some(b);
              }
            }
            "ThresholdValues" => thresh = Some(replacement.as_ref().clone()),
            _ => {}
          }
        }
      }
      Some((kind, padding, basis, thresh))
    }
    _ => None,
  }
}

// ---------------------------------------------------------------------------
// Symbolic (exact) 1D transform steps
// ---------------------------------------------------------------------------

fn sqrt2() -> Expr {
  Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![Expr::Integer(2)].into(),
  }
}

fn eval(e: Expr) -> Expr {
  crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e)
}

/// Sqrt[2] * Sum coef_i * x_i, evaluated to canonical form.
fn symbolic_dot(pairs: Vec<(Expr, Expr)>) -> Expr {
  let terms: Vec<Expr> = pairs
    .into_iter()
    .map(|(c, x)| Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![c, x].into(),
    })
    .collect();
  eval(Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      sqrt2(),
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      },
    ]
    .into(),
  })
}

/// Exact filters of a wavelet as (index, Expr) pairs, if available.
fn exact_filters(
  spec: &WaveletSpec,
) -> Option<(Vec<(i64, Expr)>, Vec<(i64, Expr)>)> {
  let f = super::wavelet_filters(spec)?;
  let dual = f.dual_lo_exact.clone()?;
  let primal = f.primal_lo_exact.clone()?;
  Some((primal, dual))
}

/// One symbolic decimated step over a periodically-extended list, mirroring
/// `dwt_step_1d` (same output length and offsets).
fn symbolic_dwt_step(x: &[Expr], filter: &[(i64, Expr)]) -> Vec<Expr> {
  // The symbolic path is periodic-only: pad odd lengths by one wrapped
  // sample, then run the canonical circular transform.
  let mut x = x.to_vec();
  if x.len() % 2 == 1 {
    x.push(x[0].clone());
  }
  let n = x.len() as i64;
  (0..n / 2)
    .map(|t| {
      symbolic_dot(
        filter
          .iter()
          .map(|(i, c)| {
            (c.clone(), x[(2 * t + i).rem_euclid(n) as usize].clone())
          })
          .collect(),
      )
    })
    .collect()
}

fn symbolic_swt_step(
  x: &[Expr],
  filter: &[(i64, Expr)],
  dilation: i64,
) -> Vec<Expr> {
  let n = x.len() as i64;
  (0..n)
    .map(|t| {
      symbolic_dot(
        filter
          .iter()
          .map(|(i, c)| {
            (
              c.clone(),
              x[(t + dilation * i).rem_euclid(n) as usize].clone(),
            )
          })
          .collect(),
      )
    })
    .collect()
}

/// Full symbolic 1D forward transform (Periodic padding only).
fn symbolic_forward(
  data: &[Expr],
  spec: &WaveletSpec,
  kind: TransformKind,
  r: usize,
) -> Option<Vec<(Vec<u8>, Expr)>> {
  let (primal, dual) = exact_filters(spec)?;
  let hi = super::filters::highpass_from_exact(&primal);
  let mut out: Vec<(Vec<u8>, Vec<Expr>)> = Vec::new();
  let data: Vec<Expr> = if kind == TransformKind::Lwt {
    let mult = 1usize << r;
    let target = data.len().div_ceil(mult) * mult;
    (0..target).map(|i| data[i % data.len()].clone()).collect()
  } else {
    data.to_vec()
  };
  let mut frontier: Vec<(Vec<u8>, Vec<Expr>)> = vec![(vec![], data)];
  for level in 0..r {
    let mut next = Vec::new();
    for (wind, arr) in &frontier {
      if !(kind.is_packet() || wind.iter().all(|&d| d == 0)) {
        continue;
      }
      let (lo_child, hi_child) = if kind.is_stationary() {
        (
          symbolic_swt_step(arr, &dual, 1 << level),
          symbolic_swt_step(arr, &hi, 1 << level),
        )
      } else {
        (symbolic_dwt_step(arr, &dual), symbolic_dwt_step(arr, &hi))
      };
      let mut w0 = wind.clone();
      w0.push(0);
      let mut w1 = wind.clone();
      w1.push(1);
      next.push((w0, lo_child));
      next.push((w1, hi_child));
    }
    next.sort_by(|a, b| a.0.cmp(&b.0));
    out.extend(next.iter().cloned());
    frontier = next;
  }
  Some(
    out
      .into_iter()
      .map(|(w, coefs)| (w, Expr::List(coefs.into())))
      .collect(),
  )
}

// ---------------------------------------------------------------------------
// Forward transform entry points
// ---------------------------------------------------------------------------

/// Parse the data argument into either a numeric array or (for 1D lists
/// with symbolic entries) the raw expression list.
enum DataArg {
  Numeric(CoefArray),
  Symbolic(Vec<Expr>),
}

fn parse_data(e: &Expr) -> Option<DataArg> {
  if let Some(arr) = CoefArray::from_expr(e) {
    // Reject empty and ragged input.
    match &arr {
      CoefArray::D1(v) if v.is_empty() => return None,
      CoefArray::D2(m)
        if m.is_empty() || m.iter().any(|r| r.len() != m[0].len()) =>
      {
        return None;
      }
      _ => {}
    }
    return Some(DataArg::Numeric(arr));
  }
  if let Expr::List(items) = e
    && !items.is_empty()
    && items.iter().all(|i| !matches!(i, Expr::List(_)))
  {
    return Some(DataArg::Symbolic(items.to_vec()));
  }
  None
}

/// Shared driver for the five forward transforms.
pub fn wavelet_transform_ast(
  kind: TransformKind,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let fname = kind.name();
  let mut positional: Vec<&Expr> = Vec::new();
  let mut padding = Padding::Periodic;
  let mut exact_requested = false;
  for a in args {
    if let Expr::Rule {
      pattern,
      replacement,
    } = a
      && let Expr::Identifier(opt) = pattern.as_ref()
    {
      match opt.as_str() {
        "Padding" if !kind.is_stationary() => {
          match Padding::parse(replacement) {
            Some(p) => padding = p,
            None => {
              crate::emit_message(&format!(
                "{}::invpad: {} is not a valid Padding setting.",
                fname,
                expr_to_string(replacement)
              ));
              return Ok(unevaluated(fname, args));
            }
          }
          continue;
        }
        "WorkingPrecision" => {
          if matches!(replacement.as_ref(), Expr::Identifier(v) if v == "Infinity")
          {
            exact_requested = true;
          }
          continue;
        }
        "Method" => continue,
        _ => {}
      }
    }
    positional.push(a);
  }
  if positional.is_empty() || positional.len() > 3 {
    crate::emit_message(&format!(
      "{fname}::argt: {fname} called with an invalid number of arguments."
    ));
    return Ok(unevaluated(fname, args));
  }
  let Some(data) = parse_data(positional[0]) else {
    crate::emit_message(&format!(
      "{fname}::invdata: The data {} is not a rectangular array of numbers of rank 1 or 2.",
      expr_to_string(positional[0])
    ));
    return Ok(unevaluated(fname, args));
  };
  let default_wavelet = Expr::FunctionCall {
    name: "HaarWavelet".to_string(),
    args: vec![].into(),
  };
  let wavelet_expr = match positional.get(1) {
    Some(Expr::Identifier(a)) if a == "Automatic" => default_wavelet,
    Some(w) => {
      // A LiftingFilterData[wave] wraps the wavelet it was derived from.
      if let Expr::FunctionCall { name, args: wargs } = w
        && name == "LiftingFilterData"
        && wargs.len() == 1
      {
        wargs[0].clone()
      } else {
        (*w).clone()
      }
    }
    None => default_wavelet,
  };
  let Some(spec) = parse_discrete_wavelet(&wavelet_expr) else {
    crate::emit_message(&format!(
      "{}::invw: {} is not a valid wavelet.",
      fname,
      expr_to_string(&wavelet_expr)
    ));
    return Ok(unevaluated(fname, args));
  };
  let Some(filters) = super::wavelet_filters(&spec) else {
    return Ok(unevaluated(fname, args));
  };

  let dims = match &data {
    DataArg::Numeric(arr) => arr.dims(),
    DataArg::Symbolic(v) => vec![v.len()],
  };
  let min_dim = dims.iter().copied().min().unwrap_or(0);
  if min_dim < 2 {
    crate::emit_message(&format!(
      "{fname}::invdata: The data should have at least two elements in every dimension."
    ));
    return Ok(unevaluated(fname, args));
  }
  let r = match positional.get(2) {
    None => default_refinement(kind, &dims),
    Some(Expr::Identifier(a)) if a == "Automatic" => {
      default_refinement(kind, &dims)
    }
    Some(Expr::Identifier(f)) if f == "Full" => {
      ((min_dim as f64).log2() + 0.5).floor().max(1.0) as usize
    }
    Some(Expr::Integer(n)) if *n >= 1 => *n as usize,
    Some(other) => {
      crate::emit_message(&format!(
        "{}::invr: The refinement level {} is not a positive integer.",
        fname,
        expr_to_string(other)
      ));
      return Ok(unevaluated(fname, args));
    }
  };
  // Sanity cap: the coarsest level must still hold data.
  let max_r = ((min_dim as f64).log2() + 0.5).floor().max(1.0) as usize;
  if !kind.is_stationary() && r > max_r.max(1) + 8 {
    crate::emit_message(&format!(
      "{fname}::invr: The refinement level {r} is too large for the given data."
    ));
    return Ok(unevaluated(fname, args));
  }

  let rules: Vec<(Vec<u8>, Expr)> = match &data {
    DataArg::Symbolic(items) => match symbolic_forward(items, &spec, kind, r) {
      Some(rules) => rules,
      None => {
        crate::emit_message(&format!(
          "{}::exact: Exact computation is not supported for the wavelet {}.",
          fname,
          expr_to_string(&wavelet_expr)
        ));
        return Ok(unevaluated(fname, args));
      }
    },
    DataArg::Numeric(arr) => {
      if exact_requested
        && let Some(exact_input) = exact_list(positional[0])
        && dims.len() == 1
        && let Some(rules) = symbolic_forward(&exact_input, &spec, kind, r)
      {
        rules
      } else {
        forward_transform(arr, &filters, kind, r, &padding)
          .into_iter()
          .map(|(w, c)| (w, c.to_expr()))
          .collect()
      }
    }
  };

  let dwd = Dwd {
    rules,
    wavelet: wavelet_expr,
    kind,
    padding,
    basis_override: None,
    threshold_values: None,
    dims,
  };
  Ok(dwd.to_expr())
}

/// The elements of a 1D list when they are all exact (no Real/BigFloat).
fn exact_list(e: &Expr) -> Option<Vec<Expr>> {
  let Expr::List(items) = e else { return None };
  fn is_exact(e: &Expr) -> bool {
    match e {
      Expr::Real(_) | Expr::BigFloat(_, _) => false,
      Expr::Integer(_) | Expr::BigInteger(_) | Expr::Identifier(_) => true,
      Expr::FunctionCall { args, .. } => args.iter().all(is_exact),
      Expr::List(items) => items.iter().all(is_exact),
      Expr::BinaryOp { left, right, .. } => is_exact(left) && is_exact(right),
      Expr::UnaryOp { operand, .. } => is_exact(operand),
      _ => false,
    }
  }
  if items.iter().all(is_exact) {
    Some(items.to_vec())
  } else {
    None
  }
}

// ---------------------------------------------------------------------------
// InverseWaveletTransform
// ---------------------------------------------------------------------------

pub fn inverse_wavelet_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let fname = "InverseWaveletTransform";
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
  let Some(dwd) = Dwd::from_expr(positional[0]) else {
    crate::emit_message(&format!(
      "{}::invdwd: {} is not a valid DiscreteWaveletData object.",
      fname,
      expr_to_string(positional[0])
    ));
    return Ok(unevaluated(fname, args));
  };
  let wavelet_expr = match positional.get(1) {
    None => dwd.wavelet.clone(),
    Some(Expr::Identifier(a)) if a == "Automatic" => dwd.wavelet.clone(),
    Some(w) => (*w).clone(),
  };
  let Some(spec) = parse_discrete_wavelet(&wavelet_expr) else {
    crate::emit_message(&format!(
      "{}::invw: {} is not a valid wavelet.",
      fname,
      expr_to_string(&wavelet_expr)
    ));
    return Ok(unevaluated(fname, args));
  };
  let Some(filters) = super::wavelet_filters(&spec) else {
    return Ok(unevaluated(fname, args));
  };

  // Partial inverse: an integer r collapses the deepest r levels and
  // returns a new DiscreteWaveletData (r < refinement) or the data (r =
  // refinement).
  if let Some(Expr::Integer(rr)) = positional.get(2).copied() {
    let n = dwd.refinement() as i128;
    if *rr >= 1 && *rr < n {
      return partial_inverse(&dwd, &filters, *rr as usize);
    }
  }

  // Which coefficients to use.
  let winds: Vec<Vec<u8>> = match positional.get(2) {
    None => dwd.basis(),
    Some(Expr::Identifier(a)) if a == "Automatic" => dwd.basis(),
    Some(Expr::Identifier(a)) if a == "All" => {
      dwd.rules.iter().map(|(w, _)| w.clone()).collect()
    }
    Some(Expr::Integer(rr)) if *rr as usize == dwd.refinement() => dwd.basis(),
    Some(spec_expr) => match select_winds(&dwd, spec_expr) {
      Some(w) => w,
      None => {
        crate::emit_message(&format!(
          "{}::invwind: {} is not a valid wavelet index specification.",
          fname,
          expr_to_string(spec_expr)
        ));
        return Ok(unevaluated(fname, args));
      }
    },
  };

  // Exact path: symbolic coefficients + exact filters.
  let all_exact = winds
    .iter()
    .all(|w| dwd.coef(w).map(|c| exact_list(c).is_some()).unwrap_or(true));
  if all_exact
    && dwd.rank() == 1
    && let Some(result) = symbolic_inverse(&dwd, &spec, &winds)
  {
    return Ok(result);
  }

  let mut coeffs: BTreeMap<Vec<u8>, CoefArray> = BTreeMap::new();
  for w in &winds {
    if let Some(c) = dwd.coef(w)
      && let Some(arr) = CoefArray::from_expr(c)
    {
      coeffs.insert(w.clone(), arr);
    }
  }
  let result = inverse_transform(
    &coeffs,
    &dwd.dims,
    &filters,
    dwd.kind,
    dwd.refinement(),
    &dwd.padding,
  );
  Ok(result.to_expr())
}

/// Collapse the deepest levels of the tree: reconstruct the node {0,…,0} at
/// level `keep` from everything below it and return a new dwd with
/// refinement `keep`.
fn partial_inverse(
  dwd: &Dwd,
  filters: &super::filters::WaveletFilters,
  r: usize,
) -> Result<Expr, InterpreterError> {
  let n = dwd.refinement();
  let keep = n - r;
  let mut coeffs: BTreeMap<Vec<u8>, CoefArray> = BTreeMap::new();
  for w in dwd.basis() {
    if w.len() > keep
      && let Some(c) = dwd.coef(&w)
      && let Some(arr) = CoefArray::from_expr(c)
    {
      coeffs.insert(w.clone(), arr);
    }
  }
  // Reconstruct each level-`keep` interior node from the deeper basis nodes.
  let mut new_rules: Vec<(Vec<u8>, Expr)> = dwd
    .rules
    .iter()
    .filter(|(w, _)| w.len() <= keep)
    .cloned()
    .collect();
  let roots: std::collections::BTreeSet<Vec<u8>> = coeffs
    .keys()
    .map(|w| w[..keep.min(w.len())].to_vec())
    .collect();
  for root in roots {
    if dwd.coef(&root).is_some() {
      continue;
    }
    // Shift the subtree below `root` to be its own transform tree.
    let sub: BTreeMap<Vec<u8>, CoefArray> = coeffs
      .iter()
      .filter(|(w, _)| w.starts_with(&root))
      .map(|(w, c)| (w[root.len()..].to_vec(), c.clone()))
      .collect();
    if sub.is_empty() {
      continue;
    }
    let dims = node_dims(&root, &dwd.dims, filters, dwd.kind, n, &dwd.padding);
    let rec = inverse_transform(
      &sub,
      &dims,
      filters,
      dwd.kind,
      n - root.len(),
      &dwd.padding,
    );
    new_rules.push((root, rec.to_expr()));
  }
  new_rules.sort_by(|a, b| (a.0.len(), &a.0).cmp(&(b.0.len(), &b.0)));
  let new_dwd = Dwd {
    rules: new_rules,
    wavelet: dwd.wavelet.clone(),
    kind: dwd.kind,
    padding: dwd.padding.clone(),
    basis_override: None,
    threshold_values: dwd.threshold_values.clone(),
    dims: dwd.dims.clone(),
  };
  Ok(new_dwd.to_expr())
}

/// Symbolic inverse for exact 1D coefficient trees (Haar and the other
/// rational-filter families); returns None when exact filters are missing.
fn symbolic_inverse(
  dwd: &Dwd,
  spec: &WaveletSpec,
  winds: &[Vec<u8>],
) -> Option<Expr> {
  let (primal, dual) = exact_filters(spec)?;
  let synth_lo = primal.clone();
  let synth_hi = super::filters::highpass_from_exact(&dual);
  let analysis_lo = dual;
  let analysis_hi = super::filters::highpass_from_exact(&primal);
  let filters_num = super::wavelet_filters(spec)?;
  let r = dwd.refinement();
  let n0 = dwd.dims[0];

  let mut map: BTreeMap<Vec<u8>, Vec<Expr>> = BTreeMap::new();
  for w in winds {
    if let Some(Expr::List(items)) = dwd.coef(w) {
      map.insert(w.clone(), items.to_vec());
    }
  }

  fn zero_vec(n: usize) -> Vec<Expr> {
    vec![Expr::Integer(0); n]
  }

  #[allow(clippy::too_many_arguments)]
  fn reconstruct(
    wind: &[u8],
    map: &BTreeMap<Vec<u8>, Vec<Expr>>,
    dwd: &Dwd,
    filters_num: &super::filters::WaveletFilters,
    synth: (&[(i64, Expr)], &[(i64, Expr)]),
    analysis: (&[(i64, Expr)], &[(i64, Expr)]),
    r: usize,
  ) -> Option<Vec<Expr>> {
    if let Some(c) = map.get(wind) {
      return Some(c.clone());
    }
    if wind.len() >= r {
      return None;
    }
    let mut w0 = wind.to_vec();
    w0.push(0);
    let mut w1 = wind.to_vec();
    w1.push(1);
    let c0 = reconstruct(&w0, map, dwd, filters_num, synth, analysis, r);
    let c1 = reconstruct(&w1, map, dwd, filters_num, synth, analysis, r);
    if c0.is_none() && c1.is_none() {
      return None;
    }
    let dims =
      node_dims(wind, &dwd.dims, filters_num, dwd.kind, r, &dwd.padding);
    let n = dims[0];
    let child_dims =
      node_dims(&w0, &dwd.dims, filters_num, dwd.kind, r, &dwd.padding);
    let a = c0.unwrap_or_else(|| zero_vec(child_dims[0]));
    let d = c1.unwrap_or_else(|| zero_vec(child_dims[0]));
    Some(symbolic_idwt_step(
      &a,
      &d,
      n,
      synth,
      analysis,
      dwd.kind,
      wind.len(),
    ))
  }

  let result = reconstruct(
    &[],
    &map,
    dwd,
    &filters_num,
    (&synth_lo, &synth_hi),
    (&analysis_lo, &analysis_hi),
    r,
  )?;
  let mut result = result;
  result.truncate(n0);
  Some(eval(Expr::List(result.into())))
}

/// One symbolic inverse level (1D).
fn symbolic_idwt_step(
  a: &[Expr],
  d: &[Expr],
  n: usize,
  synth: (&[(i64, Expr)], &[(i64, Expr)]),
  analysis: (&[(i64, Expr)], &[(i64, Expr)]),
  kind: TransformKind,
  level: usize,
) -> Vec<Expr> {
  let (synth_lo, synth_hi) = synth;
  let (analysis_lo, analysis_hi) = analysis;
  let inv_sqrt2 = eval(Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(2)].into(),
      },
      Expr::Integer(-1),
    ]
    .into(),
  });
  if kind.is_stationary() {
    let dilation = 1i64 << level;
    let nn = a.len() as i64;
    return (0..nn)
      .map(|t| {
        let mut terms: Vec<Expr> = Vec::new();
        for (i, c) in synth_lo.iter() {
          terms.push(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              c.clone(),
              a[(t - dilation * i).rem_euclid(nn) as usize].clone(),
            ]
            .into(),
          });
        }
        for (i, c) in synth_hi.iter() {
          terms.push(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              c.clone(),
              d[(t - dilation * i).rem_euclid(nn) as usize].clone(),
            ]
            .into(),
          });
        }
        eval(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            inv_sqrt2.clone(),
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: terms.into(),
            },
          ]
          .into(),
        })
      })
      .collect();
  }
  // Decimated: take the canonical circular slice, then circular synthesis.
  let m = n.div_ceil(2);
  let n_even = 2 * m;
  let _ = (analysis_lo, analysis_hi);
  let a = a.to_vec();
  let d = d.to_vec();
  let mut terms: Vec<Vec<Expr>> = vec![Vec::new(); n_even];
  for (t, (at, dt)) in a.iter().zip(d.iter()).enumerate() {
    for (i, c) in synth_lo.iter() {
      let j = (2 * t as i64 + i).rem_euclid(n_even as i64) as usize;
      terms[j].push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![c.clone(), at.clone()].into(),
      });
    }
    for (i, c) in synth_hi.iter() {
      let j = (2 * t as i64 + i).rem_euclid(n_even as i64) as usize;
      terms[j].push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![c.clone(), dt.clone()].into(),
      });
    }
  }
  let mut out: Vec<Expr> = terms
    .into_iter()
    .map(|ts| {
      eval(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          sqrt2(),
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: ts.into(),
          },
        ]
        .into(),
      })
    })
    .collect();
  out.truncate(n);
  out
}

// ---------------------------------------------------------------------------
// Wind selection (indices, index lists, patterns)
// ---------------------------------------------------------------------------

/// Resolve a wavelet-index specification against the stored winds:
/// a single wind {0,1}, a list of winds {{0},{1}}, or a pattern ({0,_}).
pub fn select_winds(dwd: &Dwd, spec: &Expr) -> Option<Vec<Vec<u8>>> {
  // Single explicit wind.
  if let Some(w) = expr_to_wind(spec) {
    if !w.is_empty() {
      return Some(vec![w]);
    }
    return Some(vec![]);
  }
  // List of explicit winds.
  if let Expr::List(items) = spec {
    let winds: Option<Vec<Vec<u8>>> = items.iter().map(expr_to_wind).collect();
    if let Some(winds) = winds {
      return Some(winds);
    }
  }
  // Pattern: match against every stored wind.
  let mut matched = Vec::new();
  for (w, _) in &dwd.rules {
    let wind_expr = wind_to_expr(w);
    if crate::evaluator::pattern_matching::match_pattern(&wind_expr, spec)
      .is_some()
    {
      matched.push(w.clone());
    }
  }
  if matched.is_empty() {
    None
  } else {
    Some(matched)
  }
}

// ---------------------------------------------------------------------------
// dwd[...] property and coefficient access (curried application)
// ---------------------------------------------------------------------------

pub fn apply_dwd(func: &Expr, args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Some(dwd) = Dwd::from_expr(func) else {
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
  let form = match args.get(1) {
    None => "Rules",
    Some(Expr::String(f)) => f.as_str(),
    Some(_) => "Rules",
  };

  // String properties.
  if let Expr::String(prop) = &args[0] {
    return dwd_property(&dwd, prop);
  }

  let winds: Vec<Vec<u8>> = match &args[0] {
    Expr::Identifier(a) if a == "All" => {
      dwd.rules.iter().map(|(w, _)| w.clone()).collect()
    }
    Expr::Identifier(a) if a == "Automatic" => dwd.basis(),
    spec => match select_winds(&dwd, spec) {
      Some(w) => w,
      None => {
        crate::emit_message(&format!(
          "DiscreteWaveletData::wind: {} is not a valid wavelet index specification.",
          expr_to_string(&args[0])
        ));
        return Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        });
      }
    },
  };

  let single_explicit = expr_to_wind(&args[0])
    .map(|w| !w.is_empty())
    .unwrap_or(false);

  let mut out: Vec<Expr> = Vec::new();
  for w in &winds {
    let Some(coef) = dwd.coef(w) else { continue };
    let value = match form {
      "Values" => coef.clone(),
      "Inverse" => {
        // Inverse transform of this coefficient alone.
        let sub = Dwd {
          rules: vec![(w.clone(), coef.clone())],
          wavelet: dwd.wavelet.clone(),
          kind: dwd.kind,
          padding: dwd.padding.clone(),
          basis_override: None,
          threshold_values: None,
          dims: dwd.dims.clone(),
        };
        let iargs = vec![
          sub.to_expr(),
          Expr::Identifier("Automatic".into()),
          wind_to_expr(w),
        ];
        inverse_wavelet_transform_ast(&iargs)?
      }
      _ => coef.clone(),
    };
    if form == "Rules" {
      out.push(Expr::Rule {
        pattern: Box::new(wind_to_expr(w)),
        replacement: Box::new(value),
      });
    } else {
      out.push(value);
    }
  }
  if single_explicit && out.len() == 1 && form != "Rules" {
    return Ok(out.pop().unwrap());
  }
  Ok(Expr::List(out.into()))
}

fn dwd_property(dwd: &Dwd, prop: &str) -> Result<Expr, InterpreterError> {
  match prop {
    "Properties" => Ok(Expr::List(
      [
        "BasisIndex",
        "DataDimensions",
        "Dimensions",
        "EnergyFraction",
        "Padding",
        "Refinement",
        "ThresholdValues",
        "Transform",
        "Wavelet",
        "WaveletIndex",
      ]
      .iter()
      .map(|s| Expr::String(s.to_string()))
      .collect::<Vec<_>>()
      .into(),
    )),
    "BasisIndex" => Ok(Expr::List(
      dwd
        .basis()
        .iter()
        .map(|w| wind_to_expr(w))
        .collect::<Vec<_>>()
        .into(),
    )),
    "WaveletIndex" => Ok(Expr::List(
      dwd
        .rules
        .iter()
        .map(|(w, _)| wind_to_expr(w))
        .collect::<Vec<_>>()
        .into(),
    )),
    "Refinement" => Ok(Expr::Integer(dwd.refinement() as i128)),
    "Wavelet" => Ok(dwd.wavelet.clone()),
    "Transform" => Ok(Expr::String(dwd.kind.name().to_string())),
    "Padding" => Ok(dwd.padding.to_expr()),
    "DataDimensions" => Ok(dims_to_expr(&dwd.dims)),
    "Dimensions" => Ok(Expr::List(
      dwd
        .rules
        .iter()
        .filter_map(|(w, c)| {
          CoefArray::from_expr(c).map(|arr| Expr::Rule {
            pattern: Box::new(wind_to_expr(w)),
            replacement: Box::new(dims_to_expr(&arr.dims())),
          })
        })
        .collect::<Vec<_>>()
        .into(),
    )),
    "EnergyFraction" => {
      let basis = dwd.basis();
      let total: f64 = basis
        .iter()
        .filter_map(|w| dwd.coef(w))
        .filter_map(CoefArray::from_expr)
        .map(|a| a.energy())
        .sum();
      if total <= 0.0 {
        return Ok(Expr::List(vec![].into()));
      }
      Ok(Expr::List(
        dwd
          .rules
          .iter()
          .filter_map(|(w, c)| {
            CoefArray::from_expr(c).map(|arr| Expr::Rule {
              pattern: Box::new(wind_to_expr(w)),
              replacement: Box::new(Expr::Real(arr.energy() / total)),
            })
          })
          .collect::<Vec<_>>()
          .into(),
      ))
    }
    "ThresholdValues" => Ok(
      dwd
        .threshold_values
        .clone()
        .unwrap_or_else(|| Expr::List(vec![].into())),
    ),
    _ => {
      crate::emit_message(&format!(
        "DiscreteWaveletData::prop: {prop} is not a valid property."
      ));
      Ok(Expr::FunctionCall {
        name: "Missing".to_string(),
        args: vec![Expr::String("NotAvailable".into())].into(),
      })
    }
  }
}

// ---------------------------------------------------------------------------
// WaveletThreshold
// ---------------------------------------------------------------------------

struct ThresholdFun {
  kind: String,
  delta_spec: Option<Expr>,
  extra: Vec<f64>,
}

/// Expand the short forms from the WaveletThreshold reference table.
fn parse_tspec(tspec: &Expr) -> Option<ThresholdFun> {
  let named = |kind: &str, delta: &str| ThresholdFun {
    kind: kind.to_string(),
    delta_spec: Some(Expr::String(delta.to_string())),
    extra: vec![],
  };
  match tspec {
    Expr::String(s) => match s.as_str() {
      "Universal" => Some(named("Hard", "Universal")),
      "UniversalLevel" => Some(named("Hard", "UniversalLevel")),
      "VisuShrink" => Some(named("Soft", "Universal")),
      "VisuShrinkLevel" => Some(named("Soft", "UniversalLevel")),
      "Hard" | "Soft" | "Firm" | "PiecewiseGarrote" | "SmoothGarrote"
      | "Hyperbola" => Some(named(s, "Universal")),
      _ => None,
    },
    Expr::List(items) if !items.is_empty() => {
      let Expr::String(kind) = &items[0] else {
        return None;
      };
      let known = matches!(
        kind.as_str(),
        "Hard"
          | "Soft"
          | "Firm"
          | "PiecewiseGarrote"
          | "SmoothGarrote"
          | "Hyperbola"
          | "LargestCoefficients"
      );
      if !known {
        return None;
      }
      let delta_spec = items.get(1).cloned();
      let extra: Vec<f64> = items.iter().skip(2).filter_map(num).collect();
      Some(ThresholdFun {
        kind: kind.clone(),
        delta_spec,
        extra,
      })
    }
    _ => None,
  }
}

fn median(sorted: &mut [f64]) -> f64 {
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let n = sorted.len();
  if n == 0 {
    return 0.0;
  }
  if n % 2 == 1 {
    sorted[n / 2]
  } else {
    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
  }
}

/// Donoho–Johnstone noise estimate: MedianDeviation of the finest detail
/// coefficients divided by Quantile[NormalDistribution[], 3/4].
fn sigma_mad(values: &[f64]) -> f64 {
  let mut v = values.to_vec();
  let med = median(&mut v);
  let mut dev: Vec<f64> = values.iter().map(|x| (x - med).abs()).collect();
  median(&mut dev) / 0.6744897501960817
}

pub fn wavelet_threshold_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let fname = "WaveletThreshold";
  if args.is_empty() || args.len() > 3 {
    crate::emit_message(&format!(
      "{fname}::argt: {fname} called with an invalid number of arguments."
    ));
    return Ok(unevaluated(fname, args));
  }
  let Some(dwd) = Dwd::from_expr(&args[0]) else {
    crate::emit_message(&format!(
      "{}::invdwd: {} is not a valid DiscreteWaveletData object.",
      fname,
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated(fname, args));
  };
  let default_tspec = Expr::String("Universal".into());
  let tspec = args.get(1).unwrap_or(&default_tspec);
  let Some(tfun) = parse_tspec(tspec) else {
    crate::emit_message(&format!(
      "{}::invtspec: {} is not a valid threshold specification.",
      fname,
      expr_to_string(tspec)
    ));
    return Ok(unevaluated(fname, args));
  };

  // Default: detail coefficients (winds containing a nonzero digit) among
  // the basis coefficients.
  let winds: Vec<Vec<u8>> = match args.get(2) {
    None => dwd
      .basis()
      .into_iter()
      .filter(|w| w.iter().any(|&d| d != 0))
      .collect(),
    Some(Expr::Identifier(a)) if a == "Automatic" => dwd
      .basis()
      .into_iter()
      .filter(|w| w.iter().any(|&d| d != 0))
      .collect(),
    Some(Expr::Identifier(a)) if a == "All" => {
      dwd.rules.iter().map(|(w, _)| w.clone()).collect()
    }
    Some(spec) => match select_winds(&dwd, spec) {
      Some(w) => w,
      None => {
        crate::emit_message(&format!(
          "{}::invwind: {} is not a valid wavelet index specification.",
          fname,
          expr_to_string(spec)
        ));
        return Ok(unevaluated(fname, args));
      }
    },
  };

  // Global noise estimate from the finest detail coefficients.
  let n_data: usize = dwd.dims.iter().product();
  let finest: Vec<f64> = dwd
    .rules
    .iter()
    .filter(|(w, _)| w.len() == 1 && w.iter().any(|&d| d != 0))
    .filter_map(|(_, c)| CoefArray::from_expr(c))
    .flat_map(|arr| match arr {
      CoefArray::D1(v) => v,
      CoefArray::D2(m) => m.into_iter().flatten().collect(),
    })
    .collect();
  let universal_delta =
    sigma_mad(&finest) * (2.0 * (n_data as f64).ln()).sqrt();

  let delta_for = |values: &[f64]| -> Option<f64> {
    match &tfun.delta_spec {
      None => Some(universal_delta),
      Some(e) => {
        if let Some(d) = num(e) {
          return Some(d);
        }
        match e {
          Expr::String(s) if s == "Universal" => Some(universal_delta),
          Expr::Identifier(s) if s == "Automatic" => Some(universal_delta),
          Expr::String(s) if s == "UniversalLevel" => {
            Some(sigma_mad(values) * (2.0 * (n_data as f64).ln()).sqrt())
          }
          _ => None,
        }
      }
    }
  };

  // "LargestCoefficients" keeps the k largest |x| across all selected winds.
  let mut keep_threshold: Option<f64> = None;
  if tfun.kind == "LargestCoefficients" {
    let k = tfun
      .delta_spec
      .as_ref()
      .and_then(num)
      .map(|v| v as usize)
      .unwrap_or(1);
    let mut all: Vec<f64> = Vec::new();
    for w in &winds {
      if let Some(c) = dwd.coef(w)
        && let Some(arr) = CoefArray::from_expr(c)
      {
        match arr {
          CoefArray::D1(v) => all.extend(v.iter().map(|x| x.abs())),
          CoefArray::D2(m) => all.extend(m.iter().flatten().map(|x| x.abs())),
        }
      }
    }
    all.sort_by(|a, b| b.partial_cmp(a).unwrap());
    keep_threshold = Some(if k == 0 || all.is_empty() {
      f64::INFINITY
    } else {
      all[(k - 1).min(all.len() - 1)]
    });
  }

  let apply = |x: f64, delta: f64| -> f64 {
    match tfun.kind.as_str() {
      "Hard" => {
        if x.abs() <= delta {
          0.0
        } else {
          x
        }
      }
      "Soft" => {
        if x.abs() <= delta {
          0.0
        } else {
          x.signum() * (x.abs() - delta)
        }
      }
      "Firm" => {
        let r = tfun.extra.first().copied().unwrap_or(1.0);
        let p = tfun.extra.get(1).copied().unwrap_or(0.5);
        let lo = delta - delta * p * r;
        let hi = lo + delta * r;
        if x.abs() <= lo {
          0.0
        } else if x.abs() <= hi {
          x.signum() * (delta + delta * r - delta * p * r) * (x.abs() - lo)
            / (delta * r)
        } else {
          x
        }
      }
      "PiecewiseGarrote" => {
        if x.abs() <= delta {
          0.0
        } else {
          x - delta * delta / x
        }
      }
      "SmoothGarrote" => {
        let n = tfun.extra.first().copied().unwrap_or(1.0);
        x.powf(2.0 * n + 1.0) / (x.powf(2.0 * n) + delta.powf(2.0 * n))
      }
      "Hyperbola" => {
        if x.abs() <= delta {
          0.0
        } else {
          x.signum() * (x * x - delta * delta).sqrt()
        }
      }
      "LargestCoefficients" => {
        if x.abs() >= keep_threshold.unwrap_or(f64::INFINITY) {
          x
        } else {
          0.0
        }
      }
      _ => x,
    }
  };

  let mut new_rules = dwd.rules.clone();
  let mut threshold_rules: Vec<Expr> = Vec::new();
  for w in &winds {
    let Some(pos) = new_rules.iter().position(|(rw, _)| rw == w) else {
      continue;
    };
    let Some(arr) = CoefArray::from_expr(&new_rules[pos].1) else {
      continue;
    };
    let values: Vec<f64> = match &arr {
      CoefArray::D1(v) => v.clone(),
      CoefArray::D2(m) => m.iter().flatten().copied().collect(),
    };
    let Some(delta) = delta_for(&values) else {
      crate::emit_message(&format!(
        "{}::invtspec: {} is not a supported threshold value specification.",
        fname,
        expr_to_string(tspec)
      ));
      return Ok(unevaluated(fname, args));
    };
    let new_arr = match arr {
      CoefArray::D1(v) => {
        CoefArray::D1(v.into_iter().map(|x| apply(x, delta)).collect())
      }
      CoefArray::D2(m) => CoefArray::D2(
        m.into_iter()
          .map(|row| row.into_iter().map(|x| apply(x, delta)).collect())
          .collect(),
      ),
    };
    new_rules[pos].1 = new_arr.to_expr();
    threshold_rules.push(Expr::Rule {
      pattern: Box::new(wind_to_expr(w)),
      replacement: Box::new(Expr::Real(delta)),
    });
  }

  let new_dwd = Dwd {
    rules: new_rules,
    wavelet: dwd.wavelet.clone(),
    kind: dwd.kind,
    padding: dwd.padding.clone(),
    basis_override: dwd.basis_override.clone(),
    threshold_values: Some(Expr::List(threshold_rules.into())),
    dims: dwd.dims.clone(),
  };
  Ok(new_dwd.to_expr())
}

// ---------------------------------------------------------------------------
// WaveletMapIndexed
// ---------------------------------------------------------------------------

pub fn wavelet_map_indexed_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let fname = "WaveletMapIndexed";
  if args.len() < 2 || args.len() > 3 {
    crate::emit_message(&format!(
      "{fname}::argt: {fname} called with an invalid number of arguments."
    ));
    return Ok(unevaluated(fname, args));
  }
  let f = &args[0];
  if let Some(dwd) = Dwd::from_expr(&args[1]) {
    let winds: Vec<Vec<u8>> = match args.get(2) {
      None => dwd.rules.iter().map(|(w, _)| w.clone()).collect(),
      Some(Expr::Identifier(a)) if a == "All" => {
        dwd.rules.iter().map(|(w, _)| w.clone()).collect()
      }
      Some(Expr::Identifier(a)) if a == "Automatic" => dwd.basis(),
      Some(spec) => match select_winds(&dwd, spec) {
        Some(w) => w,
        None => {
          crate::emit_message(&format!(
            "{}::invwind: {} is not a valid wavelet index specification.",
            fname,
            expr_to_string(args.get(2).unwrap())
          ));
          return Ok(unevaluated(fname, args));
        }
      },
    };
    let mut new_rules = dwd.rules.clone();
    for w in &winds {
      let Some(pos) = new_rules.iter().position(|(rw, _)| rw == w) else {
        continue;
      };
      let coef = new_rules[pos].1.clone();
      let mapped = crate::evaluator::function_application::apply_curried_call(
        f,
        &[coef, wind_to_expr(w)],
      )?;
      new_rules[pos].1 = mapped;
    }
    let new_dwd = Dwd {
      rules: new_rules,
      wavelet: dwd.wavelet.clone(),
      kind: dwd.kind,
      padding: dwd.padding.clone(),
      basis_override: dwd.basis_override.clone(),
      threshold_values: dwd.threshold_values.clone(),
      dims: dwd.dims.clone(),
    };
    return Ok(new_dwd.to_expr());
  }
  if let Some(result) =
    super::continuous::cwd_map_indexed(f, &args[1], args.get(2))?
  {
    return Ok(result);
  }
  crate::emit_message(&format!(
    "{}::invwd: {} is not a valid DiscreteWaveletData or ContinuousWaveletData object.",
    fname,
    expr_to_string(&args[1])
  ));
  Ok(unevaluated(fname, args))
}

// ---------------------------------------------------------------------------
// WaveletBestBasis
// ---------------------------------------------------------------------------

pub fn wavelet_best_basis_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let fname = "WaveletBestBasis";
  if args.is_empty() || args.len() > 2 {
    crate::emit_message(&format!(
      "{fname}::argt: {fname} called with an invalid number of arguments."
    ));
    return Ok(unevaluated(fname, args));
  }
  let Some(dwd) = Dwd::from_expr(&args[0]) else {
    crate::emit_message(&format!(
      "{}::invdwd: {} is not a valid DiscreteWaveletData object.",
      fname,
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated(fname, args));
  };
  if !dwd.kind.is_packet() {
    crate::emit_message(&format!(
      "{fname}::invtrans: The wavelet data should come from a wavelet packet transform."
    ));
    return Ok(unevaluated(fname, args));
  }
  let default_cspec = Expr::String("ShannonEntropy".into());
  let cspec = args.get(1).unwrap_or(&default_cspec);

  let cost = |values: &[f64]| -> Option<f64> {
    match cspec {
      Expr::String(s) if s == "ShannonEntropy" => Some(
        -values
          .iter()
          .map(|&w| {
            let w2 = w * w;
            if w2 > 0.0 { w2 * w2.ln() } else { 0.0 }
          })
          .sum::<f64>(),
      ),
      Expr::String(s) if s == "LogEnergy" => Some(
        values
          .iter()
          .map(|&w| {
            let w2 = w * w;
            if w2 > 0.0 { w2.ln() } else { 0.0 }
          })
          .sum(),
      ),
      Expr::List(items)
        if items.len() == 2
          && matches!(&items[0], Expr::String(s) if s == "Norm") =>
      {
        let p = num(&items[1])?;
        let s: f64 = values.iter().map(|w| w.abs().powf(p)).sum();
        Some(if p < 2.0 { s } else { -s })
      }
      Expr::List(items)
        if items.len() == 2
          && matches!(&items[0], Expr::String(s) if s == "Threshold") =>
      {
        let delta = num(&items[1])?;
        Some(values.iter().filter(|w| w.abs() > delta).count() as f64)
      }
      _ => None,
    }
  };
  let user_fn_cost = |values_expr: &Expr| -> Option<f64> {
    let applied = crate::evaluator::function_application::apply_curried_call(
      cspec,
      &[values_expr.clone()],
    )
    .ok()?;
    num(&applied)
  };

  // Cost per stored node.
  let mut node_cost: BTreeMap<Vec<u8>, f64> = BTreeMap::new();
  for (w, c) in &dwd.rules {
    let arr = CoefArray::from_expr(c);
    let flat: Option<Vec<f64>> = arr.map(|a| match a {
      CoefArray::D1(v) => v,
      CoefArray::D2(m) => m.into_iter().flatten().collect(),
    });
    let cost_val = match &flat {
      Some(values) => cost(values).or_else(|| user_fn_cost(c)),
      None => user_fn_cost(c),
    };
    let Some(cv) = cost_val else {
      crate::emit_message(&format!(
        "{}::invcspec: {} is not a valid cost specification.",
        fname,
        expr_to_string(cspec)
      ));
      return Ok(unevaluated(fname, args));
    };
    node_cost.insert(w.clone(), cv);
  }

  // Bottom-up best-basis selection (Coifman–Wickerhauser).
  let r = dwd.refinement();
  let digits: Vec<u8> = if dwd.rank() >= 2 {
    vec![0, 1, 2, 3]
  } else {
    vec![0, 1]
  };
  fn best(
    wind: &[u8],
    r: usize,
    digits: &[u8],
    node_cost: &BTreeMap<Vec<u8>, f64>,
  ) -> Option<(f64, Vec<Vec<u8>>)> {
    let own = node_cost.get(wind).copied();
    if wind.len() >= r {
      return own.map(|c| (c, vec![wind.to_vec()]));
    }
    let mut child_total = 0.0;
    let mut child_basis: Vec<Vec<u8>> = Vec::new();
    let mut have_children = true;
    for &d in digits {
      let mut w = wind.to_vec();
      w.push(d);
      match best(&w, r, digits, node_cost) {
        Some((c, b)) => {
          child_total += c;
          child_basis.extend(b);
        }
        None => {
          have_children = false;
          break;
        }
      }
    }
    match (own, have_children) {
      (Some(o), true) => {
        if o <= child_total {
          Some((o, vec![wind.to_vec()]))
        } else {
          Some((child_total, child_basis))
        }
      }
      (Some(o), false) => Some((o, vec![wind.to_vec()])),
      (None, true) => Some((child_total, child_basis)),
      (None, false) => None,
    }
  }
  let mut basis: Vec<Vec<u8>> = Vec::new();
  for &d in &digits {
    if let Some((_, b)) = best(&[d], r, &digits, &node_cost) {
      basis.extend(b);
    }
  }
  basis.sort();

  let new_dwd = Dwd {
    rules: dwd.rules.clone(),
    wavelet: dwd.wavelet.clone(),
    kind: dwd.kind,
    padding: dwd.padding.clone(),
    basis_override: Some(basis),
    threshold_values: dwd.threshold_values.clone(),
    dims: dwd.dims.clone(),
  };
  Ok(new_dwd.to_expr())
}
