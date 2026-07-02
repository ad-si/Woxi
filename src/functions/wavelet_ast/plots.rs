//! Wavelet visualization: WaveletScalogram, WaveletListPlot,
//! WaveletMatrixPlot, and WaveletImagePlot. These assemble coefficient
//! layouts and delegate the rendering to the existing plotting/image
//! machinery (ArrayPlot, ListLinePlot, MatrixPlot, Image).

use super::continuous::Cwd;
use super::data::{Dwd, select_winds};
use super::transforms::CoefArray;
use super::unevaluated;
use crate::InterpreterError;
use crate::syntax::{Expr, expr_to_string};

fn call(name: &str, args: Vec<Expr>) -> Result<Expr, InterpreterError> {
  crate::evaluator::dispatch::evaluate_function_call_ast(name, &args)
}

fn apply_func(
  func: Option<&Expr>,
  coef: &Expr,
  wind: Expr,
) -> Option<Vec<f64>> {
  let mapped = match func {
    Some(f) => crate::evaluator::function_application::apply_curried_call(
      f,
      &[coef.clone(), wind],
    )
    .ok()?,
    None => coef.clone(),
  };
  match CoefArray::from_expr(&mapped)? {
    CoefArray::D1(v) => Some(v),
    CoefArray::D2(m) => Some(m.into_iter().flatten().collect()),
  }
}

/// Winds to plot: the basis by default, or an explicit/pattern spec.
fn plot_winds(dwd: &Dwd, spec: Option<&Expr>) -> Option<Vec<Vec<u8>>> {
  match spec {
    None => Some(dwd.basis()),
    Some(Expr::Identifier(a)) if a == "Automatic" => Some(dwd.basis()),
    Some(Expr::Identifier(a)) if a == "All" => {
      Some(dwd.rules.iter().map(|(w, _)| w.clone()).collect())
    }
    Some(s) => select_winds(dwd, s),
  }
}

// ---------------------------------------------------------------------------
// WaveletScalogram
// ---------------------------------------------------------------------------

pub fn wavelet_scalogram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let fname = "WaveletScalogram";
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
  let func = positional.get(2).copied();

  // Rows of |coefficients|, stretched to the data length.
  let mut rows: Vec<Vec<f64>> = Vec::new();
  if let Some(dwd) = Dwd::from_expr(positional[0]) {
    if dwd.rank() != 1 {
      crate::emit_message(&format!(
        "{fname}::invdata: WaveletScalogram requires wavelet data from a 1D transform."
      ));
      return Ok(unevaluated(fname, args));
    }
    let Some(winds) = plot_winds(&dwd, positional.get(1).copied()) else {
      crate::emit_message(&format!(
        "{}::invwind: {} is not a valid wavelet index specification.",
        fname,
        expr_to_string(positional[1])
      ));
      return Ok(unevaluated(fname, args));
    };
    let n = dwd.dims[0].max(1);
    // Finest detail on top, coarser below (rows ordered by refinement).
    let mut winds = winds;
    winds.sort_by_key(|w| (w.len(), w.clone()));
    for w in &winds {
      let Some(coef) = dwd.coef(w) else { continue };
      let values = match apply_func(func, coef, super::data::wind_to_expr(w)) {
        Some(v) => v,
        None => continue,
      };
      let values: Vec<f64> = if func.is_none() {
        values.iter().map(|v| v.abs()).collect()
      } else {
        values
      };
      if values.is_empty() {
        continue;
      }
      let stretched: Vec<f64> =
        (0..n).map(|j| values[(j * values.len()) / n]).collect();
      rows.push(stretched);
    }
  } else if let Some(cwd) = Cwd::from_expr(positional[0]) {
    for (_, coefs) in &cwd.rules {
      rows.push(
        coefs
          .iter()
          .map(|(re, im)| (re * re + im * im).sqrt())
          .collect(),
      );
    }
  } else {
    crate::emit_message(&format!(
      "{}::invwd: {} is not a valid DiscreteWaveletData or ContinuousWaveletData object.",
      fname,
      expr_to_string(positional[0])
    ));
    return Ok(unevaluated(fname, args));
  }
  if rows.is_empty() {
    return Ok(unevaluated(fname, args));
  }
  let matrix = Expr::List(
    rows
      .into_iter()
      .map(|r| {
        Expr::List(r.into_iter().map(Expr::Real).collect::<Vec<_>>().into())
      })
      .collect::<Vec<_>>()
      .into(),
  );
  call("ArrayPlot", vec![matrix])
}

// ---------------------------------------------------------------------------
// WaveletListPlot
// ---------------------------------------------------------------------------

pub fn wavelet_list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let fname = "WaveletListPlot";
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
  if dwd.rank() != 1 {
    crate::emit_message(&format!(
      "{fname}::invdata: WaveletListPlot works for DiscreteWaveletData coming from 1D data."
    ));
    return Ok(unevaluated(fname, args));
  }
  let func = positional.get(2).copied();
  let Some(mut winds) = plot_winds(&dwd, positional.get(1).copied()) else {
    crate::emit_message(&format!(
      "{}::invwind: {} is not a valid wavelet index specification.",
      fname,
      expr_to_string(positional[1])
    ));
    return Ok(unevaluated(fname, args));
  };
  winds.sort_by_key(|w| (w.len(), w.clone()));
  let n = dwd.dims[0].max(1) as f64;
  // Stack against a common x axis: each coefficient list occupies its own
  // vertical band, separately scaled (the "CommonXAxis" layout).
  let mut series: Vec<Expr> = Vec::new();
  for (row, w) in winds.iter().enumerate() {
    let Some(coef) = dwd.coef(w) else { continue };
    let Some(values) = apply_func(func, coef, super::data::wind_to_expr(w))
    else {
      continue;
    };
    if values.is_empty() {
      continue;
    }
    let max_abs = values
      .iter()
      .fold(0.0f64, |acc, v| acc.max(v.abs()))
      .max(1e-300);
    let len = values.len() as f64;
    let offset = (winds.len() - 1 - row) as f64;
    let pairs: Vec<Expr> = values
      .iter()
      .enumerate()
      .map(|(j, v)| {
        Expr::List(
          vec![
            Expr::Real((j as f64 + 1.0) * n / len),
            Expr::Real(offset + 0.5 + v / (2.2 * max_abs)),
          ]
          .into(),
        )
      })
      .collect();
    series.push(Expr::List(pairs.into()));
  }
  if series.is_empty() {
    return Ok(unevaluated(fname, args));
  }
  call("ListLinePlot", vec![Expr::List(series.into())])
}

// ---------------------------------------------------------------------------
// WaveletMatrixPlot / WaveletImagePlot
// ---------------------------------------------------------------------------

/// Assemble the pyramid layout matrix for a 2D decimated transform:
/// the coarse block in the top-left corner and the detail blocks around it.
fn pyramid_matrix(
  dwd: &Dwd,
  max_level: usize,
  func: Option<&Expr>,
  normalize_blocks: bool,
) -> Option<Vec<Vec<f64>>> {
  fn block(
    dwd: &Dwd,
    prefix: &[u8],
    level: usize,
    func: Option<&Expr>,
    normalize_blocks: bool,
  ) -> Option<Vec<Vec<f64>>> {
    // Leaf: the stored coefficient array (coarse continues recursing).
    let leaf = |wind: &[u8]| -> Option<Vec<Vec<f64>>> {
      let coef = dwd.coef(wind)?;
      let mut m = match CoefArray::from_expr(coef)? {
        CoefArray::D2(m) => m,
        CoefArray::D1(v) => vec![v],
      };
      if let Some(f) = func {
        let mapped =
          crate::evaluator::function_application::apply_curried_call(
            f,
            &[coef.clone(), super::data::wind_to_expr(wind)],
          )
          .ok()?;
        if let Some(CoefArray::D2(m2)) = CoefArray::from_expr(&mapped) {
          m = m2;
        }
      } else {
        for row in &mut m {
          for v in row {
            *v = v.abs();
          }
        }
      }
      if normalize_blocks {
        let max = m
          .iter()
          .flatten()
          .fold(0.0f64, |acc, v| acc.max(v.abs()))
          .max(1e-300);
        for row in &mut m {
          for v in row {
            *v /= max;
          }
        }
      }
      Some(m)
    };
    if level == 0 {
      return leaf(prefix);
    }
    let sub = |digit: u8| -> Option<Vec<Vec<f64>>> {
      let mut w = prefix.to_vec();
      w.push(digit);
      if digit == 0 && level > 1 {
        block(dwd, &w, level - 1, func, normalize_blocks)
      } else {
        leaf(&w)
      }
    };
    let (ll, lh, hl, hh) = (sub(0)?, sub(1)?, sub(2)?, sub(3)?);
    // Pad to matching block sizes, then assemble [[LL, LH], [HL, HH]].
    let rows_top = ll.len().max(lh.len());
    let rows_bottom = hl.len().max(hh.len());
    let cols_left = ll
      .first()
      .map_or(0, |r| r.len())
      .max(hl.first().map_or(0, |r| r.len()));
    let cols_right = lh
      .first()
      .map_or(0, |r| r.len())
      .max(hh.first().map_or(0, |r| r.len()));
    let mut out = Vec::new();
    for i in 0..rows_top {
      let mut row = Vec::new();
      for j in 0..cols_left {
        row.push(ll.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0));
      }
      for j in 0..cols_right {
        row.push(lh.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0));
      }
      out.push(row);
    }
    for i in 0..rows_bottom {
      let mut row = Vec::new();
      for j in 0..cols_left {
        row.push(hl.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0));
      }
      for j in 0..cols_right {
        row.push(hh.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0));
      }
      out.push(row);
    }
    Some(out)
  }
  block(dwd, &[], max_level, func, normalize_blocks)
}

fn matrix_or_image_plot(
  fname: &str,
  args: &[Expr],
  image: bool,
) -> Result<Expr, InterpreterError> {
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
  if dwd.rank() != 2 {
    crate::emit_message(&format!(
      "{fname}::invdata: {fname} requires wavelet data from a 2D (matrix) transform."
    ));
    return Ok(unevaluated(fname, args));
  }
  let r = match positional.get(1) {
    None => dwd.refinement(),
    Some(Expr::Identifier(a)) if a == "Automatic" => dwd.refinement(),
    Some(Expr::Integer(n)) if *n >= 1 => (*n as usize).min(dwd.refinement()),
    Some(other) => {
      crate::emit_message(&format!(
        "{}::invr: {} is not a valid refinement level.",
        fname,
        expr_to_string(other)
      ));
      return Ok(unevaluated(fname, args));
    }
  };
  let func = positional.get(2).copied();
  let Some(matrix) = pyramid_matrix(&dwd, r, func, image) else {
    crate::emit_message(&format!(
      "{fname}::invdata: The wavelet coefficients could not be assembled into a pyramid layout."
    ));
    return Ok(unevaluated(fname, args));
  };
  let matrix_expr = Expr::List(
    matrix
      .into_iter()
      .map(|row| {
        Expr::List(row.into_iter().map(Expr::Real).collect::<Vec<_>>().into())
      })
      .collect::<Vec<_>>()
      .into(),
  );
  if image {
    call("Image", vec![matrix_expr])
  } else {
    call("MatrixPlot", vec![matrix_expr])
  }
}

pub fn wavelet_matrix_plot_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  matrix_or_image_plot("WaveletMatrixPlot", args, false)
}

pub fn wavelet_image_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  matrix_or_image_plot("WaveletImagePlot", args, true)
}
