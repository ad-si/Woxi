//! Wavelet visualization: WaveletScalogram, WaveletListPlot,
//! WaveletMatrixPlot, and WaveletImagePlot. These assemble coefficient
//! layouts and delegate the rendering to the existing plotting/image
//! machinery (ArrayPlot, ListLinePlot, MatrixPlot, Image).

use super::continuous::Cwd;
use super::data::{Dwd, select_winds};
use super::transforms::CoefArray;
#[allow(unused_imports)]
use super::*;

fn call(name: &str, args: &[Expr]) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_function_call_ast(name, args)
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
  let positional: Vec<&Expr> = args.iter().filter(|a| !is_option(a)).collect();
  if positional.is_empty() || positional.len() > 3 {
    crate::emit_message(&format!(
      "{fname}::argt: {fname} called with an invalid number of arguments."
    ));
    return Ok(unevaluated(fname, args));
  }
  let func = positional.get(2).copied();

  // Rows of |coefficients|, stretched to the data length.
  let mut rows: Vec<Vec<f64>> = Vec::new();
  // For continuous data: the first row of each octave, which the
  // automatic vertical ticks label with the octave number.
  let mut octave_rows: Vec<(i64, usize)> = Vec::new();
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
      let Some(values) = apply_func(func, coef, super::data::wind_to_expr(w))
      else {
        continue;
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
    for ((octave, _), coefs) in &cwd.rules {
      if !octave_rows.iter().any(|(o, _)| o == octave) {
        octave_rows.push((*octave, rows.len()));
      }
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
  let n_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
  let options: Vec<&Expr> = args
    .iter()
    .filter(|a| matches!(a, Expr::Rule { .. }))
    .collect();
  let option = |name: &str| {
    options.iter().rev().find_map(|o| match o {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == name) => {
        Some(replacement.as_ref().clone())
      }
      _ => None,
    })
  };
  let rule = |name: &str, value: Expr| Expr::Rule {
    pattern: Box::new(Expr::Identifier(name.to_string())),
    replacement: Box::new(value),
  };
  // Without a frame there are no frame ticks either.
  let frameless = matches!(option("Frame").as_ref(),
    Some(Expr::Identifier(f)) if f == "False" || f == "None");
  let frame_ticks = if frameless {
    Expr::Identifier("None".to_string())
  } else {
    scalogram_frame_ticks(option("FrameTicks").as_ref(), n_cols, &octave_rows)
  };
  let matrix = Expr::List(
    rows
      .into_iter()
      .map(|r| {
        Expr::List(r.into_iter().map(Expr::Real).collect::<Vec<_>>().into())
      })
      .collect::<Vec<_>>()
      .into(),
  );
  // A scalogram is a framed, landscape picture of the coefficient
  // magnitudes (time across, scale down), whatever the matrix shape; the
  // remaining plot options (ColorFunction, ImageSize, …) go to the
  // underlying ArrayPlot unchanged.
  let mut plot_args = vec![matrix];
  for (name, default) in [
    (
      "AspectRatio",
      call(
        "Power",
        &[Expr::Identifier("GoldenRatio".into()), Expr::Integer(-1)],
      )?,
    ),
    ("PlotRangePadding", call("Scaled", &[Expr::Real(0.02)])?),
    ("Frame", Expr::Identifier("True".into())),
  ] {
    plot_args.push(rule(name, option(name).unwrap_or(default)));
  }
  plot_args.push(rule("FrameTicks", frame_ticks));
  plot_args.extend(
    options
      .into_iter()
      .filter(|o| {
        !matches!(o, Expr::Rule { pattern, .. }
          if matches!(pattern.as_ref(), Expr::Identifier(n)
            if matches!(n.as_str(), "AspectRatio" | "PlotRangePadding" | "Frame" | "FrameTicks")))
      })
      .cloned(),
  );
  call("ArrayPlot", &plot_args)
}

/// The scalogram's `FrameTicks` as explicit `{{left, right}, {bottom,
/// top}}` tick lists in `ArrayPlot` coordinates (the center of column `j`
/// and of row `i` from the top sit at `j` and `i`).
///
/// Scalogram coordinates run along the samples horizontally (the left edge
/// of the first sample is 0) and along the octaves vertically (octave `o`
/// starts at `o`). By default the octaves are labeled on the right edge and
/// the sample positions along the bottom; a two-element spec gives the
/// vertical and the horizontal edges, each either a `{first, second}` pair
/// or one spec for both edges, where `Automatic` keeps that edge's default.
fn scalogram_frame_ticks(
  spec: Option<&Expr>,
  n_cols: usize,
  octave_rows: &[(i64, usize)],
) -> Expr {
  let none = || Expr::Identifier("None".to_string());
  let tick =
    |pos: f64, label: Expr| Expr::List(vec![Expr::Real(pos), label].into());
  let octave_ticks = || {
    if octave_rows.is_empty() {
      return none();
    }
    Expr::List(
      octave_rows
        .iter()
        .map(|(o, row)| tick(*row as f64 + 0.5, Expr::Integer(*o as i128)))
        .collect::<Vec<_>>()
        .into(),
    )
  };
  let sample_ticks = || {
    if n_cols == 0 {
      return none();
    }
    let span = n_cols as f64;
    let step = crate::functions::plot::nice_step(span, 5).max(1.0);
    let mut ticks = Vec::new();
    let mut k = 0.0;
    while k * step <= span + 1e-9 {
      let v = k * step;
      ticks.push(tick(v + 0.5, Expr::Integer(v.round() as i128)));
      k += 1.0;
    }
    Expr::List(ticks.into())
  };
  // A scalogram's octave position `o` begins at the first row of octave
  // `o`; positions between octaves interpolate linearly.
  let octave_pos = |o: f64| -> f64 {
    let voices = match octave_rows {
      [(_, a), (_, b), ..] => (*b - *a) as f64,
      _ => 1.0,
    };
    let first = octave_rows.first().map_or(1, |(o, _)| *o) as f64;
    (o - first) * voices + 0.5
  };
  let is = |e: &Expr, names: &[&str]| matches!(e, Expr::Identifier(n) if names.contains(&n.as_str()));
  // One edge's ticks: its default, none, or explicit positions mapped
  // into ArrayPlot coordinates.
  let edge = |e: &Expr, default: &dyn Fn() -> Expr, vertical: bool| -> Expr {
    if is(e, &["Automatic", "True", "All"]) {
      return default();
    }
    let Expr::List(items) = e else {
      return none();
    };
    let map_pos = |p: f64| if vertical { octave_pos(p) } else { p + 0.5 };
    Expr::List(
      items
        .iter()
        .filter_map(|item| match item {
          Expr::List(parts) if !parts.is_empty() => {
            let p = crate::functions::math_ast::try_eval_to_f64(&parts[0])?;
            let mut out = parts.to_vec();
            out[0] = Expr::Real(map_pos(p));
            Some(Expr::List(out.into()))
          }
          other => {
            let p = crate::functions::math_ast::try_eval_to_f64(other)?;
            Some(tick(map_pos(p), other.clone()))
          }
        })
        .collect::<Vec<_>>()
        .into(),
    )
  };
  let automatic = Expr::Identifier("Automatic".to_string());
  let (vertical, horizontal) = match spec {
    None => (automatic.clone(), automatic.clone()),
    Some(e) if is(e, &["Automatic", "True", "All"]) => {
      (automatic.clone(), automatic.clone())
    }
    Some(Expr::List(items)) if items.len() == 2 => {
      (items[0].clone(), items[1].clone())
    }
    Some(_) => (none(), none()),
  };
  // Split a spec into its two edges; a lone spec covers both, with
  // `Automatic` meaning each edge's own default.
  // A two-element list is an edge pair when one of its parts is a
  // symbolic spec or a whole tick list; otherwise it is itself a list of
  // two ticks.
  let split = |e: &Expr| match e {
    Expr::List(p)
      if p.len() == 2
        && p.iter().any(|x| match x {
          Expr::Identifier(_) => true,
          Expr::List(t) => t.iter().any(|y| matches!(y, Expr::List(_))),
          _ => false,
        }) =>
    {
      (p[0].clone(), p[1].clone())
    }
    other => (other.clone(), other.clone()),
  };
  let (left, right) = split(&vertical);
  let (bottom, top) = split(&horizontal);
  Expr::List(
    vec![
      Expr::List(
        vec![edge(&left, &none, true), edge(&right, &octave_ticks, true)]
          .into(),
      ),
      Expr::List(
        vec![
          edge(&bottom, &sample_ticks, false),
          edge(&top, &none, false),
        ]
        .into(),
      ),
    ]
    .into(),
  )
}

/// Whether a call argument is an option (`name -> value` or `name :> value`).
fn is_option(e: &Expr) -> bool {
  matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. })
}

/// The plot data followed by every option rule of the original call.
fn with_plot_options(data: Expr, args: &[Expr]) -> Vec<Expr> {
  let mut plot_args = vec![data];
  plot_args.extend(args.iter().filter(|a| is_option(a)).cloned());
  plot_args
}

// ---------------------------------------------------------------------------
// WaveletListPlot
// ---------------------------------------------------------------------------

pub fn wavelet_list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let fname = "WaveletListPlot";
  let positional: Vec<&Expr> = args.iter().filter(|a| !is_option(a)).collect();
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
  call(
    "ListLinePlot",
    &with_plot_options(Expr::List(series.into()), args),
  )
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
      .map_or(0, std::vec::Vec::len)
      .max(hl.first().map_or(0, std::vec::Vec::len));
    let cols_right = lh
      .first()
      .map_or(0, std::vec::Vec::len)
      .max(hh.first().map_or(0, std::vec::Vec::len));
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
  let positional: Vec<&Expr> = args.iter().filter(|a| !is_option(a)).collect();
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
  // Plot options (ColorFunction, Frame, ImageSize, …) go to the underlying
  // plot / image constructor unchanged.
  call(
    if image { "Image" } else { "MatrixPlot" },
    &with_plot_options(matrix_expr, args),
  )
}

pub fn wavelet_matrix_plot_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  matrix_or_image_plot("WaveletMatrixPlot", args, false)
}

pub fn wavelet_image_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  matrix_or_image_plot("WaveletImagePlot", args, true)
}
