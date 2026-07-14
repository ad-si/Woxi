//! Wavelet analysis: wavelet families, filter coefficients, discrete,
//! stationary, packet, lifting, and continuous wavelet transforms, plus
//! coefficient manipulation and visualization.

pub mod continuous;
pub mod data;
pub mod filters;
pub mod phipsi;
pub mod plots;
pub mod tables;
pub mod transforms;

use crate::InterpreterError;
use crate::syntax::{Expr, expr_to_string, unevaluated};

/// A validated discrete wavelet family (one that has filter coefficients
/// and works with the discrete transforms).
#[derive(Clone, Debug, PartialEq)]
pub enum WaveletSpec {
  Haar,
  Daubechies(usize),
  Symlet(usize),
  Coiflet(usize),
  BattleLemarie(u32, f64),
  BiorthogonalSpline(u32, u32),
  ReverseBiorthogonalSpline(u32, u32),
  Cdf(bool), // true = "9/7", false = "5/3"
  Meyer(u32, f64),
  Shannon(f64),
}

/// A continuous wavelet family used by ContinuousWaveletTransform,
/// WaveletPsi, and WaveletPhi. Parameters are kept as expressions so the
/// symbolic formulas stay exact.
#[derive(Clone, Debug)]
pub enum ContinuousWaveletSpec {
  MexicanHat(Expr),
  Morlet,
  Paul(Expr),
  DGaussian(Expr),
  Gabor(Expr),
}

fn pos_int(e: &Expr) -> Option<i128> {
  match e {
    Expr::Integer(n) if *n > 0 => Some(*n),
    _ => None,
  }
}

fn pos_real(e: &Expr) -> Option<f64> {
  crate::functions::math_ast::expr_to_num(e).filter(|v| *v > 0.0)
}

/// Recognize (and validate) a discrete wavelet family expression,
/// filling in the documented default parameters.
pub fn parse_discrete_wavelet(e: &Expr) -> Option<WaveletSpec> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  match (name.as_str(), args.len()) {
    ("HaarWavelet", 0) => Some(WaveletSpec::Haar),
    ("DaubechiesWavelet", 0) => Some(WaveletSpec::Daubechies(2)),
    ("DaubechiesWavelet", 1) => {
      let n = pos_int(&args[0])?;
      // Root finding at machine precision degrades for very large orders.
      if n <= 60 {
        Some(WaveletSpec::Daubechies(n as usize))
      } else {
        None
      }
    }
    ("SymletWavelet", 0) => Some(WaveletSpec::Symlet(4)),
    ("SymletWavelet", 1) => {
      let n = pos_int(&args[0])?;
      if n < 20 {
        Some(WaveletSpec::Symlet(n as usize))
      } else {
        None
      }
    }
    ("CoifletWavelet", 0) => Some(WaveletSpec::Coiflet(2)),
    ("CoifletWavelet", 1) => {
      let n = pos_int(&args[0])?;
      if n <= 5 {
        Some(WaveletSpec::Coiflet(n as usize))
      } else {
        None
      }
    }
    ("BattleLemarieWavelet", 0) => Some(WaveletSpec::BattleLemarie(3, 10.0)),
    ("BattleLemarieWavelet", 1) => {
      let n = pos_int(&args[0])?;
      if n < 15 {
        Some(WaveletSpec::BattleLemarie(n as u32, 10.0))
      } else {
        None
      }
    }
    ("BattleLemarieWavelet", 2) => {
      let n = pos_int(&args[0])?;
      let lim = pos_real(&args[1])?;
      if n < 15 {
        Some(WaveletSpec::BattleLemarie(n as u32, lim))
      } else {
        None
      }
    }
    ("BiorthogonalSplineWavelet", 0) => {
      Some(WaveletSpec::BiorthogonalSpline(4, 2))
    }
    ("BiorthogonalSplineWavelet", 2) => {
      let n = pos_int(&args[0])?;
      let m = pos_int(&args[1])?;
      if (n + m) % 2 == 0 && n <= 9 && m <= 9 {
        Some(WaveletSpec::BiorthogonalSpline(n as u32, m as u32))
      } else {
        None
      }
    }
    ("ReverseBiorthogonalSplineWavelet", 0) => {
      Some(WaveletSpec::ReverseBiorthogonalSpline(4, 2))
    }
    ("ReverseBiorthogonalSplineWavelet", 2) => {
      let n = pos_int(&args[0])?;
      let m = pos_int(&args[1])?;
      if (n + m) % 2 == 0 && n <= 9 && m <= 9 {
        Some(WaveletSpec::ReverseBiorthogonalSpline(n as u32, m as u32))
      } else {
        None
      }
    }
    ("CDFWavelet", 0) => Some(WaveletSpec::Cdf(true)),
    ("CDFWavelet", 1) => match &args[0] {
      Expr::String(s) if s == "9/7" || s == "CDF9/7" => {
        Some(WaveletSpec::Cdf(true))
      }
      Expr::String(s) if s == "5/3" || s == "CDF5/3" => {
        Some(WaveletSpec::Cdf(false))
      }
      _ => None,
    },
    ("MeyerWavelet", 0) => Some(WaveletSpec::Meyer(3, 8.0)),
    ("MeyerWavelet", 1) => {
      let n = pos_int(&args[0])?;
      Some(WaveletSpec::Meyer(n as u32, 8.0))
    }
    ("MeyerWavelet", 2) => {
      let n = pos_int(&args[0])?;
      let lim = pos_real(&args[1])?;
      Some(WaveletSpec::Meyer(n as u32, lim))
    }
    ("ShannonWavelet", 0) => Some(WaveletSpec::Shannon(10.0)),
    ("ShannonWavelet", 1) => Some(WaveletSpec::Shannon(pos_real(&args[0])?)),
    _ => None,
  }
}

/// Recognize a continuous wavelet family expression with defaults.
pub fn parse_continuous_wavelet(e: &Expr) -> Option<ContinuousWaveletSpec> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  match (name.as_str(), args.len()) {
    ("MexicanHatWavelet", 0) => {
      Some(ContinuousWaveletSpec::MexicanHat(Expr::Integer(1)))
    }
    ("MexicanHatWavelet", 1) => {
      Some(ContinuousWaveletSpec::MexicanHat(args[0].clone()))
    }
    ("MorletWavelet", 0) => Some(ContinuousWaveletSpec::Morlet),
    ("PaulWavelet", 0) => Some(ContinuousWaveletSpec::Paul(Expr::Integer(4))),
    ("PaulWavelet", 1) => Some(ContinuousWaveletSpec::Paul(args[0].clone())),
    ("DGaussianWavelet", 0) => {
      Some(ContinuousWaveletSpec::DGaussian(Expr::Integer(2)))
    }
    ("DGaussianWavelet", 1) => {
      Some(ContinuousWaveletSpec::DGaussian(args[0].clone()))
    }
    ("GaborWavelet", 0) => Some(ContinuousWaveletSpec::Gabor(Expr::Integer(6))),
    ("GaborWavelet", 1) => Some(ContinuousWaveletSpec::Gabor(args[0].clone())),
    _ => None,
  }
}

/// Canonical expression for a discrete wavelet spec.
pub fn spec_to_expr(spec: &WaveletSpec) -> Expr {
  let call = |name: &str, args: Vec<Expr>| Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  };
  let int = |n: i128| Expr::Integer(n);
  match spec {
    WaveletSpec::Haar => call("HaarWavelet", vec![]),
    WaveletSpec::Daubechies(n) => {
      call("DaubechiesWavelet", vec![int(*n as i128)])
    }
    WaveletSpec::Symlet(n) => call("SymletWavelet", vec![int(*n as i128)]),
    WaveletSpec::Coiflet(n) => call("CoifletWavelet", vec![int(*n as i128)]),
    WaveletSpec::BattleLemarie(n, lim) => call(
      "BattleLemarieWavelet",
      vec![int(*n as i128), Expr::Real(*lim)],
    ),
    WaveletSpec::BiorthogonalSpline(n, m) => call(
      "BiorthogonalSplineWavelet",
      vec![int(*n as i128), int(*m as i128)],
    ),
    WaveletSpec::ReverseBiorthogonalSpline(n, m) => call(
      "ReverseBiorthogonalSplineWavelet",
      vec![int(*n as i128), int(*m as i128)],
    ),
    WaveletSpec::Cdf(lossy) => call(
      "CDFWavelet",
      vec![Expr::String(if *lossy { "9/7" } else { "5/3" }.to_string())],
    ),
    WaveletSpec::Meyer(n, lim) => {
      call("MeyerWavelet", vec![int(*n as i128), Expr::Real(*lim)])
    }
    WaveletSpec::Shannon(lim) => call("ShannonWavelet", vec![Expr::Real(*lim)]),
  }
}

/// Filters for a validated discrete wavelet spec.
pub fn wavelet_filters(spec: &WaveletSpec) -> Option<filters::WaveletFilters> {
  match spec {
    WaveletSpec::Haar => Some(filters::haar_filters()),
    WaveletSpec::Daubechies(n) => Some(filters::daubechies_filters(*n)),
    WaveletSpec::Symlet(n) => filters::symlet_filters(*n),
    WaveletSpec::Coiflet(n) => filters::coiflet_filters(*n),
    WaveletSpec::BattleLemarie(n, lim) => {
      Some(filters::battle_lemarie_filters(*n, *lim))
    }
    WaveletSpec::BiorthogonalSpline(n, m) => {
      Some(filters::biorthogonal_spline_filters(*n, *m))
    }
    WaveletSpec::ReverseBiorthogonalSpline(n, m) => {
      Some(filters::reverse_biorthogonal_spline_filters(*n, *m))
    }
    WaveletSpec::Cdf(lossy) => Some(if *lossy {
      filters::cdf_97_filters()
    } else {
      filters::cdf_53_filters()
    }),
    WaveletSpec::Meyer(n, lim) => Some(filters::meyer_filters(*n, *lim)),
    WaveletSpec::Shannon(lim) => Some(filters::shannon_filters(*lim)),
  }
}

/// WaveletFilterCoefficients[wave], [wave, filtspec] — filter coefficients
/// as {{n, c_n}, …} pairs. Exact values are returned for the families with
/// closed-form filters; other families give machine-precision values.
pub fn wavelet_filter_coefficients_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Strip options (WorkingPrecision) from the tail.
  let mut positional: Vec<&Expr> = Vec::new();
  let mut exact_requested = false;
  for a in args {
    if let Expr::Rule {
      pattern,
      replacement,
    } = a
    {
      if let Expr::Identifier(opt) = pattern.as_ref()
        && opt == "WorkingPrecision"
        && matches!(replacement.as_ref(), Expr::Identifier(v) if v == "Infinity")
      {
        exact_requested = true;
      }
      continue;
    }
    positional.push(a);
  }
  if positional.is_empty() || positional.len() > 2 {
    crate::emit_message(
      "WaveletFilterCoefficients::argt: WaveletFilterCoefficients called with an invalid number of arguments.",
    );
    return Ok(unevaluated("WaveletFilterCoefficients", args));
  }
  let Some(spec) = parse_discrete_wavelet(positional[0]) else {
    crate::emit_message(&format!(
      "WaveletFilterCoefficients::invw: {} is not a valid wavelet.",
      expr_to_string(positional[0])
    ));
    return Ok(unevaluated("WaveletFilterCoefficients", args));
  };
  let default_spec = Expr::String("PrimalLowpass".to_string());
  let filt_spec = positional.get(1).copied().unwrap_or(&default_spec);

  // A list of filter specifications maps to a list of results.
  if let Expr::List(specs) = filt_spec {
    let mut out = Vec::new();
    for s in specs.iter() {
      out.push(single_filter_result(&spec, s, exact_requested, args)?);
    }
    return Ok(Expr::List(out.into()));
  }
  single_filter_result(&spec, filt_spec, exact_requested, args)
}

fn single_filter_result(
  spec: &WaveletSpec,
  filt_spec: &Expr,
  exact_requested: bool,
  orig_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let Expr::String(kind) = filt_spec else {
    crate::emit_message(&format!(
      "WaveletFilterCoefficients::invspec: {} is not a valid filter specification.",
      expr_to_string(filt_spec)
    ));
    return Ok(unevaluated("WaveletFilterCoefficients", orig_args));
  };
  if kind == "LiftingFilter" {
    return Ok(transforms::lifting_filter_data(spec));
  }
  let filters = match wavelet_filters(spec) {
    Some(f) => f,
    None => return Ok(unevaluated("WaveletFilterCoefficients", orig_args)),
  };
  // For the biorthogonal spline families Wolfram labels the longer
  // (complementary) filter "Primal" and the shorter B-spline filter "Dual" —
  // the opposite of the internal naming — and reports the highpass filters
  // with flipped sign. Remap the requested spec accordingly.
  let is_bior = matches!(
    spec,
    WaveletSpec::BiorthogonalSpline(_, _)
      | WaveletSpec::ReverseBiorthogonalSpline(_, _)
  );
  let eff_kind: &str = if is_bior {
    match kind.as_str() {
      "PrimalLowpass" => "DualLowpass",
      "DualLowpass" => "PrimalLowpass",
      "PrimalHighpass" => "DualHighpass",
      "DualHighpass" => "PrimalHighpass",
      other => other,
    }
  } else {
    kind.as_str()
  };
  let bior_highpass_flip =
    is_bior && matches!(kind.as_str(), "PrimalHighpass" | "DualHighpass");
  let (mut numeric, mut exact): (
    filters::Filter,
    Option<filters::ExactFilter>,
  ) = match eff_kind {
    "PrimalLowpass" => {
      (filters.primal_lo.clone(), filters.primal_lo_exact.clone())
    }
    "DualLowpass" => (filters.dual_lo.clone(), filters.dual_lo_exact.clone()),
    "PrimalHighpass" => (
      filters::highpass_from(&filters.dual_lo),
      filters
        .dual_lo_exact
        .as_ref()
        .map(filters::highpass_from_exact),
    ),
    "DualHighpass" => (
      filters::highpass_from(&filters.primal_lo),
      filters
        .primal_lo_exact
        .as_ref()
        .map(filters::highpass_from_exact),
    ),
    _ => {
      crate::emit_message(&format!(
        "WaveletFilterCoefficients::invspec: \"{}\" is not a valid filter specification.",
        kind
      ));
      return Ok(unevaluated("WaveletFilterCoefficients", orig_args));
    }
  };
  if bior_highpass_flip {
    numeric = filters::negate_filter(&numeric);
    exact = exact.as_ref().map(filters::negate_filter_exact);
  }

  // Machine-precision values are the default (matching Wolfram); exact
  // closed-form coefficients are only produced on WorkingPrecision -> Infinity.
  let use_exact = exact.is_some() && exact_requested;
  let pairs: Vec<Expr> = if use_exact {
    exact
      .unwrap()
      .into_iter()
      .map(|(i, c)| Expr::List(vec![Expr::Integer(i as i128), c].into()))
      .collect()
  } else {
    numeric
      .into_iter()
      .map(|(i, c)| {
        Expr::List(vec![Expr::Integer(i as i128), Expr::Real(c)].into())
      })
      .collect()
  };
  Ok(Expr::List(pairs.into()))
}
